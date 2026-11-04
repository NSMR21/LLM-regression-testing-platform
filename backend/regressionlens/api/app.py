"""HTTP API. Every route is scoped to the project of the caller's API token."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .. import __version__, services
from ..config import Settings
from ..db import queue
from ..db.models import Project, Trial
from ..db.session import Database
from ..providers import make_provider
from ..spec import ExecutionSpec, ModelSpec, PromptSpec, SpecError, SuiteSpec

STATIC_DIR = Path(__file__).resolve().parent / "static"


# --- request bodies -------------------------------------------------------------------------


class RunCreate(BaseModel):
    suite: dict[str, Any]
    prompt: dict[str, Any]
    model: dict[str, Any]
    execution: dict[str, Any] = Field(default_factory=dict)
    baseline: str | None = None
    baseline_run_id: str | None = None
    fresh_baseline: bool = False
    ci_meta: dict[str, Any] = Field(default_factory=dict)
    run_seed: int | None = None


class CompareBody(BaseModel):
    baseline_run_id: str
    candidate_run_id: str


class PromoteBody(BaseModel):
    run_id: str
    suite: str | None = None
    by: str | None = None


class RerunBody(BaseModel):
    force: bool = False


class RescoreBody(BaseModel):
    suite: dict[str, Any] | None = None


class LabelBody(BaseModel):
    trial_id: int
    grader_id: str
    label: float = Field(ge=0, le=1)
    labeler: str = Field(min_length=1, max_length=200)
    note: str | None = None


class TokenBody(BaseModel):
    name: str = ""
    scopes: str = "read,run"
    project: str | None = None


# --- auth -----------------------------------------------------------------------------------


@dataclass
class Principal:
    project: Project
    scopes: set[str]


def _session(request: Request) -> Iterator[Session]:
    with request.app.state.db.session() as session:
        yield session


def _principal(
    request: Request,
    session: Session = Depends(_session),
    authorization: str | None = Header(default=None),
    x_rlens_project: str | None = Header(default=None),
) -> Principal:
    settings: Settings = request.app.state.settings
    if authorization and authorization.lower().startswith("bearer "):
        found = services.authenticate(session, authorization.split(" ", 1)[1].strip())
        if found is None:
            raise HTTPException(401, "invalid or revoked API token")
        token, project = found
        return Principal(project, set(token.scopes.split(",")))
    if settings.require_auth:
        raise HTTPException(401, "missing API token (Authorization: Bearer <token>)")
    return Principal(services.ensure_project(session, x_rlens_project or settings.default_project), {"read", "run", "admin"})


def need(scope: str):
    def check(principal: Principal = Depends(_principal)) -> Principal:
        if scope not in principal.scopes:
            raise HTTPException(403, f"token lacks the {scope!r} scope")
        return principal

    return check


def _specs(body: RunCreate) -> tuple[SuiteSpec, PromptSpec, ModelSpec, ExecutionSpec]:
    try:
        return (
            SuiteSpec.model_validate(body.suite),
            PromptSpec.model_validate(body.prompt),
            ModelSpec.model_validate(body.model),
            ExecutionSpec.model_validate(body.execution),
        )
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False)) from exc


# --- app ------------------------------------------------------------------------------------


def create_app(settings: Settings | None = None, db: Database | None = None, provider_factory=make_provider) -> FastAPI:
    settings = settings or Settings()
    db = db or Database(settings.database_url)
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(_startup)
        yield
        db.engine.dispose()

    def _startup() -> None:
        if settings.auto_migrate:
            from ..db.migrate import upgrade

            upgrade(db.url)
        else:
            db.create_all()
        if settings.admin_token:
            with db.session() as s:
                project = services.ensure_project(s, settings.default_project)
                services.create_token(s, project, name="admin (RLENS_ADMIN_TOKEN)", scopes="read,run,admin", token=settings.admin_token)

    app = FastAPI(
        title="RegressionLens", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan
    )
    app.state.settings = settings
    app.state.db = db
    app.state.provider_factory = provider_factory
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(services.ServiceError)
    def _service_error(_request: Request, exc: services.ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(SpecError)
    def _spec_error(_request: Request, exc: SpecError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    url = settings.public_url

    @app.get("/api/v1/health")
    def health(session: Session = Depends(_session)) -> dict[str, Any]:
        session.execute(text("SELECT 1"))
        return {"status": "ok", "version": __version__, "jobs": queue.counts(session)}

    @app.get("/api/v1/me")
    def me(p: Principal = Depends(need("read"))) -> dict[str, Any]:
        return {"project": p.project.name, "scopes": sorted(p.scopes)}

    @app.post("/api/v1/tokens", status_code=201)
    def create_token(body: TokenBody, session: Session = Depends(_session), p: Principal = Depends(need("admin"))) -> dict[str, Any]:
        project = services.ensure_project(session, body.project) if body.project else p.project
        token = services.create_token(session, project, body.name, body.scopes)
        return {"token": token, "project": project.name, "scopes": body.scopes}

    # runs
    @app.post("/api/v1/runs", status_code=202)
    def create_run(body: RunCreate, session: Session = Depends(_session), p: Principal = Depends(need("run"))) -> dict[str, Any]:
        suite, prompt, model, execution = _specs(body)
        req = services.RunRequest(
            suite=suite, prompt=prompt, model=model, execution=execution, baseline=body.baseline,
            baseline_run_id=body.baseline_run_id, fresh_baseline=body.fresh_baseline, ci_meta=body.ci_meta, run_seed=body.run_seed,
        )
        run = services.create_run(session, p.project, req, app.state.provider_factory)
        return services.run_brief(run, url)

    @app.get("/api/v1/runs")
    def list_runs(
        suite: str | None = None, status: str | None = None, branch: str | None = None,
        limit: int = Query(50, le=500), offset: int = 0,
        session: Session = Depends(_session), p: Principal = Depends(need("read")),
    ) -> list[dict[str, Any]]:
        return [services.run_brief(r, url) for r in services.list_runs(session, p.project, suite, status, branch, limit, offset)]

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str, session: Session = Depends(_session), p: Principal = Depends(need("read"))) -> dict[str, Any]:
        run = services.get_run(session, p.project, run_id)
        return {
            **services.run_brief(run, url),
            "summary": run.summary, "verdict_report": run.verdict, "snapshot": run.snapshot,
            "environment": run.environment, "execution": run.execution, "run_seed": run.run_seed,
            "suite_spec": run.suite_version.spec, "prompt_spec": run.prompt_version.spec, "model_spec": run.model_config.spec,
        }

    @app.get("/api/v1/runs/{run_id}/verdict")
    async def get_verdict(run_id: str, wait: float = Query(0, ge=0), p: Principal = Depends(need("read"))) -> dict[str, Any]:
        deadline = time.monotonic() + min(wait, settings.long_poll_max_s)

        def fetch() -> dict[str, Any]:
            with db.session() as s:
                run = services.get_run(s, s.get(Project, p.project.id), run_id)
                return {
                    "run_id": run.id, "status": run.status, "verdict": run.verdict, "summary": run.summary,
                    "error": run.error, "progress": {"total": run.total_trials, "executed": run.executed, "graded": run.graded},
                    "url": f"{url}/runs/{run.id}",
                }

        while True:
            result = await run_in_threadpool(fetch)
            if result["status"] in services.TERMINAL or time.monotonic() >= deadline:
                return result
            await asyncio.sleep(1.0)

    @app.get("/api/v1/runs/{run_id}/trials")
    def get_trials(
        run_id: str, case_key: str | None = None, passed: bool | None = None, limit: int = Query(200, le=2000), offset: int = 0,
        session: Session = Depends(_session), p: Principal = Depends(need("read")),
    ) -> list[dict[str, Any]]:
        services.get_run(session, p.project, run_id)
        stmt = select(Trial).where(Trial.run_id == run_id)
        if case_key:
            stmt = stmt.where(Trial.case_key == case_key)
        if passed is not None:
            stmt = stmt.where(Trial.passed.is_(passed))
        rows = session.scalars(stmt.order_by(Trial.case_key, Trial.repeat_idx).limit(limit).offset(offset))
        return [{"trial_id": t.id, "status": t.status, **services.trial_record(t).model_dump(mode="json")} for t in rows]

    @app.post("/api/v1/runs/{run_id}/cancel")
    def cancel(run_id: str, session: Session = Depends(_session), p: Principal = Depends(need("run"))) -> dict[str, Any]:
        return services.run_brief(services.cancel_run(session, p.project, run_id), url)

    @app.post("/api/v1/runs/{run_id}/rerun", status_code=202)
    def rerun(run_id: str, body: RerunBody | None = None, session: Session = Depends(_session), p: Principal = Depends(need("run"))) -> dict[str, Any]:
        run = services.rerun(session, p.project, run_id, app.state.provider_factory, force=bool(body and body.force))
        return services.run_brief(run, url)

    @app.post("/api/v1/runs/{run_id}/rescore", status_code=202)
    def rescore(run_id: str, body: RescoreBody | None = None, session: Session = Depends(_session), p: Principal = Depends(need("run"))) -> dict[str, Any]:
        suite = SuiteSpec.model_validate(body.suite) if body and body.suite else None
        return services.run_brief(services.rescore(session, p.project, run_id, suite), url)

    @app.post("/api/v1/comparisons")
    def compare(body: CompareBody, session: Session = Depends(_session), p: Principal = Depends(need("read"))) -> dict[str, Any]:
        return services.compare(session, p.project, body.baseline_run_id, body.candidate_run_id)

    # baselines, suites, history
    @app.get("/api/v1/baselines")
    def baselines(suite: str | None = None, session: Session = Depends(_session), p: Principal = Depends(need("read"))) -> list[dict[str, Any]]:
        return services.list_baselines(session, p.project, suite)

    @app.put("/api/v1/baselines/{name}")
    def promote(name: str, body: PromoteBody, session: Session = Depends(_session), p: Principal = Depends(need("run"))) -> dict[str, Any]:
        row = services.promote(session, p.project, body.run_id, name, body.suite, body.by)
        return {"suite": row.suite_name, "name": row.name, "run_id": row.run_id, "updated_at": row.updated_at}

    @app.get("/api/v1/suites")
    def suites(session: Session = Depends(_session), p: Principal = Depends(need("read"))) -> list[dict[str, Any]]:
        return services.list_suites(session, p.project)

    @app.get("/api/v1/suites/{suite}/trends")
    def trends(
        suite: str, metric: str = "pass_rate", branch: str | None = None, limit: int = Query(200, le=1000),
        session: Session = Depends(_session), p: Principal = Depends(need("read")),
    ) -> dict[str, Any]:
        return services.trends(session, p.project, suite, metric, branch, limit)

    @app.get("/api/v1/suites/{suite}/cases/{case_key}/history")
    def case_history(suite: str, case_key: str, limit: int = Query(50, le=500), session: Session = Depends(_session), p: Principal = Depends(need("read"))) -> dict[str, Any]:
        return services.case_history(session, p.project, suite, case_key, limit)

    # labels & calibration
    @app.get("/api/v1/suites/{suite}/graders/{grader_id}/label-queue")
    def label_queue(
        suite: str, grader_id: str, labeler: str, n: int = Query(20, le=200),
        session: Session = Depends(_session), p: Principal = Depends(need("read")),
    ) -> list[dict[str, Any]]:
        return services.label_queue(session, p.project, suite, grader_id, labeler, n)

    @app.post("/api/v1/labels", status_code=201)
    def add_label(body: LabelBody, session: Session = Depends(_session), p: Principal = Depends(need("run"))) -> dict[str, Any]:
        row = services.add_label(session, p.project, body.trial_id, body.grader_id, body.label, body.labeler, body.note)
        return {"id": row.id, "trial_id": row.trial_id, "grader_id": row.grader_id, "label": row.label, "grader_version": row.grader_version}

    @app.get("/api/v1/suites/{suite}/graders/{grader_id}/calibration")
    def calibration(suite: str, grader_id: str, session: Session = Depends(_session), p: Principal = Depends(need("read"))) -> dict[str, Any]:
        return services.calibration(session, p.project, suite, grader_id)

    # dashboard (built SPA, if present)
    if (STATIC_DIR / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (STATIC_DIR / path).resolve()
            if path and candidate.is_file() and STATIC_DIR in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")

    return app

