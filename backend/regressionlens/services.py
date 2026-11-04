"""Server-side operations on runs, baselines, history and labels. Every function takes a Session."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import queue
from .db.models import (
    ApiToken,
    Baseline,
    BaselineEvent,
    HumanLabel,
    ModelConfig,
    Project,
    PromptVersion,
    Run,
    SuiteVersion,
    Trial,
)
from .engine import GraderSet, build_snapshot, environment_info, fingerprint, plan_trials
from .graders import GraderContext, grader_versions, validate_suite
from .local import compare_records
from .providers import Provider, ProviderError, make_provider
from .records import RunRecord, ScoreRecord, TrialRecord
from .spec import ExecutionSpec, ModelSpec, PromptSpec, SpecError, SuiteSpec
from .stats import binary_agreement
from .util import content_hash, new_id, new_run_seed, sha256_hex, utcnow

TERMINAL = {"completed", "errored", "cancelled"}
ProviderFactory = Callable[[ModelSpec], Provider]


class ServiceError(Exception):
    status_code = 400

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class NotFound(ServiceError):
    status_code = 404


# --- projects & tokens ----------------------------------------------------------------------


def ensure_project(session: Session, name: str) -> Project:
    project = session.scalar(select(Project).where(Project.name == name))
    if project is None:
        project = Project(id=new_id("prj"), name=name)
        session.add(project)
        session.flush()
    return project


def hash_token(token: str) -> str:
    return sha256_hex(token)


def create_token(session: Session, project: Project, name: str = "", scopes: str = "read,run", token: str | None = None) -> str:
    token = token or "rl_" + new_id("tok").split("_", 1)[1] + new_id("x").split("_", 1)[1]
    existing = session.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(token)))
    if existing is None:
        session.add(ApiToken(id=new_id("tok"), project_id=project.id, name=name, token_hash=hash_token(token), scopes=scopes))
        session.flush()
    return token


def authenticate(session: Session, token: str) -> tuple[ApiToken, Project] | None:
    row = session.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(token), ApiToken.revoked_at.is_(None)))
    if row is None:
        return None
    return row, session.get(Project, row.project_id)


# --- versions -------------------------------------------------------------------------------


def _upsert(session: Session, model, project: Project, name: str, spec: dict[str, Any]):
    digest = content_hash(spec)
    row = session.scalar(select(model).where(model.project_id == project.id, model.content_hash == digest))
    if row is None:
        prefix = {SuiteVersion: "suv", PromptVersion: "prv", ModelConfig: "mdl"}[model]
        row = model(id=new_id(prefix), project_id=project.id, name=name, content_hash=digest, spec=spec)
        session.add(row)
        session.flush()
    return row


def upsert_suite(session: Session, project: Project, suite: SuiteSpec) -> SuiteVersion:
    return _upsert(session, SuiteVersion, project, suite.name, suite.model_dump(mode="json"))


def upsert_prompt(session: Session, project: Project, prompt: PromptSpec) -> PromptVersion:
    return _upsert(session, PromptVersion, project, prompt.name, prompt.model_dump(mode="json"))


def upsert_model(session: Session, project: Project, model: ModelSpec) -> ModelConfig:
    return _upsert(session, ModelConfig, project, model.name, model.model_dump(mode="json"))


def exec_queue(model: ModelSpec) -> str:
    """Trials for the same model server share a queue, so its concurrency can be bounded."""
    if model.provider == "simulated":
        return "exec:simulated"
    return f"exec:{model.provider}:{model.endpoint or 'default'}"


# --- runs -----------------------------------------------------------------------------------


@dataclass
class RunRequest:
    suite: SuiteSpec
    prompt: PromptSpec
    model: ModelSpec
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    baseline: str | None = None
    baseline_run_id: str | None = None
    fresh_baseline: bool = False
    ci_meta: dict[str, Any] = field(default_factory=dict)
    run_seed: int | None = None


def _validate_request(suite: SuiteSpec, prompt: PromptSpec) -> None:
    try:
        validate_suite(suite)
        for case in suite.cases:
            prompt.render(case.input)
    except SpecError as exc:
        raise ServiceError(str(exc)) from exc


def _resolve(model: ModelSpec, suite: SuiteSpec, provider_factory: ProviderFactory):
    try:
        identity = provider_factory(model).resolve()
        ctx = GraderContext()
        try:
            judges = GraderSet(suite, ctx).describe_judges()
        finally:
            ctx.close()
    except ProviderError as exc:
        raise ServiceError(f"cannot resolve model: {exc}", status_code=502) from exc
    return identity, judges


def _insert_run(
    session: Session,
    project: Project,
    suite: SuiteSpec,
    prompt: PromptSpec,
    model: ModelSpec,
    execution: ExecutionSpec,
    identity,
    judges: dict[str, Any],
    run_seed: int,
    kind: str = "run",
    source_run_id: str | None = None,
    ci_meta: dict[str, Any] | None = None,
) -> tuple[Run, list[dict[str, Any]]]:
    suite_row = upsert_suite(session, project, suite)
    prompt_row = upsert_prompt(session, project, prompt)
    model_row = upsert_model(session, project, model)
    snapshot = build_snapshot(suite, prompt, model, identity, execution, judges)
    plans = plan_trials(suite, execution, run_seed)
    run = Run(
        id=new_id("run"),
        project_id=project.id,
        suite_name=suite.name,
        suite_version_id=suite_row.id,
        prompt_version_id=prompt_row.id,
        model_config_id=model_row.id,
        status="queued",
        fingerprint=fingerprint(snapshot),
        run_seed=run_seed,
        snapshot=snapshot,
        environment=environment_info(identity, model),
        execution=execution.model_dump(mode="json"),
        total_trials=len(plans),
        kind=kind,
        source_run_id=source_run_id,
        ci_meta=ci_meta or {},
    )
    session.add(run)
    session.flush()
    trials = [
        Trial(run_id=run.id, case_key=p.case_key, repeat_idx=p.repeat_idx, seed=p.seed, order_idx=p.order, status="pending")
        for p in plans
    ]
    session.add_all(trials)
    session.flush()
    payloads = [
        {"run_id": run.id, "trial_id": t.id, "concurrency": model.concurrency} for t in sorted(trials, key=lambda t: t.order_idx)
    ]
    return run, payloads


def _interleave(*lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    longest = max((len(x) for x in lists), default=0)
    for i in range(longest):
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
    return out


def find_baseline(session: Session, project: Project, suite_name: str, name: str) -> Run | None:
    row = session.scalar(
        select(Baseline).where(Baseline.project_id == project.id, Baseline.suite_name == suite_name, Baseline.name == name)
    )
    return session.get(Run, row.run_id) if row else None


def get_run(session: Session, project: Project, run_id: str) -> Run:
    run = session.get(Run, run_id)
    if run is None or run.project_id != project.id:
        raise NotFound(f"run {run_id} not found")
    return run


def create_run(session: Session, project: Project, req: RunRequest, provider_factory: ProviderFactory = make_provider) -> Run:
    _validate_request(req.suite, req.prompt)
    identity, judges = _resolve(req.model, req.suite, provider_factory)
    baseline_run: Run | None = None
    if req.baseline_run_id:
        baseline_run = get_run(session, project, req.baseline_run_id)
    elif req.baseline:
        baseline_run = find_baseline(session, project, req.suite.name, req.baseline)
    run_seed = req.run_seed if req.run_seed is not None else new_run_seed()
    run, payloads = _insert_run(
        session, project, req.suite, req.prompt, req.model, req.execution, identity, judges, run_seed, ci_meta=req.ci_meta
    )
    run.baseline_name = req.baseline
    batches = [payloads]
    if baseline_run is not None and req.fresh_baseline:
        base_prompt = PromptSpec.model_validate(baseline_run.prompt_version.spec)
        base_model = ModelSpec.model_validate(baseline_run.model_config.spec)
        base_exec = ExecutionSpec.model_validate(baseline_run.execution)
        _validate_request(req.suite, base_prompt)
        base_identity, base_judges = _resolve(base_model, req.suite, provider_factory)
        fresh, fresh_payloads = _insert_run(
            session, project, req.suite, base_prompt, base_model, base_exec, base_identity, base_judges,
            new_run_seed(), kind="fresh_baseline", source_run_id=baseline_run.id, ci_meta=req.ci_meta,
        )
        run.baseline_run_id = fresh.id
        if exec_queue(base_model) == exec_queue(req.model):
            batches = [_interleave(payloads, fresh_payloads)]
        else:
            queue.enqueue_many(session, "execute", fresh_payloads, queue=exec_queue(base_model))
    elif baseline_run is not None:
        run.baseline_run_id = baseline_run.id
    for batch in batches:
        queue.enqueue_many(session, "execute", batch, queue=exec_queue(req.model))
    session.flush()
    return run


def rerun(session: Session, project: Project, run_id: str, provider_factory: ProviderFactory = make_provider, force: bool = False) -> Run:
    """Execute a past run's stored configuration again, with the same seeds, and compare to it."""
    source = get_run(session, project, run_id)
    suite = SuiteSpec.model_validate(source.suite_version.spec)
    prompt = PromptSpec.model_validate(source.prompt_version.spec)
    model = ModelSpec.model_validate(source.model_config.spec)
    execution = ExecutionSpec.model_validate(source.execution)
    identity, judges = _resolve(model, suite, provider_factory)
    stored_digest = source.snapshot.get("model", {}).get("resolved", {}).get("digest")
    if stored_digest and identity.digest != stored_digest and not force:
        raise ServiceError(
            f"cannot reproduce: model weights changed ({stored_digest} -> {identity.digest}); pass force=true to run anyway",
            status_code=409,
        )
    run, payloads = _insert_run(
        session, project, suite, prompt, model, execution, identity, judges, source.run_seed,
        kind="rerun", source_run_id=source.id, ci_meta={"reproduces": source.id},
    )
    run.baseline_run_id = source.id
    queue.enqueue_many(session, "execute", payloads, queue=exec_queue(model))
    session.flush()
    return run


def rescore(session: Session, project: Project, run_id: str, suite: SuiteSpec | None = None) -> Run:
    """Re-grade a past run's stored outputs with the current graders (optionally an updated suite)."""
    source = get_run(session, project, run_id)
    if source.status != "completed":
        raise ServiceError("only completed runs can be re-scored", status_code=409)
    suite = suite or SuiteSpec.model_validate(source.suite_version.spec)
    validate_suite(suite)
    keys = {c.key for c in suite.cases}
    trials = list(session.scalars(select(Trial).where(Trial.run_id == source.id)))
    if {t.case_key for t in trials} - keys:
        raise ServiceError("the suite used for re-scoring must contain every case of the source run")
    snapshot = dict(source.snapshot)
    ctx = GraderContext()
    try:
        judges = GraderSet(suite, ctx).describe_judges()
    except ProviderError as exc:
        raise ServiceError(f"cannot resolve judge model: {exc}", status_code=502) from exc
    finally:
        ctx.close()
    snapshot["graders"] = {"versions": grader_versions(suite), "judges": judges}
    snapshot["suite"] = {**snapshot["suite"], "content_hash": suite.content_hash(), "case_hashes": suite.case_hashes()}
    suite_row = upsert_suite(session, project, suite)
    run = Run(
        id=new_id("run"), project_id=project.id, suite_name=suite.name, suite_version_id=suite_row.id,
        prompt_version_id=source.prompt_version_id, model_config_id=source.model_config_id, status="grading",
        fingerprint=fingerprint(snapshot), run_seed=source.run_seed, snapshot=snapshot, environment=source.environment,
        execution=source.execution, total_trials=len(trials), executed=len(trials), kind="rescore",
        source_run_id=source.id, ci_meta={"rescores": source.id}, started_at=utcnow(),
    )
    session.add(run)
    session.flush()
    copies = [
        Trial(
            run_id=run.id, case_key=t.case_key, repeat_idx=t.repeat_idx, seed=t.seed, order_idx=t.order_idx,
            status="executed", messages=t.messages, output=t.output, error=t.error, latency_ms=t.latency_ms,
            ttft_ms=t.ttft_ms, tokens_in=t.tokens_in, tokens_out=t.tokens_out, attempts=t.attempts, raw=t.raw,
        )
        for t in trials
    ]
    session.add_all(copies)
    session.flush()
    queue.enqueue_many(session, "grade", [{"run_id": run.id, "trial_id": t.id} for t in copies], queue="grade")
    session.flush()
    return run


def cancel_run(session: Session, project: Project, run_id: str) -> Run:
    run = get_run(session, project, run_id)
    if run.status not in TERMINAL:
        run.status = "cancelled"
        run.finished_at = utcnow()
        queue.cancel_run_jobs(session, run.id)
    return run


def trial_record(t: Trial) -> TrialRecord:
    return TrialRecord(
        case_key=t.case_key, repeat_idx=t.repeat_idx, seed=t.seed, order=t.order_idx, messages=t.messages or [],
        output=t.output, error=t.error, latency_ms=t.latency_ms, ttft_ms=t.ttft_ms, tokens_in=t.tokens_in,
        tokens_out=t.tokens_out, attempts=t.attempts or 1, raw=t.raw or {},
        scores={k: ScoreRecord.model_validate(v) for k, v in (t.scores or {}).items()}, passed=t.passed,
    )


def run_record(session: Session, run: Run, with_trials: bool = True) -> RunRecord:
    trials = []
    if with_trials:
        rows = session.scalars(select(Trial).where(Trial.run_id == run.id).order_by(Trial.order_idx))
        trials = [trial_record(t) for t in rows]
    return RunRecord(
        run_id=run.id, created_at=run.created_at, finished_at=run.finished_at, status=run.status,
        fingerprint=run.fingerprint, run_seed=run.run_seed, snapshot=run.snapshot, environment=run.environment or {},
        suite=run.suite_version.spec, prompt=run.prompt_version.spec, model=run.model_config.spec,
        execution=run.execution or {}, trials=trials, summary=run.summary, verdict=run.verdict,
        baseline_run_id=run.baseline_run_id, ci_meta=run.ci_meta or {},
    )


def run_brief(run: Run, public_url: str = "") -> dict[str, Any]:
    summary = run.summary or {}
    return {
        "run_id": run.id,
        "kind": run.kind,
        "suite": run.suite_name,
        "status": run.status,
        "verdict": run.verdict_label,
        "prompt": run.prompt_version.name if run.prompt_version else None,
        "model": run.model_config.name if run.model_config else None,
        "fingerprint": run.fingerprint,
        "baseline_run_id": run.baseline_run_id,
        "baseline_name": run.baseline_name,
        "source_run_id": run.source_run_id,
        "progress": {"total": run.total_trials, "executed": run.executed, "graded": run.graded},
        "metrics": summary.get("metrics", {}),
        "ci_meta": run.ci_meta or {},
        "error": run.error,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "url": f"{public_url}/runs/{run.id}" if public_url else None,
    }


def list_runs(
    session: Session, project: Project, suite: str | None = None, status: str | None = None,
    branch: str | None = None, limit: int = 50, offset: int = 0,
) -> list[Run]:
    stmt = select(Run).where(Run.project_id == project.id)
    if suite:
        stmt = stmt.where(Run.suite_name == suite)
    if status:
        stmt = stmt.where(Run.status == status)
    stmt = stmt.order_by(Run.created_at.desc()).limit(min(limit, 500)).offset(offset)
    runs = list(session.scalars(stmt))
    if branch:
        runs = [r for r in runs if (r.ci_meta or {}).get("branch") == branch]
    return runs


def compare(session: Session, project: Project, baseline_run_id: str, candidate_run_id: str) -> dict[str, Any]:
    base = get_run(session, project, baseline_run_id)
    cand = get_run(session, project, candidate_run_id)
    for run in (base, cand):
        if run.status != "completed":
            raise ServiceError(f"run {run.id} is {run.status}, not completed", status_code=409)
    return compare_records(run_record(session, cand), run_record(session, base)).to_dict()


def promote(session: Session, project: Project, run_id: str, name: str, suite_name: str | None = None, by: str | None = None) -> Baseline:
    run = get_run(session, project, run_id)
    if run.status != "completed":
        raise ServiceError(f"run {run_id} is {run.status}; only completed runs can be baselines", status_code=409)
    if suite_name and suite_name != run.suite_name:
        raise ServiceError(f"run {run_id} belongs to suite {run.suite_name!r}, not {suite_name!r}")
    row = session.scalar(
        select(Baseline).where(Baseline.project_id == project.id, Baseline.suite_name == run.suite_name, Baseline.name == name)
    )
    if row is None:
        row = Baseline(project_id=project.id, suite_name=run.suite_name, name=name, run_id=run.id, updated_by=by)
        session.add(row)
    else:
        row.run_id, row.updated_at, row.updated_by = run.id, utcnow(), by
    session.add(BaselineEvent(project_id=project.id, suite_name=run.suite_name, name=name, run_id=run.id))
    session.flush()
    return row


def list_baselines(session: Session, project: Project, suite: str | None = None) -> list[dict[str, Any]]:
    stmt = select(Baseline).where(Baseline.project_id == project.id)
    if suite:
        stmt = stmt.where(Baseline.suite_name == suite)
    return [
        {"suite": b.suite_name, "name": b.name, "run_id": b.run_id, "updated_at": b.updated_at, "updated_by": b.updated_by}
        for b in session.scalars(stmt.order_by(Baseline.suite_name, Baseline.name))
    ]


def list_suites(session: Session, project: Project) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Run.suite_name, func.count(Run.id), func.max(Run.created_at)).where(Run.project_id == project.id).group_by(Run.suite_name)
    ).all()
    out = []
    for name, n, last in rows:
        latest = session.scalar(
            select(SuiteVersion).where(SuiteVersion.project_id == project.id, SuiteVersion.name == name).order_by(SuiteVersion.created_at.desc())
        )
        spec = SuiteSpec.model_validate(latest.spec) if latest else None
        out.append(
            {
                "name": name, "runs": n, "last_run_at": last,
                "cases": len(spec.cases) if spec else None,
                "graders": [
                    {"id": g.id, "type": g.type} for g in {g.id: g for c in spec.cases for g in spec.effective_graders(c)}.values()
                ] if spec else [],
            }
        )
    return sorted(out, key=lambda x: x["name"])


def trends(session: Session, project: Project, suite: str, metric: str = "pass_rate", branch: str | None = None, limit: int = 200) -> dict[str, Any]:
    stmt = (
        select(Run)
        .where(Run.project_id == project.id, Run.suite_name == suite, Run.status == "completed", Run.kind.in_(["run", "rerun"]))
        .order_by(Run.created_at.desc())
        .limit(limit)
    )
    runs = list(reversed(list(session.scalars(stmt))))
    if branch:
        runs = [r for r in runs if (r.ci_meta or {}).get("branch") == branch]
    points = []
    previous_fp = None
    for r in runs:
        summary = r.summary or {}
        value = summary.get("metrics", {}).get(metric)
        if value is None:
            continue
        ci = (summary.get("metric_ci") or {}).get(metric)
        points.append(
            {
                "run_id": r.id, "created_at": r.created_at, "value": value, "ci": ci, "verdict": r.verdict_label,
                "prompt": r.prompt_version.name, "model": r.model_config.name, "fingerprint": r.fingerprint,
                "config_changed": previous_fp is not None and r.fingerprint != previous_fp,
                "branch": (r.ci_meta or {}).get("branch"),
            }
        )
        previous_fp = r.fingerprint
    events = session.scalars(
        select(BaselineEvent).where(BaselineEvent.project_id == project.id, BaselineEvent.suite_name == suite).order_by(BaselineEvent.created_at)
    )
    return {
        "suite": suite, "metric": metric, "points": points,
        "baseline_events": [{"name": e.name, "run_id": e.run_id, "created_at": e.created_at} for e in events],
    }


def case_history(session: Session, project: Project, suite: str, case_key: str, limit: int = 50) -> dict[str, Any]:
    runs = list(
        session.scalars(
            select(Run)
            .where(Run.project_id == project.id, Run.suite_name == suite, Run.status == "completed")
            .order_by(Run.created_at.desc())
            .limit(limit)
        )
    )
    by_run: dict[str, list[Trial]] = {}
    if runs:
        for t in session.scalars(select(Trial).where(Trial.run_id.in_([r.id for r in runs]), Trial.case_key == case_key)):
            by_run.setdefault(t.run_id, []).append(t)
    points = []
    for r in reversed(runs):
        trials = sorted(by_run.get(r.id, []), key=lambda t: t.repeat_idx)
        if not trials:
            continue
        outcomes = [1.0 if t.passed else 0.0 for t in trials]
        p = float(np.mean(outcomes))
        points.append(
            {
                "run_id": r.id, "created_at": r.created_at, "pass_rate": p, "repeats": len(trials),
                "flaky": 0.0 < p < 1.0, "outcomes": outcomes, "prompt": r.prompt_version.name, "model": r.model_config.name,
                "sample_output": trials[0].output, "verdict": r.verdict_label,
            }
        )
    flaky_runs = sum(1 for p in points if p["flaky"])
    return {
        "suite": suite, "case_key": case_key, "points": points,
        "flaky_fraction": flaky_runs / len(points) if points else None,
    }


# --- labels & calibration -------------------------------------------------------------------


def _grader_version(run: Run, grader_id: str) -> str:
    return run.snapshot.get("graders", {}).get("versions", {}).get(grader_id, "unknown")


def label_queue(
    session: Session, project: Project, suite: str, grader_id: str, labeler: str, n: int = 20, seed: int | None = None
) -> list[dict[str, Any]]:
    """Trials to label blind, stratified so the grader's passes and fails are equally represented."""
    labeled = set(session.scalars(select(HumanLabel.trial_id).where(HumanLabel.grader_id == grader_id, HumanLabel.labeler == labeler)))
    runs = {
        r.id: r
        for r in session.scalars(
            select(Run).where(Run.project_id == project.id, Run.suite_name == suite, Run.status == "completed").order_by(Run.created_at.desc()).limit(20)
        )
    }
    if not runs:
        return []
    pools: dict[bool, list[Trial]] = {True: [], False: []}
    for t in session.scalars(select(Trial).where(Trial.run_id.in_(list(runs)), Trial.status == "graded")):
        score = (t.scores or {}).get(grader_id)
        if score is None or t.id in labeled or t.output is None or score.get("error"):
            continue
        pools[bool(score.get("passed"))].append(t)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    picked: list[Trial] = []
    while len(picked) < n and (pools[True] or pools[False]):
        for key in (True, False):
            if pools[key] and len(picked) < n:
                picked.append(pools[key].pop())
    rng.shuffle(picked)
    out = []
    for t in picked:
        run = runs[t.run_id]
        suite_spec = SuiteSpec.model_validate(run.suite_version.spec)
        case = suite_spec.case(t.case_key)
        grader = next((g for g in suite_spec.effective_graders(case) if g.id == grader_id), None)
        out.append(
            {
                "trial_id": t.id, "run_id": t.run_id, "case_key": t.case_key, "messages": t.messages, "output": t.output,
                "expected": case.expected, "rubric": (grader.config.get("rubric") if grader else None),
                "grader_version": _grader_version(run, grader_id),
            }
        )
    return out


def add_label(session: Session, project: Project, trial_id: int, grader_id: str, label: float, labeler: str, note: str | None = None) -> HumanLabel:
    trial = session.get(Trial, trial_id)
    if trial is None:
        raise NotFound(f"trial {trial_id} not found")
    run = get_run(session, project, trial.run_id)
    if grader_id not in (trial.scores or {}):
        raise ServiceError(f"trial {trial_id} has no score from grader {grader_id!r}")
    if not 0.0 <= label <= 1.0:
        raise ServiceError("label must be in [0, 1] (1 = pass)")
    row = session.scalar(
        select(HumanLabel).where(HumanLabel.trial_id == trial_id, HumanLabel.grader_id == grader_id, HumanLabel.labeler == labeler)
    )
    if row is None:
        row = HumanLabel(trial_id=trial_id, grader_id=grader_id, grader_version=_grader_version(run, grader_id), label=label, labeler=labeler, note=note)
        session.add(row)
    else:
        row.label, row.note = label, note
    session.flush()
    return row


def calibration(session: Session, project: Project, suite: str, grader_id: str) -> dict[str, Any]:
    rows = session.execute(
        select(HumanLabel, Trial)
        .join(Trial, Trial.id == HumanLabel.trial_id)
        .join(Run, Run.id == Trial.run_id)
        .where(Run.project_id == project.id, Run.suite_name == suite, HumanLabel.grader_id == grader_id)
    ).all()
    by_version: dict[str, tuple[list[int], list[int]]] = {}
    for label, trial in rows:
        score = (trial.scores or {}).get(grader_id)
        if not score:
            continue
        g, h = by_version.setdefault(label.grader_version, ([], []))
        g.append(1 if score.get("passed") else 0)
        h.append(1 if label.label >= 0.5 else 0)
    versions = {version: binary_agreement(g, h).to_dict() for version, (g, h) in sorted(by_version.items())}
    latest = max(by_version, key=lambda v: len(by_version[v][0])) if by_version else None
    return {
        "suite": suite, "grader_id": grader_id, "versions": versions,
        "current": versions.get(latest) if latest else None, "current_version": latest,
    }


def grader_trust(session: Session, project: Project, suite: str, grader_ids: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for gid in grader_ids:
        report = calibration(session, project, suite, gid).get("current")
        if report:
            out[gid] = {"trust": report["trust"], "kappa": report["kappa"], "n": report["n"]}
    return out
