"""Queue worker: executes trials, grades them, and analyzes finished runs.

One worker process holds a thread pool. Each ``exec:<endpoint>`` queue runs at most the model's
configured concurrency at a time (default 1 for a single local GPU); grading and analysis have
their own limits. Handlers are idempotent: a job delivered twice finds its trial already done.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sqlalchemy import select, update

from .config import Settings
from .db import queue
from .db.models import Job, Run, Trial
from .db.session import Database
from .engine import GraderSet, TrialPlan, execute_trial, grade_trial, summarize, warmup_messages
from .graders import GraderContext
from .local import compare_records
from .providers import Provider, make_provider
from .services import TERMINAL, grader_trust, run_record, trial_record
from .spec import ExecutionSpec, ModelSpec, PromptSpec, SuiteSpec
from .util import utcnow

log = logging.getLogger("regressionlens.worker")
DELAYED = object()


@dataclass
class RunContext:
    suite: SuiteSpec
    prompt: PromptSpec
    model: ModelSpec
    execution: ExecutionSpec
    provider: Provider
    grader_ctx: GraderContext
    graders: GraderSet
    warmed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def ensure_warm(self) -> None:
        if not self.execution.warmup or self.warmed:
            return
        with self.lock:
            if not self.warmed:
                self.provider.warmup(warmup_messages(self.suite, self.prompt))
                self.warmed = True

    def close(self) -> None:
        self.provider.close()
        self.grader_ctx.close()


class ContextCache:
    def __init__(self, db: Database, provider_factory, size: int = 16):
        self.db = db
        self.provider_factory = provider_factory
        self.size = size
        self._items: OrderedDict[str, RunContext] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, run_id: str) -> RunContext:
        with self._lock:
            if run_id in self._items:
                self._items.move_to_end(run_id)
                return self._items[run_id]
        with self.db.session() as s:
            run = s.get(Run, run_id)
            suite = SuiteSpec.model_validate(run.suite_version.spec)
            prompt = PromptSpec.model_validate(run.prompt_version.spec)
            model = ModelSpec.model_validate(run.model_config.spec)
            execution = ExecutionSpec.model_validate(run.execution)
        grader_ctx = GraderContext(timeout_s=execution.timeout_s)
        ctx = RunContext(suite, prompt, model, execution, self.provider_factory(model), grader_ctx, GraderSet(suite, grader_ctx))
        with self._lock:
            if run_id in self._items:  # another thread won the race
                ctx.close()
                return self._items[run_id]
            self._items[run_id] = ctx
            while len(self._items) > self.size:
                _, old = self._items.popitem(last=False)
                old.close()
        return ctx


class Worker:
    def __init__(self, db: Database, settings: Settings | None = None, worker_id: str | None = None, provider_factory=make_provider):
        self.db = db
        self.settings = settings or Settings()
        self.id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.cache = ContextCache(db, provider_factory)
        self.inflight: dict[str, int] = defaultdict(int)
        self._limits: dict[str, int] = {}
        self._lock = threading.Lock()
        max_threads = int(os.environ.get("RLENS_WORKER_THREADS", "16"))
        self.pool = ThreadPoolExecutor(max_workers=max_threads, thread_name_prefix="rlens-job")
        self._last_stale_check = 0.0
        self._wake = threading.Event()  # set whenever a job finishes, so a freed slot is refilled at once

    # --- scheduling -------------------------------------------------------------------------

    def limit_for(self, queue_name: str) -> int:
        if queue_name == "grade":
            return self.settings.grade_concurrency
        if queue_name == "analyze":
            return 1
        if queue_name in self._limits:
            return self._limits[queue_name]
        with self.db.session() as s:
            payload = s.scalar(select(Job.payload).where(Job.queue == queue_name, Job.status == "queued").limit(1))
        limit = int((payload or {}).get("concurrency", 1))
        self._limits[queue_name] = max(1, limit)
        return self._limits[queue_name]

    def tick(self) -> int:
        now = time.monotonic()
        if now - self._last_stale_check > 60:
            with self.db.session() as s:
                n = queue.requeue_stale(s, self.settings.stale_job_s)
            if n:
                log.warning("requeued %d stale job(s)", n)
            self._last_stale_check = now
        dispatched = 0
        with self.db.session() as s:
            names = queue.queued_queues(s)
        # Analysis first (cheap, unblocks CI), then grading, then execution.
        names.sort(key=lambda q: (0 if q == "analyze" else 1 if q == "grade" else 2, q))
        for name in names:
            with self._lock:
                available = self.limit_for(name) - self.inflight[name]
            if available <= 0:
                continue
            with self.db.session() as s:
                jobs = queue.claim(s, name, self.id, available)
            for job in jobs:
                with self._lock:
                    self.inflight[name] += 1
                self.pool.submit(self._run_job, job)
                dispatched += 1
        return dispatched

    def busy(self) -> bool:
        with self._lock:
            return any(v > 0 for v in self.inflight.values())

    def run_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        log.info("worker %s started", self.id)
        while not stop.is_set():
            try:
                if self.tick() == 0:
                    self._wake.wait(self.settings.poll_interval_s)
                    self._wake.clear()
            except Exception:  # keep the worker alive through transient DB errors
                log.exception("worker tick failed")
                stop.wait(max(1.0, self.settings.poll_interval_s))
        self.pool.shutdown(wait=True)

    def run_until_idle(self, timeout_s: float = 120.0) -> None:
        """Process jobs until nothing is queued or running (used by tests and experiments)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            dispatched = self.tick()
            if dispatched == 0 and not self.busy():
                with self.db.session() as s:
                    pending = s.scalar(select(Job.id).where(Job.status.in_(["queued", "running"])).limit(1))
                if pending is None:
                    return
            time.sleep(0.01 if dispatched else min(0.05, self.settings.poll_interval_s))
        raise TimeoutError("worker did not become idle in time")

    # --- job execution ----------------------------------------------------------------------

    def _run_job(self, job: queue.ClaimedJob) -> None:
        try:
            handler = {"execute": self.handle_execute, "grade": self.handle_grade, "analyze": self.handle_analyze}[job.kind]
            outcome = handler(job)
            with self.db.session() as s:
                if outcome is DELAYED:
                    queue.delay(s, job, self.settings.analyze_wait_s)
                else:
                    queue.complete(s, job.id)
        except Exception as exc:
            log.exception("job %s (%s) failed", job.id, job.kind)
            with self.db.session() as s:
                retry = queue.fail(s, job, f"{type(exc).__name__}: {exc}")
                if not retry:
                    run = s.get(Run, job.payload.get("run_id"))
                    if run is not None and run.status not in TERMINAL:
                        run.status = "errored"
                        run.error = f"{job.kind} job failed after {job.attempts} attempt(s): {exc}"
                        run.finished_at = utcnow()
        finally:
            with self._lock:
                self.inflight[job.queue] -= 1
            self._wake.set()

    def handle_execute(self, job: queue.ClaimedJob):
        run_id, trial_id = job.payload["run_id"], job.payload["trial_id"]
        with self.db.session() as s:
            run = s.get(Run, run_id)
            trial = s.get(Trial, trial_id)
            if run is None or trial is None or run.status in TERMINAL or trial.status != "pending":
                return None
            if run.status == "queued":
                run.status, run.started_at = "executing", utcnow()
            plan = TrialPlan(trial.case_key, trial.repeat_idx, trial.seed, trial.order_idx)
        ctx = self.cache.get(run_id)
        ctx.ensure_warm()
        record = execute_trial(ctx.provider, ctx.prompt, ctx.model, ctx.suite.case(plan.case_key), plan, ctx.execution)
        with self.db.session() as s:
            trial = s.get(Trial, trial_id, with_for_update=True)
            if trial.status != "pending":
                return None
            trial.messages, trial.output, trial.error = record.messages, record.output, record.error
            trial.latency_ms, trial.ttft_ms = record.latency_ms, record.ttft_ms
            trial.tokens_in, trial.tokens_out = record.tokens_in, record.tokens_out
            trial.attempts, trial.raw, trial.status = record.attempts, record.raw, "executed"
            s.flush()
            executed, total = s.execute(
                update(Run).where(Run.id == run_id).values(executed=Run.executed + 1).returning(Run.executed, Run.total_trials)
            ).one()
            if executed >= total:
                run = s.get(Run, run_id)
                if run.status not in TERMINAL:
                    run.status = "grading"
                    ids = list(s.scalars(select(Trial.id).where(Trial.run_id == run_id).order_by(Trial.order_idx)))
                    queue.enqueue_many(s, "grade", [{"run_id": run_id, "trial_id": i} for i in ids], queue="grade")
        return None

    def handle_grade(self, job: queue.ClaimedJob):
        run_id, trial_id = job.payload["run_id"], job.payload["trial_id"]
        with self.db.session() as s:
            run = s.get(Run, run_id)
            trial = s.get(Trial, trial_id)
            if run is None or trial is None or run.status in TERMINAL or trial.status != "executed":
                return None
            record = trial_record(trial)
        ctx = self.cache.get(run_id)
        case = ctx.suite.case(record.case_key)
        graded = grade_trial(record, case, ctx.graders.for_case(case))
        with self.db.session() as s:
            trial = s.get(Trial, trial_id, with_for_update=True)
            if trial.status != "executed":
                return None
            trial.scores = {k: v.model_dump(mode="json") for k, v in graded.scores.items()}
            trial.passed, trial.status = graded.passed, "graded"
            s.flush()
            done, total = s.execute(
                update(Run).where(Run.id == run_id).values(graded=Run.graded + 1).returning(Run.graded, Run.total_trials)
            ).one()
            if done >= total:
                run = s.get(Run, run_id)
                if run.status not in TERMINAL:
                    run.status = "analyzing"
                    queue.enqueue(s, "analyze", {"run_id": run_id}, queue="analyze")
        return None

    def handle_analyze(self, job: queue.ClaimedJob):
        run_id = job.payload["run_id"]
        with self.db.session() as s:
            run = s.get(Run, run_id)
            if run is None or run.status in TERMINAL:
                return None
            base = s.get(Run, run.baseline_run_id) if run.baseline_run_id else None
            if base is not None and base.status not in TERMINAL:
                return DELAYED
            record = run_record(s, run)
            suite = SuiteSpec.model_validate(record.suite)
            summary = summarize(suite, record.trials, int(record.execution.get("repeats", 1)))
            record.summary = summary
            base_record = run_record(s, base) if base is not None and base.status == "completed" else None
            verdict = compare_records(record, base_record).to_dict()
            if base is not None and base.status != "completed":
                verdict["verdict"] = "error"
                verdict["reasons"].insert(0, f"baseline run {base.id} is {base.status}")
            elif run.baseline_name and base is None:
                verdict["notes"].insert(0, f"baseline {run.baseline_name!r} not found: nothing to compare against yet")
            judged = [gid for gid, info in (run.snapshot.get("graders", {}).get("judges") or {}).items() if info is not None]
            if judged:
                trust = grader_trust(s, _project(s, run), run.suite_name, judged)
                verdict["graders"] = {gid: trust.get(gid, {"trust": "uncalibrated", "n": 0}) for gid in judged}
                weak = [gid for gid, t in verdict["graders"].items() if t["trust"] != "calibrated"]
                if weak:
                    verdict["notes"].append(f"model-based grader(s) not yet calibrated against human labels: {', '.join(weak)}")
            run.summary = summary.model_dump(mode="json")
            run.verdict = verdict
            run.verdict_label = verdict["verdict"]
            run.status, run.finished_at = "completed", utcnow()
        return None


def _project(session, run: Run):
    from .db.models import Project

    return session.get(Project, run.project_id)


def main() -> None:  # pragma: no cover - process entry point
    logging.basicConfig(level=os.environ.get("RLENS_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings()
    db = Database(settings.database_url)
    if settings.auto_migrate:
        from .db.migrate import upgrade

        upgrade(settings.database_url)
    Worker(db, settings).run_forever()
