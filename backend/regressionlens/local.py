"""Run a suite in-process, without the API or database. Used by ``rlens run --local`` and tests."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .engine import (
    GraderSet,
    build_snapshot,
    environment_info,
    execute_trial,
    fingerprint,
    grade_trial,
    plan_trials,
    run_view,
    summarize,
    warmup_messages,
)
from .graders import GraderContext, validate_suite
from .providers import Provider, make_provider
from .records import RunRecord
from .spec import ExecutionSpec, ModelSpec, PromptSpec, SuiteSpec
from .stats import VerdictReport, evaluate
from .util import new_id, new_run_seed, utcnow

Progress = Callable[[str, int, int], None]


def run_local(
    suite: SuiteSpec,
    prompt: PromptSpec,
    model: ModelSpec,
    execution: ExecutionSpec | None = None,
    run_seed: int | None = None,
    provider: Provider | None = None,
    progress: Progress | None = None,
) -> RunRecord:
    execution = execution or ExecutionSpec()
    validate_suite(suite)
    run_seed = run_seed if run_seed is not None else new_run_seed()
    provider = provider or make_provider(model)
    ctx = GraderContext(timeout_s=execution.timeout_s)
    graders = GraderSet(suite, ctx)
    created = utcnow()
    try:
        identity = provider.resolve()
        snapshot = build_snapshot(suite, prompt, model, identity, execution, graders.describe_judges())
        plans = plan_trials(suite, execution, run_seed)
        cases = {c.key: c for c in suite.cases}
        if execution.warmup:
            provider.warmup(warmup_messages(suite, prompt))

        done = 0
        total = len(plans)

        def run_one(plan):
            return execute_trial(provider, prompt, model, cases[plan.case_key], plan, execution)

        trials = []
        with ThreadPoolExecutor(max_workers=model.concurrency) as pool:
            for trial in pool.map(run_one, plans):
                trials.append(trial)
                done += 1
                if progress:
                    progress("execute", done, total)

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, model.concurrency)) as pool:
            def grade_one(trial):
                case = cases[trial.case_key]
                return grade_trial(trial, case, graders.for_case(case))

            graded = []
            for trial in pool.map(grade_one, trials):
                graded.append(trial)
                done += 1
                if progress:
                    progress("grade", done, total)
    finally:
        ctx.close()

    graded.sort(key=lambda t: t.order)
    return RunRecord(
        run_id=new_id("run"),
        created_at=created,
        finished_at=utcnow(),
        status="completed",
        fingerprint=fingerprint(snapshot),
        run_seed=run_seed,
        snapshot=snapshot,
        environment=environment_info(identity, model),
        suite=suite.model_dump(mode="json"),
        prompt=prompt.model_dump(mode="json"),
        model=model.model_dump(mode="json"),
        execution=execution.model_dump(mode="json"),
        trials=graded,
        summary=summarize(suite, graded, execution.repeats),
    )


def compare_records(candidate: RunRecord, baseline: RunRecord | None, **kwargs) -> VerdictReport:
    suite = SuiteSpec.model_validate(candidate.suite)
    return evaluate(run_view(candidate), run_view(baseline) if baseline else None, suite.verdict, **kwargs)
