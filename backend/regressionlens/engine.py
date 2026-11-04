"""The run pipeline, shared by the local runner and the queue workers.

plan -> execute every trial (case x repeat) -> grade every trial -> summarize -> verdict.
Execution and grading are separate phases so a single local GPU does not keep swapping between
the model under test and a judge model.
"""

from __future__ import annotations

import platform
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import __version__
from .graders import GradeInput, Grader, GraderContext, build_grader, grader_versions
from .providers import GenerateRequest, ModelIdentity, Provider, ProviderError
from .records import RunRecord, RunSummary, ScoreRecord, TrialRecord
from .spec import CaseSpec, ExecutionSpec, ModelSpec, PromptSpec, SuiteSpec
from .stats import RunView, flaky_cases, null_difference_sd, percentile
from .stats import mde as compute_mde
from .stats.core import t_interval
from .util import canonical_json, content_hash, derive_seed

SNAPSHOT_SCHEMA_VERSION = 1
STATS_ENGINE_VERSION = "1"


@dataclass
class TrialPlan:
    case_key: str
    repeat_idx: int
    seed: int | None
    order: int


def plan_trials(suite: SuiteSpec, execution: ExecutionSpec, run_seed: int) -> list[TrialPlan]:
    pairs = [(case.key, r) for case in suite.cases for r in range(execution.repeats)]
    order = list(range(len(pairs)))
    if execution.shuffle:
        random.Random(run_seed).shuffle(order)
    plans = []
    for position, index in enumerate(order):
        key, r = pairs[index]
        seed = derive_seed(run_seed, key, r) if execution.seed_policy == "derived" else None
        plans.append(TrialPlan(key, r, seed, position))
    return plans


def build_snapshot(
    suite: SuiteSpec,
    prompt: PromptSpec,
    model: ModelSpec,
    identity: ModelIdentity,
    execution: ExecutionSpec,
    judges: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen configuration of a run. Its hash is the run's fingerprint."""
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "suite": {
            "name": suite.name,
            "content_hash": suite.content_hash(),
            "case_keys": [c.key for c in suite.cases],
            "case_hashes": suite.case_hashes(),
        },
        "prompt": {
            "name": prompt.name,
            "system": prompt.system,
            "template": prompt.template,
            "content_hash": prompt.content_hash(),
        },
        "model": {
            **model.identity_dict(),
            "resolved": {"model": identity.model, "digest": identity.digest, "details": identity.details},
        },
        "graders": {"versions": grader_versions(suite), "judges": judges or {}},
        "execution": {**execution.model_dump(mode="json"), "concurrency": model.concurrency},
        "stats": {"engine_version": STATS_ENGINE_VERSION, "policy": suite.verdict.model_dump(mode="json")},
    }


def fingerprint(snapshot: dict[str, Any]) -> str:
    return content_hash(snapshot)


def environment_info(identity: ModelIdentity, model: ModelSpec) -> dict[str, Any]:
    return {
        "provider_version": identity.server_version,
        "endpoint": model.endpoint,
        "platform_version": __version__,
        "python": platform.python_version(),
    }


def warmup_messages(suite: SuiteSpec, prompt: PromptSpec) -> list[dict[str, str]]:
    return prompt.render(suite.cases[0].input)


def execute_trial(
    provider: Provider,
    prompt: PromptSpec,
    model: ModelSpec,
    case: CaseSpec,
    plan: TrialPlan,
    execution: ExecutionSpec,
) -> TrialRecord:
    trial = TrialRecord(case_key=case.key, repeat_idx=plan.repeat_idx, seed=plan.seed, order=plan.order)
    try:
        trial.messages = prompt.render(case.input)
    except ValueError as exc:
        trial.error = str(exc)
        return trial
    last_error: Exception | None = None
    for attempt in range(execution.retries + 1):
        trial.attempts = attempt + 1
        try:
            result = provider.generate(
                GenerateRequest(
                    messages=trial.messages,
                    params=model.params,
                    seed=plan.seed,
                    timeout_s=execution.timeout_s,
                    metadata={"case_key": case.key, "expected": case.expected, "attempt": attempt},
                )
            )
        except ProviderError as exc:
            last_error = exc
            if not exc.transient or attempt == execution.retries:
                break
            time.sleep(min(8.0, provider.retry_backoff_s * (2**attempt)))
            continue
        except Exception as exc:  # unexpected provider bug: record it, don't kill the run
            last_error = exc
            break
        trial.output = result.text
        trial.latency_ms = result.latency_ms
        trial.ttft_ms = result.ttft_ms
        trial.tokens_in = result.tokens_in
        trial.tokens_out = result.tokens_out
        trial.raw = result.raw
        trial.error = None
        return trial
    trial.error = str(last_error) if last_error else "unknown error"
    return trial


class GraderSet:
    """Builds and caches grader objects per (case, grader spec)."""

    def __init__(self, suite: SuiteSpec, ctx: GraderContext):
        self.suite = suite
        self.ctx = ctx
        self._cache: dict[str, Grader] = {}

    def for_case(self, case: CaseSpec) -> list[Grader]:
        graders = []
        for spec in self.suite.effective_graders(case):
            key = canonical_json(spec.model_dump(mode="json"))
            if key not in self._cache:
                self._cache[key] = build_grader(spec, self.ctx)
            graders.append(self._cache[key])
        return graders

    def describe_judges(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for case in self.suite.cases:
            for grader in self.for_case(case):
                if not grader.deterministic and grader.spec.id not in out:
                    out[grader.spec.id] = grader.describe()
        return out


def grade_trial(trial: TrialRecord, case: CaseSpec, graders: list[Grader]) -> TrialRecord:
    scores: dict[str, ScoreRecord] = {}
    if trial.error is not None or trial.output is None:
        for grader in graders:
            scores[grader.spec.id] = ScoreRecord(value=0.0, passed=False, rationale=f"no output: {trial.error}")
    else:
        item = GradeInput(output=trial.output, case=case, messages=trial.messages)
        for grader in graders:
            try:
                result = grader.grade(item)
                scores[grader.spec.id] = ScoreRecord(**result.to_dict())
            except Exception as exc:
                scores[grader.spec.id] = ScoreRecord(value=0.0, passed=False, rationale=f"grader error: {exc}", error=True)
    trial.scores = scores
    required = [g.spec.id for g in graders if g.spec.required]
    trial.passed = trial.error is None and all(scores[g].passed for g in required)
    return trial


def metric_table(trials: list[TrialRecord]) -> dict[str, dict[str, list[float]]]:
    table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for trial in sorted(trials, key=lambda t: (t.case_key, t.repeat_idx)):
        if trial.passed is None:
            continue
        table["pass_rate"][trial.case_key].append(1.0 if trial.passed else 0.0)
        for gid, score in trial.scores.items():
            table[gid][trial.case_key].append(score.value)
    return {name: dict(cases) for name, cases in table.items()}


def latency_by_case(trials: list[TrialRecord]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        if trial.error is None and trial.latency_ms is not None:
            out[trial.case_key].append(trial.latency_ms)
    return dict(out)


def summarize(suite: SuiteSpec, trials: list[TrialRecord], repeats: int) -> RunSummary:
    table = metric_table(trials)
    metrics: dict[str, float] = {}
    metric_ci: dict[str, list[float]] = {}
    for name, cases in table.items():
        means = [float(np.mean(v)) for v in cases.values() if v]
        if means:
            metrics[name] = float(np.mean(means))
            low, high = t_interval(np.asarray(means), alpha=suite.verdict.alpha)
            metric_ci[name] = [max(0.0, low), min(1.0, high)]
    per_tag: dict[str, dict[str, float]] = {}
    tags = sorted({t for c in suite.cases for t in c.tags})
    for tag in tags:
        keys = {c.key for c in suite.cases if tag in c.tags}
        per_tag[tag] = {}
        for name, cases in table.items():
            means = [float(np.mean(v)) for k, v in cases.items() if k in keys and v]
            if means:
                per_tag[tag][name] = float(np.mean(means))
    n_trials = len(trials)
    errors = sum(1 for t in trials if t.error is not None)
    graded = [s for t in trials if t.error is None for s in t.scores.values()]
    grader_errors = sum(1 for s in graded if s.error)
    latencies = [v for vals in latency_by_case(trials).values() for v in vals]
    flaky = flaky_cases(table.get("pass_rate", {}))
    sd0 = null_difference_sd(table.get("pass_rate", {}))
    notes = []
    if repeats < 2:
        notes.append("only 1 repeat per case: flakiness cannot be measured")
    return RunSummary(
        n_cases=len(suite.cases),
        n_trials=n_trials,
        repeats=repeats,
        metrics=metrics,
        metric_ci=metric_ci,
        per_tag=per_tag,
        error_rate=errors / n_trials if n_trials else 0.0,
        grader_error_rate=grader_errors / len(graded) if graded else 0.0,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        flaky_cases=[f.to_dict() for f in flaky],
        flaky_rate=len(flaky) / len(suite.cases) if suite.cases else 0.0,
        mde=compute_mde(sd0, len(suite.cases), alpha=suite.verdict.alpha) if sd0 is not None else None,
        notes=notes,
    )


def run_view(record: RunRecord) -> RunView:
    suite = SuiteSpec.model_validate(record.suite)
    summary = record.summary
    return RunView(
        metric_table=metric_table(record.trials),
        case_hashes=suite.case_hashes(),
        grader_versions=record.snapshot.get("graders", {}).get("versions", {}),
        error_rate=summary.error_rate if summary else 0.0,
        grader_error_rate=summary.grader_error_rate if summary else 0.0,
        latency_by_case=latency_by_case(record.trials),
        critical_cases=[c.key for c in suite.cases if c.critical],
        snapshot=record.snapshot,
        environment=record.environment,
        run_id=record.run_id,
    )
