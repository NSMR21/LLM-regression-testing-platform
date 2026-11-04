"""Turn a candidate run (and optionally a baseline run) into a verdict.

Rules (docs/ARCHITECTURE.md §8.3), for each gated metric with paired differences d_c:

- FAIL if the sign-flip p-value (Holm-adjusted across gated metrics) is below alpha, the mean
  difference is negative, and the drop is at least ``min_effect``. Under identical configs the
  false-alarm rate is therefore at most alpha / 2 per metric.
- PASS if the lower bound of the (1 - alpha) bootstrap CI is at or above ``-non_inferiority_margin``
  and there are at least MIN_CASES comparable cases.
- INCONCLUSIVE otherwise.

ERROR (infrastructure problem) beats FAIL, which beats INCONCLUSIVE, which beats PASS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..spec import VerdictPolicy
from ..util import flatten
from .core import bootstrap_mean_ci, case_means, holm_adjust, paired_ci, percentile, sign_flip_pvalue
from .flaky import is_flaky
from .power import data_multiplier, mde

MIN_CASES = 5

# Snapshot paths that change between runs without changing the configuration.
_IGNORED_DIFF_PREFIXES = ("suite.case_keys", "suite.case_hashes")


@dataclass
class RunView:
    """Everything the verdict needs to know about one run."""

    metric_table: dict[str, dict[str, list[float]]]
    case_hashes: dict[str, str]
    grader_versions: dict[str, str]
    error_rate: float = 0.0
    grader_error_rate: float = 0.0
    latency_by_case: dict[str, list[float]] = field(default_factory=dict)
    critical_cases: list[str] = field(default_factory=list)
    snapshot: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None


@dataclass
class MetricResult:
    name: str
    gated: bool
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    n_cases: int
    mde: float | None
    status: str  # fail | pass | inconclusive (gated) ; worse | better | no_change (info)


@dataclass
class GateResult:
    metric: str
    value: float | None
    min: float | None
    max: float | None
    passed: bool


@dataclass
class CaseDiff:
    case_key: str
    baseline: float
    candidate: float
    delta: float
    category: str  # broken | fixed | flaky | stable_pass | stable_fail | changed


@dataclass
class VerdictReport:
    verdict: str
    reasons: list[str]
    notes: list[str]
    metrics: list[MetricResult]
    gates: list[GateResult]
    critical_failures: list[str]
    case_diff: list[CaseDiff]
    excluded_cases: dict[str, str]
    config_diff: dict[str, dict[str, Any]]
    flaky_share_of_drop: float | None
    baseline_run_id: str | None
    candidate_run_id: str | None
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    def metric(self, name: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.name == name), None)


def _clean(obj: Any) -> Any:
    """Replace NaN/inf with None so the report is valid JSON."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def pts(x: float) -> str:
    return f"{x * 100:+.1f} pts"


def metric_value(view: RunView, name: str, exclude: Sequence[str] = ()) -> float | None:
    """Absolute value of a metric on one run: mean over cases of the per-case mean."""
    if name == "error_rate":
        return view.error_rate
    if name in {"latency_p50_ms", "latency_p95_ms"}:
        values = [v for vals in view.latency_by_case.values() for v in vals]
        return percentile(values, 50 if name == "latency_p50_ms" else 95)
    table = view.metric_table.get(name)
    if not table:
        return None
    means = [m for k, m in case_means(table).items() if k not in exclude]
    return float(np.mean(means)) if means else None


def config_diff(base: Mapping[str, Any], cand: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fb, fc = flatten(dict(base)), flatten(dict(cand))
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(set(fb) | set(fc)):
        if key.startswith(_IGNORED_DIFF_PREFIXES):
            continue
        if fb.get(key) != fc.get(key):
            out[key] = {"baseline": fb.get(key), "candidate": fc.get(key)}
    return out


def _categorize(b: float, c: float, flaky: bool) -> str:
    d = c - b
    if d <= -0.5:
        return "broken"
    if d >= 0.5:
        return "fixed"
    if flaky:
        return "flaky"
    if b == 1.0 and c == 1.0:
        return "stable_pass"
    if b == 0.0 and c == 0.0:
        return "stable_fail"
    return "changed"


def evaluate(
    candidate: RunView,
    baseline: RunView | None,
    policy: VerdictPolicy,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
) -> VerdictReport:
    reasons: list[str] = []
    notes: list[str] = []
    errors: list[str] = []
    alpha = policy.alpha
    quarantine = set(policy.quarantine)

    if candidate.error_rate > policy.max_error_rate:
        errors.append(f"generation error rate {candidate.error_rate:.1%} exceeds max_error_rate {policy.max_error_rate:.0%}")
    if candidate.grader_error_rate > policy.max_error_rate:
        errors.append(f"grader error rate {candidate.grader_error_rate:.1%} exceeds max_error_rate {policy.max_error_rate:.0%}")

    # Absolute gates and critical cases need no baseline.
    gates: list[GateResult] = []
    for gate in policy.gates:
        value = metric_value(candidate, gate.metric, exclude=quarantine)
        ok = value is not None and (gate.min is None or value >= gate.min) and (gate.max is None or value <= gate.max)
        gates.append(GateResult(gate.metric, value, gate.min, gate.max, ok))
        if not ok:
            bound = f">= {gate.min}" if gate.min is not None else f"<= {gate.max}"
            shown = "missing" if value is None else f"{value:.3f}"
            reasons.append(f"gate failed: {gate.metric} = {shown}, required {bound}")

    critical_failures: list[str] = []
    pass_table = candidate.metric_table.get("pass_rate", {})
    for key in candidate.critical_cases:
        values = pass_table.get(key)
        if values and float(np.mean(values)) < 0.5:
            critical_failures.append(key)
    if critical_failures:
        reasons.append(f"critical case(s) failing in most repeats: {', '.join(critical_failures[:5])}")

    metrics: list[MetricResult] = []
    case_diff: list[CaseDiff] = []
    excluded: dict[str, str] = {}
    diff: dict[str, dict[str, Any]] = {}
    flaky_share: float | None = None

    if baseline is None:
        notes.append("no baseline: only error rate, absolute gates and critical cases were checked")
    else:
        diff = config_diff(baseline.snapshot, candidate.snapshot)
        for gid, version in candidate.grader_versions.items():
            old = baseline.grader_versions.get(gid)
            if old is not None and old != version:
                errors.append(f"grader {gid!r} differs between runs ({old} -> {version}); re-score the baseline before comparing")
        bv, cv = baseline.environment.get("provider_version"), candidate.environment.get("provider_version")
        if bv and cv and bv != cv:
            notes.append(f"model server version changed ({bv} -> {cv}); consider a fresh A/B comparison")

        for key, digest in candidate.case_hashes.items():
            if key not in baseline.case_hashes:
                excluded[key] = "not in baseline"
            elif baseline.case_hashes[key] != digest:
                excluded[key] = "case changed since baseline"
        for key in baseline.case_hashes:
            if key not in candidate.case_hashes:
                excluded[key] = "removed from suite"
        for key in sorted(quarantine):
            excluded.setdefault(key, "quarantined")
        if excluded:
            notes.append(f"{len(excluded)} case(s) excluded from the comparison")
        keys = sorted(k for k in candidate.case_hashes if k in baseline.case_hashes and k not in excluded)
        if not keys:
            errors.append("no comparable cases between baseline and candidate")
        else:
            gated_names = policy.metrics_to_gate
            other = sorted((set(candidate.metric_table) & set(baseline.metric_table)) - set(gated_names))
            raw: list[tuple[str, bool, dict[str, Any]]] = []
            for name in [*gated_names, *other]:
                if name not in candidate.metric_table or name not in baseline.metric_table:
                    if name in gated_names:
                        errors.append(f"gated metric {name!r} missing from one of the runs")
                    continue
                bm, cm = case_means(baseline.metric_table[name]), case_means(candidate.metric_table[name])
                mkeys = [k for k in keys if k in bm and k in cm]
                d = np.array([cm[k] - bm[k] for k in mkeys], dtype=float)
                if d.size == 0:
                    continue
                low, high = paired_ci(d, alpha=alpha, n_boot=n_boot, seed=seed)
                p = sign_flip_pvalue(d, n_perm=n_perm, seed=seed)
                sd = float(d.std(ddof=1)) if d.size > 1 else math.nan
                raw.append(
                    (
                        name,
                        name in gated_names,
                        {
                            "baseline": float(np.mean([bm[k] for k in mkeys])),
                            "candidate": float(np.mean([cm[k] for k in mkeys])),
                            "delta": float(d.mean()),
                            "low": low,
                            "high": high,
                            "p": p,
                            "n": int(d.size),
                            "mde": mde(sd, int(d.size), alpha=alpha),
                        },
                    )
                )
            gated_raw = [r for r in raw if r[1]]
            adjusted = dict(zip([r[0] for r in gated_raw], holm_adjust([r[2]["p"] for r in gated_raw]), strict=True))
            for name, gated, r in raw:
                p_adj = adjusted.get(name, r["p"])
                significant = p_adj < alpha
                if gated:
                    if significant and r["delta"] < 0 and -r["delta"] >= policy.min_effect:
                        status = "fail"
                        reasons.append(
                            f"{name} dropped: {pts(r['delta'])}, {1 - alpha:.0%} CI [{pts(r['low'])}, {pts(r['high'])}], p = {p_adj:.4g}"
                        )
                    elif r["n"] >= MIN_CASES and r["low"] >= -policy.non_inferiority_margin:
                        status = "pass"
                        if significant and r["delta"] > 0:
                            notes.append(f"{name} improved: {pts(r['delta'])}, p = {p_adj:.4g}")
                    else:
                        status = "inconclusive"
                        reasons.append(_inconclusive_reason(name, r, policy))
                else:
                    if significant and r["delta"] < 0:
                        status = "worse"
                    elif significant and r["delta"] > 0:
                        status = "better"
                    else:
                        status = "no_change"
                metrics.append(
                    MetricResult(
                        name=name,
                        gated=gated,
                        baseline=r["baseline"],
                        candidate=r["candidate"],
                        delta=r["delta"],
                        ci_low=r["low"],
                        ci_high=r["high"],
                        p_value=r["p"],
                        p_adjusted=p_adj,
                        n_cases=r["n"],
                        mde=r["mde"],
                        status=status,
                    )
                )

            primary = policy.primary_metric
            if primary in candidate.metric_table and primary in baseline.metric_table:
                bt, ct = baseline.metric_table[primary], candidate.metric_table[primary]
                total_drop = 0.0
                flaky_drop = 0.0
                flaky_contributors = 0
                for key in keys:
                    if key not in bt or key not in ct:
                        continue
                    b, c = float(np.mean(bt[key])), float(np.mean(ct[key]))
                    flaky = is_flaky(bt[key]) or is_flaky(ct[key])
                    case_diff.append(CaseDiff(key, b, c, c - b, _categorize(b, c, flaky)))
                    if c < b:
                        total_drop += b - c
                        if is_flaky(bt[key]):  # already noisy before the change
                            flaky_drop += b - c
                            flaky_contributors += 1
                case_diff.sort(key=lambda x: (x.delta, x.case_key))
                if total_drop > 0:
                    flaky_share = flaky_drop / total_drop
                    concentrated = flaky_contributors <= max(3, len(keys) // 10)
                    if flaky_share >= 0.5 and concentrated and any(m.status == "fail" for m in metrics):
                        notes.append(
                            f"{flaky_share:.0%} of the drop comes from {flaky_contributors} case(s) that were already "
                            "flaky in the baseline; consider more repeats before acting on it"
                        )

            latency_note = _latency_note(baseline, candidate, keys, alpha, n_boot, seed)
            if latency_note:
                notes.append(latency_note)

    if errors:
        verdict = "error"
        reasons = errors + reasons
    elif any(not g.passed for g in gates) or critical_failures or any(m.gated and m.status == "fail" for m in metrics):
        verdict = "fail"
    elif any(m.gated and m.status == "inconclusive" for m in metrics):
        verdict = "inconclusive"
    else:
        verdict = "pass"

    return VerdictReport(
        verdict=verdict,
        reasons=reasons,
        notes=notes,
        metrics=metrics,
        gates=gates,
        critical_failures=critical_failures,
        case_diff=case_diff,
        excluded_cases=excluded,
        config_diff=diff,
        flaky_share_of_drop=flaky_share,
        baseline_run_id=baseline.run_id if baseline else None,
        candidate_run_id=candidate.run_id,
        policy=policy.model_dump(mode="json"),
    )


def _inconclusive_reason(name: str, r: dict[str, Any], policy: VerdictPolicy) -> str:
    base = f"{name}: {pts(r['delta'])}, CI [{pts(r['low'])}, {pts(r['high'])}] is too wide to decide"
    if r["n"] < MIN_CASES:
        return f"{base}; only {r['n']} comparable case(s), need at least {MIN_CASES}"
    half_width = (r["high"] - r["low"]) / 2.0
    room = r["delta"] + policy.non_inferiority_margin
    factor = data_multiplier(half_width, room) if room > 0 else None
    if factor is not None:
        return f"{base}; about {factor:.1f}x more cases or repeats would resolve it"
    return f"{base}; the observed drop is larger than the tolerated margin ({policy.non_inferiority_margin:.0%})"


def _latency_note(
    baseline: RunView, candidate: RunView, keys: Sequence[str], alpha: float, n_boot: int, seed: int
) -> str | None:
    pairs = [
        (float(np.median(baseline.latency_by_case[k])), float(np.median(candidate.latency_by_case[k])))
        for k in keys
        if baseline.latency_by_case.get(k) and candidate.latency_by_case.get(k)
    ]
    if len(pairs) < MIN_CASES:
        return None
    d = np.array([c - b for b, c in pairs])
    low, high = bootstrap_mean_ci(d, alpha=alpha, n_boot=min(n_boot, 2000), seed=seed)
    mean_base = float(np.mean([b for b, _ in pairs]))
    if mean_base <= 0:
        return None
    if low > 0 or high < 0:
        direction = "slower" if d.mean() > 0 else "faster"
        return f"median latency per case is {abs(d.mean()) / mean_base:.0%} {direction} (CI [{low:+.0f}, {high:+.0f}] ms)"
    return None
