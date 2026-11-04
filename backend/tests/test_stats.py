import math

import numpy as np
import pytest
from scipy import stats as sps

from regressionlens.spec import VerdictPolicy
from regressionlens.stats import (
    RunView,
    binary_agreement,
    bootstrap_mean_ci,
    cohen_kappa,
    evaluate,
    flakiness_score,
    flaky_cases,
    holm_adjust,
    mcnemar_exact,
    mde,
    null_difference_sd,
    sign_flip_pvalue,
    weighted_kappa,
)
from regressionlens.stats.core import paired_ci, t_quantile


def test_t_quantile_matches_scipy():
    for df in [1, 2, 3, 4, 5, 8, 10, 20, 30, 60, 120]:
        for p in [0.9, 0.95, 0.975, 0.995]:
            assert t_quantile(p, df) == pytest.approx(sps.t.ppf(p, df), rel=1e-7)


def test_sign_flip_exact_small_n():
    # All 5 differences positive: only the all-plus and all-minus flips are as extreme.
    assert sign_flip_pvalue(np.ones(5)) == pytest.approx(2 / 32)
    assert sign_flip_pvalue(np.zeros(10)) == 1.0
    assert sign_flip_pvalue(np.array([0.0, 0.0, 1.0])) == 1.0


def test_sign_flip_equals_mcnemar_for_binary_single_repeat():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2, 40)
    cand = np.where(rng.random(40) < 0.3, 1 - base, base)
    d = (cand - base).astype(float)
    b, c = int(np.sum(d > 0)), int(np.sum(d < 0))
    expected = sps.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    assert mcnemar_exact(b, c) == pytest.approx(expected)
    assert sign_flip_pvalue(d, n_perm=200_000) == pytest.approx(expected, abs=0.01)


def test_sign_flip_monte_carlo_large_n():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1.0, 200)
    p = sign_flip_pvalue(d)
    t_p = sps.ttest_1samp(d, 0).pvalue
    assert p == pytest.approx(t_p, abs=0.01)


def test_holm():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]) == []


def test_bootstrap_ci_degenerate_and_coverage():
    assert bootstrap_mean_ci(np.array([])) == (pytest.approx(math.nan, nan_ok=True), pytest.approx(math.nan, nan_ok=True))
    assert bootstrap_mean_ci(np.zeros(10)) == (0.0, 0.0)
    rng = np.random.default_rng(3)
    covered = 0
    for i in range(200):
        d = rng.normal(0.1, 1.0, 50)
        lo, hi = paired_ci(d, n_boot=1000, seed=i)
        covered += lo <= 0.1 <= hi
    assert covered / 200 >= 0.9


def test_paired_ci_is_never_narrower_than_t_interval():
    d = np.array([0.4, 0.2, 0.0, 0.6, 0.2, -0.2])
    lo, hi = paired_ci(d, n_boot=2000)
    half = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    assert lo <= d.mean() - half + 1e-3 and hi >= d.mean() + half - 1e-3


def test_mde_and_null_sd():
    assert mde(0.2, 100) == pytest.approx((1.959964 + 0.841621) * 0.02, rel=1e-4)
    assert mde(0.2, 0) is None
    table = {"a": [1.0, 0.0, 1.0, 0.0], "b": [1.0, 1.0, 1.0, 1.0]}
    expected = math.sqrt(np.mean([2 * np.var([1, 0, 1, 0], ddof=1) / 4, 0.0]))
    assert null_difference_sd(table) == pytest.approx(expected)
    assert null_difference_sd({"a": [1.0]}) is None


def test_flakiness():
    assert flakiness_score([1, 1, 1]) == 0
    assert flakiness_score([1, 0, 1, 0]) == 1
    cases = flaky_cases({"stable": [1, 1, 1], "flaky": [1, 0, 1], "single": [0]})
    assert [c.case_key for c in cases] == ["flaky"]
    assert cases[0].pass_rate == pytest.approx(2 / 3)


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # Classic worked example: 20/5/10/15 table -> kappa = 0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], range(1, 6)) == pytest.approx(1.0)
    assert weighted_kappa([1, 2, 3], [3, 2, 1], [1, 2, 3]) < 0
    report = binary_agreement(a, b, n_boot=300)
    assert report.n == 50 and report.kappa == pytest.approx(0.4)
    assert report.trust == "weak"
    assert report.kappa_ci[0] < 0.4 < report.kappa_ci[1]
    assert report.false_pass_rate == pytest.approx(5 / 25)


# --- verdict --------------------------------------------------------------------------------


def view(p: np.ndarray, k: int, rng, shift: float = 0.0, **kw) -> RunView:
    table = {f"c{i}": list((rng.random(k) < min(1, max(0, pi + shift))).astype(float)) for i, pi in enumerate(p)}
    return RunView(
        metric_table={"pass_rate": table, "exact": table},
        case_hashes={key: "h" for key in table},
        grader_versions={"exact": "exact_match@1.0"},
        **kw,
    )


POLICY = VerdictPolicy(non_inferiority_margin=0.05)


def test_verdict_pass_on_identical_deterministic_runs():
    rng = np.random.default_rng(0)
    p = np.where(rng.random(40) < 0.8, 1.0, 0.0)
    report = evaluate(view(p, 3, rng), view(p, 3, rng), POLICY, n_boot=500, n_perm=500)
    assert report.verdict == "pass"
    assert report.metric("pass_rate").delta == 0


def test_verdict_fail_on_clear_regression_and_diff():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.6, 1.0, 60)
    base, cand = view(p, 5, rng), view(p, 5, rng, shift=-0.3)
    report = evaluate(cand, base, POLICY, n_boot=500, n_perm=2000)
    assert report.verdict == "fail"
    m = report.metric("pass_rate")
    assert m.status == "fail" and m.ci_high < 0 and m.p_adjusted < 0.05
    assert report.case_diff[0].delta <= report.case_diff[-1].delta
    assert any("dropped" in r for r in report.reasons)


def test_verdict_inconclusive_when_noisy_and_small():
    rng = np.random.default_rng(2)
    p = np.full(8, 0.5)
    report = evaluate(view(p, 2, rng), view(p, 2, rng), POLICY, n_boot=500, n_perm=500)
    assert report.verdict in {"inconclusive", "fail"}
    if report.verdict == "inconclusive":
        assert any("too wide" in r for r in report.reasons)


def test_verdict_error_conditions():
    rng = np.random.default_rng(3)
    p = np.full(20, 0.9)
    base = view(p, 3, rng)
    assert evaluate(view(p, 3, rng, error_rate=0.2), base, POLICY).verdict == "error"
    changed = view(p, 3, rng)
    changed.grader_versions = {"exact": "exact_match@2.0"}
    report = evaluate(changed, base, POLICY)
    assert report.verdict == "error" and "re-score" in report.reasons[0]
    disjoint = view(p, 3, rng)
    disjoint.case_hashes = {k: "other" for k in disjoint.case_hashes}
    assert evaluate(disjoint, base, POLICY).verdict == "error"


def test_gates_and_critical_cases_without_baseline():
    rng = np.random.default_rng(4)
    p = np.full(10, 1.0)
    p[0] = 0.0
    policy = VerdictPolicy(gates=[{"metric": "pass_rate", "min": 0.95}])
    report = evaluate(view(p, 3, rng, critical_cases=["c0"]), None, policy)
    assert report.verdict == "fail"
    assert report.critical_failures == ["c0"]
    assert not report.gates[0].passed
    ok = evaluate(view(np.full(10, 1.0), 3, rng), None, policy)
    assert ok.verdict == "pass" and "no baseline" in ok.notes[0]


def test_changed_and_quarantined_cases_are_excluded():
    rng = np.random.default_rng(5)
    p = np.full(12, 1.0)
    base = view(p, 2, rng)
    cand = view(p, 2, rng)
    cand.metric_table["pass_rate"]["c0"] = [0.0, 0.0]
    cand.case_hashes["c1"] = "changed"
    policy = VerdictPolicy(quarantine=["c0"])
    report = evaluate(cand, base, policy, n_boot=200, n_perm=200)
    assert report.excluded_cases == {"c1": "case changed since baseline", "c0": "quarantined"}
    assert report.metric("pass_rate").n_cases == 10
    assert report.verdict == "pass"


def test_report_is_json_serializable():
    import json

    rng = np.random.default_rng(6)
    p = np.full(5, 0.5)
    report = evaluate(view(p, 1, rng), view(p, 1, rng), POLICY, n_boot=100, n_perm=100)
    json.dumps(report.to_dict())


def test_false_alarm_rate_is_controlled():
    """A/A: same configuration twice. FAIL should happen at most ~alpha/2 of the time."""
    rng = np.random.default_rng(42)
    fails = 0
    sims = 300
    for _ in range(sims):
        p = rng.uniform(0.3, 1.0, 40)
        report = evaluate(view(p, 3, rng), view(p, 3, rng), POLICY, n_boot=200, n_perm=500)
        fails += report.verdict == "fail"
    assert fails / sims <= 0.05


def test_power_against_a_real_drop():
    """60 cases x 5 repeats, true drop of 15 pts: SE of the mean difference is ~3.6 pts, so the
    expected power at alpha = 0.05 is ~0.99."""
    rng = np.random.default_rng(7)
    detected = 0
    sims = 60
    for _ in range(sims):
        p = rng.uniform(0.5, 1.0, 60)
        report = evaluate(view(p, 5, rng, shift=-0.15), view(p, 5, rng), POLICY, n_boot=200, n_perm=1000)
        detected += report.verdict == "fail"
    assert detected / sims >= 0.9
