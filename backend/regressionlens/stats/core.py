"""Paired comparison primitives.

The unit of analysis is the test case. For case c, ``d_c`` is the candidate's mean score over its
repeats minus the baseline's. Resampling or sign-flipping whole ``d_c`` values stays valid however
the repeats of one case are correlated (deterministic decoding, batching, server state), which is
what keeps identical configurations from raising false alarms.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_PERMUTATIONS = 10_000
EXACT_PERMUTATION_MAX_N = 16


def case_means(table: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {key: float(np.mean(values)) for key, values in table.items() if len(values)}


def paired_differences(
    baseline: Mapping[str, float], candidate: Mapping[str, float], keys: Sequence[str]
) -> np.ndarray:
    return np.array([candidate[k] - baseline[k] for k in keys], dtype=float)


def bootstrap_mean_ci(
    d: np.ndarray, alpha: float = 0.05, n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(d), resampling cases with replacement."""
    d = np.asarray(d, dtype=float)
    n = d.size
    if n == 0:
        return (math.nan, math.nan)
    if n == 1 or np.all(d == d[0]):
        m = float(d.mean())
        return (m, m)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, min(n_boot, 2_000_000 // n))
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        means[start : start + size] = d[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(low), float(high))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: int) -> float:
    """Student-t quantile by bisection on the exact CDF (no scipy dependency)."""
    if df <= 0 or not 0.0 < p < 1.0:
        return math.nan
    if df == 1:
        return math.tan(math.pi * (p - 0.5))
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, max(1.0, NormalDist().inv_cdf(p)) * 2.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def t_interval(d: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    n = d.size
    if n < 2:
        m = float(d.mean()) if n else math.nan
        return (m, m)
    half = t_quantile(1 - alpha / 2, n - 1) * float(d.std(ddof=1)) / math.sqrt(n)
    m = float(d.mean())
    return (m - half, m + half)


def paired_ci(
    d: np.ndarray, alpha: float = 0.05, n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0
) -> tuple[float, float]:
    """CI for the mean paired difference: the union of the percentile bootstrap and the Student-t
    interval. They agree for large suites; for small ones the bootstrap is too narrow and the
    t interval keeps the CI honest."""
    b_low, b_high = bootstrap_mean_ci(d, alpha=alpha, n_boot=n_boot, seed=seed)
    t_low, t_high = t_interval(d, alpha=alpha)
    return (min(b_low, t_low), max(b_high, t_high))


def sign_flip_pvalue(d: np.ndarray, n_perm: int = DEFAULT_PERMUTATIONS, seed: int = 0) -> float:
    """Two-sided paired permutation test of mean(d) == 0.

    Under the null (baseline and candidate are exchangeable), each d_c is equally likely to have
    either sign. Exact enumeration for up to 16 non-zero differences, Monte Carlo otherwise.
    With one repeat and binary scores this is the exact McNemar test.
    """
    d = np.asarray(d, dtype=float)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return 1.0
    observed = abs(d.sum())
    tol = 1e-9 * max(1.0, observed)
    if n <= EXACT_PERMUTATION_MAX_N:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        sums = np.abs(signs @ d)
        return float(np.mean(sums >= observed - tol))
    rng = np.random.default_rng(seed)
    hits = 0
    chunk = max(1, min(n_perm, 2_000_000 // n))
    for start in range(0, n_perm, chunk):
        size = min(chunk, n_perm - start)
        signs = rng.integers(0, 2, size=(size, n)) * 2 - 1
        hits += int(np.sum(np.abs(signs @ d) >= observed - tol))
    return (hits + 1) / (n_perm + 1)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment (controls the family-wise error rate)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[i])
        running = max(running, value)
        adjusted[i] = running
    return adjusted


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))
