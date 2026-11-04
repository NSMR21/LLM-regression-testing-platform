"""Agreement between a grader and human labels: accuracy, Cohen's kappa, confusion, bootstrap CIs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

KAPPA_CALIBRATED = 0.6
MIN_LABELS = 30


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Unweighted Cohen's kappa for categorical labels."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    n = a_arr.size
    if n == 0:
        return math.nan
    categories = np.union1d(a_arr, b_arr)
    observed = float(np.mean(a_arr == b_arr))
    expected = sum(float(np.mean(a_arr == c)) * float(np.mean(b_arr == c)) for c in categories)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int]) -> float:
    """Quadratic-weighted kappa for ordinal labels such as a 1-5 scale."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    cats = list(categories)
    k = len(cats)
    if a_arr.size == 0 or k < 2:
        return math.nan
    index = {c: i for i, c in enumerate(cats)}
    observed = np.zeros((k, k))
    for x, y in zip(a_arr, b_arr, strict=True):
        observed[index[int(x)], index[int(y)]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    denom = float((weights * expected).sum())
    if denom == 0:
        return 1.0
    return 1.0 - float((weights * observed).sum()) / denom


@dataclass
class AgreementReport:
    n: int
    accuracy: float
    kappa: float
    kappa_ci: tuple[float, float]
    accuracy_ci: tuple[float, float]
    false_pass_rate: float | None  # grader said pass, human said fail
    false_fail_rate: float | None  # grader said fail, human said pass
    confusion: dict[str, int]
    trust: str  # calibrated | weak | uncalibrated

    def to_dict(self) -> dict:
        out = asdict(self)
        out["kappa_ci"] = list(self.kappa_ci)
        out["accuracy_ci"] = list(self.accuracy_ci)
        return out


def binary_agreement(
    grader: Sequence[int], human: Sequence[int], n_boot: int = 2000, seed: int = 0
) -> AgreementReport:
    g = np.asarray(grader, dtype=int)
    h = np.asarray(human, dtype=int)
    n = g.size
    if n == 0:
        return AgreementReport(0, math.nan, math.nan, (math.nan, math.nan), (math.nan, math.nan), None, None, {}, "uncalibrated")
    accuracy = float(np.mean(g == h))
    kappa = cohen_kappa(g, h)
    rng = np.random.default_rng(seed)
    kappas, accs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        kappas.append(cohen_kappa(g[idx], h[idx]))
        accs.append(float(np.mean(g[idx] == h[idx])))
    kappas_arr = np.asarray([k for k in kappas if not math.isnan(k)])
    kappa_ci = (float(np.quantile(kappas_arr, 0.025)), float(np.quantile(kappas_arr, 0.975))) if kappas_arr.size else (math.nan, math.nan)
    accuracy_ci = (float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975)))
    tp = int(np.sum((g == 1) & (h == 1)))
    fp = int(np.sum((g == 1) & (h == 0)))
    fn = int(np.sum((g == 0) & (h == 1)))
    tn = int(np.sum((g == 0) & (h == 0)))
    false_pass = fp / (fp + tp) if (fp + tp) else None
    false_fail = fn / (fn + tn) if (fn + tn) else None
    if n >= MIN_LABELS and kappa >= KAPPA_CALIBRATED:
        trust = "calibrated"
    elif n >= MIN_LABELS:
        trust = "weak"
    else:
        trust = "uncalibrated"
    return AgreementReport(
        n=n,
        accuracy=accuracy,
        kappa=kappa,
        kappa_ci=kappa_ci,
        accuracy_ci=accuracy_ci,
        false_pass_rate=false_pass,
        false_fail_rate=false_fail,
        confusion={"grader_pass_human_pass": tp, "grader_pass_human_fail": fp, "grader_fail_human_pass": fn, "grader_fail_human_fail": tn},
        trust=trust,
    )
