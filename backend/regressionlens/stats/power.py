"""Minimum detectable effect: how big a drop this suite can reliably catch."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np

_Z = NormalDist()


def mde(sd_d: float, n_cases: int, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Smallest true mean difference detected with probability ``power`` at two-sided level ``alpha``."""
    if n_cases <= 0 or sd_d is None or math.isnan(sd_d):
        return None
    return (_Z.inv_cdf(1 - alpha / 2) + _Z.inv_cdf(power)) * sd_d / math.sqrt(n_cases)


def null_difference_sd(table: Mapping[str, Sequence[float]]) -> float | None:
    """SD of a per-case difference between two runs of this same configuration, estimated from
    within-case variation of one run: Var(d_c) = 2 * s_c^2 / k_c. Needs >= 2 repeats per case."""
    variances = []
    for values in table.values():
        k = len(values)
        if k < 2:
            return None
        variances.append(2.0 * float(np.var(values, ddof=1)) / k)
    if not variances:
        return None
    return math.sqrt(float(np.mean(variances)))


def data_multiplier(current_half_width: float, target_half_width: float) -> float | None:
    """How many times more cases (or, for noise-dominated suites, repeats) shrink the CI to target."""
    if target_half_width <= 0 or current_half_width <= 0:
        return None
    return (current_half_width / target_half_width) ** 2
