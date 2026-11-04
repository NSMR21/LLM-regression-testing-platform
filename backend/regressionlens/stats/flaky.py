"""Flaky tests: cases whose outcome changes between repeats of an identical configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class FlakyCase:
    case_key: str
    pass_rate: float
    score: float  # 0 = stable, 1 = coin flip
    repeats: int

    def to_dict(self) -> dict:
        return asdict(self)


def flakiness_score(values: Sequence[float]) -> float:
    """4 p (1 - p) for binary outcomes; scaled within-case SD for continuous scores."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    if np.all((arr == 0.0) | (arr == 1.0)):
        p = float(arr.mean())
        return 4.0 * p * (1.0 - p)
    return float(min(1.0, 2.0 * arr.std(ddof=1)))


def flaky_cases(table: Mapping[str, Sequence[float]], threshold: float = 0.0) -> list[FlakyCase]:
    out = []
    for key, values in table.items():
        if len(values) < 2:
            continue
        score = flakiness_score(values)
        if score > threshold:
            out.append(FlakyCase(key, float(np.mean(values)), score, len(values)))
    out.sort(key=lambda f: (-f.score, f.case_key))
    return out


def is_flaky(values: Sequence[float]) -> bool:
    return len(values) >= 2 and flakiness_score(values) > 0.0
