"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

from regressionlens.spec import ModelSpec, SuiteSpec

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "support-qa"


def make_suite(n: int = 30, **verdict) -> SuiteSpec:
    return SuiteSpec.model_validate(
        {
            "name": "synthetic",
            "graders": [{"id": "exact", "type": "exact_match", "normalize": ["strip", "lower"]}],
            "verdict": verdict,
            "cases": [{"key": f"case-{i:03d}", "input": {"q": f"question {i}"}, "expected": f"answer {i}"} for i in range(n)],
        }
    )


def sim_model(**simulation) -> ModelSpec:
    return ModelSpec(name="sim", provider="simulated", model="sim-1", params={"temperature": 0.7}, simulation=simulation)
