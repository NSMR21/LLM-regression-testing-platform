"""Serializable records of a run. The same shapes are used by local JSON files and the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ScoreRecord(BaseModel):
    value: float
    passed: bool
    rationale: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    error: bool = False


class TrialRecord(BaseModel):
    case_key: str
    repeat_idx: int
    seed: int | None = None
    order: int = 0
    messages: list[dict[str, str]] = Field(default_factory=list)
    output: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    attempts: int = 1
    raw: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, ScoreRecord] = Field(default_factory=dict)
    passed: bool | None = None


class RunSummary(BaseModel):
    n_cases: int
    n_trials: int
    repeats: int
    metrics: dict[str, float]
    metric_ci: dict[str, list[float]] = Field(default_factory=dict)
    per_tag: dict[str, dict[str, float]] = Field(default_factory=dict)
    error_rate: float = 0.0
    grader_error_rate: float = 0.0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    flaky_cases: list[dict[str, Any]] = Field(default_factory=list)
    flaky_rate: float = 0.0
    mde: float | None = None
    notes: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    run_id: str
    created_at: datetime
    finished_at: datetime | None = None
    status: str = "completed"
    fingerprint: str
    run_seed: int
    snapshot: dict[str, Any]
    environment: dict[str, Any]
    suite: dict[str, Any]
    prompt: dict[str, Any]
    model: dict[str, Any]
    execution: dict[str, Any]
    trials: list[TrialRecord] = Field(default_factory=list)
    summary: RunSummary | None = None
    verdict: dict[str, Any] | None = None
    baseline_run_id: str | None = None
    ci_meta: dict[str, Any] = Field(default_factory=dict)
