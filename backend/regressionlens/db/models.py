"""Database schema. Postgres in production; SQLite works for a single-process dev setup and tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..util import utcnow

JSONType = JSON().with_variant(JSONB(), "postgresql")
BigId = BigInteger().with_variant(Integer(), "sqlite")


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApiToken(Base):
    __tablename__ = "api_tokens"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200), default="")
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    scopes: Mapped[str] = mapped_column(String(100), default="read,run")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class _Versioned:
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    content_hash: Mapped[str] = mapped_column(String(80))
    spec: Mapped[dict[str, Any]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SuiteVersion(_Versioned, Base):
    __tablename__ = "suite_versions"
    __table_args__ = (UniqueConstraint("project_id", "content_hash"), Index("ix_suite_versions_name", "project_id", "name"))


class PromptVersion(_Versioned, Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("project_id", "content_hash"),)


class ModelConfig(_Versioned, Base):
    __tablename__ = "model_configs"
    __table_args__ = (UniqueConstraint("project_id", "content_hash"),)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_project_suite_created", "project_id", "suite_name", "created_at"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    suite_name: Mapped[str] = mapped_column(String(200))
    suite_version_id: Mapped[str] = mapped_column(ForeignKey("suite_versions.id"))
    prompt_version_id: Mapped[str] = mapped_column(ForeignKey("prompt_versions.id"))
    model_config_id: Mapped[str] = mapped_column(ForeignKey("model_configs.id"))
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    fingerprint: Mapped[str] = mapped_column(String(80), index=True)
    run_seed: Mapped[int] = mapped_column(BigInteger)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONType)
    environment: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    execution: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    total_trials: Mapped[int] = mapped_column(Integer, default=0)
    executed: Mapped[int] = mapped_column(Integer, default=0)
    graded: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    verdict: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    verdict_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    baseline_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    baseline_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    kind: Mapped[str] = mapped_column(String(20), default="run")  # run | rerun | rescore | fresh_baseline
    source_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    ci_meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    suite_version: Mapped[SuiteVersion] = relationship()
    prompt_version: Mapped[PromptVersion] = relationship()
    model_config: Mapped[ModelConfig] = relationship()


class Trial(Base):
    __tablename__ = "trials"
    __table_args__ = (UniqueConstraint("run_id", "case_key", "repeat_idx"),)

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    case_key: Mapped[str] = mapped_column(String(200))
    repeat_idx: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    order_idx: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | executed | graded
    messages: Mapped[list[dict[str, str]]] = mapped_column(JSONType, default=list)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class Baseline(Base):
    __tablename__ = "baselines"
    __table_args__ = (UniqueConstraint("project_id", "suite_name", "name"),)

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    suite_name: Mapped[str] = mapped_column(String(200))
    name: Mapped[str] = mapped_column(String(200))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True)


class BaselineEvent(Base):
    """History of baseline moves, for trend markers."""

    __tablename__ = "baseline_events"

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    suite_name: Mapped[str] = mapped_column(String(200))
    name: Mapped[str] = mapped_column(String(200))
    run_id: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class HumanLabel(Base):
    __tablename__ = "human_labels"
    __table_args__ = (UniqueConstraint("trial_id", "grader_id", "labeler"),)

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    trial_id: Mapped[int] = mapped_column(ForeignKey("trials.id", ondelete="CASCADE"), index=True)
    grader_id: Mapped[str] = mapped_column(String(200))
    grader_version: Mapped[str] = mapped_column(String(100))
    label: Mapped[float] = mapped_column(Float)
    labeler: Mapped[str] = mapped_column(String(200))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_claim", "queue", "status", "run_after", "id"),)

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    queue: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued | running | done | failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    locked_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
