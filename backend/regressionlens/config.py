"""Server settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get("RLENS_DATABASE_URL", "sqlite:///./rlens.sqlite3"))
    public_url: str = field(default_factory=lambda: os.environ.get("RLENS_PUBLIC_URL", "http://localhost:8080").rstrip("/"))
    admin_token: str | None = field(default_factory=lambda: os.environ.get("RLENS_ADMIN_TOKEN"))
    require_auth: bool = field(default_factory=lambda: _bool("RLENS_REQUIRE_AUTH", True))
    auto_migrate: bool = field(default_factory=lambda: _bool("RLENS_AUTO_MIGRATE", True))
    default_project: str = field(default_factory=lambda: os.environ.get("RLENS_DEFAULT_PROJECT", "default"))
    # Worker
    poll_interval_s: float = field(default_factory=lambda: float(os.environ.get("RLENS_POLL_INTERVAL", "0.5")))
    grade_concurrency: int = field(default_factory=lambda: int(os.environ.get("RLENS_GRADE_CONCURRENCY", "2")))
    stale_job_s: float = field(default_factory=lambda: float(os.environ.get("RLENS_STALE_JOB_SECONDS", "1800")))
    analyze_wait_s: float = field(default_factory=lambda: float(os.environ.get("RLENS_ANALYZE_WAIT_SECONDS", "2")))
    long_poll_max_s: float = field(default_factory=lambda: float(os.environ.get("RLENS_LONG_POLL_MAX", "60")))


def get_settings() -> Settings:
    return Settings()
