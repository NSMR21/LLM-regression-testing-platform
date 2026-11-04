"""Run Alembic migrations programmatically (``rlens migrate`` and server start-up)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

HERE = Path(__file__).resolve().parent


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(HERE / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)
