"""Run Alembic migrations programmatically (``rlens migrate`` and server/worker start-up).

The API and the workers may start at the same time and both migrate. Migrations are serialized
with a process-wide lock (Alembic's context is global) and, on Postgres, an advisory lock so
separate containers take turns; whoever comes second finds the schema already at head.
"""

from __future__ import annotations

import threading
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from .session import make_engine

HERE = Path(__file__).resolve().parent
_LOCK = threading.Lock()
_ADVISORY_LOCK_ID = 72_635_401


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(HERE / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    with _LOCK:
        engine = make_engine(url)
        try:
            if engine.dialect.name == "postgresql":
                with engine.connect() as conn:
                    conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _ADVISORY_LOCK_ID})
                    try:
                        command.upgrade(alembic_config(url), revision)
                    finally:
                        conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _ADVISORY_LOCK_ID})
            else:
                command.upgrade(alembic_config(url), revision)
        finally:
            engine.dispose()
