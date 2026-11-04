"""Fixtures for tests that run the API, worker and database together.

Uses Postgres when RLENS_TEST_DATABASE_URL is set (e.g. postgresql://rlens:rlens@localhost/regressionlens_test),
otherwise a temporary SQLite file.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from regressionlens.api.app import create_app
from regressionlens.cli.client import Client
from regressionlens.config import Settings
from regressionlens.db import queue
from regressionlens.db.models import Base
from regressionlens.db.session import Database
from regressionlens.providers import make_provider
from regressionlens.spec import load_model, load_prompt, load_suite
from regressionlens.worker import Worker

from .helpers import EXAMPLE

ADMIN = "test-admin-token"


def database_url(tmp_path) -> str:
    return os.environ.get("RLENS_TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/rlens-test.sqlite3"


def reset(db: Database) -> None:
    Base.metadata.drop_all(db.engine)
    with db.engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))


class ProviderSwitch:
    """Provider factory whose behaviour tests can change (e.g. to simulate changed weights)."""

    def __init__(self):
        self.digest_override: str | None = None
        self.broken = False

    def __call__(self, spec):
        if self.broken:
            raise RuntimeError("provider construction failed")
        provider = make_provider(spec)
        if self.digest_override is not None:
            original = provider.resolve

            def resolve():
                identity = original()
                identity.digest = self.digest_override
                return identity

            provider.resolve = resolve
        return provider


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "RETRY_BASE_S", 0.01)
    url = database_url(tmp_path)
    db = Database(url)
    reset(db)
    settings = Settings(
        database_url=url, require_auth=True, admin_token=ADMIN, auto_migrate=True, poll_interval_s=0.02,
        analyze_wait_s=0.05, long_poll_max_s=3, public_url="http://rlens.test",
    )
    providers = ProviderSwitch()
    app = create_app(settings, db, provider_factory=providers)
    with TestClient(app) as http:
        http.headers["Authorization"] = f"Bearer {ADMIN}"
        worker = Worker(db, settings, worker_id="test-worker", provider_factory=providers)
        yield SimpleNamespace(http=http, db=db, worker=worker, settings=settings, providers=providers, app=app)
    worker.pool.shutdown(wait=True)
    reset(db)
    db.engine.dispose()


@pytest.fixture
def background_worker(stack):
    stop = threading.Event()
    thread = threading.Thread(target=stack.worker.run_forever, args=(stop,), daemon=True)
    thread.start()
    yield stack.worker
    stop.set()
    thread.join(timeout=10)


def example_body(prompt: str = "v1.yaml", repeats: int = 3, **extra) -> dict:
    suite = load_suite(EXAMPLE / "suite.yaml")
    body = {
        "suite": suite.model_dump(mode="json"),
        "prompt": load_prompt(EXAMPLE / "prompts" / prompt).model_dump(mode="json"),
        "model": load_model(EXAMPLE / "models" / "simulated.yaml").model_dump(mode="json"),
        "execution": {"repeats": repeats},
    }
    body.update(extra)
    return body


def client_for(stack) -> Client:
    return Client(token=ADMIN, http=stack.http)
