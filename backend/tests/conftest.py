from __future__ import annotations

import pytest

from regressionlens.spec import ExecutionSpec, PromptSpec, SuiteSpec

from .helpers import make_suite
from .server_fixtures import background_worker, stack  # noqa: F401  (fixtures)


@pytest.fixture
def suite() -> SuiteSpec:
    return make_suite()


@pytest.fixture
def prompt() -> PromptSpec:
    return PromptSpec(name="p1", system="Be brief.", template="Q: {{ q }}")


@pytest.fixture
def execution() -> ExecutionSpec:
    return ExecutionSpec(repeats=4)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("RLENS_HOME", str(tmp_path / ".rlens"))
