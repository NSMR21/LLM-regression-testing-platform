"""Specifications users write: test suites, prompt versions, model configs, execution settings.

Everything here is plain data validated with pydantic. Files may reference other files with
``*_file`` keys inside grader specs (e.g. ``rubric_file`` or ``schema_file``); the loader
inlines their content so a suite's content hash covers everything that affects scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from jinja2 import StrictUndefined
from jinja2.exceptions import TemplateError
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .util import content_hash

RESERVED_METRICS = {"pass_rate", "error_rate", "latency_ms", "latency_p50_ms", "latency_p95_ms"}


class SpecError(ValueError):
    """A suite, prompt or model spec is invalid."""


class GraderSpec(BaseModel):
    """One grader attached to a case (or to every case, when listed at suite level).

    ``id`` names the metric this grader produces; ``type`` selects the implementation.
    Any further keys are the grader's configuration.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    type: str = Field(min_length=1)
    required: bool = True

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class Gate(BaseModel):
    """An absolute threshold on a candidate metric, e.g. schema validity >= 0.99."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _has_bound(self) -> Gate:
        if self.min is None and self.max is None:
            raise ValueError(f"gate on {self.metric!r} needs min or max")
        return self


class VerdictPolicy(BaseModel):
    """How a comparison turns into pass / fail / inconclusive. See docs/ARCHITECTURE.md §8.3."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str = "pass_rate"
    gated_metrics: list[str] | None = None
    alpha: float = Field(0.05, gt=0, lt=0.5)
    non_inferiority_margin: float = Field(0.02, ge=0, le=1)
    min_effect: float = Field(0.0, ge=0, le=1)
    gates: list[Gate] = Field(default_factory=list)
    inconclusive: Literal["warn", "fail", "pass"] = "warn"
    max_error_rate: float = Field(0.05, ge=0, le=1)
    quarantine: list[str] = Field(default_factory=list)

    @property
    def metrics_to_gate(self) -> list[str]:
        return list(self.gated_metrics) if self.gated_metrics else [self.primary_metric]


class CaseSpec(BaseModel):
    """One test case: inputs for the prompt template, an optional expected value, graders."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=200)
    input: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None
    graders: list[GraderSpec] = Field(default_factory=list)
    use_default_graders: bool = True
    tags: list[str] = Field(default_factory=list)
    critical: bool = False


class SuiteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    graders: list[GraderSpec] = Field(default_factory=list)
    verdict: VerdictPolicy = Field(default_factory=VerdictPolicy)
    cases: list[CaseSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> SuiteSpec:
        seen: set[str] = set()
        for case in self.cases:
            if case.key in seen:
                raise ValueError(f"duplicate case key {case.key!r}")
            seen.add(case.key)
        default_ids = [g.id for g in self.graders]
        if len(default_ids) != len(set(default_ids)):
            raise ValueError("duplicate grader id in suite-level graders")
        for case in self.cases:
            graders = self.effective_graders(case)
            if not graders:
                raise ValueError(f"case {case.key!r} has no graders")
            for g in graders:
                if g.id in RESERVED_METRICS:
                    raise ValueError(f"grader id {g.id!r} is reserved")
        return self

    def effective_graders(self, case: CaseSpec) -> list[GraderSpec]:
        """Suite defaults, overridden or extended by the case's own graders (matched by id)."""
        merged: dict[str, GraderSpec] = {g.id: g for g in self.graders} if case.use_default_graders else {}
        for g in case.graders:
            merged[g.id] = g
        return list(merged.values())

    def case(self, key: str) -> CaseSpec:
        for c in self.cases:
            if c.key == key:
                return c
        raise KeyError(key)

    def content_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))

    def case_hash(self, case: CaseSpec) -> str:
        """Hash of everything that determines how a case is scored. Cases whose hash differs
        between two runs are not compared."""
        return content_hash(
            {
                "input": case.input,
                "expected": case.expected,
                "graders": [g.model_dump(mode="json") for g in self.effective_graders(case)],
            }
        )

    def case_hashes(self) -> dict[str, str]:
        return {c.key: self.case_hash(c) for c in self.cases}

    def grader_ids(self) -> list[str]:
        ids: list[str] = []
        for case in self.cases:
            for g in self.effective_graders(case):
                if g.id not in ids:
                    ids.append(g.id)
        return ids


class PromptSpec(BaseModel):
    """A prompt version: a system prompt and a user template (Jinja2, sandboxed)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    system: str = ""
    template: str = Field(min_length=1)

    def content_hash(self) -> str:
        return content_hash({"system": self.system, "template": self.template})

    def render(self, variables: dict[str, Any]) -> list[dict[str, str]]:
        try:
            messages: list[dict[str, str]] = []
            if self.system:
                messages.append({"role": "system", "content": _JINJA.from_string(self.system).render(**variables)})
            messages.append({"role": "user", "content": _JINJA.from_string(self.template).render(**variables)})
        except TemplateError as exc:
            raise SpecError(f"prompt {self.name!r} failed to render: {exc}") from exc
        return messages


_JINJA = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)

ProviderName = Literal["ollama", "openai_compat", "simulated"]


class ModelSpec(BaseModel):
    """Which model to call and how. ``endpoint`` and ``api_key_env`` are deployment details and
    are not part of the config fingerprint; the resolved model digest is."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    provider: ProviderName
    model: str = Field(min_length=1)
    endpoint: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    concurrency: int = Field(1, ge=1, le=64)
    api_key_env: str | None = None
    digest: str | None = None
    simulation: dict[str, Any] | None = None

    def identity_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"endpoint", "api_key_env", "concurrency", "name"})

    def content_hash(self) -> str:
        return content_hash(self.identity_dict())


class ExecutionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeats: int = Field(3, ge=1, le=200)
    seed_policy: Literal["derived", "none"] = "derived"
    timeout_s: float = Field(120.0, gt=0, le=3600)
    retries: int = Field(2, ge=0, le=10)
    warmup: bool = True
    shuffle: bool = True


# ---------------------------------------------------------------------------------------------
# Loading from files
# ---------------------------------------------------------------------------------------------


def _read_structured(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read {path}: {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"{path} is not valid YAML/JSON: {exc}") from exc


def _inline_file_refs(grader: dict[str, Any], base: Path) -> dict[str, Any]:
    out = dict(grader)
    for key in [k for k in grader if k.endswith("_file")]:
        ref = grader[key]
        if not isinstance(ref, str):
            raise SpecError(f"{key} must be a path string")
        target = (base / ref).resolve()
        if target.suffix.lower() in {".json", ".yaml", ".yml"}:
            value: Any = _read_structured(target)
        else:
            try:
                value = target.read_text(encoding="utf-8")
            except OSError as exc:
                raise SpecError(f"cannot read {target}: {exc}") from exc
        out.pop(key)
        out[key[: -len("_file")]] = value
    return out


def _validate(model: type[BaseModel], data: Any, source: str) -> Any:
    if not isinstance(data, dict):
        raise SpecError(f"{source}: expected a mapping at the top level")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise SpecError(f"{source}: {exc}") from exc


def suite_from_dict(data: Any, source: str = "suite") -> SuiteSpec:
    return _validate(SuiteSpec, data, source)


def load_suite(path: str | Path) -> SuiteSpec:
    path = Path(path)
    data = _read_structured(path)
    if isinstance(data, dict):
        base = path.parent
        # Top-level keys starting with "x-" hold YAML anchors and are ignored.
        data = {k: v for k, v in data.items() if not str(k).startswith("x-")}
        data["graders"] = [_inline_file_refs(g, base) for g in data.get("graders") or []]
        cases = []
        for case in data.get("cases") or []:
            if isinstance(case, dict) and case.get("graders"):
                case = dict(case)
                case["graders"] = [_inline_file_refs(g, base) for g in case["graders"]]
            cases.append(case)
        data["cases"] = cases
    return _validate(SuiteSpec, data, str(path))


def load_prompt(path: str | Path) -> PromptSpec:
    path = Path(path)
    data = _read_structured(path)
    if isinstance(data, dict) and "template_file" in data:
        data = dict(data)
        data["template"] = (path.parent / data.pop("template_file")).read_text(encoding="utf-8")
    if isinstance(data, dict) and "system_file" in data:
        data = dict(data)
        data["system"] = (path.parent / data.pop("system_file")).read_text(encoding="utf-8")
    return _validate(PromptSpec, data, str(path))


def load_model(path: str | Path) -> ModelSpec:
    path = Path(path)
    return _validate(ModelSpec, _read_structured(path), str(path))


def dump_json(obj: BaseModel) -> str:
    return json.dumps(obj.model_dump(mode="json"), indent=2, ensure_ascii=False)
