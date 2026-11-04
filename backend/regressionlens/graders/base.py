"""Grader interface and registry."""

from __future__ import annotations

import json
import re
import string
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..providers import Provider, make_provider
from ..spec import CaseSpec, GraderSpec, ModelSpec, SpecError, SuiteSpec


@dataclass
class GradeResult:
    value: float
    passed: bool
    rationale: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "passed": self.passed,
            "rationale": self.rationale,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class GradeInput:
    output: str
    case: CaseSpec
    messages: list[dict[str, str]] = field(default_factory=list)


class GraderContext:
    """Shared resources for graders: currently a cache of judge-model providers."""

    def __init__(self, timeout_s: float = 120.0):
        self.timeout_s = timeout_s
        self._providers: dict[str, Provider] = {}
        self._lock = threading.Lock()

    def provider_for(self, spec: ModelSpec) -> Provider:
        key = spec.model_dump_json()
        with self._lock:
            if key not in self._providers:
                self._providers[key] = make_provider(spec)
            return self._providers[key]

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()


class Grader:
    type: ClassVar[str]
    version: ClassVar[str] = "1.0"
    needs_expected: ClassVar[bool] = False
    deterministic: ClassVar[bool] = True

    def __init__(self, spec: GraderSpec, ctx: GraderContext | None = None):
        self.spec = spec
        self.cfg = spec.config
        self.ctx = ctx
        self.setup()

    def setup(self) -> None:
        """Validate and pre-compile configuration. Raise SpecError on bad config."""

    def check_case(self, case: CaseSpec) -> None:
        if self.needs_expected and case.expected is None and "value" not in self.cfg and "values" not in self.cfg:
            raise SpecError(f"grader {self.spec.id!r} ({self.type}) on case {case.key!r} needs `expected` or `value`")

    def describe(self) -> dict[str, Any]:
        """Extra identity for the run snapshot (e.g. the judge model's digest)."""
        return {}

    def grade(self, item: GradeInput) -> GradeResult:
        raise NotImplementedError

    def expected_values(self, case: CaseSpec) -> list[Any]:
        if "values" in self.cfg:
            return list(self.cfg["values"])
        if "value" in self.cfg:
            return [self.cfg["value"]]
        if isinstance(case.expected, list):
            return list(case.expected)
        return [case.expected]


_REGISTRY: dict[str, type[Grader]] = {}


def register(cls: type[Grader]) -> type[Grader]:
    _REGISTRY[cls.type] = cls
    return cls


def grader_class(type_name: str) -> type[Grader]:
    try:
        return _REGISTRY[type_name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise SpecError(f"unknown grader type {type_name!r} (known: {known})") from exc


def registered_types() -> dict[str, str]:
    return {name: cls.version for name, cls in sorted(_REGISTRY.items())}


def build_grader(spec: GraderSpec, ctx: GraderContext | None = None) -> Grader:
    return grader_class(spec.type)(spec, ctx)


def validate_suite(suite: SuiteSpec) -> None:
    """Check every grader's configuration against every case it applies to."""
    for case in suite.cases:
        for spec in suite.effective_graders(case):
            grader = build_grader(spec)
            grader.check_case(case)
    for metric in suite.verdict.metrics_to_gate + [g.metric for g in suite.verdict.gates]:
        known = {"pass_rate", "error_rate", "latency_p50_ms", "latency_p95_ms", *suite.grader_ids()}
        if metric not in known:
            raise SpecError(f"verdict refers to unknown metric {metric!r} (known: {', '.join(sorted(known))})")


def grader_versions(suite: SuiteSpec) -> dict[str, str]:
    """``{grader_id: "type@version"}`` for every grader used by the suite."""
    out: dict[str, str] = {}
    for case in suite.cases:
        for spec in suite.effective_graders(case):
            out[spec.id] = f"{spec.type}@{grader_class(spec.type).version}"
    return out


# --- text helpers ---------------------------------------------------------------------------

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str, ops: list[str] | None) -> str:
    for op in ops or []:
        if op == "strip":
            text = text.strip()
        elif op == "lower":
            text = text.lower()
        elif op == "collapse_whitespace":
            text = re.sub(r"\s+", " ", text).strip()
        elif op == "strip_punctuation":
            text = text.translate(_PUNCT)
        elif op == "strip_quotes":
            text = text.strip().strip("\"'`")
        else:
            raise SpecError(f"unknown normalization {op!r}")
    return text


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON from model output: the whole text, a fenced block, or the outermost {...}/[...]."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start, end = text.find(open_char), text.rfind(close_char)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise ValueError("no JSON value found in output")
