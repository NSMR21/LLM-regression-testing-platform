"""Deterministic graders: exact match, contains, regex, choice, numeric, JSON schema, JSON field."""

from __future__ import annotations

import math
import re
from typing import Any

import jsonschema

from ..spec import SpecError
from .base import GradeInput, Grader, GradeResult, extract_json, normalize, register


def _result(passed: bool, rationale: str = "", **details: Any) -> GradeResult:
    return GradeResult(value=1.0 if passed else 0.0, passed=passed, rationale=rationale, details=details)


def _negate(grader: Grader, result: GradeResult) -> GradeResult:
    if grader.cfg.get("negate"):
        return _result(not result.passed, f"negated: {result.rationale}", **result.details)
    return result


@register
class ExactMatch(Grader):
    """Output equals the expected value (or any of several) after optional normalization."""

    type = "exact_match"
    needs_expected = True

    def setup(self) -> None:
        self.ops = self.cfg.get("normalize", ["strip"])
        normalize("", self.ops)

    def grade(self, item: GradeInput) -> GradeResult:
        got = normalize(item.output, self.ops)
        targets = [normalize(str(v), self.ops) for v in self.expected_values(item.case)]
        if got in targets:
            return _result(True, "exact match")
        return _result(False, f"expected {targets[0]!r}, got {got[:120]!r}")


@register
class Contains(Grader):
    """Output contains the given substring(s). ``mode``: any (default) or all."""

    type = "contains"
    needs_expected = True

    def setup(self) -> None:
        self.mode = self.cfg.get("mode", "any")
        if self.mode not in {"any", "all"}:
            raise SpecError("contains.mode must be 'any' or 'all'")
        self.case_sensitive = bool(self.cfg.get("case_sensitive", False))

    def grade(self, item: GradeInput) -> GradeResult:
        hay = item.output if self.case_sensitive else item.output.lower()
        needles = [str(v) if self.case_sensitive else str(v).lower() for v in self.expected_values(item.case)]
        hits = [n for n in needles if n in hay]
        passed = bool(hits) if self.mode == "any" else len(hits) == len(needles)
        missing = [n for n in needles if n not in hits]
        return _negate(self, _result(passed, "found" if passed else f"missing {missing[:3]}", hits=hits))


@register
class Regex(Grader):
    type = "regex"

    def setup(self) -> None:
        pattern = self.cfg.get("pattern")
        if not isinstance(pattern, str):
            raise SpecError(f"regex grader {self.spec.id!r} needs a `pattern`")
        flags = 0
        for name in self.cfg.get("flags", []):
            flags |= getattr(re, str(name).upper(), 0)
        try:
            self.pattern = re.compile(pattern, flags)
        except re.error as exc:
            raise SpecError(f"bad regex {pattern!r}: {exc}") from exc
        self.fullmatch = self.cfg.get("match", "search") == "fullmatch"

    def grade(self, item: GradeInput) -> GradeResult:
        text = item.output.strip() if self.fullmatch else item.output
        match = self.pattern.fullmatch(text) if self.fullmatch else self.pattern.search(text)
        return _negate(self, _result(match is not None, "matched" if match else "no match"))


@register
class Choice(Grader):
    """Extract the first of a fixed set of labels (yes/no, A-D, ...) and compare to expected."""

    type = "choice"
    needs_expected = True

    def setup(self) -> None:
        choices = self.cfg.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SpecError(f"choice grader {self.spec.id!r} needs a non-empty `choices` list")
        self.choices = [str(c) for c in choices]
        self.patterns = [(c, re.compile(rf"(?<![\w]){re.escape(c)}(?![\w])", re.IGNORECASE)) for c in self.choices]

    def check_case(self, case) -> None:
        super().check_case(case)
        lowered = {c.lower() for c in self.choices}
        for value in self.expected_values(case):
            if str(value).lower() not in lowered:
                raise SpecError(f"case {case.key!r}: expected {value!r} is not one of {self.choices}")

    def extract(self, text: str) -> str | None:
        best: tuple[int, str] | None = None
        for choice, pattern in self.patterns:
            m = pattern.search(text)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), choice)
        return best[1] if best else None

    def grade(self, item: GradeInput) -> GradeResult:
        got = self.extract(item.output)
        if got is None:
            return _result(False, "no choice found in output", extracted=None)
        targets = {str(v).lower() for v in self.expected_values(item.case)}
        return _result(got.lower() in targets, f"extracted {got!r}", extracted=got)


_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


@register
class Numeric(Grader):
    """Parse a number from the output and compare within ``abs_tol`` / ``rel_tol``."""

    type = "numeric"
    needs_expected = True

    def setup(self) -> None:
        self.abs_tol = float(self.cfg.get("abs_tol", 0.0))
        self.rel_tol = float(self.cfg.get("rel_tol", 0.0))
        self.pick = self.cfg.get("pick", "first")
        if self.pick not in {"first", "last"}:
            raise SpecError("numeric.pick must be 'first' or 'last'")

    def grade(self, item: GradeInput) -> GradeResult:
        found = _NUMBER.findall(item.output)
        if not found:
            return _result(False, "no number in output")
        raw = found[0] if self.pick == "first" else found[-1]
        got = float(raw.replace(",", ""))
        for target in self.expected_values(item.case):
            target = float(target)
            if math.isclose(got, target, rel_tol=self.rel_tol, abs_tol=self.abs_tol) or got == target:
                return _result(True, f"{got} matches {target}", parsed=got)
        return _result(False, f"{got} != {self.expected_values(item.case)[0]}", parsed=got)


@register
class JsonSchema(Grader):
    """Output parses as JSON and validates against a JSON Schema (draft 2020-12)."""

    type = "json_schema"

    def setup(self) -> None:
        schema = self.cfg.get("schema")
        if not isinstance(schema, dict):
            raise SpecError(f"json_schema grader {self.spec.id!r} needs a `schema` (or `schema_file`)")
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise SpecError(f"invalid JSON schema: {exc.message}") from exc
        self.validator = jsonschema.Draft202012Validator(schema)

    def grade(self, item: GradeInput) -> GradeResult:
        try:
            value = extract_json(item.output)
        except ValueError:
            return _result(False, "output is not JSON")
        errors = sorted(self.validator.iter_errors(value), key=lambda e: list(e.path))
        if errors:
            first = errors[0]
            where = "/".join(str(p) for p in first.path) or "(root)"
            return _result(False, f"{where}: {first.message}", errors=len(errors))
        return _result(True, "valid")


_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def get_path(value: Any, path: str) -> Any:
    for name, index in _PATH_TOKEN.findall(path):
        if name:
            if not isinstance(value, dict) or name not in value:
                raise KeyError(path)
            value = value[name]
        else:
            i = int(index)
            if not isinstance(value, list) or i >= len(value):
                raise KeyError(path)
            value = value[i]
    return value


@register
class JsonField(Grader):
    """A field of the JSON output (dotted path, e.g. ``order.items[0].sku``) equals the expected value."""

    type = "json_field"
    needs_expected = True

    def setup(self) -> None:
        if not isinstance(self.cfg.get("path"), str):
            raise SpecError(f"json_field grader {self.spec.id!r} needs a `path`")
        self.ops = self.cfg.get("normalize", [])

    def targets(self, case) -> list[Any]:
        if "value" not in self.cfg and "values" not in self.cfg and isinstance(case.expected, dict):
            return [get_path(case.expected, self.cfg["path"])]
        return self.expected_values(case)

    def check_case(self, case) -> None:
        super().check_case(case)
        try:
            self.targets(case)
        except KeyError as exc:
            raise SpecError(f"case {case.key!r}: expected has no field {self.cfg['path']!r}") from exc

    def grade(self, item: GradeInput) -> GradeResult:
        try:
            got = get_path(extract_json(item.output), self.cfg["path"])
        except ValueError:
            return _result(False, "output is not JSON")
        except KeyError:
            return _result(False, f"missing field {self.cfg['path']}")
        for target in self.targets(item.case):
            if isinstance(got, str) and isinstance(target, str):
                if normalize(got, self.ops) == normalize(target, self.ops):
                    return _result(True, "field matches")
            elif got == target:
                return _result(True, "field matches")
        return _result(False, f"{self.cfg['path']} = {str(got)[:80]!r}")
