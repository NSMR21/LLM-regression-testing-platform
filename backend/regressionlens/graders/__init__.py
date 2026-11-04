"""Graders turn a model output into a score in [0, 1] and a pass flag."""

from . import builtin, judge  # noqa: F401  (registers grader types)
from .base import (
    GradeInput,
    Grader,
    GraderContext,
    GradeResult,
    build_grader,
    extract_json,
    grader_class,
    grader_versions,
    normalize,
    registered_types,
    validate_suite,
)

__all__ = [
    "GradeInput",
    "GradeResult",
    "Grader",
    "GraderContext",
    "build_grader",
    "extract_json",
    "grader_class",
    "grader_versions",
    "normalize",
    "registered_types",
    "validate_suite",
]
