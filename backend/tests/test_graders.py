import pytest

from regressionlens.graders import GradeInput, GraderContext, build_grader, extract_json
from regressionlens.spec import CaseSpec, GraderSpec, SpecError


def grade(spec: dict, output: str, expected=None, ctx=None, messages=None):
    grader = build_grader(GraderSpec.model_validate(spec), ctx)
    case = CaseSpec(key="k", expected=expected, input={"q": "x"})
    grader.check_case(case)
    return grader.grade(GradeInput(output=output, case=case, messages=messages or []))


def test_exact_match_normalization_and_alternatives():
    assert grade({"id": "g", "type": "exact_match", "normalize": ["strip", "lower"]}, "  Paris \n", "paris").passed
    assert not grade({"id": "g", "type": "exact_match"}, "Paris.", "Paris").passed
    assert grade({"id": "g", "type": "exact_match", "normalize": ["strip", "strip_punctuation"]}, "Paris.", "Paris").passed
    assert grade({"id": "g", "type": "exact_match"}, "Lyon", ["Paris", "Lyon"]).passed


def test_contains_modes_and_negate():
    assert grade({"id": "g", "type": "contains", "values": ["30 days", "receipt"]}, "Within 30 days.").passed
    assert not grade({"id": "g", "type": "contains", "values": ["30 days", "receipt"], "mode": "all"}, "Within 30 days.").passed
    assert not grade({"id": "g", "type": "contains", "value": "as an ai", "negate": True}, "As an AI, I cannot").passed


def test_regex():
    assert grade({"id": "g", "type": "regex", "pattern": r"^\d{3}$", "match": "fullmatch"}, " 123 ").passed
    assert not grade({"id": "g", "type": "regex", "pattern": r"^\d{3}$", "match": "fullmatch"}, "1234").passed
    with pytest.raises(SpecError):
        build_grader(GraderSpec(id="g", type="regex", pattern="("))


def test_choice_extracts_first_label():
    spec = {"id": "g", "type": "choice", "choices": ["yes", "no"]}
    assert grade(spec, "Yes — within 30 days you can.", "yes").passed
    assert not grade(spec, "No, but yes if you have a receipt", "yes").passed
    result = grade(spec, "I cannot say", "yes")
    assert not result.passed and "no choice" in result.rationale
    assert not grade(spec, "Nobody knows", "no").passed  # whole-word match only


def test_numeric():
    assert grade({"id": "g", "type": "numeric"}, "It costs $15.", 15).passed
    assert grade({"id": "g", "type": "numeric"}, "about 1,250 units", 1250).passed
    assert grade({"id": "g", "type": "numeric", "rel_tol": 0.01}, "3.1416", 3.14159).passed
    assert not grade({"id": "g", "type": "numeric"}, "30 or 60 days", 60).passed
    assert grade({"id": "g", "type": "numeric", "pick": "last"}, "30 or 60 days", 60).passed


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Here you go: {"a": 3} hope it helps') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("no json here")


SCHEMA = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False}


def test_json_schema_and_field():
    assert grade({"id": "s", "type": "json_schema", "schema": SCHEMA}, '{"id": "A-1"}').passed
    bad = grade({"id": "s", "type": "json_schema", "schema": SCHEMA}, '{"id": 3}')
    assert not bad.passed and "id" in bad.rationale
    assert not grade({"id": "s", "type": "json_schema", "schema": SCHEMA}, "nope").passed
    with pytest.raises(SpecError):
        build_grader(GraderSpec(id="s", type="json_schema", schema={"type": "not-a-type"}))
    field = {"id": "f", "type": "json_field", "path": "order.items[0].sku"}
    out = '{"order": {"items": [{"sku": "X1"}]}}'
    assert grade(field, out, "X1").passed
    assert grade(field, out, {"order": {"items": [{"sku": "X1"}]}}).passed
    assert not grade(field, '{"order": {}}', "X1").passed


def _judge_spec(accuracy: float, scale: str = "binary") -> dict:
    return {
        "id": "judge",
        "type": "llm_judge",
        "rubric": "The response must state the correct answer.",
        "scale": scale,
        "judge_model": {"provider": "simulated", "model": "sim-judge", "params": {"temperature": 0}, "simulation": {"judge_accuracy": accuracy}},
    }


def test_llm_judge_with_simulated_judge():
    ctx = GraderContext()
    good = grade(_judge_spec(1.0), "Paris", "Paris", ctx=ctx, messages=[{"role": "user", "content": "Capital of France?"}])
    assert good.passed and good.value == 1.0 and good.rationale
    bad = grade(_judge_spec(1.0), "Lyon", "Paris", ctx=ctx)
    assert not bad.passed
    inverted = grade(_judge_spec(0.0), "Paris", "Paris", ctx=ctx)
    assert not inverted.passed
    scaled = grade(_judge_spec(1.0, "1-5"), "Paris", "Paris", ctx=ctx)
    assert scaled.passed and scaled.value == 1.0


def test_llm_judge_config_errors():
    with pytest.raises(SpecError, match="rubric"):
        build_grader(GraderSpec.model_validate({"id": "j", "type": "llm_judge", "judge_model": {"provider": "simulated", "model": "x"}}))
    with pytest.raises(SpecError, match="judge_model"):
        build_grader(GraderSpec.model_validate({"id": "j", "type": "llm_judge", "rubric": "r"}))
    grader = build_grader(GraderSpec.model_validate(_judge_spec(1.0)))
    assert grader.judge_spec.params["temperature"] == 0
    assert "judge_identity" in build_grader(GraderSpec.model_validate(_judge_spec(1.0)), GraderContext()).describe()
