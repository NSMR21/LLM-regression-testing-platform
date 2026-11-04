import pytest

from regressionlens.graders import validate_suite
from regressionlens.spec import PromptSpec, SpecError, SuiteSpec, load_model, load_prompt, load_suite

from .helpers import EXAMPLE, make_suite


def test_example_suite_loads_and_inlines_files():
    suite = load_suite(EXAMPLE / "suite.yaml")
    validate_suite(suite)
    assert len(suite.cases) == 30
    extract = suite.case("extract-late-1")
    schema = next(g for g in suite.effective_graders(extract) if g.id == "schema")
    assert "schema_file" not in schema.config
    assert schema.config["schema"]["required"] == ["order_id", "issue"]
    assert suite.case("return-final-sale").critical


def test_example_prompts_render_every_case():
    suite = load_suite(EXAMPLE / "suite.yaml")
    for name in ("v1.yaml", "v2-shorter.yaml"):
        prompt = load_prompt(EXAMPLE / "prompts" / name)
        for case in suite.cases:
            messages = prompt.render(case.input)
            assert messages[-1]["role"] == "user"
            assert case.input["question"] in messages[-1]["content"]
    assert load_model(EXAMPLE / "models" / "simulated.yaml").provider == "simulated"


def test_content_hash_is_stable_and_sensitive():
    a, b = make_suite(5), make_suite(5)
    assert a.content_hash() == b.content_hash()
    data = b.model_dump(mode="json")
    data["cases"][0]["expected"] = "something else"
    c = SuiteSpec.model_validate(data)
    assert c.content_hash() != a.content_hash()
    assert c.case_hash(c.cases[0]) != a.case_hash(a.cases[0])
    assert c.case_hash(c.cases[1]) == a.case_hash(a.cases[1])


def test_duplicate_case_keys_rejected():
    with pytest.raises(ValueError, match="duplicate case key"):
        SuiteSpec.model_validate(
            {"name": "x", "graders": [{"id": "e", "type": "exact_match"}], "cases": [{"key": "a", "expected": 1}, {"key": "a", "expected": 2}]}
        )


def test_reserved_grader_id_rejected():
    with pytest.raises(ValueError, match="reserved"):
        SuiteSpec.model_validate({"name": "x", "graders": [{"id": "pass_rate", "type": "exact_match"}], "cases": [{"key": "a", "expected": 1}]})


def test_case_graders_override_defaults_by_id():
    suite = SuiteSpec.model_validate(
        {
            "name": "x",
            "graders": [{"id": "g", "type": "exact_match"}, {"id": "h", "type": "contains", "value": "x"}],
            "cases": [
                {"key": "a", "expected": "1", "graders": [{"id": "g", "type": "regex", "pattern": "1"}]},
                {"key": "b", "expected": "1", "use_default_graders": False, "graders": [{"id": "z", "type": "regex", "pattern": "1"}]},
            ],
        }
    )
    assert [g.type for g in suite.effective_graders(suite.case("a"))] == ["regex", "contains"]
    assert [g.id for g in suite.effective_graders(suite.case("b"))] == ["z"]


def test_validation_errors():
    with pytest.raises(SpecError, match="unknown grader type"):
        validate_suite(SuiteSpec.model_validate({"name": "x", "graders": [{"id": "g", "type": "nope"}], "cases": [{"key": "a"}]}))
    with pytest.raises(SpecError, match="needs `expected`"):
        validate_suite(SuiteSpec.model_validate({"name": "x", "graders": [{"id": "g", "type": "exact_match"}], "cases": [{"key": "a"}]}))
    with pytest.raises(SpecError, match="not one of"):
        validate_suite(
            SuiteSpec.model_validate(
                {"name": "x", "graders": [{"id": "g", "type": "choice", "choices": ["yes", "no"]}], "cases": [{"key": "a", "expected": "maybe"}]}
            )
        )
    with pytest.raises(SpecError, match="unknown metric"):
        validate_suite(make_suite(3, gates=[{"metric": "nonexistent", "min": 0.5}]))


def test_prompt_render_strict_undefined():
    prompt = PromptSpec(name="p", template="Hello {{ name }}")
    assert prompt.render({"name": "Ada"}) == [{"role": "user", "content": "Hello Ada"}]
    with pytest.raises(SpecError, match="failed to render"):
        prompt.render({})


def test_prompt_template_is_sandboxed():
    prompt = PromptSpec(name="p", template="{{ ''.__class__.__mro__[1].__subclasses__() }}")
    with pytest.raises(SpecError):
        prompt.render({})
