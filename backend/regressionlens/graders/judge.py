"""Model-based grader: a pinned judge model applies a rubric and returns a structured verdict."""

from __future__ import annotations

from typing import Any

from ..providers import GenerateRequest, ProviderError
from ..spec import ModelSpec, SpecError
from .base import GradeInput, Grader, GradeResult, extract_json, register

JUDGE_SYSTEM = (
    "You are a strict, consistent evaluator. Judge the RESPONSE against the RUBRIC. "
    "Do not reward length or style unless the rubric asks for it. "
    "Reply with JSON only."
)

JUDGE_SCHEMA_BINARY = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "verdict": {"type": "string", "enum": ["pass", "fail"]}},
    "required": ["reasoning", "verdict"],
}

JUDGE_SCHEMA_SCALE = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["reasoning", "score"],
}


@register
class LLMJudge(Grader):
    """Config:

    - ``rubric`` (or ``rubric_file``): what a good response looks like.
    - ``judge_model``: a model spec (provider, model, endpoint, params). Defaults to temperature 0, seed 0.
    - ``scale``: ``binary`` (default) or ``1-5``. 1-5 scores are normalized to [0, 1].
    - ``pass_threshold``: minimum 1-5 score that counts as a pass (default 4).
    - ``include_reference``: show the case's expected value to the judge (default true when present).
    - ``samples``: judge the same output this many times and average (default 1).
    """

    type = "llm_judge"
    version = "1.0"
    deterministic = False

    def setup(self) -> None:
        if not isinstance(self.cfg.get("rubric"), str) or not self.cfg["rubric"].strip():
            raise SpecError(f"llm_judge grader {self.spec.id!r} needs a `rubric` (or `rubric_file`)")
        model = self.cfg.get("judge_model")
        if not isinstance(model, dict):
            raise SpecError(f"llm_judge grader {self.spec.id!r} needs a `judge_model`")
        model = dict(model)
        model.setdefault("name", f"judge-{self.spec.id}")
        params = {"temperature": 0, **(model.get("params") or {})}
        model["params"] = params
        try:
            self.judge_spec = ModelSpec.model_validate(model)
        except ValueError as exc:
            raise SpecError(f"llm_judge {self.spec.id!r}: bad judge_model: {exc}") from exc
        self.scale = self.cfg.get("scale", "binary")
        if self.scale not in {"binary", "1-5"}:
            raise SpecError("llm_judge.scale must be 'binary' or '1-5'")
        self.pass_threshold = int(self.cfg.get("pass_threshold", 4))
        self.samples = max(1, int(self.cfg.get("samples", 1)))
        self.judge_seed = int(self.cfg.get("seed", 0))

    def describe(self) -> dict[str, Any]:
        if self.ctx is None:
            return {"judge_model": self.judge_spec.identity_dict()}
        identity = self.ctx.provider_for(self.judge_spec).resolve()
        return {"judge_model": self.judge_spec.identity_dict(), "judge_identity": identity.to_dict()}

    def build_messages(self, item: GradeInput) -> list[dict[str, str]]:
        task = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in item.messages) or str(item.case.input)
        parts = [f"RUBRIC:\n{self.cfg['rubric'].strip()}", f"TASK GIVEN TO THE MODEL:\n{task}"]
        include_reference = self.cfg.get("include_reference", item.case.expected is not None)
        if include_reference and item.case.expected is not None:
            parts.append(f"REFERENCE ANSWER:\n{item.case.expected}")
        parts.append(f"RESPONSE:\n{item.output}")
        if self.scale == "binary":
            parts.append('Return {"reasoning": "<one or two sentences>", "verdict": "pass" | "fail"}.')
        else:
            parts.append('Return {"reasoning": "<one or two sentences>", "score": <integer 1-5>}.')
        return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]

    def parse(self, text: str) -> tuple[float, str]:
        data = extract_json(text)
        if not isinstance(data, dict):
            raise ValueError("judge did not return an object")
        reasoning = str(data.get("reasoning", ""))[:500]
        if self.scale == "binary":
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict not in {"pass", "fail"}:
                raise ValueError(f"judge verdict {verdict!r} is not pass/fail")
            return (1.0 if verdict == "pass" else 0.0), reasoning
        score = int(data.get("score"))
        if not 1 <= score <= 5:
            raise ValueError(f"judge score {score} out of range")
        return (score - 1) / 4.0, reasoning

    def grade(self, item: GradeInput) -> GradeResult:
        if self.ctx is None:
            raise RuntimeError("llm_judge needs a GraderContext")
        provider = self.ctx.provider_for(self.judge_spec)
        messages = self.build_messages(item)
        schema = JUDGE_SCHEMA_BINARY if self.scale == "binary" else JUDGE_SCHEMA_SCALE
        values: list[float] = []
        reasons: list[str] = []
        raw_outputs: list[str] = []
        for i in range(self.samples):
            metadata = {
                "role": "judge",
                "case_key": item.case.key,
                "output": item.output,
                "reference": item.case.expected,
                "scale": self.scale,
            }
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    result = provider.generate(
                        GenerateRequest(
                            messages=messages,
                            params=self.judge_spec.params,
                            seed=self.judge_seed + i,
                            timeout_s=self.ctx.timeout_s,
                            metadata={**metadata, "attempt": attempt},
                            response_schema=schema,
                        )
                    )
                    value, reason = self.parse(result.text)
                    values.append(value)
                    reasons.append(reason)
                    raw_outputs.append(result.text[:1000])
                    last_error = None
                    break
                except (ProviderError, ValueError, TypeError) as exc:
                    last_error = exc
                    if isinstance(exc, ProviderError) and not exc.transient:
                        break
            if last_error is not None:
                return GradeResult(0.0, False, f"judge error: {last_error}", error=True)
        value = sum(values) / len(values)
        if self.scale == "binary":
            passed = value >= 0.5
        else:
            passed = value >= (self.pass_threshold - 1) / 4.0
        return GradeResult(
            value=value,
            passed=passed,
            rationale=reasons[0] if reasons else "",
            details={"samples": values, "judge_raw": raw_outputs},
        )
