"""A simulated model with a known ground truth.

Each case gets a fixed probability of being answered correctly. A correct answer is the case's
``expected`` value; a wrong answer is a fixed wrong string. Because the true per-case success
probabilities are known, the simulated provider measures the statistics engine directly:
false-alarm rate when nothing changed, and detection power when accuracy is shifted.

``simulation`` options (all optional):

- ``accuracy`` (0.8): mean per-case success probability.
- ``spread`` (0.3): per-case probabilities are spread uniformly over accuracy +/- spread.
- ``shift`` (0.0): added to every case's probability, to inject a regression or improvement.
- ``overrides``: ``{case_key: probability}``.
- ``prompt_effects``: ``{substring: delta}`` added when the rendered prompt contains the substring,
  so editing a prompt changes simulated quality like it would for a real model.
- ``case_seed`` (0): which draw of per-case difficulties to use.
- ``latency_ms`` (50): median simulated latency.
- ``error_rate`` (0.0): probability that a call fails with a transient error.
- ``wrong_answer`` ("I am not sure."): the text returned for a wrong answer.

With ``temperature: 0`` the outcome per case is deterministic (no dependence on the seed).
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from ..util import canonical_json, derive_seed
from .base import GenerateRequest, GenerateResult, ModelIdentity, Provider, ProviderError


def render_expected(expected: Any) -> str:
    if isinstance(expected, list) and expected:
        expected = expected[0]
    if isinstance(expected, dict | list):
        return json.dumps(expected)
    return "" if expected is None else str(expected)


class SimulatedProvider(Provider):
    name = "simulated"
    retry_backoff_s = 0.0

    def __init__(self, spec):
        super().__init__(spec)
        sim = spec.simulation or {}
        self.accuracy = float(sim.get("accuracy", 0.8))
        self.spread = float(sim.get("spread", 0.3))
        self.shift = float(sim.get("shift", 0.0))
        self.overrides: dict[str, float] = {str(k): float(v) for k, v in (sim.get("overrides") or {}).items()}
        self.prompt_effects: dict[str, float] = {str(k): float(v) for k, v in (sim.get("prompt_effects") or {}).items()}
        self.case_seed = sim.get("case_seed", 0)
        self.latency_ms = float(sim.get("latency_ms", 50.0))
        self.error_rate = float(sim.get("error_rate", 0.0))
        self.wrong_answer = str(sim.get("wrong_answer", "I am not sure."))
        self.temperature = float(spec.params.get("temperature", 0.7))
        self.judge_accuracy = float(sim.get("judge_accuracy", 0.9))

    def resolve(self) -> ModelIdentity:
        return ModelIdentity(
            provider="simulated",
            model=self.spec.model,
            digest=f"sim:{derive_seed(canonical_json(self.spec.simulation or {}), self.spec.model)}",
            details={"simulation": self.spec.simulation or {}},
            server_version="simulated-1",
        )

    def case_probability(self, case_key: str, prompt_text: str = "") -> float:
        if case_key in self.overrides:
            p = self.overrides[case_key]
        else:
            u = derive_seed("difficulty", self.case_seed, case_key) / 0x7FFFFFFF
            p = self.accuracy + self.spread * (2.0 * u - 1.0)
        p += self.shift
        for needle, delta in self.prompt_effects.items():
            if needle in prompt_text:
                p += delta
        return min(1.0, max(0.0, p))

    def warmup(self, messages: list[dict[str, str]]) -> None:
        return None

    def generate(self, request: GenerateRequest) -> GenerateResult:
        case_key = str(request.metadata.get("case_key", ""))
        attempt = int(request.metadata.get("attempt", 0))
        prompt_text = "\n".join(m.get("content", "") for m in request.messages)
        if request.seed is None:
            rng = random.Random()
        else:
            rng = random.Random(derive_seed(request.seed, attempt))
        if self.error_rate and rng.random() < self.error_rate:
            raise ProviderError("simulated server error", transient=True)
        if request.metadata.get("role") == "judge":
            return self._judge(request, rng)
        p = self.case_probability(case_key, prompt_text)
        if self.temperature == 0.0:
            # Greedy decoding: same answer every time for the same case.
            draw = derive_seed("greedy", self.case_seed, case_key) / 0x7FFFFFFF
        else:
            draw = rng.random()
        correct = draw < p
        text = render_expected(request.metadata.get("expected")) if correct else self.wrong_answer
        latency = max(1.0, rng.lognormvariate(math.log(self.latency_ms), 0.25))
        return GenerateResult(
            text=text,
            latency_ms=latency,
            tokens_in=len(prompt_text.split()),
            tokens_out=len(text.split()),
            raw={"p": p, "correct": correct},
        )

    def _judge(self, request: GenerateRequest, rng: random.Random) -> GenerateResult:
        """Simulated judge: knows the truth (output == reference) and agrees with it with
        probability ``judge_accuracy``. With temperature 0 the verdict is fixed per output."""
        output = str(request.metadata.get("output", "")).strip().lower()
        reference = render_expected(request.metadata.get("reference")).strip().lower()
        truth = output == reference
        if self.temperature == 0.0:
            draw = derive_seed("judge", self.case_seed, request.metadata.get("case_key"), output) / 0x7FFFFFFF
        else:
            draw = rng.random()
        verdict = truth if draw < self.judge_accuracy else not truth
        if request.metadata.get("scale") == "1-5":
            body = {"reasoning": "simulated judgement", "score": 5 if verdict else 1}
        else:
            body = {"reasoning": "simulated judgement", "verdict": "pass" if verdict else "fail"}
        return GenerateResult(text=json.dumps(body), latency_ms=self.latency_ms, raw={"truth": truth})
