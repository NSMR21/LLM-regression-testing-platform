"""OpenAI-compatible chat completions: llama.cpp server, vLLM, LM Studio and similar."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..util import content_hash
from .base import GenerateRequest, GenerateResult, ModelIdentity, Provider, ProviderError, http_error

_PASSTHROUGH = {"temperature", "top_p", "max_tokens", "stop", "presence_penalty", "frequency_penalty", "top_k", "min_p"}


class OpenAICompatProvider(Provider):
    name = "openai_compat"

    def __init__(self, spec, client: httpx.Client | None = None):
        super().__init__(spec)
        base = (spec.endpoint or "http://localhost:8000").rstrip("/")
        self.base = base if base.endswith("/v1") else base + "/v1"
        headers = {}
        if spec.api_key_env:
            key = os.environ.get(spec.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        self.client = client or httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0), headers=headers)

    def resolve(self) -> ModelIdentity:
        try:
            response = self.client.get(self.base + "/models")
            response.raise_for_status()
            entries = response.json().get("data", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise http_error(exc, f"{self.base}/models") from exc
        for entry in entries:
            if entry.get("id") == self.spec.model:
                # Most servers expose no weights digest; prefer the user's pinned one.
                digest = self.spec.digest or content_hash(entry)
                return ModelIdentity(provider=self.name, model=self.spec.model, digest=digest, details=entry)
        raise ProviderError(f"model {self.spec.model!r} not served at {self.base}")

    def generate(self, request: GenerateRequest) -> GenerateResult:
        body: dict[str, Any] = {"model": self.spec.model, "messages": request.messages}
        for key, value in request.params.items():
            if key in _PASSTHROUGH:
                body[key] = value
        if request.seed is not None:
            body["seed"] = request.seed
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": request.response_schema},
            }
        started = time.perf_counter()
        try:
            response = self.client.post(self.base + "/chat/completions", json=body, timeout=request.timeout_s)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise http_error(exc, f"{self.base}/chat/completions") from exc
        wall_ms = (time.perf_counter() - started) * 1000.0
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return GenerateResult(
            text=text,
            latency_ms=wall_ms,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            raw={"finish_reason": data["choices"][0].get("finish_reason"), "wall_ms": wall_ms},
        )

    def close(self) -> None:
        self.client.close()
