"""Ollama adapter (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import GenerateRequest, GenerateResult, ModelIdentity, Provider, ProviderError, http_error

# Parameters whose names differ between our generic spec and Ollama's ``options``.
_RENAMES = {"max_tokens": "num_predict"}


def ollama_options(params: dict[str, Any], seed: int | None) -> dict[str, Any]:
    options = {_RENAMES.get(k, k): v for k, v in params.items()}
    if seed is not None:
        options["seed"] = seed
    return options


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, spec, client: httpx.Client | None = None):
        super().__init__(spec)
        base = spec.endpoint or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not base.startswith(("http://", "https://")):
            base = "http://" + base
        self.base = base.rstrip("/")
        self.client = client or httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))

    def _get(self, path: str) -> Any:
        try:
            response = self.client.get(self.base + path)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise http_error(exc, f"ollama {self.base}{path}") from exc

    def resolve(self) -> ModelIdentity:
        version = (self._get("/api/version") or {}).get("version")
        models = (self._get("/api/tags") or {}).get("models", [])
        wanted = {self.spec.model, f"{self.spec.model}:latest"}
        for entry in models:
            if entry.get("name") in wanted or entry.get("model") in wanted:
                digest = entry.get("digest")
                if digest and not digest.startswith("sha256:"):
                    digest = "sha256:" + digest
                return ModelIdentity(
                    provider="ollama",
                    model=entry.get("name", self.spec.model),
                    digest=digest,
                    details=entry.get("details") or {},
                    server_version=version,
                )
        raise ProviderError(
            f"model {self.spec.model!r} is not available on {self.base}; run `ollama pull {self.spec.model}`"
        )

    def generate(self, request: GenerateRequest) -> GenerateResult:
        body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": request.messages,
            "stream": False,
            "options": ollama_options(request.params, request.seed),
        }
        if request.response_schema is not None:
            body["format"] = request.response_schema
        started = time.perf_counter()
        try:
            response = self.client.post(self.base + "/api/chat", json=body, timeout=request.timeout_s)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise http_error(exc, f"ollama {self.base}/api/chat") from exc
        wall_ms = (time.perf_counter() - started) * 1000.0
        message = data.get("message") or {}
        if "content" not in message:
            raise ProviderError(f"unexpected ollama response: {str(data)[:200]}")
        total = data.get("total_duration")
        load = data.get("load_duration") or 0
        latency_ms = (total - load) / 1e6 if total else wall_ms
        raw = {k: data.get(k) for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "done_reason")}
        raw["wall_ms"] = wall_ms
        return GenerateResult(
            text=message["content"],
            latency_ms=latency_ms,
            tokens_in=data.get("prompt_eval_count"),
            tokens_out=data.get("eval_count"),
            raw=raw,
        )

    def close(self) -> None:
        self.client.close()
