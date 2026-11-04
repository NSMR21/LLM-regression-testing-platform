"""Provider interface: one way to resolve a model's identity and generate text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..spec import ModelSpec


class ProviderError(Exception):
    """A model call failed. ``transient`` errors (timeouts, 5xx, refused connections) are retried."""

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


@dataclass
class ModelIdentity:
    provider: str
    model: str
    digest: str | None
    details: dict[str, Any] = field(default_factory=dict)
    server_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "digest": self.digest,
            "details": self.details,
            "server_version": self.server_version,
        }


@dataclass
class GenerateRequest:
    messages: list[dict[str, str]]
    params: dict[str, Any]
    seed: int | None = None
    timeout_s: float = 120.0
    metadata: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] | None = None


@dataclass
class GenerateResult:
    text: str
    latency_ms: float
    ttft_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    name: str = "base"
    retry_backoff_s: float = 0.5

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def resolve(self) -> ModelIdentity:
        """Return the exact identity (digest, server version) of the configured model."""

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResult:
        """Generate one completion. Raise ProviderError on failure."""

    def warmup(self, messages: list[dict[str, str]]) -> None:
        """Load the model before timed calls so load time does not pollute latency."""
        try:
            self.generate(GenerateRequest(messages=messages, params={**self.spec.params, "max_tokens": 1}))
        except ProviderError:
            pass

    def close(self) -> None:  # noqa: B027  (optional hook; most providers hold no resources)
        """Release network clients."""


def http_error(exc: Exception, where: str) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(f"timeout calling {where}", transient=True)
    if isinstance(exc, httpx.TransportError):
        return ProviderError(f"cannot reach {where}: {exc}", transient=True)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = exc.response.text[:300]
        return ProviderError(f"{where} returned HTTP {status}: {body}", transient=status >= 500 or status == 429)
    return ProviderError(f"{where}: {exc}")
