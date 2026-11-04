"""Model providers. Workers never know which server they are talking to."""

from __future__ import annotations

from ..spec import ModelSpec
from .base import GenerateRequest, GenerateResult, ModelIdentity, Provider, ProviderError
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider
from .simulated import SimulatedProvider

_PROVIDERS: dict[str, type[Provider]] = {
    "ollama": OllamaProvider,
    "openai_compat": OpenAICompatProvider,
    "simulated": SimulatedProvider,
}


def make_provider(spec: ModelSpec) -> Provider:
    try:
        cls = _PROVIDERS[spec.provider]
    except KeyError as exc:
        raise ProviderError(f"unknown provider {spec.provider!r}") from exc
    return cls(spec)


__all__ = [
    "GenerateRequest",
    "GenerateResult",
    "ModelIdentity",
    "Provider",
    "ProviderError",
    "make_provider",
]
