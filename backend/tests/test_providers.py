import json

import httpx
import pytest

from regressionlens.providers import GenerateRequest, ProviderError, make_provider
from regressionlens.providers.ollama import OllamaProvider
from regressionlens.providers.openai_compat import OpenAICompatProvider
from regressionlens.spec import ModelSpec

from .helpers import sim_model


def ollama_transport(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.7"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b", "digest": "abc123", "details": {"quantization_level": "Q4_K_M"}}]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            if body["model"] == "broken":
                return httpx.Response(500, text="boom")
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "yes"},
                    "total_duration": 900_000_000,
                    "load_duration": 400_000_000,
                    "prompt_eval_count": 12,
                    "eval_count": 2,
                    "done_reason": "stop",
                },
            )
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


def test_ollama_resolve_and_generate():
    seen: list = []
    spec = ModelSpec(name="m", provider="ollama", model="llama3.2:3b", endpoint="localhost:11434", params={"temperature": 0.5, "max_tokens": 50})
    provider = OllamaProvider(spec, client=httpx.Client(transport=ollama_transport(seen)))
    identity = provider.resolve()
    assert identity.digest == "sha256:abc123"
    assert identity.server_version == "0.5.7"
    assert identity.details["quantization_level"] == "Q4_K_M"
    result = provider.generate(GenerateRequest(messages=[{"role": "user", "content": "hi"}], params=spec.params, seed=7))
    assert result.text == "yes"
    assert result.latency_ms == pytest.approx(500.0)  # model load time excluded
    assert (result.tokens_in, result.tokens_out) == (12, 2)
    body = json.loads(seen[-1].content)
    assert body["options"] == {"temperature": 0.5, "num_predict": 50, "seed": 7}
    assert body["stream"] is False


def test_ollama_missing_model_and_server_errors():
    provider = OllamaProvider(ModelSpec(name="m", provider="ollama", model="missing"), client=httpx.Client(transport=ollama_transport([])))
    with pytest.raises(ProviderError, match="ollama pull missing") as exc:
        provider.resolve()
    assert not exc.value.transient
    broken = OllamaProvider(ModelSpec(name="m", provider="ollama", model="broken"), client=httpx.Client(transport=ollama_transport([])))
    with pytest.raises(ProviderError) as exc:
        broken.generate(GenerateRequest(messages=[], params={}))
    assert exc.value.transient


def test_ollama_unreachable_is_transient():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    provider = OllamaProvider(ModelSpec(name="m", provider="ollama", model="x"), client=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(ProviderError) as exc:
        provider.generate(GenerateRequest(messages=[], params={}))
    assert exc.value.transient


def test_openai_compat():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen2.5-1.5b", "object": "model"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    spec = ModelSpec(name="m", provider="openai_compat", model="qwen2.5-1.5b", endpoint="http://srv:8080", params={"temperature": 0.2, "num_ctx": 4096})
    provider = OpenAICompatProvider(spec, client=httpx.Client(transport=httpx.MockTransport(handler)))
    identity = provider.resolve()
    assert identity.digest.startswith("sha256:")
    result = provider.generate(GenerateRequest(messages=[{"role": "user", "content": "q"}], params=spec.params, seed=3, response_schema={"type": "object"}))
    assert result.text == "42" and result.tokens_out == 1
    body = json.loads(seen[-1].content)
    assert body["temperature"] == 0.2 and body["seed"] == 3 and "num_ctx" not in body
    assert body["response_format"]["type"] == "json_schema"
    assert str(seen[-1].url) == "http://srv:8080/v1/chat/completions"


def test_simulated_determinism_and_effects():
    provider = make_provider(sim_model(accuracy=0.5, spread=0.0, prompt_effects={"careful": 0.5}))
    def req(seed, text="x"):
        return GenerateRequest(messages=[{"role": "user", "content": text}], params={}, seed=seed, metadata={"case_key": "a", "expected": "ok"})
    assert provider.generate(req(1)).text == provider.generate(req(1)).text
    outcomes = {provider.generate(req(s)).text for s in range(50)}
    assert outcomes == {"ok", "I am not sure."}
    assert all(provider.generate(req(s, "be careful")).text == "ok" for s in range(20))
    greedy = make_provider(ModelSpec(name="s", provider="simulated", model="g", params={"temperature": 0}, simulation={"accuracy": 0.5}))
    assert len({greedy.generate(req(s)).text for s in range(20)}) == 1
    assert provider.resolve().digest == make_provider(sim_model(accuracy=0.5, spread=0.0, prompt_effects={"careful": 0.5})).resolve().digest
    assert provider.resolve().digest != make_provider(sim_model(accuracy=0.6)).resolve().digest
