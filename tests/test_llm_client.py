"""LLM client: retries on transient errors, fallback model, reasoning params - with a mocked SDK."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from src.llm.llm_client import OpenAILLMClient
from src.utils.exceptions import LLMError


def _rate_limit() -> openai.RateLimitError:
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)


def _bad_request() -> openai.BadRequestError:
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)


class FakeResponses:
    def __init__(self, script: list[Any]) -> None:
        self.script, self.calls = script, []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(output_text=item, usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def _client(script: list[Any], **kw: Any) -> tuple[OpenAILLMClient, FakeResponses]:
    c = OpenAILLMClient(
        api_key="sk-test",
        model="gpt-6-luna",
        fallback_model="gpt-5.4-mini",
        retry_kwargs={"max_attempts": 3, "initial": 0.01, "max_wait": 0.02, "jitter": 0},
        **kw,
    )
    fake = FakeResponses(script)
    c._client = SimpleNamespace(responses=fake)
    return c, fake


def test_retries_transient_errors_then_succeeds() -> None:
    client, fake = _client([_rate_limit(), _rate_limit(), "ok [1]"])
    result = client.generate("sys", [{"role": "user", "content": "hi"}])
    assert result.text == "ok [1]" and len(fake.calls) == 3 and not result.used_fallback


def test_falls_back_to_secondary_model() -> None:
    client, fake = _client([_rate_limit(), _rate_limit(), _rate_limit(), "from fallback"])
    result = client.generate("sys", [{"role": "user", "content": "hi"}])
    assert result.used_fallback and result.model == "gpt-5.4-mini"
    assert fake.calls[-1]["model"] == "gpt-5.4-mini"


def test_client_errors_are_not_retried() -> None:
    client, fake = _client([_bad_request(), _bad_request()])
    with pytest.raises(LLMError):
        client.generate("sys", [{"role": "user", "content": "hi"}])
    assert len(fake.calls) == 2  # one per model, no retries


def test_reasoning_params_and_prompt_cache_key() -> None:
    client, fake = _client(["x"], reasoning_effort="low")
    client.generate("sys", [{"role": "user", "content": "hi"}], prompt_cache_key="v1")
    params = fake.calls[0]
    assert params["reasoning"] == {"effort": "low"}
    assert params["prompt_cache_key"] == "v1" and params["store"] is False
    assert "temperature" not in params
