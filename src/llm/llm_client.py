"""OpenAI LLM client (Responses API) with retries, model fallback, streaming, structured output,
prompt caching and LangSmith tracing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from src.utils.exceptions import ConfigurationError, LLMError
from src.utils.helpers import with_retry
from src.utils.logger import get_logger
from src.utils.metrics import LLM_ERRORS, LLM_TOKENS
from src.utils.tracing import wrap_openai_client

logger = get_logger(__name__)

_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    used_fallback: bool = False
    raw: Any = field(default=None, repr=False)


class LLMClient:
    """Interface used by the pipeline. ``messages`` follow the chat format: [{"role", "content"}]."""

    model: str

    def generate(self, instructions: str, messages: list[dict[str, str]], **kw: Any) -> LLMResult:
        raise NotImplementedError

    def stream(self, instructions: str, messages: list[dict[str, str]], **kw: Any) -> Iterator[str | LLMResult]:
        raise NotImplementedError

    def generate_json(
        self, instructions: str, messages: list[dict[str, str]], schema: dict[str, Any], name: str, **kw: Any
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        reasoning_effort: str | None = "low",
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        timeout: float = 60,
        base_url: str | None = None,
        retry_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from openai import OpenAI

        raw = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self._client = wrap_openai_client(raw)  # LangSmith LLM runs incl. token usage
        self.model, self.fallback_model = model, fallback_model
        self.reasoning_effort, self.max_output_tokens, self.temperature = (
            reasoning_effort,
            max_output_tokens,
            temperature,
        )
        self._retry = with_retry(**(retry_kwargs or {}))

    # ------------------------------------------------------------------ request building
    def _params(self, model: str, instructions: str, messages: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": messages,
            "max_output_tokens": kw.get("max_output_tokens") or self.max_output_tokens,
            "store": False,  # don't persist HR conversations on the provider side
        }
        if model.startswith(_REASONING_PREFIXES):
            effort = kw.get("reasoning_effort", self.reasoning_effort)
            if effort:
                params["reasoning"] = {"effort": effort}
        temperature = kw.get("temperature", self.temperature)
        if temperature is not None:
            params["temperature"] = temperature
        if kw.get("prompt_cache_key"):
            # same static prefix -> provider-side prompt caching (cheaper + faster)
            params["prompt_cache_key"] = kw["prompt_cache_key"]
        if kw.get("text_format"):
            params["text"] = {"format": kw["text_format"]}
        return params

    def _models(self, override: str | None) -> list[str]:
        primary = override or self.model
        return [primary] + ([self.fallback_model] if self.fallback_model and self.fallback_model != primary else [])

    def _record(self, model: str, usage: Any) -> tuple[int, int]:
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        LLM_TOKENS.labels(model, "input").inc(inp)
        LLM_TOKENS.labels(model, "output").inc(out)
        return inp, out

    # ------------------------------------------------------------------ calls
    def generate(self, instructions: str, messages: list[dict[str, str]], **kw: Any) -> LLMResult:
        last_exc: Exception | None = None
        for i, model in enumerate(self._models(kw.pop("model", None))):
            t0 = time.perf_counter()
            try:
                resp = self._retry(self._client.responses.create)(**self._params(model, instructions, messages, **kw))
                inp, out = self._record(model, resp.usage)
                return LLMResult(
                    text=resp.output_text or "",
                    model=model,
                    input_tokens=inp,
                    output_tokens=out,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                    used_fallback=i > 0,
                    raw=resp,
                )
            except Exception as exc:
                LLM_ERRORS.labels(model).inc()
                logger.error("LLM call failed", extra={"model": model, "error": str(exc)})
                last_exc = exc
        raise LLMError(f"All LLM models failed: {last_exc}") from last_exc

    def stream(self, instructions: str, messages: list[dict[str, str]], **kw: Any) -> Iterator[str | LLMResult]:
        """Yields text deltas, then a final ``LLMResult`` with usage. Falls back only before the first token."""
        last_exc: Exception | None = None
        for i, model in enumerate(self._models(kw.pop("model", None))):
            t0 = time.perf_counter()
            emitted, parts = False, []
            try:
                events = self._retry(self._client.responses.create)(
                    **self._params(model, instructions, messages, **kw), stream=True
                )
                usage = None
                for event in events:
                    etype = getattr(event, "type", "")
                    if etype == "response.output_text.delta":
                        emitted = True
                        parts.append(event.delta)
                        yield event.delta
                    elif etype == "response.completed":
                        usage = event.response.usage
                    elif etype in ("response.failed", "error"):
                        raise LLMError(f"Streaming error event: {event}")
                inp, out = self._record(model, usage)
                yield LLMResult("".join(parts), model, inp, out, round((time.perf_counter() - t0) * 1000, 1), i > 0)
                return
            except Exception as exc:
                LLM_ERRORS.labels(model).inc()
                logger.error("LLM stream failed", extra={"model": model, "error": str(exc)})
                last_exc = exc
                if emitted:  # can't transparently switch models mid-answer
                    break
        raise LLMError(f"LLM streaming failed: {last_exc}") from last_exc

    def generate_json(
        self, instructions: str, messages: list[dict[str, str]], schema: dict[str, Any], name: str, **kw: Any
    ) -> dict[str, Any]:
        fmt = {"type": "json_schema", "name": name, "schema": schema, "strict": True}
        result = self.generate(instructions, messages, text_format=fmt, **kw)
        try:
            return json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON for {name}") from exc


class FakeLLMClient(LLMClient):
    """Deterministic offline LLM for tests/CI: answers with the first sentence of the first context block."""

    model = "fake-llm"

    def generate(self, instructions: str, messages: list[dict[str, str]], **kw: Any) -> LLMResult:
        content = messages[-1]["content"]
        if "<context>" in content and "[1]" in content:
            block = content.split("[1]", 1)[1]
            body = block.split("\n", 1)[1] if "\n" in block else block
            first = body.strip().split("\n")[0].split(". ")[0].strip()
            text = f"{first}. [1]"
        elif "standalone" in instructions.lower():
            text = content.rsplit("Follow-up question:", 1)[-1].strip()
        else:
            text = "I couldn't find this in the HR documents available to me."
        return LLMResult(text=text, model=self.model, input_tokens=len(content) // 4, output_tokens=len(text) // 4)

    def stream(self, instructions: str, messages: list[dict[str, str]], **kw: Any) -> Iterator[str | LLMResult]:
        result = self.generate(instructions, messages, **kw)
        for word in result.text.split(" "):
            yield word + " "
        yield result

    def generate_json(self, instructions, messages, schema, name, **kw):  # type: ignore[no-untyped-def]
        return {"scores": []}


def build_llm(settings: Any) -> LLMClient:
    cfg = settings.llm
    if cfg.provider == "fake":
        return FakeLLMClient()
    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is required for llm.provider=openai")
    r = settings.retry
    return OpenAILLMClient(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        model=cfg.model,
        fallback_model=cfg.fallback_model,
        reasoning_effort=cfg.reasoning_effort,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
        timeout=cfg.timeout_seconds,
        retry_kwargs={
            "max_attempts": r.max_attempts,
            "initial": r.initial_wait_seconds,
            "max_wait": r.max_wait_seconds,
            "jitter": r.jitter_seconds,
        },
    )
