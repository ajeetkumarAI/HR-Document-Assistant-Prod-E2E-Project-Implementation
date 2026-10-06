"""Second-stage rerankers.

Hybrid retrieval is optimised for recall (top_k ~ 20); a reranker then scores each
(query, passage) pair jointly for precision and keeps ``top_n`` (~5) for the LLM.

* ``cohere``        - Cohere Rerank API
* ``llm``           - OpenAI model grades passages via structured output (default, no extra deps)
* ``none``          - keep retrieval order
All rerankers fail open (configurable): an outage degrades quality, not availability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from src.ingestion.models import RetrievedChunk
from src.prompts.prompt_templates import RERANK_PROMPT
from src.utils.exceptions import ConfigurationError
from src.utils.helpers import with_retry
from src.utils.logger import get_logger
from src.utils.tracing import traceable

logger = get_logger(__name__)


class Reranker(ABC):
    name = "base"

    def __init__(self, top_n: int = 5, score_threshold: float | None = None, fail_open: bool = True) -> None:
        self.top_n, self.score_threshold, self.fail_open = top_n, score_threshold, fail_open

    @abstractmethod
    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]: ...

    @traceable(
        name="rerank",
        run_type="chain",
        process_inputs=lambda i: {"query": i.get("query"), "n_candidates": len(i.get("chunks") or [])},
        process_outputs=lambda o: {"documents": [c.to_langsmith() for c in (o or [])]},
    )
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int | None = None) -> list[RetrievedChunk]:
        top_n = top_n or self.top_n
        if not chunks:
            return []
        try:
            scores = self._score(query, chunks)
        except Exception as exc:
            if not self.fail_open:
                raise
            logger.warning(
                "Reranker failed - falling back to retrieval order", extra={"reranker": self.name, "error": str(exc)}
            )
            return chunks[:top_n]
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.rerank_score = float(score)
        ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
        if self.score_threshold is not None:
            ranked = [c for c in ranked if (c.rerank_score or 0.0) >= self.score_threshold]
        return ranked[:top_n]


class NoopReranker(Reranker):
    name = "none"

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        return [c.score for c in chunks]


class CohereReranker(Reranker):
    name = "cohere"

    def __init__(self, api_key: str, model: str = "rerank-v3.5", **kw: Any) -> None:
        super().__init__(**kw)
        self._client = httpx.Client(
            base_url="https://api.cohere.com", headers={"Authorization": f"Bearer {api_key}"}, timeout=15
        )
        self.model = model
        self._post = with_retry(max_attempts=3)(self._raw_post)

    def _raw_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post("/v2/rerank", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        data = self._post({"model": self.model, "query": query, "documents": [c.text for c in chunks]})
        scores = [0.0] * len(chunks)
        for r in data["results"]:
            scores[r["index"]] = r["relevance_score"]
        return scores


class LLMReranker(Reranker):
    name = "llm"
    _SCHEMA = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "score": {"type": "number"}},
                    "required": ["id", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }

    def __init__(self, llm: Any, model: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.llm, self.model = llm, model

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        passages = "\n\n".join(f"<passage id={i}>\n{c.text[:1500]}\n</passage>" for i, c in enumerate(chunks))
        data = self.llm.generate_json(
            RERANK_PROMPT,
            [{"role": "user", "content": f"Question: {query}\n\n{passages}"}],
            schema=self._SCHEMA,
            name="rerank_scores",
            model=self.model,
            reasoning_effort="none",
        )
        scores = [0.0] * len(chunks)
        for item in data.get("scores", []):
            if 0 <= int(item["id"]) < len(chunks):
                scores[int(item["id"])] = float(item["score"])
        return scores


def build_reranker(settings: Any, llm: Any = None) -> Reranker:
    cfg = settings.reranker
    common = {"top_n": cfg.top_n, "score_threshold": cfg.score_threshold, "fail_open": cfg.fail_open}
    try:
        if cfg.provider == "cohere":
            if not settings.cohere_api_key:
                raise ConfigurationError("COHERE_API_KEY is required for reranker.provider=cohere")
            return CohereReranker(settings.cohere_api_key.get_secret_value(), cfg.cohere_model, **common)
        if cfg.provider == "llm" and llm is not None:
            return LLMReranker(llm, settings.llm.utility_model, **common)
    except ConfigurationError as exc:
        if not cfg.fail_open:
            raise
        logger.error("Reranker unavailable, using retrieval order", extra={"error": str(exc)})
    return NoopReranker(**common)
