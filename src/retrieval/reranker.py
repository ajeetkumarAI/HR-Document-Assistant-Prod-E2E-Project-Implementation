"""Second-stage rerankers.

WHY A SECOND STAGE?
-------------------
Search (stage 1) is fast but rough: it compares the question and each chunk SEPARATELY
(vector vs vector). A reranker (stage 2) reads the question and each passage TOGETHER,
which is much more accurate - but too slow to run on every chunk. So:

    all chunks ──search──► top 20 candidates ──rerank──► best 5 ──► LLM
                (fast, broad)                  (slow, precise)

Hybrid retrieval is optimised for recall (top_k ~ 20); a reranker then scores each
(query, passage) pair jointly for precision and keeps ``top_n`` (~5) for the LLM.

* ``llm``           - OpenAI model grades passages via structured output (default, no extra deps)
* ``cohere``        - Cohere Rerank API
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
    """Base class: the sort / threshold / fail-open logic is written ONCE here.
    Each provider only implements `_score()` (one number per passage)."""

    name = "base"

    def __init__(self, top_n: int = 5, score_threshold: float | None = None, fail_open: bool = True) -> None:
        self.top_n = top_n  # how many passages to keep for the LLM
        self.score_threshold = score_threshold  # optional: drop passages scoring below this
        self.fail_open = fail_open  # True = if reranking fails, continue with search order

    @abstractmethod
    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]: ...

    # Shows up in LangSmith as a "rerank" step. process_inputs/outputs control WHAT is logged:
    # we log the question + number of candidates (not 20 full passages twice) and the final documents.
    @traceable(
        name="rerank",
        run_type="chain",
        process_inputs=lambda i: {"query": i.get("query"), "n_candidates": len(i.get("chunks") or [])},
        process_outputs=lambda o: {"documents": [c.to_langsmith() for c in (o or [])]},
    )
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int | None = None) -> list[RetrievedChunk]:
        # top_n is passed per call (not stored on self) so concurrent requests can't affect each other
        top_n = top_n or self.top_n
        if not chunks:
            return []
        try:
            scores = self._score(query, chunks)
        except Exception as exc:
            if not self.fail_open:
                raise
            # FAIL OPEN: the answer is still useful with search order, so an OpenAI/Cohere hiccup
            # lowers quality slightly instead of returning an error to the user.
            logger.warning(
                "Reranker failed - falling back to retrieval order", extra={"reranker": self.name, "error": str(exc)}
            )
            return chunks[:top_n]

        # Attach the score to each chunk (it appears as "rerank_score" in the API response)
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.rerank_score = float(score)
        # Highest score first
        ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
        if self.score_threshold is not None:
            # e.g. threshold 3 -> passages the model rated 0-2 ("irrelevant") never reach the LLM.
            # If nothing passes, the pipeline answers "I couldn't find this" instead of guessing.
            ranked = [c for c in ranked if (c.rerank_score or 0.0) >= self.score_threshold]
        return ranked[:top_n]


class NoopReranker(Reranker):
    """provider: none - keep the search order, just cut to top_n. Zero cost, zero latency."""

    name = "none"

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        return [c.score for c in chunks]  # reuse the search score -> order unchanged


class CohereReranker(Reranker):
    """provider: cohere - a model built specifically for reranking (fast, ~200 ms)."""

    name = "cohere"

    def __init__(self, api_key: str, model: str = "rerank-v3.5", **kw: Any) -> None:
        super().__init__(**kw)
        # One HTTP client reused for every request (keeps the connection open = faster)
        self._client = httpx.Client(
            base_url="https://api.cohere.com", headers={"Authorization": f"Bearer {api_key}"}, timeout=15
        )
        self.model = model
        self._post = with_retry(max_attempts=3)(self._raw_post)  # retry 429 / 5xx / timeouts

    def _raw_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post("/v2/rerank", json=payload)
        resp.raise_for_status()  # turn 4xx/5xx into an exception (so retry / fail-open can react)
        return resp.json()

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        data = self._post({"model": self.model, "query": query, "documents": [c.text for c in chunks]})
        # Cohere returns results sorted by relevance, each with the ORIGINAL index -> put back in order
        scores = [0.0] * len(chunks)
        for r in data["results"]:
            scores[r["index"]] = r["relevance_score"]
        return scores


class LLMReranker(Reranker):
    """provider: llm (default) - an OpenAI model reads all passages and scores each 0-10.

    Request sent to the model:
        Question: How many sick leaves do I get?
        <passage id=0> ...earned leave text... </passage>
        <passage id=1> ...sick leave text...   </passage>
    Reply (forced to this exact JSON shape):
        {"scores": [{"id": 0, "score": 3}, {"id": 1, "score": 10}]}
    """

    name = "llm"

    # JSON Schema for OpenAI "structured outputs": the model MUST reply in exactly this shape,
    # so we never have to parse free text like "Passage 2 seems relevant...".
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
        self.llm, self.model = llm, model  # model = llm.utility_model from config.yaml

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        # Each passage capped at 1,500 characters: keeps the request small (cost + latency)
        # while still giving the model enough text to judge relevance.
        passages = "\n\n".join(f"<passage id={i}>\n{c.text[:1500]}\n</passage>" for i, c in enumerate(chunks))
        data = self.llm.generate_json(
            RERANK_PROMPT,
            [{"role": "user", "content": f"Question: {query}\n\n{passages}"}],
            schema=self._SCHEMA,
            name="rerank_scores",
            model=self.model,
            reasoning_effort="none",  # simple grading task -> no "thinking" needed, much faster
        )
        scores = [0.0] * len(chunks)  # any passage the model forgot to score gets 0
        for item in data.get("scores", []):
            # Ignore made-up ids (e.g. id 99 when there are only 20 passages)
            if 0 <= int(item["id"]) < len(chunks):
                scores[int(item["id"])] = float(item["score"])
        return scores


def build_reranker(settings: Any, llm: Any = None) -> Reranker:
    """Factory: choose the reranker from config.yaml -> reranker.provider."""
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
        # Misconfigured reranker -> start anyway without reranking, but log it loudly
        logger.error("Reranker unavailable, using retrieval order", extra={"error": str(exc)})
    return NoopReranker(**common)
