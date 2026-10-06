"""End-to-end RAG orchestration.

guardrails -> condense follow-up (history) -> exact cache -> embed -> semantic cache
  -> hybrid retrieve (filters + RBAC) -> rerank -> context budget -> LLM (retry + fallback)
  -> citation validation -> cache + memory write -> metrics / trace metadata
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.embeddings.embedder import Embedder
from src.guardrails.guards import check_input, extract_citations
from src.ingestion.models import RetrievedChunk
from src.llm.llm_client import LLMClient, LLMResult
from src.prompts.prompt_templates import (
    CONDENSE_PROMPT,
    CONDENSE_USER,
    NO_CONTEXT_ANSWER,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
    format_context,
    format_history,
)
from src.retrieval.filters import MetadataFilters
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import HybridRetriever
from src.utils.cache import ConversationMemory, ResponseCache
from src.utils.config import Settings
from src.utils.helpers import StageTimer
from src.utils.logger import get_logger, request_id_var
from src.utils.metrics import NO_ANSWER, observe_stages
from src.utils.tracing import add_trace_metadata, current_run_id, traceable

logger = get_logger(__name__)


@dataclass
class QueryRequest:
    question: str
    role: str = "employee"
    filters: MetadataFilters | None = None
    session_id: str | None = None
    top_n: int | None = None
    use_cache: bool = True


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm: LLMClient,
        response_cache: ResponseCache,
        memory: ConversationMemory,
    ) -> None:
        self.settings, self.embedder, self.retriever = settings, embedder, retriever
        self.reranker, self.llm, self.cache, self.memory = reranker, llm, response_cache, memory

    # ================================================================== helpers
    def _access_levels(self, role: str) -> list[str]:
        return self.settings.security.role_access.get(role, ["public"])

    @traceable(name="condense_question", run_type="chain")
    def _condense(self, question: str, history: list[dict[str, str]]) -> str:
        if not history or not self.settings.retrieval.use_query_rewrite:
            return question
        try:
            result = self.llm.generate(
                CONDENSE_PROMPT,
                [{"role": "user", "content": CONDENSE_USER.format(history=format_history(history), question=question)}],
                model=self.settings.llm.utility_model,
                reasoning_effort="none",
                max_output_tokens=200,
            )
            rewritten = result.text.strip().strip('"')
            return rewritten or question
        except Exception as exc:  # rewriting is an optimisation - never fail the request on it
            logger.warning("Query condensation failed; using raw question", extra={"error": str(exc)})
            return question

    @staticmethod
    def _citations(used: list[RetrievedChunk], cited: list[int]) -> list[dict[str, Any]]:
        ids = cited or list(range(1, len(used) + 1))
        out = []
        for n in ids:
            c = used[n - 1]
            m = c.metadata
            out.append(
                {
                    "id": n,
                    "doc_id": m.get("doc_id"),
                    "chunk_id": c.chunk_id,
                    "source": m.get("source"),
                    "title": m.get("title"),
                    "page": m.get("page"),
                    "section": m.get("section"),
                    "effective_date": m.get("effective_date"),
                    "score": round(c.score, 4),
                    "rerank_score": round(c.rerank_score, 4) if c.rerank_score is not None else None,
                    "snippet": c.text[:300],
                    "cited": n in cited,
                }
            )
        return out

    def _prepare(self, req: QueryRequest, timer: StageTimer) -> dict[str, Any]:
        """Everything before generation. Returns a dict with either a ready response or LLM inputs."""
        g = self.settings.guardrails
        with timer.stage("guardrails"):
            question = check_input(req.question, g.max_query_chars, g.block_prompt_injection)

        history = self.memory.get(req.session_id)
        with timer.stage("condense"):
            standalone = self._condense(question, history)

        filters_key = req.filters.cache_key() if req.filters else {}
        scope = self.cache.scope(req.role, filters_key)
        use_cache = req.use_cache and not history  # answers to follow-ups depend on the conversation

        if use_cache:
            with timer.stage("cache_exact"):
                hit = self.cache.get_exact(standalone, scope)
            if hit:
                return {"cached": {**hit, "cache": "exact"}, "question": question, "standalone": standalone}

        with timer.stage("embed_query"):
            qvec = self.embedder.embed_query(standalone)

        if use_cache:
            with timer.stage("cache_semantic"):
                hit = self.cache.get_semantic(qvec, scope)
            if hit:
                return {"cached": {**hit, "cache": "semantic"}, "question": question, "standalone": standalone}

        with timer.stage("retrieve"):
            candidates = self.retriever.retrieve(
                standalone, query_vector=qvec, filters=req.filters, access_levels=self._access_levels(req.role)
            )
        with timer.stage("rerank"):
            ranked = self.reranker.rerank(standalone, candidates, top_n=req.top_n)

        context, used = format_context(ranked, self.settings.llm.context_token_budget)
        return {
            "question": question,
            "standalone": standalone,
            "scope": scope,
            "qvec": qvec,
            "used": used,
            "use_cache": use_cache,
            "messages": history
            + [{"role": "user", "content": USER_PROMPT.format(context=context, question=standalone)}],
        }

    def _finalize(
        self, req: QueryRequest, prep: dict[str, Any], llm_result: LLMResult | None, timer: StageTimer
    ) -> dict[str, Any]:
        used: list[RetrievedChunk] = prep["used"]
        if llm_result is None:
            answer, cited = NO_CONTEXT_ANSWER, []
        else:
            answer, cited = extract_citations(llm_result.text, len(used))
        response = {
            "answer": answer,
            "citations": self._citations(used, cited) if used else [],
            "standalone_question": prep["standalone"],
            "model": llm_result.model if llm_result else None,
            "used_fallback_model": bool(llm_result and llm_result.used_fallback),
            "usage": {
                "input_tokens": llm_result.input_tokens if llm_result else 0,
                "output_tokens": llm_result.output_tokens if llm_result else 0,
            },
            "prompt_version": PROMPT_VERSION,
            "cache": None,
        }
        # Only cache grounded answers - never cache "not found" (the doc may be uploaded later)
        if prep["use_cache"] and used and llm_result is not None:
            self.cache.put(prep["standalone"], prep["scope"], response, prep["qvec"])
        self.memory.append(req.session_id, prep["question"], answer)
        return response

    def _envelope(self, response: dict[str, Any], timer: StageTimer, req: QueryRequest) -> dict[str, Any]:
        response = {k: v for k, v in response.items() if not k.startswith("_")}
        response.update(
            timings_ms={**timer.timings, "total": timer.total_ms},
            request_id=request_id_var.get(),
            run_id=current_run_id(),
        )
        observe_stages(timer.timings)
        add_trace_metadata(
            role=req.role, cache=response.get("cache"), prompt_version=PROMPT_VERSION, request_id=response["request_id"]
        )
        logger.info(
            "Query answered",
            extra={
                "cache": response.get("cache"),
                "citations": len(response["citations"]),
                "model": response.get("model"),
                "total_ms": timer.total_ms,
            },
        )
        return response

    # ================================================================== public API
    @traceable(
        name="rag_query",
        run_type="chain",
        process_inputs=lambda i: {
            "question": i["req"].question,
            "role": i["req"].role,
            "session_id": i["req"].session_id,
        },
    )
    def answer(self, req: QueryRequest) -> dict[str, Any]:
        timer = StageTimer()
        prep = self._prepare(req, timer)
        if "cached" in prep:
            self.memory.append(req.session_id, prep["question"], prep["cached"]["answer"])
            return self._envelope(prep["cached"], timer, req)

        llm_result = None
        if prep["used"]:
            with timer.stage("generate"):
                llm_result = self.llm.generate(SYSTEM_PROMPT, prep["messages"], prompt_cache_key=PROMPT_VERSION)
        else:
            NO_ANSWER.inc()
        return self._envelope(self._finalize(req, prep, llm_result, timer), timer, req)

    def stream(self, req: QueryRequest) -> Iterator[dict[str, Any]]:
        """Server-sent-events friendly generator: ``sources`` -> ``token``* -> ``done``."""
        timer = StageTimer()
        prep = self._prepare(req, timer)
        if "cached" in prep:
            self.memory.append(req.session_id, prep["question"], prep["cached"]["answer"])
            final = self._envelope(prep["cached"], timer, req)
            yield {"type": "token", "content": final["answer"]}
            yield {"type": "done", "response": final}
            return

        used = prep["used"]
        yield {"type": "sources", "citations": self._citations(used, []) if used else []}
        llm_result = None
        if used:
            with timer.stage("generate"):
                for item in self.llm.stream(SYSTEM_PROMPT, prep["messages"], prompt_cache_key=PROMPT_VERSION):
                    if isinstance(item, LLMResult):
                        llm_result = item
                    else:
                        yield {"type": "token", "content": item}
        else:
            NO_ANSWER.inc()
            yield {"type": "token", "content": NO_CONTEXT_ANSWER}
        yield {"type": "done", "response": self._envelope(self._finalize(req, prep, llm_result, timer), timer, req)}
