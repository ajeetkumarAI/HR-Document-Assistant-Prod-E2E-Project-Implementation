"""End-to-end RAG orchestration.

This is the "conductor": it calls every other component in the right order.

    guardrails -> condense follow-up (history) -> exact cache -> embed -> semantic cache
      -> hybrid retrieve (filters + RBAC) -> rerank -> context budget -> LLM (retry + fallback)
      -> citation validation -> cache + memory write -> metrics / trace metadata

Code layout:
    _prepare()   steps BEFORE the LLM (can return early with a cached answer)
    _finalize()  steps AFTER the LLM  (citations, cache write, memory)
    _envelope()  adds timings / request_id / run_id, logs, metrics
    answer()     = _prepare + generate + _finalize + _envelope   (normal JSON response)
    stream()     = same, but yields events for Server-Sent Events
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
    """Internal request object (the API layer converts the HTTP body into this)."""

    question: str
    role: str = "employee"  # decides which access levels can be searched
    filters: MetadataFilters | None = None
    session_id: str | None = None  # enables follow-up questions
    top_n: int | None = None  # override how many passages reach the LLM
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
        # All parts are passed in (built by container.py) rather than created here.
        # WHY: tests can pass fakes, and swapping e.g. the reranker needs no change in this file.
        self.settings, self.embedder, self.retriever = settings, embedder, retriever
        self.reranker, self.llm, self.cache, self.memory = reranker, llm, response_cache, memory

    # ================================================================== helpers
    def _access_levels(self, role: str) -> list[str]:
        """role -> allowed document levels (config.yaml security.role_access). Unknown role = public only."""
        return self.settings.security.role_access.get(role, ["public"])

    @traceable(name="condense_question", run_type="chain")
    def _condense(self, question: str, history: list[dict[str, str]]) -> str:
        """Turn a follow-up into a standalone question.

        History:  "How many sick leaves do I get?" -> "12 days per year."
        Follow-up: "Can I carry them forward?"
        Becomes:   "Can I carry forward unused sick leave to the next year?"

        WHY: search only sees this one question. "Can I carry them forward?" alone
        contains no word about sick leave, so search would find the wrong policy.
        """
        if not history or not self.settings.retrieval.use_query_rewrite:
            return question  # first question in a session -> nothing to rewrite, no LLM call
        try:
            result = self.llm.generate(
                CONDENSE_PROMPT,
                [{"role": "user", "content": CONDENSE_USER.format(history=format_history(history), question=question)}],
                model=self.settings.llm.utility_model,  # cheap/fast model is enough
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
        """Build the `citations` list of the API response.

        cited = numbers the LLM actually wrote, e.g. [1, 3].
        If the LLM cited nothing, return all passages it was given (cited=False) for transparency.
        """
        ids = cited or list(range(1, len(used) + 1))
        out = []
        for n in ids:
            c = used[n - 1]  # [1] in the answer = first passage = index 0
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
                    "snippet": c.text[:300],  # preview for the UI, not the whole chunk
                    "cited": n in cited,
                }
            )
        return out

    def _prepare(self, req: QueryRequest, timer: StageTimer) -> dict[str, Any]:
        """Everything before generation. Returns a dict with either a ready response or LLM inputs."""
        g = self.settings.guardrails

        # ---- 1. Guardrails: reject bad input before spending money --------------
        with timer.stage("guardrails"):  # timer records how long each stage takes (timings_ms)
            question = check_input(req.question, g.max_query_chars, g.block_prompt_injection)

        # ---- 2. Follow-up -> standalone question ------------------------------
        history = self.memory.get(req.session_id)
        with timer.stage("condense"):
            standalone = self._condense(question, history)

        # ---- 3. Cache scope ------------------------------------------------------
        # The SAME question can have different correct answers for different roles/filters
        # (an hr_admin can see confidential docs). So the cache is partitioned by role + filters.
        filters_key = req.filters.cache_key() if req.filters else {}
        scope = self.cache.scope(req.role, filters_key)
        use_cache = req.use_cache and not history  # answers to follow-ups depend on the conversation

        # ---- 4. Exact cache: identical question asked before? (no API call needed) --
        if use_cache:
            with timer.stage("cache_exact"):
                hit = self.cache.get_exact(standalone, scope)
            if hit:
                return {"cached": {**hit, "cache": "exact"}, "question": question, "standalone": standalone}

        # ---- 5. Embed the question (reused below for search -> only one embedding call) --
        with timer.stage("embed_query"):
            qvec = self.embedder.embed_query(standalone)

        # ---- 6. Semantic cache: a question with the same MEANING asked before? --------
        if use_cache:
            with timer.stage("cache_semantic"):
                hit = self.cache.get_semantic(qvec, scope)
            if hit:
                return {"cached": {**hit, "cache": "semantic"}, "question": question, "standalone": standalone}

        # ---- 7. Hybrid search (meaning + keywords, with filters + role access) ------
        with timer.stage("retrieve"):
            candidates = self.retriever.retrieve(
                standalone, query_vector=qvec, filters=req.filters, access_levels=self._access_levels(req.role)
            )

        # ---- 8. Rerank: 20 candidates -> best 5 ----------------------------------
        with timer.stage("rerank"):
            ranked = self.reranker.rerank(standalone, candidates, top_n=req.top_n)

        # ---- 9. Build the numbered context, within the token budget --------------
        context, used = format_context(ranked, self.settings.llm.context_token_budget)
        return {
            "question": question,
            "standalone": standalone,
            "scope": scope,
            "qvec": qvec,
            "used": used,  # passages actually sent to the LLM (may be < ranked if over budget)
            "use_cache": use_cache,
            # Previous turns + the new message containing context and question
            "messages": history
            + [{"role": "user", "content": USER_PROMPT.format(context=context, question=standalone)}],
        }

    def _finalize(
        self, req: QueryRequest, prep: dict[str, Any], llm_result: LLMResult | None, timer: StageTimer
    ) -> dict[str, Any]:
        """Everything after generation: clean citations, build the response, save cache + memory."""
        used: list[RetrievedChunk] = prep["used"]
        if llm_result is None:
            # Nothing relevant was found -> safe fixed answer, the LLM was never called
            answer, cited = NO_CONTEXT_ANSWER, []
        else:
            # Remove [n] markers that point to no real passage (e.g. "[7]" when we sent 5)
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
            "cache": None,  # fresh answer
        }
        # Only cache grounded answers - never cache "not found" (the doc may be uploaded later)
        if prep["use_cache"] and used and llm_result is not None:
            self.cache.put(prep["standalone"], prep["scope"], response, prep["qvec"])
        # Remember this turn so the next question in the session can be a follow-up
        self.memory.append(req.session_id, prep["question"], answer)
        return response

    def _envelope(self, response: dict[str, Any], timer: StageTimer, req: QueryRequest) -> dict[str, Any]:
        """Final touches shared by fresh and cached answers."""
        # Drop internal fields (e.g. "_similarity" from the semantic cache)
        response = {k: v for k, v in response.items() if not k.startswith("_")}
        response.update(
            timings_ms={**timer.timings, "total": timer.total_ms},  # per-stage latency for debugging
            request_id=request_id_var.get(),  # same id as in logs/app.log
            run_id=current_run_id(),  # LangSmith trace id (null if tracing is off)
        )
        observe_stages(timer.timings)  # -> Prometheus /metrics
        # Extra searchable tags on the LangSmith trace
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
    # The whole answer() call becomes ONE trace in LangSmith ("rag_query"); every @traceable
    # function called inside it (condense, embed, retrieve, rerank, OpenAI) appears nested under it.
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
        """Used by POST /api/v1/query."""
        timer = StageTimer()
        prep = self._prepare(req, timer)

        # Cache hit -> return immediately (milliseconds, no LLM cost)
        if "cached" in prep:
            self.memory.append(req.session_id, prep["question"], prep["cached"]["answer"])
            return self._envelope(prep["cached"], timer, req)

        llm_result = None
        if prep["used"]:
            with timer.stage("generate"):
                # prompt_cache_key = prompt version -> OpenAI reuses the cached system prompt
                llm_result = self.llm.generate(SYSTEM_PROMPT, prep["messages"], prompt_cache_key=PROMPT_VERSION)
        else:
            # No passages survived filters/reranking -> don't let the LLM guess
            NO_ANSWER.inc()
        return self._envelope(self._finalize(req, prep, llm_result, timer), timer, req)

    def stream(self, req: QueryRequest) -> Iterator[dict[str, Any]]:
        """Server-sent-events friendly generator: ``sources`` -> ``token``* -> ``done``.

        Used by POST /api/v1/query/stream. The UI can show the sources first,
        then type the answer out word by word, then read the final metadata.
        """
        timer = StageTimer()
        prep = self._prepare(req, timer)
        if "cached" in prep:
            self.memory.append(req.session_id, prep["question"], prep["cached"]["answer"])
            final = self._envelope(prep["cached"], timer, req)
            yield {"type": "token", "content": final["answer"]}  # whole cached answer as one token
            yield {"type": "done", "response": final}
            return

        used = prep["used"]
        # Event 1: which documents will be used (lets the UI show sources immediately)
        yield {"type": "sources", "citations": self._citations(used, []) if used else []}
        llm_result = None
        if used:
            with timer.stage("generate"):
                for item in self.llm.stream(SYSTEM_PROMPT, prep["messages"], prompt_cache_key=PROMPT_VERSION):
                    if isinstance(item, LLMResult):
                        llm_result = item  # final object (full text + token usage)
                    else:
                        yield {"type": "token", "content": item}  # Event 2..n: answer pieces
        else:
            NO_ANSWER.inc()
            yield {"type": "token", "content": NO_CONTEXT_ANSWER}
        # Last event: the complete response (same shape as /query)
        yield {"type": "done", "response": self._envelope(self._finalize(req, prep, llm_result, timer), timer, req)}
