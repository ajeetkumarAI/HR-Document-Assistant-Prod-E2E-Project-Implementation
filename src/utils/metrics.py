"""Prometheus metrics exposed on /metrics."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "path", "status"], registry=REGISTRY)
REQUEST_LATENCY = Histogram("rag_http_request_seconds", "HTTP request latency", ["path"], registry=REGISTRY)
STAGE_LATENCY = Histogram(
    "rag_stage_seconds",
    "Latency per RAG stage",
    ["stage"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)
CACHE_EVENTS = Counter("rag_cache_events_total", "Cache hits/misses", ["cache", "result"], registry=REGISTRY)
LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["model", "kind"], registry=REGISTRY)
LLM_ERRORS = Counter("rag_llm_errors_total", "LLM errors", ["model"], registry=REGISTRY)
NO_ANSWER = Counter("rag_no_context_answers_total", "Queries answered without context", registry=REGISTRY)
GUARDRAIL_BLOCKS = Counter("rag_guardrail_blocks_total", "Blocked requests", ["reason"], registry=REGISTRY)


def observe_stages(timings_ms: dict[str, float]) -> None:
    for stage, ms in timings_ms.items():
        STAGE_LATENCY.labels(stage=stage).observe(ms / 1000)
