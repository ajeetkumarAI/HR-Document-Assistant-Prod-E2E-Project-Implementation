"""Request / response contracts (OpenAPI docs are generated from these)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.retrieval.filters import MetadataFilters


class QueryRequestModel(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, examples=["How many sick leaves do I get per year?"])
    filters: MetadataFilters | None = Field(default=None, description="Optional metadata filters")
    session_id: str | None = Field(default=None, max_length=128, description="Enables follow-up questions")
    top_n: int | None = Field(default=None, ge=1, le=20, description="Passages sent to the LLM after reranking")
    use_cache: bool = True


class Citation(BaseModel):
    id: int
    doc_id: str | None
    chunk_id: str
    source: str | None
    title: str | None
    page: int | None = None
    section: str | None = None
    effective_date: str | None = None
    score: float
    rerank_score: float | None = None
    snippet: str
    cited: bool


class QueryResponseModel(BaseModel):
    answer: str
    citations: list[Citation]
    standalone_question: str
    model: str | None
    used_fallback_model: bool = False
    usage: dict[str, int]
    prompt_version: str
    cache: str | None
    timings_ms: dict[str, float]
    request_id: str
    run_id: str | None = Field(default=None, description="LangSmith run id - pass to /feedback")


class IngestResponseModel(BaseModel):
    summary: dict[str, int]
    results: list[dict[str, Any]]


class DocumentInfo(BaseModel):
    doc_id: str
    source: str | None
    title: str | None
    department: str | None = None
    doc_type: str | None = None
    category: str | None = None
    access_level: str | None = None
    effective_date: str | None = None
    version: str | None = None
    checksum: str | None = None
    chunks: int


class FeedbackRequest(BaseModel):
    run_id: str
    score: float = Field(..., ge=0, le=1, description="1 = helpful, 0 = not helpful")
    comment: str | None = Field(default=None, max_length=2000)


class HealthResponse(BaseModel):
    status: str
    version: str
    checks: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    details: dict[str, Any] = {}
