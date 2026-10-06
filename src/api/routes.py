"""HTTP endpoints.

Handlers are plain ``def`` on purpose: the pipeline does blocking I/O (HTTP to OpenAI/Qdrant),
so FastAPI runs them in its threadpool instead of blocking the event loop.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.dependencies import Principal, get_container_dep, get_principal, require_admin
from src.api.schemas import (
    DocumentInfo,
    FeedbackRequest,
    HealthResponse,
    IngestResponseModel,
    QueryRequestModel,
    QueryResponseModel,
)
from src.pipeline.container import Container
from src.pipeline.rag_pipeline import QueryRequest
from src.utils.exceptions import IngestionError, RAGError
from src.utils.logger import get_logger
from src.utils.metrics import REGISTRY
from src.utils.tracing import log_feedback

logger = get_logger(__name__)

system_router = APIRouter(tags=["system"])
router = APIRouter(prefix="/api/v1")


# =============================================================================== system
@system_router.get("/health", response_model=HealthResponse)
def health(container: Container = Depends(get_container_dep)) -> HealthResponse:
    """Liveness: the process is up."""
    return HealthResponse(status="ok", version=container.settings.app.version)


@system_router.get("/ready", response_model=HealthResponse)
def ready(response: Response, container: Container = Depends(get_container_dep)) -> HealthResponse:
    """Readiness: dependencies reachable (used by k8s / load balancers)."""
    vectordb_ok = container.store.healthy()
    checks = {
        "vectordb": vectordb_ok,
        "indexed_chunks": container.store.count() if vectordb_ok else None,
        "llm_model": getattr(container.llm, "model", None),
        "reranker": container.reranker.name,
    }
    if not vectordb_ok:
        response.status_code = 503
    return HealthResponse(
        status="ready" if vectordb_ok else "degraded", version=container.settings.app.version, checks=checks
    )


@system_router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# =============================================================================== query
def _to_request(body: QueryRequestModel, principal: Principal) -> QueryRequest:
    return QueryRequest(
        question=body.question,
        role=principal.role,
        filters=body.filters,
        # namespace sessions per caller so one key can't read another key's history
        session_id=f"{principal.key_id}:{body.session_id}" if body.session_id else None,
        top_n=body.top_n,
        use_cache=body.use_cache,
    )


@router.post("/query", response_model=QueryResponseModel, tags=["query"])
def query(
    body: QueryRequestModel,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container_dep),
) -> dict[str, Any]:
    """Ask a question about HR policies. Returns a grounded answer with citations."""
    return container.rag.answer(_to_request(body, principal))


@router.post("/query/stream", tags=["query"])
def query_stream(
    body: QueryRequestModel,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container_dep),
) -> StreamingResponse:
    """Same as /query but streams Server-Sent Events: `sources`, `token`..., `done` (or `error`)."""
    req = _to_request(body, principal)

    def events() -> Iterator[str]:
        try:
            for event in container.rag.stream(req):
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
        except RAGError as exc:
            yield f"event: error\ndata: {json.dumps({'error': exc.code, 'message': exc.message})}\n\n"
        except Exception:
            logger.exception("Streaming failed")
            yield 'event: error\ndata: {"error": "internal_error", "message": "Unexpected error"}\n\n'

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", tags=["query"])
def feedback(body: FeedbackRequest, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """Thumbs up/down on an answer - stored as LangSmith feedback for evaluation."""
    sent = log_feedback(body.run_id, body.score, body.comment)
    logger.info("Feedback received", extra={"run_id": body.run_id, "score": body.score, "sent": sent})
    return {"accepted": True, "forwarded_to_langsmith": sent}


@router.delete("/sessions/{session_id}", tags=["query"])
def clear_session(
    session_id: str,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container_dep),
) -> dict[str, Any]:
    container.memory.clear(f"{principal.key_id}:{session_id}")
    return {"cleared": True}


# =============================================================================== documents (admin)
@router.post("/documents", response_model=IngestResponseModel, tags=["documents"])
async def upload_documents(
    files: list[UploadFile] = File(...),
    department: str | None = Form(None),
    doc_type: str | None = Form(None),
    category: str | None = Form(None),
    access_level: str | None = Form(None, description="public | manager | confidential"),
    region: str | None = Form(None),
    effective_date: date | None = Form(None),
    version: str | None = Form(None),
    tags: str | None = Form(None, description="comma separated"),
    force: bool = Form(False, description="re-index even if unchanged"),
    _: Principal = Depends(require_admin),
    container: Container = Depends(get_container_dep),
) -> dict[str, Any]:
    """Upload and index one or more documents (hr_admin only)."""
    from starlette.concurrency import run_in_threadpool

    meta = {
        k: v
        for k, v in {
            "department": department,
            "doc_type": doc_type,
            "category": category,
            "access_level": access_level,
            "region": region,
            "effective_date": effective_date,
            "version": version,
            "tags": tags,
        }.items()
        if v not in (None, "")
    }
    limit = container.settings.app.max_upload_mb * 1024 * 1024
    combined: dict[str, Any] = {"summary": {}, "results": []}
    for f in files:
        content = await f.read(limit + 1)
        if len(content) > limit:
            raise IngestionError(f"{f.filename} exceeds {container.settings.app.max_upload_mb} MB")
        report = await run_in_threadpool(container.ingestion.ingest_bytes, content, f.filename or "upload", meta, force)
        combined["results"].extend(report.to_dict()["results"])
    for r in combined["results"]:
        combined["summary"][r["status"]] = combined["summary"].get(r["status"], 0) + 1
    combined["summary"]["chunks"] = sum(r["chunks"] for r in combined["results"])
    return combined


@router.post("/documents/reindex", response_model=IngestResponseModel, tags=["documents"])
def reindex(
    force: bool = False,
    _: Principal = Depends(require_admin),
    container: Container = Depends(get_container_dep),
) -> dict[str, Any]:
    """Index everything under ``ingestion.raw_data_dir`` (unchanged files are skipped)."""
    directory = container.settings.resolve_path(container.settings.ingestion.raw_data_dir)
    return container.ingestion.ingest_directory(directory, force=force).to_dict()


@router.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents(
    _: Principal = Depends(require_admin), container: Container = Depends(get_container_dep)
) -> list[dict[str, Any]]:
    return container.store.list_documents()


@router.delete("/documents/{doc_id}", tags=["documents"])
def delete_document(
    doc_id: str, _: Principal = Depends(require_admin), container: Container = Depends(get_container_dep)
) -> dict[str, Any]:
    container.ingestion.delete_document(doc_id)
    return {"deleted": doc_id}


@router.delete("/cache", tags=["system"])
def clear_cache(
    _: Principal = Depends(require_admin), container: Container = Depends(get_container_dep)
) -> dict[str, Any]:
    return {"cleared_entries": container.response_cache.clear()}
