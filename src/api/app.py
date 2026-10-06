"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import TokenBucketLimiter
from src.api.middleware import RequestContextMiddleware
from src.api.routes import router, system_router
from src.pipeline.container import Container, build_container
from src.utils.config import Settings, get_settings
from src.utils.exceptions import RAGError
from src.utils.logger import get_logger, request_id_var, setup_logging
from src.utils.tracing import configure_tracing

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(
        settings.app.log_level,
        settings.resolve_path(settings.app.log_dir),
        settings.app.log_json,
        settings.guardrails.redact_pii_in_logs,
    )
    configure_tracing(
        settings.tracing.langsmith_enabled,
        settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else None,
        settings.tracing.project,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or build_container(settings)
        app.state.rate_limiter = TokenBucketLimiter(settings.security.rate_limit_per_minute)
        if settings.security.auth_enabled and not settings.api_keys:
            logger.warning("Auth is enabled but API_KEYS is empty - every request will be rejected")
        logger.info("Application started", extra={"env": settings.app.environment, "version": settings.app.version})
        yield
        logger.info("Application shutdown")

    app = FastAPI(
        title="HR Document Assistant",
        description="Production RAG API over HR policy documents - hybrid search, reranking, caching, "
        "metadata filtering, RBAC and LangSmith tracing.",
        version=settings.app.version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(RAGError)
    async def rag_error_handler(_: Request, exc: RAGError) -> JSONResponse:
        level = logger.error if exc.status_code >= 500 else logger.warning
        level("Request failed", extra={"error": exc.code, "detail": exc.message})
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "message": exc.message,
                "request_id": request_id_var.get(),
                "details": exc.details,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Invalid request",
                "request_id": request_id_var.get(),
                "details": {"errors": jsonable_errors(exc)},
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "Unexpected error", "request_id": request_id_var.get()},
        )

    app.include_router(system_router)
    app.include_router(router)
    return app


def jsonable_errors(exc: RequestValidationError) -> list[dict[str, object]]:
    return [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
