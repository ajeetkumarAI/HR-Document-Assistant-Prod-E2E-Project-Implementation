"""LangSmith tracing helpers.

Tracing is switched on only when ``LANGSMITH_API_KEY`` is present and ``tracing.langsmith_enabled``
is true. Everything here degrades to a no-op otherwise, so the app never depends on LangSmith
being reachable.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

from src.utils.logger import get_logger

logger = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])

try:
    from langsmith import Client as _LSClient
    from langsmith import traceable as _ls_traceable
    from langsmith.run_helpers import get_current_run_tree as _get_run_tree

    _LANGSMITH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGSMITH_AVAILABLE = False


def configure_tracing(enabled: bool, api_key: str | None, project: str) -> bool:
    """Export the env vars the LangSmith SDK reads. Returns True when tracing is active."""
    if not (_LANGSMITH_AVAILABLE and enabled and api_key):
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info("LangSmith tracing disabled")
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    logger.info("LangSmith tracing enabled", extra={"langsmith_project": os.environ["LANGSMITH_PROJECT"]})
    return True


def traceable(*, name: str | None = None, run_type: str = "chain", **kwargs: Any) -> Callable[[F], F]:
    """Thin wrapper over ``langsmith.traceable`` that is a no-op when the SDK is missing."""

    def decorator(fn: F) -> F:
        if not _LANGSMITH_AVAILABLE:
            return fn
        return _ls_traceable(name=name or fn.__name__, run_type=run_type, **kwargs)(fn)  # type: ignore[return-value]

    return decorator


def current_run_id() -> str | None:
    """Run id of the active LangSmith trace (returned to clients so they can send feedback)."""
    if not _LANGSMITH_AVAILABLE or os.environ.get("LANGSMITH_TRACING") != "true":
        return None
    try:
        run = _get_run_tree()
        return str(run.id) if run else None
    except Exception:
        return None


def add_trace_metadata(**metadata: Any) -> None:
    if not _LANGSMITH_AVAILABLE or os.environ.get("LANGSMITH_TRACING") != "true":
        return
    try:
        run = _get_run_tree()
        if run is not None:
            run.metadata.update(metadata)
    except Exception:
        pass


def log_feedback(run_id: str, score: float, comment: str | None = None, key: str = "user_rating") -> bool:
    if not _LANGSMITH_AVAILABLE or os.environ.get("LANGSMITH_TRACING") != "true":
        return False
    try:
        _LSClient().create_feedback(run_id=run_id, key=key, score=score, comment=comment)
        return True
    except Exception as exc:
        logger.warning("Failed to send LangSmith feedback", extra={"error": str(exc)})
        return False


def wrap_openai_client(client: Any) -> Any:
    """Wrap the OpenAI client so every call shows up as an LLM run with token usage."""
    if not _LANGSMITH_AVAILABLE or os.environ.get("LANGSMITH_TRACING") != "true":
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception:  # pragma: no cover
        return client
