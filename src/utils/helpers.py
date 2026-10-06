"""Small, dependency-light helpers shared across the codebase."""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)
T = TypeVar("T")

# Stable namespace so the same (doc, chunk) always maps to the same point id -> idempotent upserts
_UUID_NAMESPACE = uuid.UUID("6f1c2a52-3b1e-4c1d-9b7a-2f4e8d5c9a10")


# --------------------------------------------------------------------------- text
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_query(query: str) -> str:
    """Canonical form used for exact-match cache keys.

    "  How many Sick leaves?? "  ->  "how many sick leaves"
    so trivial differences in case / spacing / punctuation still hit the cache.
    """
    return re.sub(r"\s+", " ", query.strip().lower()).rstrip("?!. ")


# --------------------------------------------------------------------------- hashing / ids
def sha256(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def stable_uuid(*parts: Any) -> str:
    """Same inputs -> same UUID, on every machine and every run (uuid5 = hash-based).
    Used for doc_id and chunk_id so re-ingesting overwrites instead of duplicating."""
    return str(uuid.uuid5(_UUID_NAMESPACE, "::".join(str(p) for p in parts)))


# --------------------------------------------------------------------------- tokens
@lru_cache(maxsize=1)
def _encoder() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # offline / blocked download -> heuristic fallback
        logger.warning("tiktoken encoding unavailable; falling back to heuristic token counting")
        return None


def count_tokens(text: str) -> int:
    """Tokens = the units LLMs read and bill (~4 characters of English each).
    Chunk sizes and the context budget are measured in tokens, not characters."""
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # ~4 chars per token for English prose is a well-known approximation
    return max(1, len(text) // 4) if text else 0


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _encoder()
    if enc is not None:
        ids = enc.encode(text, disallowed_special=())
        return text if len(ids) <= max_tokens else enc.decode(ids[:max_tokens])
    return text[: max_tokens * 4]


# --------------------------------------------------------------------------- iteration
def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------- timing
class StageTimer:
    """Collects per-stage latencies (ms) for a request: ``with timer.stage("retrieve"): ...``

    Produces the `timings_ms` you see in every API response:
        {"embed_query": 380, "retrieve": 21, "rerank": 3981, "generate": 1762, "total": 6200}
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round((time.perf_counter() - t0) * 1000, 2)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)


# --------------------------------------------------------------------------- retry
def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient network / rate-limit / 5xx errors only - never on 4xx client errors.

    RETRY (temporary, may work in a second):   429 rate limit, 5xx server error, timeout, connection drop
    DON'T (will fail again identically):        400 bad request, 401 wrong key, 404 unknown model
    Retrying a 401 four times only wastes time before showing the real error.
    """
    try:
        import openai

        if isinstance(
            exc,
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    except ImportError:  # pragma: no cover
        pass
    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (408, 429) or exc.response.status_code >= 500
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, (TimeoutError, ConnectionError))


def with_retry(
    max_attempts: int = 4, initial: float = 0.5, max_wait: float = 20, jitter: float = 0.5
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Exponential backoff with jitter for transient failures (works for sync and async functions).

    Wait before attempt 2, 3, 4  ~  0.5s, 1s, 2s ... (doubling, capped at max_wait)
    + random jitter so 100 clients that failed together don't all retry at the same instant.
    reraise=True -> after the last attempt the ORIGINAL error is raised (clear message).
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=initial, max=max_wait, jitter=jitter),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, 30),  # WARNING
    )


def retry_from_settings() -> Callable[[Callable[..., T]], Callable[..., T]]:
    from src.utils.config import get_settings

    r = get_settings().retry
    return with_retry(r.max_attempts, r.initial_wait_seconds, r.max_wait_seconds, r.jitter_seconds)
