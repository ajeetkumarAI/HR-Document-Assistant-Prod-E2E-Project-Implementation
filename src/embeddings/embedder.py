"""Dense embedding providers with batching, caching and retries.

WHAT IS AN EMBEDDING?
---------------------
A list of numbers (a vector) that captures the MEANING of a text:

    "How many sick days do I get?"   ──►  [0.012, -0.044, 0.091, ... 1536 numbers]
    "Sick leave entitlement per year" ──►  [0.010, -0.041, 0.088, ...]   <- very close!
    "Hotel limit in Mumbai"           ──►  [-0.07, 0.033, -0.012, ...]   <- far away

Texts with similar meaning get similar vectors, so "find similar vectors" =
"find text about the same thing", even if the words are different.

Providers
---------
* ``openai``                - text-embedding-3-small/large (supports reduced ``dimensions``)
* ``hashing``               - deterministic, offline bag-of-words vectors for tests / CI
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.utils.cache import EmbeddingCache
from src.utils.exceptions import ConfigurationError
from src.utils.helpers import batched, sha256, truncate_to_tokens, with_retry
from src.utils.logger import get_logger
from src.utils.tracing import traceable

logger = get_logger(__name__)

_MAX_EMBED_TOKENS = 8000  # OpenAI embedding input limit is 8192 - keep a small safety margin


class Embedder(ABC):
    """Base class. Subclasses only implement `_embed_batch`; caching/batching logic lives here once."""

    model: str  # e.g. "text-embedding-3-small" (also part of the cache key)
    dimensions: int  # vector length, e.g. 1536 (must match the Qdrant collection)

    def __init__(self, cache: EmbeddingCache | None = None, batch_size: int = 96) -> None:
        self.cache = cache
        self.batch_size = batch_size  # texts per API request (fewer requests = faster ingestion)

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts: cache lookup -> batch the misses -> write back.

        Example with 5 texts where 2 are cached and 2 are identical:
            texts   = [A, B, C, C, D]
            cached  = [vA, None, None, None, vD]       <- A and D already known
            missing = [1, 2, 3]
            unique  = {B: [1], C: [2, 3]}              <- C embedded only ONCE
            API call with [B, C]  -> fill positions 1, 2, 3
        """
        if not texts:
            return []

        # 1. Ask the cache first. Unknown texts come back as None.
        cached = self.cache.get_many(texts) if self.cache else [None] * len(texts)
        missing = [i for i, v in enumerate(cached) if v is None]

        if missing:
            # 2. de-duplicate identical texts inside the request (don't pay twice)
            unique: dict[str, list[int]] = {}  # text -> every position it appears at
            for i in missing:
                unique.setdefault(texts[i], []).append(i)
            to_embed = list(unique)

            # 3. Call the provider in batches (e.g. 96 texts per request)
            vectors: list[list[float]] = []
            for batch in batched(to_embed, self.batch_size):
                vectors.extend(self._embed_batch(batch))

            # 4. Put each new vector back into every position that needed it
            for text, vec in zip(to_embed, vectors, strict=True):
                for i in unique[text]:
                    cached[i] = vec

            # 5. Remember them so the next ingest / same question costs nothing
            if self.cache:
                self.cache.set_many(to_embed, vectors)
            logger.debug("Embedded texts", extra={"embedded": len(to_embed), "cached": len(texts) - len(missing)})
        return cached  # type: ignore[return-value]

    # @traceable -> shows up in LangSmith as an "embedding" step with its latency
    @traceable(name="embed_query", run_type="embedding")
    def embed_query(self, text: str) -> list[float]:
        """Embed ONE text (the user's question). Same code path, so it's cached too."""
        return self.embed_documents([text])[0]


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        dimensions: int,
        api_key: str,
        base_url: str | None = None,  # set for Azure OpenAI or a company gateway
        cache: EmbeddingCache | None = None,
        batch_size: int = 96,
        retry_kwargs: dict[str, Any] | None = None,
        timeout: float = 30,  # seconds before a hung request is abandoned (then retried)
    ) -> None:
        super().__init__(cache, batch_size)
        from openai import OpenAI

        self.model, self.dimensions = model, dimensions
        # SDK retries disabled: we own the retry policy (logged, jittered, configurable)
        # If both retried we'd get attempts x attempts (e.g. 4 x 3 = 12 calls) on an outage.
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)
        # Wrap the raw call with our retry decorator: backoff on 429 / 5xx / timeouts only
        self._call = with_retry(**(retry_kwargs or {}))(self._raw_call)

    def _raw_call(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        # text-embedding-3-* can return SHORTER vectors (e.g. 512 instead of 1536)
        # -> less memory in Qdrant with a small quality loss. Older models don't support it.
        if self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embeddings.create(**kwargs)
        # Sort by index so vector i always belongs to text i
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Too-long text would make the API reject the WHOLE batch -> trim each one first.
        # Empty strings are also rejected -> replace with a single space.
        safe = [truncate_to_tokens(t, _MAX_EMBED_TOKENS) or " " for t in texts]
        return self._call(safe)


class HashingEmbedder(Embedder):
    """Deterministic feature-hashing embedder (word unigrams + bigrams). Offline; for tests & CI.

    NOT for production: it only matches shared words, it doesn't understand meaning.
    It exists so the 43 tests run without an API key, without cost, and give the same result every time.
    """

    def __init__(self, dimensions: int = 384, cache: EmbeddingCache | None = None, batch_size: int = 256) -> None:
        super().__init__(cache, batch_size)
        self.model, self.dimensions = "hashing", dimensions

    def _vec(self, text: str) -> list[float]:
        from src.embeddings.sparse import tokenize  # stop-words + light stemming

        words = tokenize(text)
        # features = single words + neighbouring pairs ("sick_leav") for a bit of word order
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        v = np.zeros(self.dimensions, dtype=np.float32)
        for f in feats:
            h = int(sha256(f)[:8], 16)  # stable hash of the feature
            # bucket = h % dimensions; sign from another hash bit reduces collisions cancelling out
            v[h % self.dimensions] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()  # normalise to length 1 so cosine similarity works

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def build_embedder(settings: Any, cache: EmbeddingCache | None) -> Embedder:
    """Factory: choose the embedder from config.yaml -> embeddings.provider."""
    cfg = settings.embeddings
    if cfg.provider == "openai":
        # Fail at STARTUP with a clear message, not on the first user request
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for embeddings.provider=openai")
        r = settings.retry
        return OpenAIEmbedder(
            model=cfg.model,
            dimensions=cfg.dimensions,
            api_key=settings.openai_api_key.get_secret_value(),  # SecretStr -> plain string only here
            base_url=settings.openai_base_url,
            cache=cache,
            batch_size=cfg.batch_size,
            retry_kwargs={
                "max_attempts": r.max_attempts,
                "initial": r.initial_wait_seconds,
                "max_wait": r.max_wait_seconds,
                "jitter": r.jitter_seconds,
            },
        )
    return HashingEmbedder(cfg.dimensions, cache=cache)
