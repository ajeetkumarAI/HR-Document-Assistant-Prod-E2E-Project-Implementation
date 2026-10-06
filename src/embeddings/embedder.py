"""Dense embedding providers with batching, caching and retries.

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

_MAX_EMBED_TOKENS = 8000  # OpenAI embedding input limit is 8192


class Embedder(ABC):
    model: str
    dimensions: int

    def __init__(self, cache: EmbeddingCache | None = None, batch_size: int = 96) -> None:
        self.cache = cache
        self.batch_size = batch_size

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts: cache lookup -> batch the misses -> write back."""
        if not texts:
            return []
        cached = self.cache.get_many(texts) if self.cache else [None] * len(texts)
        missing = [i for i, v in enumerate(cached) if v is None]
        if missing:
            # de-duplicate identical texts inside the request
            unique: dict[str, list[int]] = {}
            for i in missing:
                unique.setdefault(texts[i], []).append(i)
            to_embed = list(unique)
            vectors: list[list[float]] = []
            for batch in batched(to_embed, self.batch_size):
                vectors.extend(self._embed_batch(batch))
            for text, vec in zip(to_embed, vectors, strict=True):
                for i in unique[text]:
                    cached[i] = vec
            if self.cache:
                self.cache.set_many(to_embed, vectors)
            logger.debug("Embedded texts", extra={"embedded": len(to_embed), "cached": len(texts) - len(missing)})
        return cached  # type: ignore[return-value]

    @traceable(name="embed_query", run_type="embedding")
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        dimensions: int,
        api_key: str,
        base_url: str | None = None,
        cache: EmbeddingCache | None = None,
        batch_size: int = 96,
        retry_kwargs: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> None:
        super().__init__(cache, batch_size)
        from openai import OpenAI

        self.model, self.dimensions = model, dimensions
        # SDK retries disabled: we own the retry policy (logged, jittered, configurable)
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)
        self._call = with_retry(**(retry_kwargs or {}))(self._raw_call)

    def _raw_call(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embeddings.create(**kwargs)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        safe = [truncate_to_tokens(t, _MAX_EMBED_TOKENS) or " " for t in texts]
        return self._call(safe)


class HashingEmbedder(Embedder):
    """Deterministic feature-hashing embedder (word unigrams + bigrams). Offline; for tests & CI."""

    def __init__(self, dimensions: int = 384, cache: EmbeddingCache | None = None, batch_size: int = 256) -> None:
        super().__init__(cache, batch_size)
        self.model, self.dimensions = "hashing", dimensions

    def _vec(self, text: str) -> list[float]:
        from src.embeddings.sparse import tokenize  # stop-words + light stemming

        words = tokenize(text)
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        v = np.zeros(self.dimensions, dtype=np.float32)
        for f in feats:
            h = int(sha256(f)[:8], 16)
            v[h % self.dimensions] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def build_embedder(settings: Any, cache: EmbeddingCache | None) -> Embedder:
    cfg = settings.embeddings
    if cfg.provider == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for embeddings.provider=openai")
        r = settings.retry
        return OpenAIEmbedder(
            model=cfg.model,
            dimensions=cfg.dimensions,
            api_key=settings.openai_api_key.get_secret_value(),
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
