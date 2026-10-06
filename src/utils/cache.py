"""Caching layer.

* ``KVStore``          - pluggable key/value backend (in-process TTL LRU or Redis)
* ``EmbeddingCache``   - avoids re-embedding identical text (saves cost on re-ingest + repeated queries)
* ``ResponseCache``    - exact-match answer cache + semantic cache (near-duplicate questions)
* ``ConversationMemory`` - short rolling chat history per session

All answer-cache keys include the *corpus version*, which is bumped on every ingest/delete,
so a document update can never serve a stale answer.

HOW THE ANSWER CACHE KEY IS BUILT
---------------------------------
    scope = hash( corpus_version | role | filters )        e.g. "a91f..."
    key   = "resp:" + scope + ":" + hash(normalised question)

    corpus_version 7, employee, no filters, "how many sick leaves"  -> resp:a91f..:3c2e..
    after a policy is re-uploaded the version becomes 8 -> a NEW scope -> old entry never matches.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Protocol

import numpy as np
from cachetools import TTLCache

from src.utils.helpers import normalize_query, sha256
from src.utils.logger import get_logger
from src.utils.metrics import CACHE_EVENTS

logger = get_logger(__name__)


# =========================================================================== backends
class KVStore(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def incr(self, key: str) -> int: ...
    def clear(self, prefix: str = "") -> int: ...


class MemoryKVStore:
    """Thread-safe TTL + LRU store. Good for a single instance / dev; use Redis when scaling out.

    TTL  = entries expire after `ttl` seconds (stale data cleans itself up)
    LRU  = when `max_entries` is reached, the least-recently-used entry is dropped (bounded RAM)
    Lock = FastAPI serves requests on several threads; the lock stops two threads corrupting the dict.
    Limitation: each process has its own copy -> with 2+ workers use Redis so they share one cache.
    """

    def __init__(self, max_entries: int = 5000, ttl: int = 86400) -> None:
        self._data: TTLCache[str, Any] = TTLCache(maxsize=max_entries, ttl=ttl)
        # counters that must not expire (the corpus version must survive the TTL, or old
        # answers could become reachable again)
        self._persistent: dict[str, Any] = {}  # counters that must not expire
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._persistent.get(key, self._data.get(key))

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._persistent.pop(key, None)

    def incr(self, key: str) -> int:
        with self._lock:
            self._persistent[key] = int(self._persistent.get(key, 0)) + 1
            return self._persistent[key]

    def clear(self, prefix: str = "") -> int:
        with self._lock:
            keys = [k for k in list(self._data.keys()) if k.startswith(prefix)]
            for k in keys:
                self._data.pop(k, None)
            return len(keys)


class RedisKVStore:
    """Shared cache for multiple API workers / servers. Values are stored as JSON strings."""

    def __init__(self, url: str, ttl: int = 86400) -> None:
        import redis

        self._r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        self._ttl = ttl
        self._r.ping()  # fail NOW (at startup) if Redis is unreachable, not on the first request

    def get(self, key: str) -> Any | None:
        raw = self._r.get(key)
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._r.set(key, json.dumps(value, default=str), ex=ttl or self._ttl)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def incr(self, key: str) -> int:
        return int(self._r.incr(key))

    def clear(self, prefix: str = "") -> int:
        n = 0
        for key in self._r.scan_iter(match=f"{prefix}*", count=500):
            if not key.endswith("corpus_version"):
                self._r.delete(key)
                n += 1
        return n


def build_kv_store(backend: str, redis_url: str, max_entries: int, ttl: int) -> KVStore:
    """config.yaml cache.backend -> store. If Redis is down we log an error and use memory:
    the app keeps working (just without a shared cache) instead of refusing to start."""
    if backend == "redis":
        try:
            return RedisKVStore(redis_url, ttl)
        except Exception as exc:  # fail-safe: caching must never take the service down
            logger.error("Redis unavailable, falling back to in-memory cache", extra={"error": str(exc)})
    return MemoryKVStore(max_entries=max_entries, ttl=ttl)


# =========================================================================== embedding cache
class EmbeddingCache:
    """text -> vector. Key includes model + dimensions so switching models never returns
    a vector of the wrong size/kind."""

    PREFIX = "emb:"

    def __init__(self, store: KVStore, model: str, dimensions: int, enabled: bool = True) -> None:
        self.store, self.enabled = store, enabled
        self._ns = f"{self.PREFIX}{model}:{dimensions}:"

    def _key(self, text: str) -> str:
        return self._ns + sha256(text)

    def get_many(self, texts: list[str]) -> list[list[float] | None]:
        if not self.enabled:
            return [None] * len(texts)
        out = [self.store.get(self._key(t)) for t in texts]
        hits = sum(v is not None for v in out)
        CACHE_EVENTS.labels("embedding", "hit").inc(hits)
        CACHE_EVENTS.labels("embedding", "miss").inc(len(texts) - hits)
        return out

    def set_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        if self.enabled:
            for t, v in zip(texts, vectors, strict=True):
                self.store.set(self._key(t), list(map(float, v)))


# =========================================================================== response cache
class ResponseCache:
    """Exact + semantic answer cache, scoped by (corpus version, role, filters)."""

    PREFIX = "resp:"
    _VERSION_KEY = "rag:corpus_version"
    _MAX_SEMANTIC_PER_SCOPE = 500

    def __init__(
        self,
        store: KVStore,
        enabled: bool = True,
        semantic: bool = True,
        threshold: float = 0.95,
        ttl: int = 86400,
    ) -> None:
        self.store, self.enabled, self.semantic = store, enabled, semantic
        self.threshold, self.ttl = threshold, ttl
        self._lock = threading.Lock()

    # -- corpus versioning -------------------------------------------------
    @property
    def corpus_version(self) -> int:
        return int(self.store.get(self._VERSION_KEY) or 0)

    def invalidate(self) -> int:
        """Called after ingest/delete. Old entries become unreachable and expire via TTL."""
        version = self.store.incr(self._VERSION_KEY)
        logger.info("Response cache invalidated", extra={"corpus_version": version})
        return version

    def scope(self, role: str, filters: dict[str, Any] | None) -> str:
        return sha256(f"{self.corpus_version}|{role}|{json.dumps(filters or {}, sort_keys=True)}")[:24]

    # -- lookups ---------------------------------------------------------------
    def get_exact(self, query: str, scope: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        hit = self.store.get(f"{self.PREFIX}{scope}:{sha256(normalize_query(query))}")
        CACHE_EVENTS.labels("response_exact", "hit" if hit else "miss").inc()
        return hit

    def get_semantic(self, query_vec: list[float], scope: str) -> dict[str, Any] | None:
        """Find a previously answered question with (almost) the same MEANING.

        "How many sick leaves do I get?"  vs  "Sick leave days per year?"  -> similarity 0.96 -> HIT
        Uses cosine similarity: both vectors are normalised to length 1, so a dot product = cosine.
        Threshold 0.95 is strict on purpose - a wrong cached answer is worse than a cache miss.
        """
        if not (self.enabled and self.semantic):
            return None
        index = self.store.get(f"{self.PREFIX}sem:{scope}") or []
        if not index:
            CACHE_EVENTS.labels("response_semantic", "miss").inc()
            return None
        q = np.asarray(query_vec, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        mat = np.asarray([e["vec"] for e in index], dtype=np.float32)  # one row per cached question
        sims = mat @ q  # similarity of the new question to every cached one, in one matrix multiply
        best = int(np.argmax(sims))
        if float(sims[best]) >= self.threshold:
            hit = self.store.get(index[best]["key"])
            if hit:
                CACHE_EVENTS.labels("response_semantic", "hit").inc()
                return {**hit, "_similarity": round(float(sims[best]), 4)}
        CACHE_EVENTS.labels("response_semantic", "miss").inc()
        return None

    def put(self, query: str, scope: str, response: dict[str, Any], query_vec: list[float] | None) -> None:
        if not self.enabled:
            return
        key = f"{self.PREFIX}{scope}:{sha256(normalize_query(query))}"
        self.store.set(key, response, ttl=self.ttl)
        if self.semantic and query_vec is not None:
            v = np.asarray(query_vec, dtype=np.float32)
            v /= np.linalg.norm(v) or 1.0
            with self._lock:
                idx_key = f"{self.PREFIX}sem:{scope}"
                index = self.store.get(idx_key) or []
                index.append({"key": key, "vec": v.round(5).tolist()})  # rounded -> smaller to store
                # keep only the newest 500 questions per scope -> the similarity check stays fast
                self.store.set(idx_key, index[-self._MAX_SEMANTIC_PER_SCOPE :], ttl=self.ttl)

    def clear(self) -> int:
        return self.store.clear(self.PREFIX)


# =========================================================================== conversation memory
class ConversationMemory:
    """Last N question/answer pairs per session_id, so follow-ups like "and for managers?" work.

    Stored as chat messages:  [{"role": "user", ...}, {"role": "assistant", ...}, ...]
    Expires after `ttl` (default 1 hour of inactivity).
    """

    PREFIX = "chat:"

    def __init__(self, store: KVStore, max_turns: int = 6, ttl: int = 3600, enabled: bool = True) -> None:
        self.store, self.max_turns, self.ttl, self.enabled = store, max_turns, ttl, enabled

    def get(self, session_id: str | None) -> list[dict[str, str]]:
        if not (self.enabled and session_id):
            return []
        return self.store.get(self.PREFIX + session_id) or []

    def append(self, session_id: str | None, question: str, answer: str) -> None:
        if not (self.enabled and session_id):
            return
        history = self.get(session_id)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        # 1 turn = 2 messages (user + assistant). Keep only the newest turns -> bounded prompt size.
        self.store.set(self.PREFIX + session_id, history[-2 * self.max_turns :], ttl=self.ttl)

    def clear(self, session_id: str) -> None:
        self.store.delete(self.PREFIX + session_id)
