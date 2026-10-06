"""Runnable solutions for the coding questions in the interview bank.

Run:  python preparation/coding_solutions.py      -> executes every self-check and prints "All checks passed"
Each solution is explained in 09_coding_round.md.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


# ============================================================================ 1. retry with backoff (Wipro, EPAM)
class TransientError(Exception):
    """429 / 5xx / timeout -> worth retrying."""


class PermanentError(Exception):
    """400 / 401 / 404 -> retrying will fail the same way."""


def retry_with_backoff(
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    retry_on: tuple[type[BaseException], ...] = (TransientError, TimeoutError, ConnectionError),
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: exponential backoff + full jitter, retry ONLY transient errors."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on:
                    if attempt == max_attempts:
                        raise  # out of attempts -> surface the original error
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))  # 0.5, 1, 2, 4 ...
                    sleep(random.uniform(0, delay))  # full jitter: spreads out retry storms
            raise AssertionError("unreachable")

        return wrapper

    return decorator


# ============================================================================ 2. LLM response cache (Wipro)
class LLMCache:
    """Thread-safe in-memory cache with TTL + LRU eviction for identical prompts."""

    def __init__(
        self, max_entries: int = 1000, ttl_seconds: float = 3600, clock: Callable[[], float] = time.time
    ) -> None:
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.max_entries, self.ttl, self.clock = max_entries, ttl_seconds, clock
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(model: str, prompt: str) -> str:
        normalised = re.sub(r"\s+", " ", prompt.strip().lower())
        return f"{model}::{normalised}"  # model in the key: different models -> different answers

    def get(self, model: str, prompt: str) -> str | None:
        k = self.key(model, prompt)
        with self._lock:
            item = self._data.get(k)
            if item is None or self.clock() - item[0] > self.ttl:
                self._data.pop(k, None)  # expired -> drop
                self.misses += 1
                return None
            self._data.move_to_end(k)  # mark as recently used
            self.hits += 1
            return item[1]

    def set(self, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self._data[self.key(model, prompt)] = (self.clock(), response)
            self._data.move_to_end(self.key(model, prompt))
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # evict least recently used

    def get_or_call(self, model: str, prompt: str, call: Callable[[str], str]) -> str:
        cached = self.get(model, prompt)
        if cached is not None:
            return cached
        response = call(prompt)
        self.set(model, prompt, response)
        return response


# ======================================================== 3. top-k with near-duplicate removal (Wipro)
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def top_k_unique(docs: list[dict[str, Any]], k: int = 5, dup_threshold: float = 0.95) -> list[dict[str, Any]]:
    """docs: [{"id", "score", "embedding"}]. Highest score first, skipping near-duplicates of already-chosen docs."""
    chosen: list[dict[str, Any]] = []
    for doc in sorted(docs, key=lambda d: d["score"], reverse=True):
        if all(cosine(doc["embedding"], c["embedding"]) < dup_threshold for c in chosen):
            chosen.append(doc)
        if len(chosen) == k:
            break
    return chosen


def mmr(query: list[float], docs: list[dict[str, Any]], k: int = 5, lam: float = 0.7) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance: balance relevance to the query against similarity to already-selected docs."""
    selected: list[dict[str, Any]] = []
    candidates = list(docs)
    while candidates and len(selected) < k:
        best = max(
            candidates,
            key=lambda d: (
                lam * cosine(query, d["embedding"])
                - (1 - lam) * max((cosine(d["embedding"], s["embedding"]) for s in selected), default=0.0)
            ),
        )
        selected.append(best)
        candidates.remove(best)
    return selected


# ============================================================================ 4. robust JSON from an LLM (Wipro)
def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from messy LLM output: ```json fences, 'Sure! here it is:' prefixes, trailing text."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)) if start != -1 else []:
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return json.loads(text[start : i + 1])
    raise ValueError("no JSON object found")


def get_valid_json(
    call_llm: Callable[[str], str], prompt: str, required: set[str], max_attempts: int = 3
) -> dict[str, Any]:
    """Retry with the validation error fed back to the model; give up with a clear error."""
    feedback = ""
    for _ in range(max_attempts):
        raw = call_llm(prompt + feedback)
        try:
            data = extract_json(raw)
            missing = required - data.keys()
            if not missing:
                return data
            feedback = f"\n\nYour last reply was missing keys {sorted(missing)}. Return ONLY valid JSON."
        except (ValueError, json.JSONDecodeError) as exc:
            feedback = f"\n\nYour last reply was not valid JSON ({exc}). Return ONLY a JSON object."
    raise ValueError(f"LLM failed to return valid JSON after {max_attempts} attempts")


# ============================================================================ 5. chunking with overlap (Wipro)
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Word-based chunks with overlap; cut on sentence boundaries when possible."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    window: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if window and len(window) + len(words) > chunk_size:
            chunks.append(" ".join(window))
            window = window[-overlap:]  # carry the tail forward
        window.extend(words)
    if window:
        chunks.append(" ".join(window))
    return chunks


# ============================================================================ 6. pandas (PwC)
def mark_failures(df: Any) -> Any:
    """Students with Marks < 30 -> Result = 'Fail' (others 'Pass')."""
    import numpy as np

    df = df.copy()
    df["Result"] = np.where(df["Marks"] < 30, "Fail", "Pass")
    return df


# ============================================================================ 7. Python / DSA (Google, Deloitte)
def group_by_tuple_index(data: dict[str, tuple[Any, ...]], index: int) -> dict[Any, list[str]]:
    """{'a': ('HR', 1), 'b': ('IT', 2), 'c': ('HR', 3)}, index 0 -> {'HR': ['a', 'c'], 'IT': ['b']}"""
    groups: dict[Any, list[str]] = defaultdict(list)
    for key, values in data.items():
        groups[values[index]].append(key)
    return dict(groups)


def second_largest_distinct(nums: list[int]) -> int | None:
    first = second = None
    for n in nums:
        if first is None or n > first:
            first, second = n, first
        elif n != first and (second is None or n > second):
            second = n
    return second


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    seen: dict[int, int] = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return seen[target - n], i
        seen[n] = i
    return None


def char_frequency(s: str) -> dict[str, int]:
    return dict(Counter(s))


def binary_search(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def clean_text(text: str) -> str:
    """NLP live-coding: normalise messy real-world text but keep useful numbers and currency."""
    import html
    import unicodedata

    text = html.unescape(text)  # &amp; -> &
    text = unicodedata.normalize("NFKC", text)  # full-width / odd unicode -> standard
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " <EMAIL> ", text)  # mask emails
    text = text.lower()
    text = re.sub(r"[^a-z0-9₹$%.,\s<>]", " ", text)  # keep numbers, currency, basic punctuation
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================ self-checks
def _self_checks() -> None:
    # 1 retry
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=4, sleep=lambda s: None)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("429")
        return "ok"

    assert flaky() == "ok" and calls["n"] == 3
    perm_calls = {"n": 0}

    @retry_with_backoff(sleep=lambda s: None)
    def bad_request() -> None:
        perm_calls["n"] += 1
        raise PermanentError("400")

    with contextlib.suppress(PermanentError):
        bad_request()
    assert perm_calls["n"] == 1  # 400 is NOT retried

    # 2 cache
    now = [0.0]
    cache = LLMCache(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    llm_calls = {"n": 0}

    def fake_llm(p: str) -> str:
        llm_calls["n"] += 1
        return p.upper()

    cache.get_or_call("m", "Hello  World", fake_llm)
    cache.get_or_call("m", "hello world", fake_llm)  # normalised -> hit
    assert llm_calls["n"] == 1 and cache.hits == 1
    now[0] = 11
    cache.get_or_call("m", "hello world", fake_llm)  # expired -> miss
    assert llm_calls["n"] == 2
    cache.set("m", "a", "A")
    cache.set("m", "b", "B")  # evicts the oldest
    assert cache.get("m", "hello world") is None

    # 3 top-k + mmr
    docs = [
        {"id": 1, "score": 0.9, "embedding": [1.0, 0.0]},
        {"id": 2, "score": 0.89, "embedding": [0.999, 0.01]},  # near-duplicate of 1
        {"id": 3, "score": 0.5, "embedding": [0.0, 1.0]},
    ]
    assert [d["id"] for d in top_k_unique(docs, k=2)] == [1, 3]
    # lam=0.3 weights diversity: after picking doc 1, its near-duplicate (doc 2) is penalised
    assert [d["id"] for d in mmr([1.0, 0.0], docs, k=2, lam=0.3)] == [1, 3]

    # 4 json
    assert extract_json('Sure! Here it is:\n```json\n{"a": 1, "b": {"c": 2}}\n``` hope that helps') == {
        "a": 1,
        "b": {"c": 2},
    }
    replies = iter(["oops not json", '{"name": "x"}', '{"name": "x", "age": 3}'])
    assert get_valid_json(lambda p: next(replies), "give json", {"name", "age"}) == {"name": "x", "age": 3}

    # 5 chunking
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 5 and all(len(c.split()) <= 60 for c in chunks)
    assert chunks[1].split()[:10] == chunks[0].split()[-10:]  # overlap carried forward

    # 6 pandas
    try:
        import pandas as pd

        out = mark_failures(pd.DataFrame({"Student": ["A", "B"], "Marks": [25, 80]}))
        assert out["Result"].tolist() == ["Fail", "Pass"]
    except ImportError:
        pass

    # 7 DSA
    assert group_by_tuple_index({"a": ("HR", 1), "b": ("IT", 2), "c": ("HR", 3)}, 0) == {"HR": ["a", "c"], "IT": ["b"]}
    assert second_largest_distinct([5, 5, 4, 3]) == 4 and second_largest_distinct([7, 7]) is None
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert char_frequency("aab") == {"a": 2, "b": 1}
    assert binary_search([1, 3, 5, 7], 5) == 2 and binary_search([1, 3], 4) == -1
    assert clean_text("<p>Pay ₹8,000 &amp; mail hr@acme.com NOW!!! https://x.io</p>") == "pay ₹8,000 mail <email> now"
    print("All checks passed")


if __name__ == "__main__":
    _self_checks()
