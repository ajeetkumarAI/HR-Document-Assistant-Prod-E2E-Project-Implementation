from __future__ import annotations

from src.guardrails.guards import extract_citations
from src.utils.cache import MemoryKVStore, ResponseCache
from src.utils.logger import redact_pii


def test_pii_redaction() -> None:
    text = redact_pii("Mail ajeet@example.com, PAN ABCDE1234F, phone 9876543210, key sk-abcdefghijklmnopqrstu")
    assert "@" not in text and "ABCDE1234F" not in text and "9876543210" not in text and "sk-abc" not in text


def test_invalid_citations_removed() -> None:
    text, cited = extract_citations("Twelve days [1]. Also [7] and [2].", n_sources=3)
    assert cited == [1, 2] and "[7]" not in text


def test_response_cache_scope_and_versioning() -> None:
    cache = ResponseCache(MemoryKVStore(), threshold=0.9)
    scope = cache.scope("employee", {})
    cache.put("How many sick leaves?", scope, {"answer": "12"}, [1.0, 0.0, 0.0])
    assert cache.get_exact("how many sick leaves", scope)["answer"] == "12"
    assert cache.get_semantic([0.99, 0.05, 0.0], scope)["answer"] == "12"
    assert cache.get_semantic([0.0, 1.0, 0.0], scope) is None
    cache.invalidate()
    assert cache.get_exact("how many sick leaves", cache.scope("employee", {})) is None


def test_stemmer_aligns_word_forms() -> None:
    from src.embeddings.sparse import tokenize

    pairs = [("leaves", "leave"), ("employees", "employee"), ("approved", "approve"), ("policies", "policy")]
    for a, b in pairs:
        assert tokenize(a) == tokenize(b), (a, b)
    assert tokenize("How many sick leaves do I get?") == ["many", "sick", "leav", "get"]
