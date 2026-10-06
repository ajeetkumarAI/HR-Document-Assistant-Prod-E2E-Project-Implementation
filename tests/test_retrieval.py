from __future__ import annotations

from datetime import date

from src.ingestion.models import RetrievedChunk
from src.pipeline.container import Container
from src.retrieval.filters import MetadataFilters
from src.retrieval.reranker import Reranker


def _retrieve(c: Container, q: str, levels: list[str] | None = None, **filters: object) -> list:
    return c.rag.retriever.retrieve(
        q, filters=MetadataFilters(**filters) if filters else None, access_levels=levels or ["public"]
    )


def test_hybrid_retrieval_finds_relevant_section(indexed_container: Container) -> None:
    results = _retrieve(indexed_container, "medical certificate for sick leave")
    assert results
    assert any("medical certificate" in r.text for r in results[:3])


def test_keyword_match_via_sparse_vectors(indexed_container: Container) -> None:
    results = _retrieve(indexed_container, "POSH Internal Committee")
    assert results[0].metadata["source"] == "code_of_conduct.md"


def test_metadata_filter_restricts_results(indexed_container: Container) -> None:
    results = _retrieve(indexed_container, "allowance per day", category="expenses")
    assert results and all(r.metadata["category"] == "expenses" for r in results)


def test_date_range_filter(indexed_container: Container) -> None:
    results = _retrieve(indexed_container, "policy", effective_after=date(2026, 1, 1))
    assert results and all(r.metadata["effective_ts"] >= 20260101 for r in results)


def test_rbac_hides_confidential_docs_from_employees(indexed_container: Container) -> None:
    employee = _retrieve(indexed_container, "salary band increment matrix")
    assert all(r.metadata["access_level"] == "public" for r in employee)
    admin = _retrieve(indexed_container, "salary band increment matrix", ["public", "manager", "confidential"])
    assert admin[0].metadata["source"] == "compensation_bands.md"


def test_reranker_reorders_and_fails_open() -> None:
    class LengthReranker(Reranker):
        name = "length"

        def _score(self, query, chunks):  # type: ignore[no-untyped-def]
            return [len(c.text) for c in chunks]

    class BrokenReranker(Reranker):
        name = "broken"

        def _score(self, query, chunks):  # type: ignore[no-untyped-def]
            raise RuntimeError("model server down")

    chunks = [RetrievedChunk(str(i), "x" * (i + 1), {}, score=1 / (i + 1)) for i in range(5)]
    ranked = LengthReranker(top_n=2).rerank("q", chunks)
    assert [c.chunk_id for c in ranked] == ["4", "3"]
    fallback = BrokenReranker(top_n=3).rerank("q", chunks)
    assert [c.chunk_id for c in fallback] == ["0", "1", "2"]


def test_llm_reranker_uses_model_scores() -> None:
    from src.retrieval.reranker import LLMReranker

    class ScoringLLM:
        def generate_json(self, instructions, messages, schema, name, **kw):  # type: ignore[no-untyped-def]
            assert name == "rerank_scores" and "Question: sick leave" in messages[0]["content"]
            return {
                "scores": [{"id": 0, "score": 2}, {"id": 1, "score": 9}, {"id": 2, "score": 5}, {"id": 99, "score": 10}]
            }

    chunks = [RetrievedChunk(str(i), f"passage {i}", {}, score=1.0) for i in range(3)]
    ranked = LLMReranker(ScoringLLM(), top_n=2).rerank("sick leave", chunks)
    assert [c.chunk_id for c in ranked] == ["1", "2"]  # invalid id 99 ignored
    assert ranked[0].rerank_score == 9
