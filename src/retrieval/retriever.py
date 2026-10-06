"""Hybrid retriever: dense (semantic) + BM25 sparse (lexical), fused with RRF inside Qdrant,
with metadata + access-control filters applied during the HNSW search.

FLOW
----
    question ─► dense vector  (embedder.py, or reused from the semantic-cache step)
             ─► sparse vector (sparse.py, keywords)
             ─► filter        (filters.py: user filters + role access levels)
                      │
                      ▼
              vector_store.search()  ─►  top 20 RetrievedChunk
"""

from __future__ import annotations

from src.embeddings.embedder import Embedder
from src.embeddings.sparse import BM25SparseEncoder
from src.ingestion.models import RetrievedChunk
from src.retrieval.filters import MetadataFilters, build_qdrant_filter
from src.utils.config import RetrievalConfig
from src.utils.logger import get_logger
from src.utils.tracing import traceable
from src.vectordb.vector_store import QdrantVectorStore

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        store: QdrantVectorStore,
        embedder: Embedder,
        sparse_encoder: BM25SparseEncoder,
        cfg: RetrievalConfig,
    ) -> None:
        self.store, self.embedder, self.sparse, self.cfg = store, embedder, sparse_encoder, cfg

    # run_type="retriever" makes LangSmith show the returned documents as a nice list.
    # process_inputs: log the question, filters and access levels (not the 1536-number vector).
    @traceable(
        name="hybrid_retrieve",
        run_type="retriever",
        process_inputs=lambda i: {
            "query": i.get("query"),
            "filters": i["filters"].cache_key() if i.get("filters") else None,
            "access_levels": i.get("access_levels"),
        },
        process_outputs=lambda o: {"documents": [c.to_langsmith() for c in (o or [])]},
    )
    def retrieve(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,  # already computed by the pipeline? reuse it (saves an API call)
        filters: MetadataFilters | None = None,  # optional user filters (category, dates...)
        access_levels: list[str] | None = None,  # from the caller's role, e.g. ["public"]
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.cfg.top_k
        mode = self.cfg.mode  # hybrid | dense | sparse (config.yaml -> retrieval.mode)

        # Meaning vector - only needed for hybrid/dense
        dense = None
        if mode in ("hybrid", "dense"):
            dense = query_vector if query_vector is not None else self.embedder.embed_query(query)

        # Keyword vector - only needed for hybrid/sparse
        sparse = self.sparse.encode_query(query) if mode in ("hybrid", "sparse") else None

        # A question made only of stop-words ("what is this?") has no keywords left.
        # An empty sparse query would return nothing, so fall back to meaning-only search.
        if mode == "hybrid" and sparse is not None and not sparse[0]:
            mode = "dense"  # query had only stop-words

        results = self.store.search(
            dense=dense,
            sparse=sparse,
            # ONE filter = user filters AND role access. Applied inside both searches.
            query_filter=build_qdrant_filter(filters, access_levels),
            limit=top_k,  # final number after fusion (20)
            # Each branch fetches more than needed (20 x 2 = 40) so fusion has enough to choose from
            prefetch_limit=top_k * self.cfg.prefetch_multiplier,
            dense_score_threshold=self.cfg.dense_score_threshold,
            mode=mode,
        )
        logger.info("Retrieved candidates", extra={"mode": mode, "candidates": len(results)})
        return results
