"""Composition root: builds every component once from settings (simple dependency injection).

Swap implementations (embedder, reranker, LLM, cache backend) via config.yaml - no code changes.
Tests build their own container with offline fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.chunking.chunker import Chunker
from src.embeddings.embedder import Embedder, build_embedder
from src.embeddings.sparse import BM25SparseEncoder
from src.ingestion.loader import DocumentLoader
from src.llm.llm_client import LLMClient, build_llm
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.reranker import Reranker, build_reranker
from src.retrieval.retriever import HybridRetriever
from src.utils.cache import ConversationMemory, EmbeddingCache, ResponseCache, build_kv_store
from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger
from src.vectordb.vector_store import QdrantVectorStore

logger = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    store: QdrantVectorStore
    embedder: Embedder
    llm: LLMClient
    reranker: Reranker
    response_cache: ResponseCache
    memory: ConversationMemory
    ingestion: IngestionPipeline
    rag: RAGPipeline


def build_container(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    reranker: Reranker | None = None,
) -> Container:
    s = settings or get_settings()
    c = s.cache
    kv = build_kv_store(c.backend, c.redis_url, c.max_entries, c.ttl_seconds)

    embedder = embedder or build_embedder(
        s, EmbeddingCache(kv, s.embeddings.model, s.embeddings.dimensions, enabled=c.embedding_cache)
    )
    llm = llm or build_llm(s)
    reranker = reranker or build_reranker(s, llm)

    store = QdrantVectorStore(
        s.vectordb,
        dimensions=embedder.dimensions,
        api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None,
        local_path=str(s.resolve_path(s.vectordb.local_path)),
    )
    store.ensure_collection()

    sparse = BM25SparseEncoder()
    response_cache = ResponseCache(kv, c.response_cache, c.semantic_cache, c.semantic_threshold, c.ttl_seconds)
    memory = ConversationMemory(kv, s.memory.max_turns, s.memory.ttl_seconds, s.memory.enabled)

    ingestion = IngestionPipeline(
        loader=DocumentLoader(s.ingestion.default_metadata, s.ingestion.supported_extensions),
        chunker=Chunker(
            s.chunking.chunk_size_tokens,
            s.chunking.chunk_overlap_tokens,
            s.chunking.min_chunk_tokens,
            s.chunking.strategy,
            s.chunking.add_context_header,
        ),
        embedder=embedder,
        sparse_encoder=sparse,
        store=store,
        response_cache=response_cache,
    )
    retriever = HybridRetriever(store, embedder, sparse, s.retrieval)
    rag = RAGPipeline(s, embedder, retriever, reranker, llm, response_cache, memory)
    logger.info(
        "Container ready",
        extra={
            "embedder": embedder.model,
            "llm": getattr(llm, "model", "?"),
            "reranker": reranker.name,
            "vectordb_mode": s.vectordb.mode,
            "cache": c.backend,
        },
    )
    return Container(s, store, embedder, llm, reranker, response_cache, memory, ingestion, rag)


@lru_cache(maxsize=1)
def get_container() -> Container:
    return build_container()
