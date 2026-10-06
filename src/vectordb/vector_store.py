"""Qdrant vector store: HNSW-tuned dense vectors + BM25 sparse vectors + payload indexes.

WHAT IS STORED (one "point" per chunk)
--------------------------------------
    point id : "4a38526c-..."                       (chunk_id, deterministic)
    vectors  : dense  -> [0.012, -0.044, ...]       (meaning, 1536 numbers)
               sparse -> {18822: 0.9, 9937121: 1.1} (keywords)
    payload  : {"text": "...", "source": "leave_policy.md", "category": "leave",
                "access_level": "public", "effective_ts": 20260101, ...}

Collection layout
-----------------
* named vector ``dense``  : HNSW (m / ef_construct from config), optional int8 quantization
* named vector ``sparse`` : BM25 term weights, IDF computed server-side (Modifier.IDF)
* payload                 : chunk text + all metadata; keyword/integer indexes on filter fields

WHAT IS HNSW?
-------------
Comparing a question with EVERY stored vector is slow at millions of chunks.
HNSW builds a "small-world graph" where each vector links to its nearest neighbours,
so a search hops through the graph and checks only a few hundred vectors.
    m            = links per vector        (more = better recall, more RAM)
    ef_construct = effort when building    (more = better graph, slower ingestion)
    ef_search    = effort when searching   (more = better recall, slower query)

Modes: ``memory`` (tests), ``local`` (embedded on-disk, no server - exact search), ``server``
(Docker / Qdrant Cloud - real HNSW + payload indexes). Use ``server`` in staging/prod.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

from qdrant_client import QdrantClient, models

from src.ingestion.models import Chunk, RetrievedChunk
from src.utils.config import VectorDBConfig
from src.utils.exceptions import ConfigurationError, RetrievalError
from src.utils.helpers import batched
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Names of the two vectors stored per point
DENSE, SPARSE = "dense", "sparse"

# config.yaml "distance" -> Qdrant enum. Cosine = compare direction (standard for OpenAI embeddings).
_DISTANCE = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT, "euclid": models.Distance.EUCLID}

# Local mode ignores HNSW params / payload indexes and warns about it - that is expected.
warnings.filterwarnings("ignore", message=".*local Qdrant.*")
warnings.filterwarnings("ignore", message=".*Local mode.*")


class QdrantVectorStore:
    def __init__(self, cfg: VectorDBConfig, dimensions: int, api_key: str | None = None, local_path: str | None = None):
        self.cfg, self.dimensions, self.collection = cfg, dimensions, cfg.collection

        # Same code, three ways to run Qdrant (chosen in config.yaml -> vectordb.mode):
        if cfg.mode == "memory":
            # RAM only, gone when the process ends -> perfect for tests
            self.client = QdrantClient(location=":memory:")
        elif cfg.mode == "local":
            # Files in data/qdrant, no server to install. Only ONE process may open it at a time
            # (that's why ingest and the API server can't run together in this mode).
            self.client = QdrantClient(path=local_path or cfg.local_path)
        else:
            # Real Qdrant server (docker-compose or Qdrant Cloud) - many workers can share it
            self.client = QdrantClient(url=cfg.url, api_key=api_key, timeout=30, prefer_grpc=False)
        self.is_server = cfg.mode == "server"

    # ------------------------------------------------------------------ schema
    def ensure_collection(self) -> None:
        """Create the collection the first time; on later starts just verify it's compatible."""
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            size = vectors[DENSE].size if isinstance(vectors, dict) else None
            # Mixing vector sizes is impossible. This happens if you switch the embedding model
            # or `dimensions` in config.yaml but keep the old data -> stop with clear instructions.
            if size is not None and size != self.dimensions:
                raise ConfigurationError(
                    f"Collection '{self.collection}' has dim={size} but embedder produces {self.dimensions}. "
                    "Use a new collection name or re-index (python -m scripts.ingest --recreate)."
                )
            return

        hnsw = self.cfg.hnsw

        # Optional compression: store each number in 1 byte instead of 4 (4x less RAM).
        # quantile=0.99 ignores the 1% most extreme values when choosing the scale (less precision loss).
        quantization = (
            models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=self.cfg.quantization.always_ram
                )
            )
            if self.cfg.quantization.enabled
            else None
        )

        self.client.create_collection(
            collection_name=self.collection,
            # Dense vector: fixed length, compared by cosine similarity
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.dimensions,
                    distance=_DISTANCE[self.cfg.distance],
                    on_disk=hnsw.on_disk,  # true = keep vectors on disk for very large corpora
                )
            },
            # Sparse vector: Qdrant computes IDF itself and keeps it updated as documents change
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            # HNSW graph settings (see top of file)
            hnsw_config=models.HnswConfigDiff(
                m=hnsw.m,
                ef_construct=hnsw.ef_construct,
                full_scan_threshold=hnsw.full_scan_threshold,  # below this many points, brute force is faster
                on_disk=hnsw.on_disk,
            ),
            quantization_config=quantization,
            # Build the HNSW index once 10k vectors exist; smaller collections are searched directly
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
        )

        # Payload indexes = like database indexes on the metadata fields we filter by.
        # WHY: with a filter like category="leave", Qdrant uses the index to stay fast
        # and still return the full top_k results ("filterable HNSW").
        if self.is_server:  # embedded/local mode has no payload indexes (it brute-forces)
            for field in self.cfg.keyword_indexes:  # exact-match text fields
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
            for field in self.cfg.integer_indexes:  # number fields for range filters (effective_ts)
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.INTEGER)
        logger.info(
            "Created Qdrant collection",
            extra={
                "collection": self.collection,
                "dim": self.dimensions,
                "hnsw_m": hnsw.m,
                "ef_construct": hnsw.ef_construct,
            },
        )

    def recreate(self) -> None:
        """Delete everything and start empty (used by `scripts.ingest --recreate`)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure_collection()

    # ------------------------------------------------------------------ writes
    def upsert(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[tuple[list[int], list[float]]],
        batch_size: int = 128,
    ) -> int:
        """Insert or overwrite chunks. "Upsert" = update if the id exists, insert if not."""
        # Build one Qdrant point per chunk. zip(strict=True) crashes loudly if the three
        # lists ever have different lengths (a bug we want to catch, not hide).
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={DENSE: d, SPARSE: models.SparseVector(indices=s[0], values=s[1])},
                payload={**c.metadata, "text": c.text},  # text stored so search results include it
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        # Send in batches: one huge request could hit size limits / timeouts.
        # wait=True -> return only after Qdrant has stored them (so a query right after sees them).
        for batch in batched(points, batch_size):
            self.client.upsert(self.collection, points=batch, wait=True)
        return len(points)

    def delete_document(self, doc_id: str) -> None:
        """Delete EVERY chunk of one document (all chunks share the same doc_id)."""
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
            ),
            wait=True,
        )

    # ------------------------------------------------------------------ reads
    def get_document_checksum(self, doc_id: str) -> str | None:
        """Checksum of the stored version of a document, or None if it was never indexed.

        Used by ingestion: same checksum -> file unchanged -> skip (no embedding cost).
        """
        points, _ = self.client.scroll(  # scroll = read points by filter, no vector search
            self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            limit=1,  # any one chunk is enough - they all carry the same checksum
            with_payload=["checksum"],  # fetch only this field, not the text
        )
        return points[0].payload.get("checksum") if points else None

    def list_documents(self) -> list[dict[str, Any]]:
        """One row per document with its chunk count (for GET /api/v1/documents)."""
        fields = [
            "doc_id",
            "source",
            "title",
            "department",
            "doc_type",
            "category",
            "access_level",
            "effective_date",
            "version",
            "checksum",
        ]
        docs: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = defaultdict(int)
        offset = None
        # Qdrant returns results page by page (512 at a time); `offset` points to the next page.
        while True:
            points, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=fields)
            for p in points:
                doc_id = p.payload["doc_id"]
                counts[doc_id] += 1
                docs.setdefault(doc_id, {k: p.payload.get(k) for k in fields})  # keep the first chunk's info
            if offset is None:  # no more pages
                break
        return [{**d, "chunks": counts[k]} for k, d in sorted(docs.items(), key=lambda kv: kv[1]["source"] or "")]

    def count(self) -> int:
        """Total number of chunks stored (shown by GET /ready)."""
        return self.client.count(self.collection, exact=True).count

    def healthy(self) -> bool:
        """Used by GET /ready: can we reach the collection right now?"""
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False

    def search(
        self,
        *,
        dense: list[float] | None,
        sparse: tuple[list[int], list[float]] | None,
        query_filter: models.Filter | None,
        limit: int,
        prefetch_limit: int,
        dense_score_threshold: float | None = None,
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        """Find the chunks most relevant to the question.

        HYBRID (default):

            question ─┬─► dense search  (top 40 by meaning,  WITH filter) ─┐
                      │                                                    ├─► RRF merge ─► top 20
                      └─► sparse search (top 40 by keywords, WITH filter) ─┘

        RRF (Reciprocal Rank Fusion) scores each chunk by its RANK in each list:
            score = 1/(60+rank_dense) + 1/(60+rank_sparse)
        A chunk ranked well in BOTH lists wins. Ranks are used instead of raw scores
        because cosine scores and BM25 scores are on different scales.
        """
        # ef_search: how hard HNSW looks (higher = better recall, slower). exact=False -> use HNSW.
        params = models.SearchParams(hnsw_ef=self.cfg.hnsw.ef_search, exact=False)
        try:
            if mode == "hybrid" and dense is not None and sparse is not None and sparse[0]:
                # Both branches run with the same filter; Qdrant fuses them with Reciprocal Rank Fusion.
                # The filter is applied INSIDE each search (not afterwards) so we still get `limit` results
                # and confidential chunks are never even considered for an employee.
                result = self.client.query_points(
                    self.collection,
                    prefetch=[
                        models.Prefetch(
                            query=dense,
                            using=DENSE,
                            filter=query_filter,
                            limit=prefetch_limit,
                            score_threshold=dense_score_threshold,  # drop very weak meaning matches
                            params=params,
                        ),
                        models.Prefetch(
                            query=models.SparseVector(indices=sparse[0], values=sparse[1]),
                            using=SPARSE,
                            filter=query_filter,
                            limit=prefetch_limit,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    with_payload=True,  # return text + metadata with each hit
                )
            elif mode == "sparse" and sparse is not None:
                # Keyword-only search (retrieval.mode = sparse)
                result = self.client.query_points(
                    self.collection,
                    query=models.SparseVector(indices=sparse[0], values=sparse[1]),
                    using=SPARSE,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            else:
                # Meaning-only search (retrieval.mode = dense, or the question was all stop-words)
                result = self.client.query_points(
                    self.collection,
                    query=dense,
                    using=DENSE,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=dense_score_threshold,
                    search_params=params,
                    with_payload=True,
                )
        except Exception as exc:
            # Turn any Qdrant/network error into one error type -> API returns HTTP 503
            logger.exception("Vector search failed")
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        # Convert Qdrant results into our own RetrievedChunk objects
        out: list[RetrievedChunk] = []
        for p in result.points:
            payload = dict(p.payload or {})
            text = payload.pop("text", "")  # separate the text from the metadata
            out.append(RetrievedChunk(chunk_id=str(p.id), text=text, metadata=payload, score=float(p.score)))
        return out
