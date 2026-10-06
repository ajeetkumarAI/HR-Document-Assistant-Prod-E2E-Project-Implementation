"""Qdrant vector store: HNSW-tuned dense vectors + BM25 sparse vectors + payload indexes.

Collection layout
-----------------
* named vector ``dense``  : HNSW (m / ef_construct from config), optional int8 quantization
* named vector ``sparse`` : BM25 term weights, IDF computed server-side (Modifier.IDF)
* payload                 : chunk text + all metadata; keyword/integer indexes on filter fields

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

DENSE, SPARSE = "dense", "sparse"
_DISTANCE = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT, "euclid": models.Distance.EUCLID}
# Local mode ignores HNSW params / payload indexes and warns about it - that is expected.
warnings.filterwarnings("ignore", message=".*local Qdrant.*")
warnings.filterwarnings("ignore", message=".*Local mode.*")


class QdrantVectorStore:
    def __init__(self, cfg: VectorDBConfig, dimensions: int, api_key: str | None = None, local_path: str | None = None):
        self.cfg, self.dimensions, self.collection = cfg, dimensions, cfg.collection
        if cfg.mode == "memory":
            self.client = QdrantClient(location=":memory:")
        elif cfg.mode == "local":
            self.client = QdrantClient(path=local_path or cfg.local_path)
        else:
            self.client = QdrantClient(url=cfg.url, api_key=api_key, timeout=30, prefer_grpc=False)
        self.is_server = cfg.mode == "server"

    # ------------------------------------------------------------------ schema
    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            size = vectors[DENSE].size if isinstance(vectors, dict) else None
            if size is not None and size != self.dimensions:
                raise ConfigurationError(
                    f"Collection '{self.collection}' has dim={size} but embedder produces {self.dimensions}. "
                    "Use a new collection name or re-index (python scripts/ingest.py --recreate)."
                )
            return

        hnsw = self.cfg.hnsw
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
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.dimensions,
                    distance=_DISTANCE[self.cfg.distance],
                    on_disk=hnsw.on_disk,
                )
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            hnsw_config=models.HnswConfigDiff(
                m=hnsw.m,
                ef_construct=hnsw.ef_construct,
                full_scan_threshold=hnsw.full_scan_threshold,
                on_disk=hnsw.on_disk,
            ),
            quantization_config=quantization,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
        )
        if self.is_server:  # embedded/local mode has no payload indexes (it brute-forces)
            for field in self.cfg.keyword_indexes:
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
            for field in self.cfg.integer_indexes:
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
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={DENSE: d, SPARSE: models.SparseVector(indices=s[0], values=s[1])},
                payload={**c.metadata, "text": c.text},
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for batch in batched(points, batch_size):
            self.client.upsert(self.collection, points=batch, wait=True)
        return len(points)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
            ),
            wait=True,
        )

    # ------------------------------------------------------------------ reads
    def get_document_checksum(self, doc_id: str) -> str | None:
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=["checksum"],
        )
        return points[0].payload.get("checksum") if points else None

    def list_documents(self) -> list[dict[str, Any]]:
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
        while True:
            points, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=fields)
            for p in points:
                doc_id = p.payload["doc_id"]
                counts[doc_id] += 1
                docs.setdefault(doc_id, {k: p.payload.get(k) for k in fields})
            if offset is None:
                break
        return [{**d, "chunks": counts[k]} for k, d in sorted(docs.items(), key=lambda kv: kv[1]["source"] or "")]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def healthy(self) -> bool:
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
        params = models.SearchParams(hnsw_ef=self.cfg.hnsw.ef_search, exact=False)
        try:
            if mode == "hybrid" and dense is not None and sparse is not None and sparse[0]:
                # Both branches run with the same filter; Qdrant fuses them with Reciprocal Rank Fusion.
                result = self.client.query_points(
                    self.collection,
                    prefetch=[
                        models.Prefetch(
                            query=dense,
                            using=DENSE,
                            filter=query_filter,
                            limit=prefetch_limit,
                            score_threshold=dense_score_threshold,
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
                    with_payload=True,
                )
            elif mode == "sparse" and sparse is not None:
                result = self.client.query_points(
                    self.collection,
                    query=models.SparseVector(indices=sparse[0], values=sparse[1]),
                    using=SPARSE,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            else:
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
            logger.exception("Vector search failed")
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        out: list[RetrievedChunk] = []
        for p in result.points:
            payload = dict(p.payload or {})
            text = payload.pop("text", "")
            out.append(RetrievedChunk(chunk_id=str(p.id), text=text, metadata=payload, score=float(p.score)))
        return out
