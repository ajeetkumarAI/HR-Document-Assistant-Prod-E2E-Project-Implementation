"""Ingestion pipeline: load -> dedupe/version check -> chunk -> embed (dense + sparse) -> upsert.

Idempotent by design:
* ``doc_id`` is derived from the file name and chunk ids from (doc_id, position), so re-ingesting
  is an upsert, never a duplicate.
* If the checksum is unchanged the file is skipped (no embedding cost).
* If the checksum changed, the old version's chunks are deleted first (no orphaned chunks).
* The answer cache is invalidated whenever the corpus changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.chunking.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.embeddings.sparse import BM25SparseEncoder
from src.ingestion.loader import DocumentLoader
from src.ingestion.models import Document
from src.utils.cache import ResponseCache
from src.utils.exceptions import IngestionError
from src.utils.logger import get_logger
from src.utils.tracing import traceable
from src.vectordb.vector_store import QdrantVectorStore

logger = get_logger(__name__)


@dataclass
class IngestResult:
    source: str
    status: str  # indexed | updated | skipped | failed
    doc_id: str | None = None
    chunks: int = 0
    error: str | None = None


@dataclass
class IngestReport:
    results: list[IngestResult] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        out["chunks"] = sum(r.chunks for r in self.results)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "results": [asdict(r) for r in self.results]}


class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        sparse_encoder: BM25SparseEncoder,
        store: QdrantVectorStore,
        response_cache: ResponseCache | None = None,
    ) -> None:
        self.loader, self.chunker, self.embedder = loader, chunker, embedder
        self.sparse, self.store, self.response_cache = sparse_encoder, store, response_cache

    @traceable(name="ingest_documents", run_type="chain")
    def ingest_documents(self, docs: list[Document], force: bool = False) -> IngestResult:
        doc_id, source, checksum = docs[0].doc_id, docs[0].metadata["source"], docs[0].metadata["checksum"]
        existing = self.store.get_document_checksum(doc_id)
        if existing == checksum and not force:
            logger.info("Unchanged document skipped", extra={"source": source})
            return IngestResult(source, "skipped", doc_id)

        chunks = self.chunker.split_documents(docs)
        if not chunks:
            raise IngestionError(f"No chunks produced for {source}")
        dense = self.embedder.embed_documents([c.embed_text for c in chunks])
        sparse = [self.sparse.encode_document(c.embed_text) for c in chunks]

        if existing is not None:
            self.store.delete_document(doc_id)  # remove the previous version's chunks
        self.store.upsert(chunks, dense, sparse)
        status = "updated" if existing is not None else "indexed"
        logger.info("Document ingested", extra={"source": source, "status": status, "chunks": len(chunks)})
        return IngestResult(source, status, doc_id, len(chunks))

    def ingest_bytes(
        self, content: bytes, filename: str, metadata: dict[str, Any] | None = None, force: bool = False
    ) -> IngestReport:
        report = IngestReport()
        try:
            docs = self.loader.load_bytes(content, filename, metadata)
            report.results.append(self.ingest_documents(docs, force))
        except Exception as exc:
            logger.exception("Ingestion failed", extra={"source": filename})
            report.results.append(IngestResult(filename, "failed", error=str(exc)))
        self._after_change(report)
        return report

    def ingest_directory(self, directory: str | Path, force: bool = False) -> IngestReport:
        report = IngestReport()
        for path in self.loader.iter_directory(directory):
            try:
                docs = self.loader.load_file(path)
                report.results.append(self.ingest_documents(docs, force))
            except Exception as exc:  # one bad file must not stop the batch
                logger.exception("Ingestion failed", extra={"source": path.name})
                report.results.append(IngestResult(path.name, "failed", error=str(exc)))
        self._after_change(report)
        logger.info("Directory ingestion complete", extra=report.summary)
        return report

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
        if self.response_cache:
            self.response_cache.invalidate()

    def _after_change(self, report: IngestReport) -> None:
        if self.response_cache and any(r.status in ("indexed", "updated") for r in report.results):
            self.response_cache.invalidate()
