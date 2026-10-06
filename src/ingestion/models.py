"""Core data objects passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """One logical unit of source content (a whole file, or one PDF page)."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.metadata["doc_id"]


@dataclass
class Chunk:
    """A retrievable piece of a document.

    ``text`` is what the LLM sees; ``embed_text`` (text + context header) is what gets embedded.
    """

    chunk_id: str
    text: str
    embed_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float  # fusion / similarity score from the vector DB
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

    def to_langsmith(self) -> dict[str, Any]:
        """LangSmith renders retriever outputs nicely in this shape."""
        return {
            "page_content": self.text,
            "type": "Document",
            "metadata": {**self.metadata, "score": self.score, "rerank_score": self.rerank_score},
        }
