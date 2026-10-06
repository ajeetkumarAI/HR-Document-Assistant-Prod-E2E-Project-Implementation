"""Token-aware, structure-aware chunking.

1. Split the document on Markdown headings into sections (keeping the heading path,
   e.g. "Leave Policy > Sick Leave > Documentation").
2. Sections that fit in ``chunk_size_tokens`` stay whole; larger ones are split recursively
   on paragraph -> line -> sentence -> word boundaries, then greedily merged with token overlap.
3. Tiny trailing fragments are merged into their neighbour so we never index 5-token chunks.
4. A context header (title + section path) is prepended to the text that is *embedded*,
   which noticeably improves retrieval for chunks like "It is capped at 12 days per year."
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.models import Chunk, Document
from src.utils.helpers import count_tokens, stable_uuid

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", r"(?<=[;:])\s+", " "]


@dataclass
class _Section:
    path: list[str]
    text: str


class Chunker:
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        min_chunk_tokens: int = 30,
        strategy: str = "structure_aware",
        add_context_header: bool = True,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.strategy = strategy
        self.add_context_header = add_context_header

    # ------------------------------------------------------------------ public
    def split_documents(self, docs: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(self.split_document(doc))
        return chunks

    def split_document(self, doc: Document) -> list[Chunk]:
        sections = self._sections(doc.text) if self.strategy == "structure_aware" else [_Section([], doc.text)]
        pieces: list[tuple[list[str], str]] = []
        for section in sections:
            for text in self._split_text(section.text):
                pieces.append((section.path, text))
        pieces = self._merge_small(pieces)

        title = doc.metadata.get("title", doc.metadata.get("source", ""))
        page = doc.metadata.get("page")
        chunks: list[Chunk] = []
        for idx, (path, text) in enumerate(pieces):
            section = " > ".join(path)
            header = f"Document: {title}" + (f"\nSection: {section}" if section else "")
            embed_text = f"{header}\n\n{text}" if self.add_context_header else text
            chunk_index = f"{page}-{idx}" if page else str(idx)
            chunks.append(
                Chunk(
                    chunk_id=stable_uuid(doc.doc_id, chunk_index),
                    text=text,
                    embed_text=embed_text,
                    metadata={
                        **doc.metadata,
                        "section": section or None,
                        "chunk_index": idx,
                        "token_count": count_tokens(text),
                    },
                )
            )
        return chunks

    # ------------------------------------------------------------------ sections
    def _sections(self, text: str) -> list[_Section]:
        matches = list(_HEADING.finditer(text))
        if not matches:
            return [_Section([], text)]
        sections: list[_Section] = []
        if matches[0].start() > 0 and text[: matches[0].start()].strip():
            sections.append(_Section([], text[: matches[0].start()]))
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end() : end].strip()
            if body:
                sections.append(_Section([t for _, t in stack], body))
        return sections

    # ------------------------------------------------------------------ recursive split
    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if count_tokens(text) <= self.chunk_size:
            return [text]
        atoms = self._atomize(text, 0)
        return self._merge_with_overlap(atoms)

    def _atomize(self, text: str, level: int) -> list[str]:
        """Break text into pieces each <= chunk_size tokens, using the coarsest separator possible."""
        if count_tokens(text) <= self.chunk_size or level >= len(_SEPARATORS):
            return [text] if text.strip() else []
        parts = [p for p in re.split(_SEPARATORS[level], text) if p.strip()]
        if len(parts) <= 1:
            return self._atomize(text, level + 1)
        out: list[str] = []
        for p in parts:
            out.extend(self._atomize(p.strip(), level + 1))
        return out

    def _merge_with_overlap(self, atoms: list[str]) -> list[str]:
        chunks: list[str] = []
        window: list[str] = []
        window_tokens = 0
        for atom in atoms:
            t = count_tokens(atom) + 1  # +1: joining newline / tokenizer boundary effects
            if window and window_tokens + t > self.chunk_size:
                chunks.append("\n".join(window))
                # carry trailing atoms forward as overlap
                overlap: list[str] = []
                overlap_tokens = 0
                for prev in reversed(window):
                    pt = count_tokens(prev) + 1
                    if overlap_tokens + pt > self.chunk_overlap:
                        break
                    overlap.insert(0, prev)
                    overlap_tokens += pt
                window, window_tokens = overlap, overlap_tokens
            window.append(atom)
            window_tokens += t
        if window:
            chunks.append("\n".join(window))
        return chunks

    def _merge_small(self, pieces: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
        merged: list[tuple[list[str], str]] = []
        for path, text in pieces:
            if (
                merged
                and count_tokens(text) < self.min_chunk_tokens
                and count_tokens(merged[-1][1]) + count_tokens(text) <= self.chunk_size
            ):
                prev_path, prev_text = merged[-1]
                heading = f"{path[-1]}: " if path and path != prev_path else ""
                merged[-1] = (prev_path, f"{prev_text}\n\n{heading}{text}")
            else:
                merged.append((path, text))
        return merged
