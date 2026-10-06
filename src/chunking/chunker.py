"""Token-aware, structure-aware chunking.

WHY CHUNK AT ALL?
-----------------
We can't send a whole 50-page policy to the LLM for every question (slow, expensive,
and the answer gets lost). So we cut documents into small pieces ("chunks"), store
them, and later fetch only the few pieces relevant to the question.

HOW (visual)
------------
    # Leave Policy
    ## Earned Leave          ─┐
    ...18 days per year...    ├─► chunk 1  section = "Leave Policy > Earned Leave"
                             ─┘
    ## Sick Leave            ─┐
    ...12 days per year...    ├─► chunk 2  section = "Leave Policy > Sick Leave"
                             ─┘
    ## Maternity Leave (very long, 900 tokens)
    ...part A...              ──► chunk 3  ┐ last ~60 tokens of chunk 3 are repeated
    ...part B...              ──► chunk 4  ┘ at the start of chunk 4 (overlap)

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

# Finds Markdown headings, one per line:  "## Sick Leave"
#   group(1) = "##"         -> heading level (number of #)
#   group(2) = "Sick Leave" -> heading title
# re.MULTILINE makes ^ and $ match at every line, not just the start/end of the text.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)

# Split points, from "best place to cut" to "last resort":
#   "\n\n"            paragraph break
#   "\n"              line break (bullet points)
#   (?<=[.!?])\s+     after a sentence end  (lookbehind keeps the "." in the text)
#   (?<=[;:])\s+      after ; or :
#   " "               between words (only for a single giant sentence)
_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", r"(?<=[;:])\s+", " "]


@dataclass
class _Section:
    path: list[str]  # e.g. ["Leave Policy", "Sick Leave"]
    text: str  # body text under that heading


class Chunker:
    def __init__(
        self,
        chunk_size: int = 400,  # max tokens per chunk (~300 words)
        chunk_overlap: int = 60,  # tokens repeated between neighbouring chunks
        min_chunk_tokens: int = 30,  # smaller pieces get merged into the previous chunk
        strategy: str = "structure_aware",  # or "recursive" = ignore headings
        add_context_header: bool = True,  # prepend "Document/Section" before embedding
    ) -> None:
        # Overlap >= size would create an infinite loop (each new chunk = previous chunk)
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.strategy = strategy
        self.add_context_header = add_context_header

    # ------------------------------------------------------------------ public
    def split_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk several Documents (e.g. every page of one PDF)."""
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(self.split_document(doc))
        return chunks

    def split_document(self, doc: Document) -> list[Chunk]:
        # ---- Step 1: sections (by heading) ---------------------------------
        sections = self._sections(doc.text) if self.strategy == "structure_aware" else [_Section([], doc.text)]

        # ---- Step 2: cut each section to size --------------------------------
        # pieces = list of (heading path, text)
        pieces: list[tuple[list[str], str]] = []
        for section in sections:
            for text in self._split_text(section.text):
                pieces.append((section.path, text))

        # ---- Step 3: glue tiny leftovers onto their neighbour -----------------
        pieces = self._merge_small(pieces)

        # ---- Step 4: build Chunk objects --------------------------------------
        title = doc.metadata.get("title", doc.metadata.get("source", ""))
        page = doc.metadata.get("page")  # only set for PDFs
        chunks: list[Chunk] = []
        for idx, (path, text) in enumerate(pieces):
            section = " > ".join(path)  # ["Leave Policy","Sick Leave"] -> "Leave Policy > Sick Leave"

            # Context header. WHY: a chunk like "It cannot be carried forward." alone
            # doesn't say WHAT can't be carried forward. With the header the embedding
            # knows it is about "Leave Policy > Sick Leave", so the right question finds it.
            header = f"Document: {title}" + (f"\nSection: {section}" if section else "")
            embed_text = f"{header}\n\n{text}" if self.add_context_header else text

            # Position inside the document. PDFs include the page so ids stay unique across pages.
            chunk_index = f"{page}-{idx}" if page else str(idx)
            chunks.append(
                Chunk(
                    # Deterministic id: same document + same position -> same id every run.
                    # So re-ingesting OVERWRITES chunks instead of creating duplicates.
                    chunk_id=stable_uuid(doc.doc_id, chunk_index),
                    text=text,  # what the LLM reads (no header - saves tokens)
                    embed_text=embed_text,  # what gets turned into a vector
                    metadata={
                        **doc.metadata,  # every chunk inherits the document's metadata (filters!)
                        "section": section or None,
                        "chunk_index": idx,
                        "token_count": count_tokens(text),
                    },
                )
            )
        return chunks

    # ------------------------------------------------------------------ sections
    def _sections(self, text: str) -> list[_Section]:
        """Split text at headings and remember each section's full heading path."""
        matches = list(_HEADING.finditer(text))
        if not matches:
            return [_Section([], text)]  # no headings -> whole text is one section

        sections: list[_Section] = []

        # Text BEFORE the first heading (an intro paragraph) is kept as its own section
        if matches[0].start() > 0 and text[: matches[0].start()].strip():
            sections.append(_Section([], text[: matches[0].start()]))

        # `stack` tracks the current heading path. Example walk-through:
        #   "# Leave Policy"   stack = [(1,"Leave Policy")]
        #   "## Sick Leave"    stack = [(1,"Leave Policy"), (2,"Sick Leave")]
        #   "## Casual Leave"  pop level-2, push -> [(1,"Leave Policy"), (2,"Casual Leave")]
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level, title = len(m.group(1)), m.group(2).strip()
            # Close every heading at the same or deeper level - they are siblings/children that ended
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

            # Body = text between this heading and the next one (or the end of the document)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end() : end].strip()
            if body:  # a heading directly followed by a sub-heading has no body of its own
                sections.append(_Section([t for _, t in stack], body))
        return sections

    # ------------------------------------------------------------------ recursive split
    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        # Most HR sections are short -> keep them whole (best for answer quality)
        if count_tokens(text) <= self.chunk_size:
            return [text]
        # Too big: break into small "atoms", then pack atoms into chunks with overlap
        atoms = self._atomize(text, 0)
        return self._merge_with_overlap(atoms)

    def _atomize(self, text: str, level: int) -> list[str]:
        """Break text into pieces each <= chunk_size tokens, using the coarsest separator possible.

        Recursive: try paragraphs first; only a paragraph that is STILL too big is split by
        lines, then sentences, etc. This keeps sentences intact whenever possible.
        """
        if count_tokens(text) <= self.chunk_size or level >= len(_SEPARATORS):
            return [text] if text.strip() else []
        parts = [p for p in re.split(_SEPARATORS[level], text) if p.strip()]
        if len(parts) <= 1:
            # This separator doesn't occur in the text -> try the next, finer one
            return self._atomize(text, level + 1)
        out: list[str] = []
        for p in parts:
            out.extend(self._atomize(p.strip(), level + 1))
        return out

    def _merge_with_overlap(self, atoms: list[str]) -> list[str]:
        """Pack atoms into chunks of <= chunk_size tokens, repeating the tail of each chunk.

        atoms:   [a][b][c][d][e][f]
        chunk 1: [a][b][c]
        chunk 2:       [c][d][e]     <- [c] repeated (overlap) so context isn't cut mid-thought
        chunk 3:             [e][f]
        """
        chunks: list[str] = []
        window: list[str] = []  # atoms in the chunk being built
        window_tokens = 0
        for atom in atoms:
            t = count_tokens(atom) + 1  # +1: joining newline / tokenizer boundary effects
            if window and window_tokens + t > self.chunk_size:
                # Adding this atom would overflow -> close the current chunk
                chunks.append("\n".join(window))

                # carry trailing atoms forward as overlap (walk backwards until the budget is used)
                overlap: list[str] = []
                overlap_tokens = 0
                for prev in reversed(window):
                    pt = count_tokens(prev) + 1
                    if overlap_tokens + pt > self.chunk_overlap:
                        break
                    overlap.insert(0, prev)  # insert at front to keep original order
                    overlap_tokens += pt
                window, window_tokens = overlap, overlap_tokens  # new chunk starts with the overlap
            window.append(atom)
            window_tokens += t
        if window:
            chunks.append("\n".join(window))  # last, partially filled chunk
        return chunks

    def _merge_small(self, pieces: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
        """Merge tiny pieces (e.g. a 10-token "Eligibility" section) into the previous chunk.

        WHY: a 10-token chunk has too little meaning to be found by search, and wastes a slot
        in the top-5 results. Merging keeps it findable together with its neighbour.
        """
        merged: list[tuple[list[str], str]] = []
        for path, text in pieces:
            if (
                merged
                and count_tokens(text) < self.min_chunk_tokens  # this piece is tiny
                and count_tokens(merged[-1][1]) + count_tokens(text) <= self.chunk_size  # and it fits
            ):
                prev_path, prev_text = merged[-1]
                # If it came from a different section, keep its heading so meaning isn't lost:
                #   "...previous text\n\nBereavement Leave: 5 days..."
                heading = f"{path[-1]}: " if path and path != prev_path else ""
                merged[-1] = (prev_path, f"{prev_text}\n\n{heading}{text}")
            else:
                merged.append((path, text))
        return merged
