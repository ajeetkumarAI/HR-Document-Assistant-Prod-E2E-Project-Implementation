"""Document loaders: PDF, DOCX, TXT, Markdown, HTML, CSV -> ``Document`` objects with rich metadata.

WHAT THIS FILE DOES
-------------------
Takes a raw file (bytes) and turns it into one or more ``Document`` objects:

    leave_policy.pdf  ──►  [ Document(page 1 text, metadata),
                             Document(page 2 text, metadata), ... ]

    leave_policy.md   ──►  [ Document(whole text, metadata from front-matter) ]

WHY HEADINGS BECOME MARKDOWN
----------------------------
Every format has its own way of marking headings (Word styles, <h2> tags...).
We convert them all to "#", "##" so the chunker (chunking/chunker.py) can use ONE
rule to find sections, like "Leave Policy > Sick Leave".
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from src.ingestion.metadata import (
    file_metadata,
    load_sidecar,
    normalize_metadata,
    split_front_matter,
)
from src.ingestion.models import Document
from src.utils.exceptions import IngestionError, UnsupportedFileTypeError
from src.utils.helpers import normalize_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A "parser" is any function: raw bytes -> list of (text, extra_metadata).
# A list because one file can produce several parts (a PDF gives one part per page).
Parser = Callable[[bytes], list[tuple[str, dict[str, Any]]]]


# =============================================================================
# Helper: bytes -> str
# =============================================================================
def _decode(content: bytes) -> str:
    # Files written on Windows are often cp1252, not UTF-8. We try the common
    # encodings in order so a "smart quote" or "₹" doesn't crash ingestion.
    #   utf-8-sig : UTF-8 with a BOM (Notepad adds this)
    #   utf-8     : the normal case
    #   cp1252    : Windows / Excel exports
    #   latin-1   : never fails (every byte maps to a character) - last resort
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    # Safety net (practically unreachable because latin-1 always succeeds)
    return content.decode("utf-8", errors="replace")


# =============================================================================
# One parser per file type
# =============================================================================
def _parse_text(content: bytes) -> list[tuple[str, dict[str, Any]]]:
    # .md / .txt files may start with a YAML block:
    #     ---
    #     category: leave
    #     access_level: public
    #     ---
    # split_front_matter() pulls that block out as metadata and returns the rest as text.
    meta, body = split_front_matter(_decode(content))
    return [(body, meta)]


def _parse_pdf(content: bytes) -> list[tuple[str, dict[str, Any]]]:
    # Imported inside the function so the app still starts if pypdf is missing
    # and nobody uploads PDFs.
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))  # BytesIO = treat bytes in memory like a file

    # PDF files carry an "info dictionary" with optional /Title and /Author.
    # Keep only the ones that are actually filled in.
    info = reader.metadata or {}
    doc_meta = {k: v for k, v in {"title": info.get("/Title"), "author": info.get("/Author")}.items() if v}

    # One entry PER PAGE, so citations can say "Leave Policy (page 3)".
    pages: list[tuple[str, dict[str, Any]]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip blank pages (cover images, separators)
            pages.append((text, {**doc_meta, "page": i, "total_pages": len(reader.pages)}))

    # A scanned PDF is just images - there is no text to extract.
    # Fail loudly with a helpful hint instead of silently indexing nothing.
    if not pages:
        raise IngestionError("PDF has no extractable text (scanned?). Run OCR first, e.g. `ocrmypdf in.pdf out.pdf`.")
    return pages


def _parse_docx(content: bytes) -> list[tuple[str, dict[str, Any]]]:
    import docx  # python-docx

    d = docx.Document(io.BytesIO(content))
    lines: list[str] = []

    for para in d.paragraphs:
        text = para.text.strip()
        if not text:
            continue  # skip empty paragraphs (Word uses them for spacing)

        # Word marks headings with styles named "Heading 1", "Heading 2", ...
        # Convert them to Markdown so the chunker can find sections:
        #     "Heading 2" + "Sick Leave"  ──►  "## Sick Leave"
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "1"  # "heading 2" -> "2"
            lines.append(f"{'#' * min(int(level), 6)} {text}")  # Markdown supports max 6 levels
        elif style == "title":
            lines.append(f"# {text}")  # the document's Title style = top-level heading
        else:
            lines.append(text)  # normal paragraph

    # Tables: write each row as "cell | cell | cell" on ONE line.
    # WHY: if cells were separate lines, "Band B3" and "15,00,000" could end up
    # in different chunks and the LLM would lose which value belongs to which row.
    for table in d.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))

    # Word's File > Properties (title / author), if filled in
    props = d.core_properties
    meta = {k: v for k, v in {"title": props.title, "author": props.author}.items() if v}
    return [("\n\n".join(lines), meta)]


def _parse_html(content: bytes) -> list[tuple[str, dict[str, Any]]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(content), "html.parser")

    # Remove page "chrome" that is not policy content - menus and footers would
    # otherwise be indexed and pollute search results.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # <h2>Sick Leave</h2>  ──►  "## Sick Leave"   (same idea as DOCX headings)
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            h.replace_with(f"\n{'#' * level} {h.get_text(strip=True)}\n")

    title = soup.title.get_text(strip=True) if soup.title else None
    return [(soup.get_text("\n"), {"title": title} if title else {})]


def _parse_csv(content: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Each row becomes 'column: value' lines - keeps tabular facts (e.g. holiday lists) retrievable.

    Example row:  2026-11-08,Diwali,Mandatory,All India
    becomes:      date: 2026-11-08
                  holiday: Diwali
                  type: Mandatory
                  locations: All India

    WHY: the column name next to each value lets search match "Diwali holiday date"
    and lets the LLM understand what each value means.
    """
    reader = csv.DictReader(io.StringIO(_decode(content)))  # first row = column names
    rows = ["\n".join(f"{k}: {v}" for k, v in row.items() if k and v not in (None, "")) for row in reader]
    # Blank line between rows so the chunker can split between rows, never inside one.
    return [("\n\n".join(rows), {"rows": len(rows)})]


# Lookup table: file extension -> parser function.
# To support a new format, write a _parse_xxx function and add it here.
PARSERS: dict[str, Parser] = {
    ".txt": _parse_text,
    ".md": _parse_text,
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".html": _parse_html,
    ".htm": _parse_html,
    ".csv": _parse_csv,
}


# =============================================================================
# Public class used by the ingestion pipeline and the upload API
# =============================================================================
class DocumentLoader:
    def __init__(self, default_metadata: dict[str, Any] | None = None, supported: list[str] | None = None):
        # default_metadata comes from config.yaml -> ingestion.default_metadata
        # (e.g. department: HR). Every document starts with these values.
        self.default_metadata = default_metadata or {}
        # Which extensions are allowed (config.yaml -> ingestion.supported_extensions).
        # Stored lower-case so ".PDF" and ".pdf" are treated the same.
        self.supported = {e.lower() for e in (supported or PARSERS.keys())}

    def load_bytes(
        self,
        content: bytes,
        filename: str,
        extra_metadata: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> list[Document]:
        """Main entry point. Used directly for API uploads (we only have bytes + a name)."""

        # ---- 1. Pick the parser from the file extension --------------------
        ext = Path(filename).suffix.lower()
        if ext not in self.supported or ext not in PARSERS:
            # Reject early (HTTP 415) - e.g. someone uploads an .exe
            raise UnsupportedFileTypeError(f"Unsupported file type '{ext}' for {filename}")

        # ---- 2. Parse ------------------------------------------------------
        try:
            parts = PARSERS[ext](content)
        except IngestionError:
            raise  # already a clear message (e.g. "scanned PDF") - pass it through
        except Exception as exc:
            # Any library error (corrupt file, wrong format) becomes ONE error type,
            # so the pipeline can report "failed" for this file and continue with others.
            raise IngestionError(f"Failed to parse {filename}: {exc}") from exc

        # ---- 3. Build the base metadata (later entries override earlier ones) --
        #   config defaults  <  file facts (checksum, doc_id...)  <  sidecar .meta.yaml
        base = {
            **self.default_metadata,
            **file_metadata(path or Path(filename), content, logical_name=filename),
            **(load_sidecar(path) if path else {}),
        }

        # ---- 4. Create one Document per part --------------------------------
        docs: list[Document] = []
        for text, part_meta in parts:
            text = normalize_text(text)  # unify unicode, collapse extra spaces/blank lines
            if not text:
                continue
            # Priority: base  <  what the parser found (page, front-matter)  <  API upload fields
            meta = normalize_metadata({**base, **part_meta, **(extra_metadata or {})})
            docs.append(Document(text=text, metadata=meta))

        if not docs:
            raise IngestionError(f"No text content found in {filename}")
        logger.info("Loaded document", extra={"source": filename, "parts": len(docs)})
        return docs

    def load_file(self, path: str | Path, extra_metadata: dict[str, Any] | None = None) -> list[Document]:
        """Load from disk. Passing `path` lets us also read a sidecar <file>.meta.yaml."""
        path = Path(path)
        return self.load_bytes(path.read_bytes(), path.name, extra_metadata, path=path)

    def iter_directory(self, directory: str | Path) -> Iterator[Path]:
        """Yield every supported file under `directory` (including sub-folders).

        sorted()  -> same order every run (reproducible logs)
        .meta.yaml files are skipped: they describe another file, they are not documents.
        """
        directory = Path(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.supported and not path.name.endswith(".meta.yaml"):
                yield path
