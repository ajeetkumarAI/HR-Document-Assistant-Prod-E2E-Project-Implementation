"""Metadata extraction & normalisation.

WHAT IS METADATA HERE?
----------------------
Extra facts stored next to every chunk in Qdrant, for example:

    {
      "source": "leave_policy.md",      <- shown in citations
      "category": "leave",              <- used by metadata filters
      "access_level": "public",         <- used by role-based access control
      "effective_date": "2026-01-01",   <- used by date filters
      "effective_ts": 20260101,         <- same date as a number (fast range filter)
      "checksum": "9f2c...",            <- detects whether the file changed
      "doc_id": "94b9..."               <- stable id for the whole document
    }

WHERE IT COMES FROM (later wins)
--------------------------------
  1. ``ingestion.default_metadata`` from config.yaml
  2. File-derived facts (name, extension, size, checksum, mtime)
  3. Embedded metadata (PDF info dict, DOCX core properties)
  4. A sidecar ``<file>.meta.yaml``  or YAML front-matter in .md files
  5. Metadata supplied explicitly at upload time via the API
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.utils.helpers import sha256, stable_uuid

# Only these three levels exist. Anything else is downgraded to "public" (see below).
ALLOWED_ACCESS_LEVELS = {"public", "manager", "confidential"}

# Matches a YAML block at the very start of a file:
#   ---            <- \A--- : must be the first line
#   key: value     <- (.*?) : captured lazily (stop at the first closing ---)
#   ---
# re.DOTALL lets "." match newlines so the block can span many lines.
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Return (metadata dict, remaining text). If there is no valid block, metadata is {}."""
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        # safe_load (not load): never executes code hidden in YAML
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        # Broken YAML -> treat the whole file as plain text rather than failing ingestion
        return {}, text
    # match.end() = position right after the closing "---" -> the real document body
    return (meta if isinstance(meta, dict) else {}), text[match.end() :]


def load_sidecar(path: Path) -> dict[str, Any]:
    """PDF/DOCX/CSV can't contain front-matter, so metadata lives in a separate file:

    data/raw/holidays_2026.csv
    data/raw/holidays_2026.csv.meta.yaml   <- this one
    """
    sidecar = path.with_name(path.name + ".meta.yaml")
    if sidecar.exists():
        data = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    return {}


def file_metadata(path: Path, content: bytes, logical_name: str | None = None) -> dict[str, Any]:
    """Facts we can always derive from the file itself."""
    # logical_name = the uploaded file name (API uploads have no real path on disk)
    name = logical_name or path.name
    stat_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) if path.exists() else None
    return {
        "source": name,
        "file_type": Path(name).suffix.lower().lstrip("."),  # ".PDF" -> "pdf"
        # Fallback title from the file name: "leave_policy.md" -> "Leave Policy"
        # (front-matter / PDF title will override this if present)
        "title": Path(name).stem.replace("_", " ").replace("-", " ").title(),
        # SHA-256 of the bytes. Same bytes -> same checksum -> ingestion can SKIP the file.
        "checksum": sha256(content),
        "size_bytes": len(content),
        "modified_at": stat_mtime.isoformat() if stat_mtime else None,
        # doc_id is derived from the logical name so re-uploading a new version replaces the old one
        # (same name -> same doc_id -> the pipeline deletes old chunks and writes new ones)
        "doc_id": stable_uuid("doc", name.lower()),
    }


def _to_date(value: Any) -> date | None:
    """Accept dates in the formats people actually type. Returns None if unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value  # YAML already turns 2026-01-01 into a date object
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):  # 2026-01-15, 15-01-2026, 15/01/2026...
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    return None


def normalize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce types so they are filterable in the vector DB.

    Qdrant filters are EXACT matches: "HR" != "hr". So everything we filter on must be
    stored in one consistent form, and the filter code (retrieval/filters.py) lower-cases too.
    """
    # Drop empty values - storing None just wastes space
    out = {k: v for k, v in meta.items() if v is not None}

    # "  Leave " -> "leave"
    for key in ("department", "doc_type", "category", "region"):
        if key in out:
            out[key] = str(out[key]).strip().lower()

    # SECURITY: a typo like "confidental" must not create a 4th, unprotected level.
    # Unknown values fall back to "public". (Mark sensitive docs carefully!)
    level = str(out.get("access_level", "public")).strip().lower()
    out["access_level"] = level if level in ALLOWED_ACCESS_LEVELS else "public"

    # Tags may arrive as a list (YAML) or "a, b, c" (upload form).
    # Result: lower-case, de-duplicated (set), sorted -> ["leave", "sick leave"]
    tags = out.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    out["tags"] = sorted({str(t).strip().lower() for t in tags if str(t).strip()})

    eff = _to_date(out.get("effective_date"))
    if eff:
        out["effective_date"] = eff.isoformat()  # "2026-01-01" - readable, shown in citations
        # integer YYYYMMDD -> cheap range filters with an integer payload index
        # e.g. "effective_after 2026-01-01" becomes  effective_ts >= 20260101
        out["effective_ts"] = int(eff.strftime("%Y%m%d"))
    else:
        out.pop("effective_date", None)  # unparseable date -> drop rather than store garbage

    # YAML reads version: 3.2 as a float; keep it as text so "3.10" doesn't become 3.1
    if "version" in out:
        out["version"] = str(out["version"])
    return out
