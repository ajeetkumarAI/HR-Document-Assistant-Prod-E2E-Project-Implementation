"""Metadata extraction & normalisation.

Metadata sources (later wins):
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

ALLOWED_ACCESS_LEVELS = {"public", "manager", "confidential"}
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    return (meta if isinstance(meta, dict) else {}), text[match.end() :]


def load_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".meta.yaml")
    if sidecar.exists():
        data = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    return {}


def file_metadata(path: Path, content: bytes, logical_name: str | None = None) -> dict[str, Any]:
    name = logical_name or path.name
    stat_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) if path.exists() else None
    return {
        "source": name,
        "file_type": Path(name).suffix.lower().lstrip("."),
        "title": Path(name).stem.replace("_", " ").replace("-", " ").title(),
        "checksum": sha256(content),
        "size_bytes": len(content),
        "modified_at": stat_mtime.isoformat() if stat_mtime else None,
        # doc_id is derived from the logical name so re-uploading a new version replaces the old one
        "doc_id": stable_uuid("doc", name.lower()),
    }


def _to_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    return None


def normalize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce types so they are filterable in the vector DB."""
    out = {k: v for k, v in meta.items() if v is not None}

    for key in ("department", "doc_type", "category", "region"):
        if key in out:
            out[key] = str(out[key]).strip().lower()

    level = str(out.get("access_level", "public")).strip().lower()
    out["access_level"] = level if level in ALLOWED_ACCESS_LEVELS else "public"

    tags = out.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    out["tags"] = sorted({str(t).strip().lower() for t in tags if str(t).strip()})

    eff = _to_date(out.get("effective_date"))
    if eff:
        out["effective_date"] = eff.isoformat()
        # integer YYYYMMDD -> cheap range filters with an integer payload index
        out["effective_ts"] = int(eff.strftime("%Y%m%d"))
    else:
        out.pop("effective_date", None)

    if "version" in out:
        out["version"] = str(out["version"])
    return out
