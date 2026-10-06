"""Metadata filters -> Qdrant ``Filter``.

EXAMPLE
-------
Request body:
    {"question": "...", "filters": {"category": "expenses", "effective_after": "2026-01-01"}}
Caller role: employee  (allowed access levels: ["public"])

Becomes ONE Qdrant filter where every condition must be true ("must" = AND):
    access_level  IN  ["public"]           <- always added (security)
    category      ==  "expenses"           <- from the request
    effective_ts  >=  20260101             <- from the request

Filters are applied *inside* the HNSW search (Qdrant's filterable HNSW + payload indexes),
not as a post-filter, so you always get ``top_k`` results that satisfy the filter.
Access control is enforced the same way: a role can never retrieve chunks above its level.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import models


class MetadataFilters(BaseModel):
    """The `filters` object accepted by POST /api/v1/query. Every field is optional.

    A field can be one value ("leave") or a list (["leave", "expenses"] = either one).
    """

    department: str | list[str] | None = None
    doc_type: str | list[str] | None = None
    category: str | list[str] | None = None
    region: str | list[str] | None = None
    source: str | list[str] | None = None  # exact file name, e.g. "leave_policy.md"
    doc_ids: list[str] | None = None
    tags: list[str] | None = Field(default=None, description="match ANY of these tags")
    effective_after: date | None = None  # policies effective on/after this date
    effective_before: date | None = None  # policies effective on/before this date

    def is_empty(self) -> bool:
        return not any(v not in (None, [], "") for v in self.model_dump().values())

    def cache_key(self) -> dict[str, Any]:
        """Only the filled-in fields, JSON-friendly. Used in cache keys and trace logs, so
        "sick leave" with category=leave and without it are cached as DIFFERENT answers."""
        return {k: v for k, v in self.model_dump(mode="json").items() if v not in (None, [], "")}


def _match(key: str, value: str | list[str]) -> models.FieldCondition:
    """One value -> exact match; a list -> match any of them. Lower-cased because ingestion
    stored these fields lower-cased (metadata.normalize_metadata), so "Leave" still matches."""
    if isinstance(value, list):
        return models.FieldCondition(key=key, match=models.MatchAny(any=[str(v).lower() for v in value]))
    return models.FieldCondition(key=key, match=models.MatchValue(value=str(value).lower()))


def build_qdrant_filter(
    filters: MetadataFilters | None, allowed_access_levels: list[str] | None
) -> models.Filter | None:
    must: list[models.Condition] = []  # every condition in this list must be true

    # SECURITY FIRST: restrict to the access levels the caller's role may see.
    # Added before (and independent of) user filters, so a request can never remove it.
    if allowed_access_levels is not None:
        must.append(models.FieldCondition(key="access_level", match=models.MatchAny(any=allowed_access_levels)))

    if filters:
        # Simple text fields
        for key in ("department", "doc_type", "category", "region"):
            value = getattr(filters, key)
            if value:
                must.append(_match(key, value))

        # File name: matched exactly (NOT lower-cased) because file names keep their case
        if filters.source:
            src = filters.source
            must.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchAny(any=src) if isinstance(src, list) else models.MatchValue(value=src),
                )
            )

        if filters.doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=filters.doc_ids)))

        # Tags are stored as a list; MatchAny = the chunk has at least one of these tags
        if filters.tags:
            must.append(_match("tags", [t.lower() for t in filters.tags]))

        # Date range on the integer YYYYMMDD field: 2026-01-01 -> 20260101
        # gte = greater than or equal, lte = less than or equal; None = no limit on that side
        if filters.effective_after or filters.effective_before:
            must.append(
                models.FieldCondition(
                    key="effective_ts",
                    range=models.Range(
                        gte=int(filters.effective_after.strftime("%Y%m%d")) if filters.effective_after else None,
                        lte=int(filters.effective_before.strftime("%Y%m%d")) if filters.effective_before else None,
                    ),
                )
            )

    # No conditions at all -> None (search everything)
    return models.Filter(must=must) if must else None
