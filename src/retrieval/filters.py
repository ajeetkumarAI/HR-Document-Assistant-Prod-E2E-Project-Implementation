"""Metadata filters -> Qdrant ``Filter``.

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
    department: str | list[str] | None = None
    doc_type: str | list[str] | None = None
    category: str | list[str] | None = None
    region: str | list[str] | None = None
    source: str | list[str] | None = None
    doc_ids: list[str] | None = None
    tags: list[str] | None = Field(default=None, description="match ANY of these tags")
    effective_after: date | None = None
    effective_before: date | None = None

    def is_empty(self) -> bool:
        return not any(v not in (None, [], "") for v in self.model_dump().values())

    def cache_key(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump(mode="json").items() if v not in (None, [], "")}


def _match(key: str, value: str | list[str]) -> models.FieldCondition:
    if isinstance(value, list):
        return models.FieldCondition(key=key, match=models.MatchAny(any=[str(v).lower() for v in value]))
    return models.FieldCondition(key=key, match=models.MatchValue(value=str(value).lower()))


def build_qdrant_filter(
    filters: MetadataFilters | None, allowed_access_levels: list[str] | None
) -> models.Filter | None:
    must: list[models.Condition] = []
    if allowed_access_levels is not None:
        must.append(models.FieldCondition(key="access_level", match=models.MatchAny(any=allowed_access_levels)))
    if filters:
        for key in ("department", "doc_type", "category", "region"):
            value = getattr(filters, key)
            if value:
                must.append(_match(key, value))
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
        if filters.tags:
            must.append(_match("tags", [t.lower() for t in filters.tags]))
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
    return models.Filter(must=must) if must else None
