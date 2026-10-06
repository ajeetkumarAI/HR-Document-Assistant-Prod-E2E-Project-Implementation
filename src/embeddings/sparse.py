"""BM25-style sparse vectors for hybrid search, computed without any model download.

We send BM25 *term-frequency* weights per hashed token; Qdrant applies the IDF part
server-side (``SparseVectorParams(modifier=Modifier.IDF)``) and keeps it up to date as the
corpus changes. Together that is real BM25 scoring inside the vector DB, so keyword-heavy
HR queries ("form 16", "PF", "LTA", policy codes) still match exactly.
"""

from __future__ import annotations

import re
from collections import Counter

from src.utils.helpers import sha256

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "my",
        "of",
        "on",
        "or",
        "our",
        "shall",
        "should",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "me",
        "am",
        "been",
        "being",
        "did",
        "may",
        "might",
        "must",
        "not",
        "no",
        "so",
        "than",
        "then",
        "they",
        "them",
        "us",
    ]
)
_VOCAB_SPACE = 2**31 - 1


def tokenize(text: str) -> list[str]:
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
    # light stemming: plural/verb suffixes - keeps "leaves"/"leave", "approved"/"approve" aligned
    out = []
    for t in tokens:
        for suffix in ("ies", "es", "s", "ed", "ing"):
            if len(t) > len(suffix) + 3 and t.endswith(suffix):
                t = t[: -len(suffix)] + ("y" if suffix == "ies" else "")
                break
        out.append(t)
    return out


def _token_id(token: str) -> int:
    return int(sha256(token)[:8], 16) % _VOCAB_SPACE


class BM25SparseEncoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 250.0) -> None:
        self.k1, self.b, self.avg_doc_len = k1, b, avg_doc_len

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        if not tokens:
            return [], []
        tf = Counter(_token_id(t) for t in tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_doc_len)
        indices = sorted(tf)
        values = [tf[i] * (self.k1 + 1) / (tf[i] + norm) for i in indices]
        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        ids = sorted({_token_id(t) for t in tokenize(text)})
        return ids, [1.0] * len(ids)
