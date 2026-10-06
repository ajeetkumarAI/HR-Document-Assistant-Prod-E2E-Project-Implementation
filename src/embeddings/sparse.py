"""BM25-style sparse vectors for hybrid search, computed without any model download.

DENSE vs SPARSE (why we need both)
----------------------------------
Dense (embedder.py)  = meaning.  Good at: "time off when unwell" -> finds "sick leave".
                                 Weak at: exact codes/acronyms like "POSH", "LTA", "Form 16".
Sparse (this file)   = keywords. Good at: exact words. Weak at: synonyms.
Hybrid search runs BOTH and merges the results -> best of both.

WHAT A SPARSE VECTOR LOOKS LIKE
-------------------------------
Only the words that appear get a value; everything else is 0, so we store just those:

    "Sick leave requires a medical certificate"
       tokens  -> ["sick", "leav", "requir", "medical", "certificat"]
       indices -> [ 18822, 9937121, 44410, 7711, 55023 ]   (hash of each word)
       values  -> [ 1.0,   1.0,     1.0,   1.0,  1.0  ]    (BM25 term-frequency weight)

We send BM25 *term-frequency* weights per hashed token; Qdrant applies the IDF part
server-side (``SparseVectorParams(modifier=Modifier.IDF)``) and keeps it up to date as the
corpus changes. Together that is real BM25 scoring inside the vector DB, so keyword-heavy
HR queries ("form 16", "PF", "LTA", policy codes) still match exactly.
(IDF = rare words like "POSH" count more than common words like "employee".)
"""

from __future__ import annotations

import re
from collections import Counter

from src.utils.helpers import sha256

# A token = letters/digits, optionally joined by - or _  ("form-16", "work_from_home")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

# Very common words that carry no meaning for search. Removing them means
# "how many days of leave" matches on "days" + "leave", not on "how"/"of".
_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has",
        "have", "how", "i", "if", "in", "is", "it", "its", "my", "of", "on", "or", "our", "shall",
        "should", "that", "the", "their", "there", "these", "this", "to", "was", "we", "what",
        "when", "where", "which", "who", "will", "with", "you", "your", "me", "am", "been", "being",
        "did", "may", "might", "must", "not", "no", "so", "than", "then", "they", "them", "us",
    ]
)  # fmt: skip

# Word ids are hashes in [0, 2^31). No vocabulary file is needed, and new words just work.
_VOCAB_SPACE = 2**31 - 1


def _stem(t: str) -> str:
    """Light stemming so different forms of a word become the SAME token.

    The same function runs on documents (ingestion) and on questions (query time),
    so both sides always line up:

        leaves / leave              -> leav
        employees / employee        -> employe
        approved / approving / approve -> approv
        policies / policy           -> policy
        processes / process         -> process
    """
    if len(t) <= 4:  # short words ("days" aside) are left alone: "pass", "bus", "fee"
        return t[:-1] if t.endswith("s") and not t.endswith("ss") and len(t) == 4 else t
    if t.endswith("ies"):  # policies -> policy
        t = t[:-3] + "y"
    elif t.endswith(("sses", "xes", "ches", "shes", "zes")):  # processes -> process, taxes -> tax
        t = t[:-2]
    elif t.endswith("s") and not t.endswith("ss"):  # leaves -> leave, employees -> employee
        t = t[:-1]
    elif t.endswith("ing") and len(t) > 6:  # approving -> approv
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 5:  # approved -> approv
        t = t[:-2]
    # Drop a final silent "e" so "leave" and "leave(s)" both end as "leav"
    if t.endswith("e") and len(t) > 4:
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    """Lower-case, split into words, drop stop-words, stem.

    "How many sick leaves do I get?"  ->  ["many", "sick", "leav", "get"]
    """
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
    return [_stem(t) for t in tokens]


def _token_id(token: str) -> int:
    """Stable integer id for a word (same word -> same id on every machine, every run)."""
    return int(sha256(token)[:8], 16) % _VOCAB_SPACE


class BM25SparseEncoder:
    """BM25 parameters:
    k1  (1.2)  : how quickly repeating a word stops adding score (saturation)
    b   (0.75) : how much long chunks are penalised (so long chunks don't win just by size)
    avg_doc_len: typical chunk length in tokens, used for that length normalisation
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 250.0) -> None:
        self.k1, self.b, self.avg_doc_len = k1, b, avg_doc_len

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """Chunk -> (word ids, BM25 term-frequency weights). Called at ingestion time."""
        tokens = tokenize(text)
        if not tokens:
            return [], []
        tf = Counter(_token_id(t) for t in tokens)  # how often each word appears
        # Length normalisation: longer chunk -> bigger norm -> each occurrence counts a bit less
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_doc_len)
        indices = sorted(tf)  # Qdrant expects sorted indices
        # BM25 TF formula: grows with frequency but flattens out (10 mentions != 10x the score)
        values = [tf[i] * (self.k1 + 1) / (tf[i] + norm) for i in indices]
        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Question -> word ids with weight 1.0 each. Qdrant multiplies by IDF + document weights."""
        ids = sorted({_token_id(t) for t in tokenize(text)})
        return ids, [1.0] * len(ids)
