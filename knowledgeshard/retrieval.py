"""Small lexical retrieval helpers.

The full plan calls for sentence-transformer embeddings. This MVP keeps the
same interface shape but uses transparent TF-IDF style scoring from stdlib so
the first local savant works immediately.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-']*", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        normalize(token.lower().strip("-'"))
        for token in WORD_RE.findall(text)
        if token.lower().strip("-'") not in STOPWORDS
    ]


def normalize(token: str) -> str:
    if len(token) > 8 and token.endswith("ment"):
        token = token[:-4]
    if len(token) > 4 and token.endswith("ies"):
        token = f"{token[:-3]}y"
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def score(query: str, document: str, corpus: Iterable[str]) -> float:
    query_terms = Counter(tokenize(query))
    document_terms = Counter(tokenize(document))
    if not query_terms or not document_terms:
        return 0.0

    corpus_tokens = [set(tokenize(item)) for item in corpus]
    corpus_size = max(len(corpus_tokens), 1)
    weighted_overlap = 0.0
    norm = 0.0

    for term, query_count in query_terms.items():
        docs_with_term = sum(1 for tokens in corpus_tokens if term in tokens)
        idf = math.log((1 + corpus_size) / (1 + docs_with_term)) + 1
        norm += query_count * idf
        weighted_overlap += min(query_count, document_terms.get(term, 0)) * idf

    coverage = weighted_overlap / norm if norm else 0.0
    density = weighted_overlap / max(sum(document_terms.values()), 1)
    return round((0.8 * coverage) + (0.2 * min(density * 4, 1.0)), 4)
