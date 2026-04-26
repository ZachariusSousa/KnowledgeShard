"""Shared data models for the local savant MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    confidence: float = 0.75
    source: str = "local"
    domain: str = "general"
    tags: tuple[str, ...] = field(default_factory=tuple)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"

    @classmethod
    def from_row(cls, row: Any) -> "Fact":
        tags = tuple(filter(None, (row["tags"] or "").split(",")))
        return cls(
            id=row["id"],
            subject=row["subject"],
            relation=row["relation"],
            object=row["object"],
            confidence=float(row["confidence"]),
            source=row["source"],
            domain=row["domain"],
            tags=tags,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class PendingFact:
    subject: str
    relation: str
    object: str
    confidence: float = 0.6
    source: str = "obsession"
    domain: str = "general"
    tags: tuple[str, ...] = field(default_factory=tuple)
    review_status: str = "pending"
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"

    @classmethod
    def from_row(cls, row: Any) -> "PendingFact":
        tags = tuple(filter(None, (row["tags"] or "").split(",")))
        return cls(
            id=row["id"],
            subject=row["subject"],
            relation=row["relation"],
            object=row["object"],
            confidence=float(row["confidence"]),
            source=row["source"],
            domain=row["domain"],
            tags=tags,
            review_status=row["review_status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class Entity:
    name: str
    domain: str
    entity_type: str = "concept"
    confidence: float = 0.75
    source: str = "local"
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_row(cls, row: Any) -> "Entity":
        return cls(
            id=row["id"],
            name=row["name"],
            domain=row["domain"],
            entity_type=row["entity_type"],
            confidence=float(row["confidence"]),
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    fact_id: str
    domain: str
    confidence: float = 0.75
    source: str = "local"
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_row(cls, row: Any) -> "Relation":
        return cls(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            fact_id=row["fact_id"],
            domain=row["domain"],
            confidence=float(row["confidence"]),
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class SourceCandidate:
    url: str
    title: str
    domain: str
    discovery_query: str
    obsession: str = ""
    source_type: str = "web"
    trust_score: float = 0.5
    relevance_score: float = 0.0
    status: str = "candidate"
    id: str = field(default_factory=lambda: uuid4().hex)
    last_seen_at: str = field(default_factory=utc_now_iso)
    last_fetched_at: str = ""

    @classmethod
    def from_row(cls, row: Any) -> "SourceCandidate":
        return cls(
            id=row["id"],
            url=row["url"],
            title=row["title"],
            domain=row["domain"],
            discovery_query=row["discovery_query"],
            obsession=row["obsession"] if "obsession" in row.keys() else "",
            source_type=row["source_type"],
            trust_score=float(row["trust_score"]),
            relevance_score=float(row["relevance_score"]),
            status=row["status"],
            last_seen_at=row["last_seen_at"],
            last_fetched_at=row["last_fetched_at"] or "",
        )


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    url: str
    title: str
    text_excerpt: str
    content_hash: str
    domain: str
    obsession: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)
    fetched_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_row(cls, row: Any) -> "SourceDocument":
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            url=row["url"],
            title=row["title"],
            text_excerpt=row["text_excerpt"],
            content_hash=row["content_hash"],
            domain=row["domain"],
            obsession=row["obsession"] if "obsession" in row.keys() else "",
            fetched_at=row["fetched_at"],
        )


@dataclass(frozen=True)
class Citation:
    fact_id: str
    source: str
    confidence: float
    excerpt: str


@dataclass(frozen=True)
class QueryResponse:
    query_id: str
    question: str
    answer: str
    confidence: float
    citations: tuple[Citation, ...]
    savant_id: str
    domain: str


@dataclass(frozen=True)
class Correction:
    query_id: str
    savant_id: str
    correction: str
    confidence: float = 1.0
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
