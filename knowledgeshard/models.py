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
    evidence_text: str = ""
    evidence_hash: str = ""
    extraction_method: str = ""
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
            evidence_text=row["evidence_text"] if "evidence_text" in row.keys() else "",
            evidence_hash=row["evidence_hash"] if "evidence_hash" in row.keys() else "",
            extraction_method=row["extraction_method"] if "extraction_method" in row.keys() else "",
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
    evidence_text: str = ""
    evidence_hash: str = ""
    extraction_method: str = ""
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
            evidence_text=row["evidence_text"] if "evidence_text" in row.keys() else "",
            evidence_hash=row["evidence_hash"] if "evidence_hash" in row.keys() else "",
            extraction_method=row["extraction_method"] if "extraction_method" in row.keys() else "",
            review_status=row["review_status"],
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
    parent_url: str = ""
    crawl_depth: int = 0
    discovery_reason: str = ""
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
            parent_url=row["parent_url"] if "parent_url" in row.keys() else "",
            crawl_depth=int(row["crawl_depth"]) if "crawl_depth" in row.keys() else 0,
            discovery_reason=row["discovery_reason"] if "discovery_reason" in row.keys() else "",
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
    full_text: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)
    fetched_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_row(cls, row: Any) -> "SourceDocument":
        full_text = row["full_text"] if "full_text" in row.keys() else ""
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            url=row["url"],
            title=row["title"],
            text_excerpt=row["text_excerpt"],
            content_hash=row["content_hash"],
            domain=row["domain"],
            obsession=row["obsession"] if "obsession" in row.keys() else "",
            full_text=full_text or row["text_excerpt"],
            fetched_at=row["fetched_at"],
        )


@dataclass(frozen=True)
class ResearchChunk:
    document_id: str
    chunk_index: int
    text: str
    char_count: int
    token_count: int
    topic: str
    domain: str
    status: str = "pending"
    priority: float = 0.5
    attempts: int = 0
    error: str = ""
    processed_at: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_row(cls, row: Any) -> "ResearchChunk":
        return cls(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=int(row["chunk_index"]),
            text=row["text"],
            char_count=int(row["char_count"]),
            token_count=int(row["token_count"]),
            topic=row["topic"],
            domain=row["domain"],
            status=row["status"],
            priority=float(row["priority"]),
            attempts=int(row["attempts"]),
            error=row["error"],
            processed_at=row["processed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class ResearchNote:
    chunk_id: str
    document_id: str
    topic: str
    domain: str
    summary: str
    claims: tuple[str, ...] = field(default_factory=tuple)
    entities: tuple[str, ...] = field(default_factory=tuple)
    relations: tuple[str, ...] = field(default_factory=tuple)
    questions: tuple[str, ...] = field(default_factory=tuple)
    evidence_quotes: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    source: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_row(cls, row: Any) -> "ResearchNote":
        return cls(
            id=row["id"],
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            topic=row["topic"],
            domain=row["domain"],
            summary=row["summary"],
            claims=tuple(json_loads(row["claims"])),
            entities=tuple(json_loads(row["entities"])),
            relations=tuple(json_loads(row["relations"])),
            questions=tuple(json_loads(row["questions"])),
            evidence_quotes=tuple(json_loads(row["evidence_quotes"])),
            confidence=float(row["confidence"]),
            source=row["source"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class ResearchSynthesisRun:
    topic: str
    domain: str
    summary: str
    promoted_pending_ids: tuple[str, ...] = field(default_factory=tuple)
    unresolved_questions: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_row(cls, row: Any) -> "ResearchSynthesisRun":
        return cls(
            id=row["id"],
            topic=row["topic"],
            domain=row["domain"],
            summary=row["summary"],
            promoted_pending_ids=tuple(json_loads(row["promoted_pending_ids"])),
            unresolved_questions=tuple(json_loads(row["unresolved_questions"])),
            errors=tuple(json_loads(row["errors"])),
            created_at=row["created_at"],
        )


def json_loads(value: str) -> list[str]:
    import json

    payload = json.loads(value or "[]")
    return [str(item) for item in payload]


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
