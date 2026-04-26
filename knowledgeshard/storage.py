"""SQLite-backed knowledge graph storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import Correction, Entity, Fact, PendingFact, Relation, SourceCandidate, SourceDocument, utc_now_iso


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return False


class KnowledgeStore:
    def __init__(self, path: str | Path = "data/knowledgeshard.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, factory=ClosingConnection)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS query_log (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    citations TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS corrections (
                    id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    savant_id TEXT NOT NULL,
                    correction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(name, domain)
                );

                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    fact_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(fact_id)
                );

                CREATE TABLE IF NOT EXISTS pending_facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts(domain);
                CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain);
                CREATE INDEX IF NOT EXISTS idx_relations_domain ON relations(domain);
                CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate);
                CREATE INDEX IF NOT EXISTS idx_pending_domain_status ON pending_facts(domain, review_status);

                CREATE TABLE IF NOT EXISTS source_candidates (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    obsession TEXT NOT NULL DEFAULT '',
                    discovery_query TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    relevance_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_fetched_at TEXT NOT NULL DEFAULT '',
                    UNIQUE(url, domain)
                );

                CREATE TABLE IF NOT EXISTS source_documents (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text_excerpt TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    obsession TEXT NOT NULL DEFAULT '',
                    fetched_at TEXT NOT NULL,
                    UNIQUE(content_hash, domain)
                );

                CREATE TABLE IF NOT EXISTS obsession_runs (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    obsession TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    discovered_count INTEGER NOT NULL,
                    fetched_count INTEGER NOT NULL,
                    extracted_count INTEGER NOT NULL,
                    pending_count INTEGER NOT NULL,
                    errors TEXT NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    memory_peak_kb REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sources_domain_status ON source_candidates(domain, status);
                CREATE INDEX IF NOT EXISTS idx_documents_domain ON source_documents(domain);

                CREATE TABLE IF NOT EXISTS discovery_queries (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    obsession TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_count INTEGER NOT NULL,
                    errors TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_discovery_queries_domain ON discovery_queries(domain, created_at DESC);
                """
            )
            self._ensure_column(db, "source_candidates", "obsession", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "source_documents", "obsession", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "obsession_runs", "obsession", "TEXT NOT NULL DEFAULT ''")

    def _ensure_column(self, db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def upsert_fact(self, fact: Fact) -> None:
        with self.connect() as db:
            self._upsert_fact_row(db, fact)
            self._upsert_graph_fact_row(db, fact)

    def _upsert_graph_fact(self, fact: Fact) -> None:
        with self.connect() as db:
            self._upsert_graph_fact_row(db, fact)

    def _upsert_fact_row(self, db: sqlite3.Connection, fact: Fact) -> None:
        db.execute(
            """
            INSERT INTO facts (
                id, subject, relation, object, confidence, source, domain,
                tags, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                subject = excluded.subject,
                relation = excluded.relation,
                object = excluded.object,
                confidence = excluded.confidence,
                source = excluded.source,
                domain = excluded.domain,
                tags = excluded.tags,
                updated_at = excluded.updated_at
            """,
            (
                fact.id,
                fact.subject,
                fact.relation,
                fact.object,
                fact.confidence,
                fact.source,
                fact.domain,
                ",".join(fact.tags),
                fact.created_at,
                utc_now_iso(),
            ),
        )

    def _upsert_graph_fact_row(self, db: sqlite3.Connection, fact: Fact) -> None:
        subject = fact.subject.strip()
        obj = fact.object.strip()
        if not subject or not obj:
            return
        now = utc_now_iso()
        for name in (subject, obj):
            db.execute(
                """
                INSERT INTO entities (
                    id, name, domain, entity_type, confidence, source, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name, domain) DO UPDATE SET
                    confidence = MAX(entities.confidence, excluded.confidence),
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    f"{fact.domain}:{name}".lower(),
                    name,
                    fact.domain,
                    "concept",
                    fact.confidence,
                    fact.source,
                    fact.created_at,
                    now,
                ),
            )
        db.execute(
            """
            INSERT INTO relations (
                id, subject, predicate, object, fact_id, domain,
                confidence, source, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fact_id) DO UPDATE SET
                subject = excluded.subject,
                predicate = excluded.predicate,
                object = excluded.object,
                domain = excluded.domain,
                confidence = excluded.confidence,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                f"relation:{fact.id}",
                subject,
                fact.relation,
                obj,
                fact.id,
                fact.domain,
                fact.confidence,
                fact.source,
                fact.created_at,
                now,
            ),
        )

    def add_facts(self, facts: Iterable[Fact]) -> int:
        count = 0
        with self.connect() as db:
            for fact in facts:
                self._upsert_fact_row(db, fact)
                self._upsert_graph_fact_row(db, fact)
                count += 1
        return count

    def list_facts(self, domain: str | None = None) -> list[Fact]:
        with self.connect() as db:
            if domain:
                rows = db.execute(
                    "SELECT * FROM facts WHERE domain = ? ORDER BY updated_at DESC",
                    (domain,),
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM facts ORDER BY updated_at DESC").fetchall()
        return [Fact.from_row(row) for row in rows]

    def count_facts(self, domain: str | None = None) -> int:
        with self.connect() as db:
            if domain:
                row = db.execute("SELECT COUNT(*) AS total FROM facts WHERE domain = ?", (domain,)).fetchone()
            else:
                row = db.execute("SELECT COUNT(*) AS total FROM facts").fetchone()
        return int(row["total"])

    def add_pending_fact(self, pending: PendingFact) -> bool:
        if self._fact_exists(pending.subject, pending.relation, pending.object, pending.domain):
            return False
        with self.connect() as db:
            existing = db.execute(
                """
                SELECT id FROM pending_facts
                WHERE subject = ? AND relation = ? AND object = ? AND domain = ?
                """,
                (pending.subject, pending.relation, pending.object, pending.domain),
            ).fetchone()
            if existing:
                return False
            db.execute(
                """
                INSERT INTO pending_facts (
                    id, subject, relation, object, confidence, source, domain,
                    tags, review_status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pending.id,
                    pending.subject,
                    pending.relation,
                    pending.object,
                    pending.confidence,
                    pending.source,
                    pending.domain,
                    ",".join(pending.tags),
                    pending.review_status,
                    pending.created_at,
                    pending.updated_at,
                ),
            )
        return True

    def _fact_exists(self, subject: str, relation: str, object: str, domain: str) -> bool:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT id FROM facts
                WHERE subject = ? AND relation = ? AND object = ? AND domain = ?
                """,
                (subject, relation, object, domain),
            ).fetchone()
        return row is not None

    def list_pending_facts(self, domain: str | None = None, review_status: str = "pending") -> list[PendingFact]:
        with self.connect() as db:
            if domain:
                rows = db.execute(
                    """
                    SELECT * FROM pending_facts
                    WHERE domain = ? AND review_status = ?
                    ORDER BY created_at ASC
                    """,
                    (domain, review_status),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT * FROM pending_facts
                    WHERE review_status = ?
                    ORDER BY created_at ASC
                    """,
                    (review_status,),
                ).fetchall()
        return [PendingFact.from_row(row) for row in rows]

    def count_pending_facts(self, domain: str | None = None, review_status: str = "pending") -> int:
        with self.connect() as db:
            if domain:
                row = db.execute(
                    "SELECT COUNT(*) AS total FROM pending_facts WHERE domain = ? AND review_status = ?",
                    (domain, review_status),
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT COUNT(*) AS total FROM pending_facts WHERE review_status = ?",
                    (review_status,),
                ).fetchone()
        return int(row["total"])

    def approve_pending_fact(self, pending_id: str) -> Fact:
        pending = self.get_pending_fact(pending_id)
        if pending is None:
            raise KeyError(f"Pending fact {pending_id} was not found.")
        fact = Fact(
            id=pending.id,
            subject=pending.subject,
            relation=pending.relation,
            object=pending.object,
            confidence=pending.confidence,
            source=pending.source,
            domain=pending.domain,
            tags=pending.tags,
            created_at=pending.created_at,
        )
        self.upsert_fact(fact)
        self.update_pending_status(pending_id, "approved")
        return fact

    def reject_pending_fact(self, pending_id: str) -> None:
        if self.get_pending_fact(pending_id) is None:
            raise KeyError(f"Pending fact {pending_id} was not found.")
        self.update_pending_status(pending_id, "rejected")

    def get_pending_fact(self, pending_id: str) -> PendingFact | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM pending_facts WHERE id = ?", (pending_id,)).fetchone()
        return PendingFact.from_row(row) if row else None

    def update_pending_status(self, pending_id: str, review_status: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE pending_facts SET review_status = ?, updated_at = ? WHERE id = ?",
                (review_status, utc_now_iso(), pending_id),
            )

    def list_entities(self, domain: str | None = None) -> list[Entity]:
        with self.connect() as db:
            if domain:
                rows = db.execute("SELECT * FROM entities WHERE domain = ? ORDER BY name", (domain,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM entities ORDER BY domain, name").fetchall()
        return [Entity.from_row(row) for row in rows]

    def list_relations(self, domain: str | None = None) -> list[Relation]:
        with self.connect() as db:
            if domain:
                rows = db.execute("SELECT * FROM relations WHERE domain = ? ORDER BY updated_at DESC", (domain,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM relations ORDER BY updated_at DESC").fetchall()
        return [Relation.from_row(row) for row in rows]

    def count_entities(self, domain: str | None = None) -> int:
        with self.connect() as db:
            if domain:
                row = db.execute("SELECT COUNT(*) AS total FROM entities WHERE domain = ?", (domain,)).fetchone()
            else:
                row = db.execute("SELECT COUNT(*) AS total FROM entities").fetchone()
        return int(row["total"])

    def count_relations(self, domain: str | None = None) -> int:
        with self.connect() as db:
            if domain:
                row = db.execute("SELECT COUNT(*) AS total FROM relations WHERE domain = ?", (domain,)).fetchone()
            else:
                row = db.execute("SELECT COUNT(*) AS total FROM relations").fetchone()
        return int(row["total"])

    def upsert_source_candidate(self, source: SourceCandidate) -> bool:
        with self.connect() as db:
            existing = db.execute(
                "SELECT id FROM source_candidates WHERE url = ? AND domain = ?",
                (source.url, source.domain),
            ).fetchone()
            db.execute(
                """
                INSERT INTO source_candidates (
                    id, url, title, domain, obsession, discovery_query, source_type,
                    trust_score, relevance_score, status, last_seen_at, last_fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url, domain) DO UPDATE SET
                    title = excluded.title,
                    obsession = excluded.obsession,
                    discovery_query = excluded.discovery_query,
                    source_type = excluded.source_type,
                    relevance_score = MAX(source_candidates.relevance_score, excluded.relevance_score),
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    source.id,
                    source.url,
                    source.title,
                    source.domain,
                    source.obsession,
                    source.discovery_query,
                    source.source_type,
                    source.trust_score,
                    source.relevance_score,
                    source.status,
                    source.last_seen_at,
                    source.last_fetched_at,
                ),
            )
        return existing is None

    def list_source_candidates(
        self,
        domain: str,
        statuses: tuple[str, ...] = ("candidate", "approved"),
        limit: int = 20,
    ) -> list[SourceCandidate]:
        placeholders = ",".join("?" for _ in statuses)
        with self.connect() as db:
            rows = db.execute(
                f"""
                SELECT * FROM source_candidates
                WHERE domain = ? AND status IN ({placeholders})
                ORDER BY (trust_score + relevance_score) DESC, last_seen_at DESC
                LIMIT ?
                """,
                (domain, *statuses, limit),
            ).fetchall()
        return [SourceCandidate.from_row(row) for row in rows]

    def update_source_candidate(
        self,
        source_id: str,
        *,
        status: str | None = None,
        trust_delta: float = 0.0,
        fetched: bool = False,
    ) -> None:
        source = self.get_source_candidate(source_id)
        if source is None:
            raise KeyError(f"Source candidate {source_id} was not found.")
        trust_score = max(0.0, min(source.trust_score + trust_delta, 1.0))
        with self.connect() as db:
            db.execute(
                """
                UPDATE source_candidates
                SET status = ?, trust_score = ?, last_fetched_at = ?
                WHERE id = ?
                """,
                (
                    status or source.status,
                    trust_score,
                    utc_now_iso() if fetched else source.last_fetched_at,
                    source_id,
                ),
            )

    def get_source_candidate(self, source_id: str) -> SourceCandidate | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM source_candidates WHERE id = ?", (source_id,)).fetchone()
        return SourceCandidate.from_row(row) if row else None

    def add_source_document(self, document: SourceDocument) -> bool:
        with self.connect() as db:
            existing = db.execute(
                "SELECT id FROM source_documents WHERE content_hash = ? AND domain = ?",
                (document.content_hash, document.domain),
            ).fetchone()
            if existing:
                return False
            db.execute(
                """
                INSERT INTO source_documents (
                    id, source_id, url, title, text_excerpt, content_hash, domain, obsession, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.source_id,
                    document.url,
                    document.title,
                    document.text_excerpt,
                    document.content_hash,
                    document.domain,
                    document.obsession,
                    document.fetched_at,
                ),
            )
        return True

    def list_source_documents(self, domain: str, limit: int = 20) -> list[SourceDocument]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM source_documents
                WHERE domain = ?
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (domain, limit),
            ).fetchall()
        return [SourceDocument.from_row(row) for row in rows]

    def list_learned_facts(self, domain: str, limit: int = 20) -> list[Fact]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM facts
                WHERE domain = ? AND tags LIKE '%auto-approved%'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (domain, limit),
            ).fetchall()
        return [Fact.from_row(row) for row in rows]

    def list_obsession_runs(self, domain: str, limit: int = 10) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM obsession_runs
                WHERE domain = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (domain, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "domain": row["domain"],
                "obsession": row["obsession"] if "obsession" in row.keys() else "",
                "status": row["status"],
                "discovered_count": int(row["discovered_count"]),
                "fetched_count": int(row["fetched_count"]),
                "extracted_count": int(row["extracted_count"]),
                "pending_count": int(row["pending_count"]),
                "errors": json.loads(row["errors"] or "[]"),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "memory_peak_kb": float(row["memory_peak_kb"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def log_discovery_query(
        self,
        domain: str,
        obsession: str,
        query: str,
        status: str,
        result_count: int,
        errors: list[str],
    ) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO discovery_queries (
                    id, domain, obsession, query, status, result_count, errors, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"discovery:{domain}:{abs(hash((obsession, query, utc_now_iso())))}",
                    domain,
                    obsession,
                    query,
                    status,
                    result_count,
                    json.dumps(errors),
                    utc_now_iso(),
                ),
            )

    def list_recent_query_questions(self, limit: int = 20) -> list[str]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT question FROM query_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [str(row["question"]) for row in rows]

    def log_obsession_run(
        self,
        domain: str,
        obsession: str,
        status: str,
        discovered_count: int,
        fetched_count: int,
        extracted_count: int,
        pending_count: int,
        errors: list[str],
        elapsed_seconds: float,
        memory_peak_kb: float,
    ) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO obsession_runs (
                    id, domain, status, discovered_count, fetched_count,
                    obsession,
                    extracted_count, pending_count, errors, elapsed_seconds,
                    memory_peak_kb, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"obsession:{domain}:{utc_now_iso()}",
                    domain,
                    status,
                    discovered_count,
                    fetched_count,
                    obsession,
                    extracted_count,
                    pending_count,
                    json.dumps(errors),
                    elapsed_seconds,
                    memory_peak_kb,
                    utc_now_iso(),
                ),
            )

    def log_query(self, query_id: str, question: str, answer: str, confidence: float, citations: list[dict]) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO query_log (id, question, answer, confidence, citations, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (query_id, question, answer, confidence, json.dumps(citations), utc_now_iso()),
            )

    def add_correction(self, correction: Correction) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO corrections (id, query_id, savant_id, correction, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    correction.id,
                    correction.query_id,
                    correction.savant_id,
                    correction.correction,
                    correction.confidence,
                    correction.created_at,
                ),
            )

    def count_queries(self) -> int:
        with self.connect() as db:
            row = db.execute("SELECT COUNT(*) AS total FROM query_log").fetchone()
        return int(row["total"])

    def count_corrections(self) -> int:
        with self.connect() as db:
            row = db.execute("SELECT COUNT(*) AS total FROM corrections").fetchone()
        return int(row["total"])
