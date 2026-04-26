"""SQLite-backed knowledge graph storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import Correction, Entity, Fact, PendingFact, Relation, utc_now_iso


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
                """
            )

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
