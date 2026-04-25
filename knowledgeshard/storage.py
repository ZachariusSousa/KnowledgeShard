"""SQLite-backed knowledge graph storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import Correction, Fact, utc_now_iso


class KnowledgeStore:
    def __init__(self, path: str | Path = "data/knowledgeshard.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
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
                """
            )

    def upsert_fact(self, fact: Fact) -> None:
        with self.connect() as db:
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

    def add_facts(self, facts: Iterable[Fact]) -> int:
        count = 0
        for fact in facts:
            self.upsert_fact(fact)
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
