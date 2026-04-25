"""Seed data loading for the MVP."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .models import Fact
from .storage import KnowledgeStore


def load_seed_facts(path: str | Path, store: KnowledgeStore, domain: str = "trains") -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    facts = [
        Fact(
            id=item.get("id") or uuid4().hex,
            subject=item["subject"],
            relation=item["relation"],
            object=item["object"],
            confidence=float(item.get("confidence", 0.8)),
            source=item.get("source", "seed"),
            domain=item.get("domain", domain),
            tags=tuple(item.get("tags", [])),
        )
        for item in payload["facts"]
    ]
    return store.add_facts(facts)
