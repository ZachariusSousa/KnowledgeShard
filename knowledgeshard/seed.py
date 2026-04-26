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
    facts.extend(_expand_templates(payload.get("fact_templates", []), domain))
    return store.add_facts(facts)


def _expand_templates(templates: list[dict], default_domain: str) -> list[Fact]:
    facts: list[Fact] = []
    for template in templates:
        subjects = template.get("subjects", [])
        objects = template.get("objects", [])
        relation = template["relation"]
        source = template.get("source", "seed")
        domain = template.get("domain", default_domain)
        tags = tuple(template.get("tags", []))
        confidence = float(template.get("confidence", 0.75))
        for subject in subjects:
            for obj in objects:
                subject_text = subject["name"] if isinstance(subject, dict) else str(subject)
                object_text = obj["text"] if isinstance(obj, dict) else str(obj)
                object_tags = tuple(obj.get("tags", [])) if isinstance(obj, dict) else ()
                subject_tags = tuple(subject.get("tags", [])) if isinstance(subject, dict) else ()
                facts.append(
                    Fact(
                        id=f"{domain}:{_slug(subject_text)}:{_slug(relation)}:{_slug(object_text)}",
                        subject=subject_text,
                        relation=relation,
                        object=object_text,
                        confidence=confidence,
                        source=source,
                        domain=domain,
                        tags=tags + subject_tags + object_tags,
                    )
                )
    return facts


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")[:80]
