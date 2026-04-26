"""Curated source monitoring with a review queue."""

from __future__ import annotations

import json
import time
import tracemalloc
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from .models import PendingFact
from .storage import KnowledgeStore


@dataclass(frozen=True)
class SourceConfig:
    name: str
    url: str
    tags: tuple[str, ...]


def load_sources(path: str | Path) -> list[SourceConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SourceConfig(
            name=item["name"],
            url=item["url"],
            tags=tuple(item.get("tags", [])),
        )
        for item in payload.get("sources", [])
    ]


class ObsessionLoop:
    def __init__(
        self,
        store: KnowledgeStore,
        domain: str = "mario-kart-wii",
        config_path: str | Path = "config/mario_kart_wii.sources.json",
    ) -> None:
        self.store = store
        self.domain = domain
        self.config_path = Path(config_path)

    def fetch(self, limit_per_source: int = 10) -> dict:
        started = time.perf_counter()
        tracemalloc.start()
        fetched = 0
        added = 0
        errors: list[str] = []

        for source in load_sources(self.config_path):
            try:
                entries = self._fetch_source(source, limit_per_source)
            except OSError as exc:
                errors.append(f"{source.name}: {exc}")
                continue
            for entry in entries:
                fetched += 1
                pending = self._entry_to_pending(source, entry)
                if self.store.add_pending_fact(pending):
                    added += 1

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "domain": self.domain,
            "fetched": fetched,
            "pending_added": added,
            "pending_total": self.store.count_pending_facts(self.domain),
            "errors": errors,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "memory_peak_kb": round(peak / 1024, 1),
        }

    def review(self, limit: int = 20) -> list[dict]:
        return [pending.__dict__ for pending in self.store.list_pending_facts(self.domain)[:limit]]

    def approve(self, pending_id: str) -> dict:
        fact = self.store.approve_pending_fact(pending_id)
        return {"approved": True, "fact_id": fact.id, "text": fact.text}

    def reject(self, pending_id: str) -> dict:
        self.store.reject_pending_fact(pending_id)
        return {"rejected": True, "pending_id": pending_id}

    def run_once(self, limit_per_source: int = 10) -> dict:
        return self.fetch(limit_per_source)

    def _fetch_source(self, source: SourceConfig, limit: int) -> list[dict[str, str]]:
        with urllib.request.urlopen(source.url, timeout=20) as response:
            raw = response.read()
        root = ET.fromstring(raw)
        items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
        entries: list[dict[str, str]] = []
        for item in items[:limit]:
            title = self._text(item, "title")
            link = self._text(item, "link")
            summary = self._text(item, "description") or self._text(item, "summary")
            if title:
                entries.append({"title": title, "link": link or source.url, "summary": summary})
        return entries

    def _entry_to_pending(self, source: SourceConfig, entry: dict[str, str]) -> PendingFact:
        title = " ".join(entry["title"].split())
        summary = " ".join(entry.get("summary", "").split())
        object_text = summary[:240] if summary else f"new curated source item titled {title}"
        stable_id = uuid5(NAMESPACE_URL, f"{self.domain}:{source.url}:{entry.get('link')}:{title}").hex
        return PendingFact(
            id=stable_id,
            subject=title[:120],
            relation="reports",
            object=object_text,
            confidence=0.55,
            source=entry.get("link") or source.url,
            domain=self.domain,
            tags=("obsession", "pending", *source.tags),
        )

    def _text(self, item: ET.Element, tag: str) -> str:
        element = item.find(tag) or item.find(f"{{http://www.w3.org/2005/Atom}}{tag}")
        if element is None:
            return ""
        if tag == "link" and "href" in element.attrib:
            return element.attrib["href"]
        return element.text or ""
