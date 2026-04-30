"""Crawl, chunk, and process local research documents."""

from __future__ import annotations

import re
import json
from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid5

from .extraction import evidence_hash, parse_confidence
from .model_runtime import OptionalModelRuntime
from .models import PendingFact, ResearchChunk, ResearchNote, ResearchSynthesisRun, SourceCandidate
from .sources import (
    DocumentFetcher,
    DuckDuckGoSearchProvider,
    SearchResult,
    SourceScorer,
    generate_agenda,
    load_research_profile,
    normalize_url,
)
from .retrieval import tokenize
from .storage import KnowledgeStore


class ResearchAgent:
    def __init__(
        self,
        store: KnowledgeStore,
        domain: str,
        topic: str,
        config_path: str = "config/mario_kart_wii.sources.json",
        search_provider: object | None = None,
        fetcher: DocumentFetcher | None = None,
        model_runtime: OptionalModelRuntime | None = None,
    ) -> None:
        self.store = store
        self.domain = domain
        self.topic = topic
        self.profile = load_research_profile(config_path, domain)
        self.search_provider = search_provider or DuckDuckGoSearchProvider()
        self.fetcher = fetcher or DocumentFetcher()
        self.model_runtime = model_runtime or OptionalModelRuntime()
        self.scorer = SourceScorer(domain, topic, self.profile)

    def ingest(self, budget: int = 8) -> dict:
        errors: list[str] = []
        angles = self.angles(budget)
        sources = self.discover_sources(angles, budget, errors)
        fetched = 0
        stored = 0
        links_discovered = 0
        for source in sources[:budget]:
            self.store.upsert_source_candidate(source)
            try:
                document = self.fetcher.fetch(source)
            except OSError as exc:
                errors.append(f"{source.url}: {exc}")
                self.store.update_source_candidate(source.id, trust_delta=-0.02)
                continue
            if not document.obsession:
                document = document.__class__(
                    id=document.id,
                    source_id=document.source_id,
                    url=document.url,
                    title=document.title,
                    text_excerpt=document.text_excerpt,
                    full_text=document.full_text,
                    content_hash=document.content_hash,
                    domain=document.domain,
                    obsession=self.topic,
                    fetched_at=document.fetched_at,
                )
            fetched += 1
            if self.store.add_source_document(document):
                stored += 1
            self.store.update_source_candidate(source.id, fetched=True)
            for link_source in self.discover_link_sources(source, angles):
                if self.store.upsert_source_candidate(link_source):
                    links_discovered += 1
        return {
            "domain": self.domain,
            "topic": self.topic,
            "sources": len(sources),
            "fetched": fetched,
            "stored": stored,
            "links_discovered": links_discovered,
            "errors": errors,
        }

    def chunk(self, limit: int = 20, chunk_chars: int = 3200, overlap_chars: int = 300) -> dict:
        created = 0
        documents = self.store.list_source_documents(self.domain, limit)
        for document in documents:
            text = document.full_text or document.text_excerpt
            for index, chunk_text in enumerate(split_chunks(text, chunk_chars, overlap_chars)):
                terms = tokenize(chunk_text)
                priority = chunk_priority(chunk_text, self.topic)
                chunk = ResearchChunk(
                    id=uuid5(NAMESPACE_URL, f"{self.domain}:{self.topic}:{document.id}:{index}").hex,
                    document_id=document.id,
                    chunk_index=index,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    token_count=len(terms),
                    topic=self.topic,
                    domain=self.domain,
                    priority=priority,
                )
                if self.store.add_research_chunk(chunk):
                    created += 1
        return {"domain": self.domain, "topic": self.topic, "documents": len(documents), "chunks_created": created}

    def process(self, chunks: int = 10) -> dict:
        if not self.model_runtime.available:
            return {
                "domain": self.domain,
                "topic": self.topic,
                "processed": 0,
                "failed": 0,
                "notes_added": 0,
                "error": self.model_runtime.error or "model runtime unavailable",
            }
        pending = self.store.list_research_chunks(self.domain, self.topic, "pending", chunks)
        if len(pending) < chunks:
            pending.extend(self.store.list_research_chunks(self.domain, self.topic, "failed", chunks - len(pending)))
        processed = 0
        failed = 0
        notes_added = 0
        errors: list[str] = []
        for chunk in pending[:chunks]:
            prompt = research_note_prompt(self.domain, self.topic, chunk.text)
            generated = self.model_runtime.generate(prompt, max_new_tokens=700)
            if not generated:
                error = self.model_runtime.error or "model returned no note"
                self.store.update_research_chunk_status(chunk.id, "failed", error=error, increment_attempts=True)
                errors.append(f"{chunk.id}: {error}")
                failed += 1
                continue
            try:
                note = parse_research_note(generated, chunk)
            except ValueError as exc:
                error = str(exc)
                self.store.update_research_chunk_status(chunk.id, "failed", error=error, increment_attempts=True)
                errors.append(f"{chunk.id}: {error}")
                failed += 1
                continue
            if self.store.add_research_note(note):
                notes_added += 1
            self.store.update_research_chunk_status(chunk.id, "processed", processed=True)
            processed += 1
        return {
            "domain": self.domain,
            "topic": self.topic,
            "processed": processed,
            "failed": failed,
            "notes_added": notes_added,
            "errors": errors,
        }

    def notes(self, limit: int = 20) -> list[dict]:
        return [asdict(note) for note in self.store.list_research_notes(self.domain, self.topic, limit)]

    def crawl_status(self) -> dict:
        status = getattr(self.fetcher, "status", None)
        if callable(status):
            return dict(status())
        return {"backend": self.fetcher.__class__.__name__, "available": True, "error": ""}

    def synthesize(self, limit: int = 20) -> dict:
        notes = self.store.list_research_notes(self.domain, self.topic, limit)
        promoted: list[str] = []
        errors: list[str] = []
        seen_claims: set[str] = set()
        for note in notes:
            for claim in note.claims:
                normalized = " ".join(claim.lower().split())
                if not normalized or normalized in seen_claims:
                    continue
                seen_claims.add(normalized)
                evidence_text = first_supporting_quote(note.evidence_quotes, claim)
                pending = PendingFact(
                    id=uuid5(NAMESPACE_URL, f"{self.domain}:{self.topic}:{normalized}").hex,
                    subject=self.topic[:200],
                    relation="claims",
                    object=claim[:500],
                    confidence=note.confidence,
                    source=note.source or f"chunk:{note.chunk_id}",
                    domain=self.domain,
                    tags=("research", "synthesized", "pending-review"),
                    evidence_text=evidence_text[:600],
                    evidence_hash=evidence_hash(evidence_text),
                    extraction_method="research-note",
                )
                if self.store.add_pending_fact(pending):
                    promoted.append(pending.id)
        summary = " ".join(note.summary for note in notes[:5])[:1000] or "No notes available for synthesis."
        questions = tuple(unique_strings([question for note in notes for question in note.questions])[:10])
        run = ResearchSynthesisRun(
            topic=self.topic,
            domain=self.domain,
            summary=summary,
            promoted_pending_ids=tuple(promoted),
            unresolved_questions=questions,
            errors=tuple(errors),
        )
        self.store.add_research_synthesis_run(run)
        return {
            "domain": self.domain,
            "topic": self.topic,
            "notes_considered": len(notes),
            "pending_added": len(promoted),
            "pending_ids": promoted,
            "unresolved_questions": list(questions),
            "synthesis_run_id": run.id,
            "errors": errors,
        }

    def angles(self, limit: int = 8) -> list[str]:
        profile_angles = generate_agenda(self.domain, self.topic, self.profile, self.store, limit)
        defaults = [
            f"{self.topic} surprising facts",
            f"{self.topic} current debates",
            f"{self.topic} best sources",
            f"{self.topic} beginner vs expert knowledge",
            f"{self.topic} recent discoveries",
        ]
        return unique_strings([*profile_angles, *defaults])[:limit]

    def discover_sources(self, angles: list[str], budget: int, errors: list[str]) -> list[SourceCandidate]:
        candidates: list[SourceCandidate] = []
        for seed_url in self.profile.seed_urls:
            candidates.append(self.scorer.score(SearchResult(seed_url, seed_url, self.topic), self.topic, reason="research-seed"))
        for angle in angles[: max(1, budget // 2)]:
            try:
                results = self.search_provider.search(angle, limit=3)
            except OSError as exc:
                errors.append(f"{angle}: {exc}")
                results = []
            if not results and not self.profile.seed_urls:
                results = [SearchResult(angle, f"https://en.wikipedia.org/wiki/{'_'.join(angle.split())}")]
            for result in results:
                candidate = self.scorer.score(result, angle, reason="research-search")
                if candidate.status == "candidate" and candidate.relevance_score >= 0.2:
                    candidates.append(candidate)
        candidates.sort(key=lambda item: (item.trust_score + item.relevance_score), reverse=True)
        return dedupe_sources(candidates)

    def discover_link_sources(self, source: SourceCandidate, angles: list[str]) -> list[SourceCandidate]:
        links = getattr(self.fetcher, "last_links", [])
        candidates: list[SourceCandidate] = []
        for link in links:
            best = best_angle(f"{link.title} {link.url}", angles)
            candidate = self.scorer.score(link, best, parent_url=source.url, crawl_depth=source.crawl_depth + 1, reason="research-link")
            if candidate.status == "candidate" and candidate.relevance_score >= 0.25 and supports_angle(f"{link.title} {link.url}", best):
                candidates.append(candidate)
        candidates.sort(key=lambda item: (item.trust_score + item.relevance_score), reverse=True)
        return candidates[:5]

def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def split_chunks(text: str, chunk_chars: int = 3200, overlap_chars: int = 300) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunk_chars = max(chunk_chars, 500)
    overlap_chars = max(min(overlap_chars, chunk_chars // 3), 0)
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(". ", start + chunk_chars // 2, end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(end - overlap_chars, start + 1)
    return [chunk for chunk in chunks if chunk]


def chunk_priority(text: str, topic: str) -> float:
    terms = set(tokenize(text))
    topic_terms = set(tokenize(topic))
    if not topic_terms:
        return 0.5
    overlap = len(terms & topic_terms) / len(topic_terms)
    evidence_bonus = 0.15 if any(marker in text.lower() for marker in ("because", "therefore", "requires", "enables")) else 0.0
    return round(min(0.4 + overlap * 0.45 + evidence_bonus, 1.0), 3)


def research_note_prompt(domain: str, topic: str, text: str) -> str:
    return (
        "Read this local research chunk and extract notes for later synthesis.\n"
        f"Domain: {domain}\n"
        f"Topic: {topic}\n"
        "Return only one JSON object with these fields: summary, claims, entities, relations, questions, evidence_quotes, confidence.\n"
        "claims, entities, relations, questions, and evidence_quotes must be arrays of strings.\n"
        "Use only the chunk text. Do not invent details. Keep evidence quotes short and exact.\n\n"
        f"Chunk text:\n{text[:7000]}\n"
    )


def parse_research_note(text: str, chunk: ResearchChunk) -> ResearchNote:
    payload = parse_json_object(text)
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        raise ValueError("model note missing summary")
    claims = tuple(json_string_list(payload.get("claims", []), 500))
    evidence_quotes = tuple(json_string_list(payload.get("evidence_quotes", []), 600))
    if not claims and not evidence_quotes:
        raise ValueError("model note missing claims and evidence quotes")
    return ResearchNote(
        id=uuid5(NAMESPACE_URL, f"{chunk.domain}:{chunk.topic}:{chunk.id}:note").hex,
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        topic=chunk.topic,
        domain=chunk.domain,
        summary=summary[:1000],
        claims=claims,
        entities=tuple(json_string_list(payload.get("entities", []), 200)),
        relations=tuple(json_string_list(payload.get("relations", []), 300)),
        questions=tuple(json_string_list(payload.get("questions", []), 300)),
        evidence_quotes=evidence_quotes,
        confidence=parse_confidence(payload.get("confidence", 0.5)),
        source=f"chunk:{chunk.id}",
    )


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model response did not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response JSON was not an object")
    return payload


def json_string_list(value: object, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if text:
            output.append(text[:max_chars])
    return output


def first_supporting_quote(quotes: tuple[str, ...], claim: str) -> str:
    claim_terms = set(tokenize(claim))
    for quote in quotes:
        if claim_terms & set(tokenize(quote)):
            return quote
    return quotes[0] if quotes else ""


def best_angle(sentence: str, angles: list[str]) -> str:
    sentence_terms = set(tokenize(sentence))
    best = angles[0] if angles else ""
    best_score = -1
    for angle in angles:
        score = len(sentence_terms & set(tokenize(angle)))
        if score > best_score:
            best = angle
            best_score = score
    return best


def novelty_score(sentence: str, topic: str, title: str) -> float:
    terms = set(tokenize(sentence))
    topic_terms = set(tokenize(topic))
    if not topic_terms or not (terms & topic_terms):
        return 0.0
    interesting_terms = {
        "best",
        "competitive",
        "advanced",
        "technique",
        "strategy",
        "record",
        "world",
        "fast",
        "faster",
        "hidden",
        "rare",
        "used",
        "dominant",
        "because",
        "however",
        "although",
        "notable",
    }
    overlap = len(terms & topic_terms) / max(len(topic_terms), 1)
    novelty = min(len(terms & interesting_terms) * 0.09, 0.45)
    specificity = min(max(len(terms) - 8, 0) * 0.015, 0.25)
    title_bonus = 0.1 if terms & set(tokenize(title)) else 0.0
    return min(overlap * 0.35 + novelty + specificity + title_bonus, 1.0)


def supports_angle(sentence: str, angle: str) -> bool:
    sentence_terms = set(tokenize(sentence))
    angle_terms = set(tokenize(angle))
    if ({"vehicle", "vehicles"} & angle_terms) and ({"stat", "stats"} & angle_terms):
        stat_terms = sentence_terms & {"stat", "stats", "speed", "weight", "acceleration", "handling", "drift", "road", "mini", "turbo"}
        has_vehicle_term = bool(sentence_terms & {"vehicle", "vehicles", "kart", "karts", "bike", "bikes"})
        return (has_vehicle_term and bool(stat_terms)) or len(stat_terms) >= 2
    if "character" in angle_terms and "weight" in angle_terms:
        return bool(sentence_terms & {"character", "characters", "driver", "drivers"}) and bool(
            sentence_terms & {"weight", "lightweight", "medium", "heavyweight", "heavy", "class", "classes"}
        )
    if "drift" in angle_terms or "turbo" in angle_terms:
        return bool(sentence_terms & {"drift", "drifting", "turbo", "mini", "boost", "technique", "techniques"})
    if "tracks" in angle_terms or "shortcuts" in angle_terms:
        return bool(sentence_terms & {"track", "tracks", "course", "courses", "shortcut", "shortcuts", "route", "routes"})
    if "viability" in angle_terms:
        return bool(sentence_terms & {"competitive", "viable", "best", "popular", "used", "dominant", "trial", "online"})
    focus_terms = angle_terms - {"mario", "kart", "wii", "competitive", "facts", "sources", "best", "current"}
    if not focus_terms:
        return bool(sentence_terms & angle_terms)
    return bool(sentence_terms & focus_terms)


def is_research_noise(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(re.findall(r"\d+\.\d+", sentence)) >= 4:
        return True
    noisy_phrases = (
        "developer ",
        "publisher ",
        "release dates",
        "ratings ",
        "languages ",
        "cero",
        "pegi",
        "usk",
        "rars",
        "classind",
        "file history",
        "jump to navigation",
    )
    return any(phrase in lowered for phrase in noisy_phrases)


def summarize_sentence(sentence: str) -> str:
    return sentence[:280].rstrip()


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = " ".join(value.lower().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def dedupe_sources(sources: list[SourceCandidate]) -> list[SourceCandidate]:
    seen: set[str] = set()
    output: list[SourceCandidate] = []
    for source in sources:
        url = normalize_url(source.url)
        if url not in seen:
            seen.add(url)
            output.append(source)
    return output
