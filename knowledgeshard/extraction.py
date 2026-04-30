"""Local model and structured-table extraction helpers."""

from __future__ import annotations

import hashlib
import json
import re
from uuid import NAMESPACE_URL, uuid5

from .model_runtime import OptionalModelRuntime
from .models import PendingFact, SourceCandidate, SourceDocument
from .retrieval import tokenize
from .sources import ResearchProfile, is_search_result_url, required_terms_present


GENERIC_RELATIONS = {
    "is_a",
    "has_property",
    "has_part",
    "located_in",
    "causes",
    "enables",
    "requires",
    "uses",
    "measured_by",
    "related_to",
}

BAD_FACT_VALUES = {"", "none", "unknown", "n/a", "na", "null", "not specified", "unspecified"}
BAD_FACT_TERMS = {"duckduckgo", "google search", "bing search", "search engine", "search page", "search result"}
BAD_RELATIONS = {"search_engine", "topic_of_search", "search_result", "has_search_result", "link"}
BAD_OBJECT_PATTERNS = (
    re.compile(r"^details?\s+in\s+.+\s+section$", re.IGNORECASE),
    re.compile(r"^details?\s+in\s+.+\s+subsection\s+under\s+.+\s+section$", re.IGNORECASE),
)
GENERIC_SECTION_OBJECTS = {"awards", "development", "gameplay", "reception", "sales", "vehicles"}


def evidence_hash(evidence_text: str) -> str:
    return hashlib.sha256(" ".join(evidence_text.split()).encode("utf-8")).hexdigest() if evidence_text else ""


def parse_confidence(value: object, default: float = 0.55) -> float:
    if isinstance(value, str):
        mapped = {
            "low": 0.35,
            "medium": 0.6,
            "moderate": 0.6,
            "high": 0.85,
            "very high": 0.95,
        }.get(value.strip().lower())
        if mapped is not None:
            return mapped
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def normalize_relation(relation: str, profile: ResearchProfile) -> str:
    normalized = "_".join(tokenize(relation))
    aliases = {
        "is": "is_a",
        "is_an": "is_a",
        "is_a": "is_a",
        "about": "related_to",
        "related": "related_to",
        "related_to": "related_to",
        **(profile.relation_aliases or {}),
    }
    return aliases.get(normalized, aliases.get(relation.lower().strip(), normalized))


def relation_allowed(relation: str, profile: ResearchProfile) -> bool:
    normalized = normalize_relation(relation, profile)
    allowed = set(profile.allowed_relations) or GENERIC_RELATIONS
    return normalized in allowed or normalized in GENERIC_RELATIONS


def meaningful_terms(value: str) -> set[str]:
    return {term for term in tokenize(value) if len(term) > 2}


def evidence_supports_fact(fact: PendingFact, evidence_text: str) -> bool:
    evidence_terms = meaningful_terms(evidence_text)
    subject_terms = meaningful_terms(fact.subject)
    object_terms = meaningful_terms(fact.object)
    if not evidence_terms:
        return False
    return (not subject_terms or bool(subject_terms & evidence_terms)) and (not object_terms or bool(object_terms & evidence_terms))


def fact_auto_score(fact: PendingFact, source: SourceCandidate | None, profile: ResearchProfile) -> float:
    score = 0.0
    if fact.evidence_text and evidence_supports_fact(fact, fact.evidence_text):
        score += 0.38
    if relation_allowed(fact.relation, profile):
        score += 0.2
    if source:
        score += min(source.trust_score, 1.0) * 0.22
        score += min(source.relevance_score, 1.0) * 0.1
        if not relation_allowed(fact.relation, profile) and source.trust_score >= 0.85 and fact.evidence_text:
            score += 0.14
    if fact.extraction_method == "table":
        score += 0.1
    elif fact.extraction_method == "model":
        score += 0.04
    return min(round(score, 3), 1.0)


def fact_quality_error(fact: PendingFact, document: SourceDocument, topic: str, profile: ResearchProfile | None = None) -> str | None:
    profile = profile or ResearchProfile(domain=document.domain)
    subject = fact.subject.strip().lower()
    relation = fact.relation.strip().lower()
    obj = fact.object.strip().lower()
    if subject in BAD_FACT_VALUES or relation in BAD_FACT_VALUES or obj in BAD_FACT_VALUES:
        return "placeholder field"
    if obj.startswith("[") or obj.startswith("{") or obj.endswith("]") or obj.endswith("}"):
        return "unsupported structured literal"
    if any(pattern.match(fact.object.strip()) for pattern in BAD_OBJECT_PATTERNS):
        return "section placeholder"
    if obj in GENERIC_SECTION_OBJECTS:
        return "generic section heading"
    if relation in BAD_RELATIONS:
        return "search relation"
    combined = f"{subject} {relation} {obj}"
    if any(term in combined for term in BAD_FACT_TERMS):
        return "search-page fact"
    if is_search_result_url(fact.source) or is_search_result_url(document.url):
        return "search-result source"
    required_context = f"{fact.subject} {fact.relation} {fact.object} {fact.evidence_text} {document.title} {document.url}"
    if not required_terms_present(required_context, profile):
        return "missing required domain terms"
    document_terms = meaningful_terms(f"{document.title} {document.text_excerpt}")
    subject_terms = meaningful_terms(fact.subject)
    object_terms = meaningful_terms(fact.object)
    topic_terms = meaningful_terms(topic)
    if subject_terms and not (subject_terms & document_terms):
        return "subject not grounded in document"
    if object_terms and not (object_terms & document_terms):
        return "object not grounded in document"
    if fact.evidence_text and not evidence_supports_fact(fact, fact.evidence_text):
        return "evidence does not support fact"
    fact_terms = meaningful_terms(f"{fact.subject} {fact.relation} {fact.object} {fact.evidence_text} {document.title}")
    if topic_terms and not (topic_terms & fact_terms):
        return "not relevant to topic"
    return None


class FactExtractor:
    def __init__(
        self,
        domain: str,
        topic: str | None = None,
        model_runtime: OptionalModelRuntime | None = None,
        profile: ResearchProfile | None = None,
    ) -> None:
        self.domain = domain
        self.topic = topic or domain.replace("-", " ")
        self.model_runtime = model_runtime or OptionalModelRuntime()
        self.profile = profile or ResearchProfile(domain=domain)

    def extract(self, document: SourceDocument, limit: int = 5) -> tuple[list[PendingFact], str | None]:
        if not self.model_runtime.available:
            return [], self.model_runtime.error or "model runtime unavailable"
        prompt = (
            "Extract concise structured facts for local storage.\n"
            f"Domain: {self.domain}\n"
            f"Topic: {self.topic}\n"
            "Return only a JSON array. Each item must have subject, relation, object, confidence, evidence_text, and tags.\n"
            "Use only the document text. Do not invent facts.\n\n"
            "Do not return section placeholders such as 'Details in Gameplay section'. "
            "The object must be a concrete value grounded in the document text.\n\n"
            f"Source URL: {document.url}\n"
            f"Document title: {document.title}\n"
            f"Document text:\n{document.text_excerpt[:6000]}\n"
        )
        generated = self.model_runtime.generate(prompt, max_new_tokens=600)
        if not generated:
            return [], self.model_runtime.error or "model returned no extraction"
        try:
            items = parse_json_array(generated)
        except ValueError as exc:
            return [], str(exc)

        pending: list[PendingFact] = []
        for item in items[:limit]:
            fact = self._validate_item(item, document)
            if fact:
                pending.append(fact)
        return pending, None

    def _validate_item(self, item: object, document: SourceDocument) -> PendingFact | None:
        if not isinstance(item, dict):
            return None
        subject = str(item.get("subject", "")).strip()
        relation = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        evidence_text = " ".join(str(item.get("evidence_text", "")).split())[:600]
        if not subject or not relation or not obj:
            return None
        relation = normalize_relation(relation, self.profile)
        combined = f"{subject} {relation} {obj}"
        relevance_terms = set(tokenize(f"{self.domain.replace('-', ' ')} {self.topic}"))
        if not (relevance_terms & set(tokenize(combined + " " + document.title))):
            return None
        confidence = parse_confidence(item.get("confidence", 0.55))
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        stable_id = uuid5(NAMESPACE_URL, f"{self.domain}:{document.url}:{subject}:{relation}:{obj}").hex
        return PendingFact(
            id=stable_id,
            subject=subject[:200],
            relation=relation[:120],
            object=obj[:500],
            confidence=confidence,
            source=document.url,
            domain=self.domain,
            evidence_text=evidence_text,
            evidence_hash=evidence_hash(evidence_text),
            extraction_method="model",
            tags=("extracted", *tuple(str(tag)[:40] for tag in tags)),
        )


def parse_json_array(text: str) -> list[object]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model response did not contain a JSON array")
    return json.loads(text[start : end + 1])


class StructuredExtractor:
    GENERIC_HEADERS = {
        "name": "name",
        "entity": "name",
        "subject": "name",
        "property": "property",
        "attribute": "property",
        "value": "value",
        "object": "value",
        "description": "value",
    }

    def __init__(self, domain: str, profile: ResearchProfile | None = None) -> None:
        self.domain = domain
        self.profile = profile or ResearchProfile(domain=domain)

    def extract(self, document: SourceDocument, tables: list[list[list[str]]], limit: int = 5) -> list[PendingFact]:
        facts: list[PendingFact] = []
        for table in tables:
            facts.extend(self.extract_table(document, table, limit - len(facts)))
            if len(facts) >= limit:
                break
        return facts[:limit]

    def extract_table(self, document: SourceDocument, table: list[list[str]], limit: int = 5) -> list[PendingFact]:
        if len(table) < 2:
            return []
        headers = [self.normalize_header(cell) for cell in table[0]]
        subject_index = self.subject_column(headers)
        if subject_index is None:
            return []
        facts: list[PendingFact] = []
        for row in table[1:]:
            if len(facts) >= limit:
                break
            if subject_index >= len(row):
                continue
            subject = row[subject_index].strip()
            if not subject:
                continue
            for index, header in enumerate(headers):
                if len(facts) >= limit:
                    break
                if index == subject_index or index >= len(row):
                    continue
                value = row[index].strip()
                if not header or not value:
                    continue
                relation = self.header_relation(header)
                evidence_text = " ".join(cell for cell in row if cell).strip()
                facts.append(
                    PendingFact(
                        id=uuid5(NAMESPACE_URL, f"{document.domain}:{document.url}:{subject}:{relation}:{value}").hex,
                        subject=subject[:200],
                        relation=relation,
                        object=value[:500],
                        confidence=0.86,
                        source=document.url,
                        domain=document.domain,
                        evidence_text=evidence_text[:600],
                        evidence_hash=evidence_hash(evidence_text),
                        extraction_method="table",
                        tags=("extracted", "table-extracted", header[:40]),
                    )
                )
        return facts

    def normalize_header(self, value: str) -> str:
        key = "_".join(tokenize(value))
        aliases = {**self.GENERIC_HEADERS, **(self.profile.table_header_aliases or {})}
        return aliases.get(key, key)

    def subject_column(self, headers: list[str]) -> int | None:
        for candidate in ("name", "entity", "subject"):
            if candidate in headers:
                return headers.index(candidate)
        return 0 if headers else None

    def header_relation(self, header: str) -> str:
        if header in {"property", "value"}:
            return "has_property"
        relation = f"has_{header}"
        return normalize_relation(relation, self.profile)
