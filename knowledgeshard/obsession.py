"""Autonomous obsession loop with review-first learning."""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import tracemalloc
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .ingest import Crawl4AIDocumentFetcher
from .model_runtime import OptionalModelRuntime
from .models import Fact, PendingFact, SourceCandidate, SourceDocument
from .retrieval import tokenize
from .storage import KnowledgeStore


@dataclass(frozen=True)
class SourceConfig:
    name: str
    url: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class ResearchProfile:
    domain: str
    seed_urls: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    denied_url_patterns: tuple[str, ...] = ()
    source_trust: dict[str, float] | None = None
    required_terms: tuple[str, ...] = ()
    agenda_templates: tuple[str, ...] = ()
    relation_aliases: dict[str, str] | None = None
    allowed_relations: tuple[str, ...] = ()
    table_header_aliases: dict[str, str] | None = None
    default_crawl_depth: int = 1

    @property
    def has_domain_rules(self) -> bool:
        return bool(self.seed_urls or self.allowed_hosts or self.allowed_relations)


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


def load_research_profile(path: str | Path, domain: str) -> ResearchProfile:
    config_path = Path(path)
    if not config_path.exists():
        return ResearchProfile(domain=domain)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    profile = payload.get("research_profile", payload)
    if profile.get("domain") and profile.get("domain") != domain:
        return ResearchProfile(domain=domain)
    return ResearchProfile(
        domain=domain,
        seed_urls=tuple(profile.get("seed_urls", [])),
        allowed_hosts=tuple(profile.get("allowed_hosts", [])),
        denied_url_patterns=tuple(profile.get("denied_url_patterns", [])),
        source_trust=dict(profile.get("source_trust", {})),
        required_terms=tuple(profile.get("required_terms", [])),
        agenda_templates=tuple(profile.get("agenda_templates", [])),
        relation_aliases=dict(profile.get("relation_aliases", {})),
        allowed_relations=tuple(profile.get("allowed_relations", [])),
        table_header_aliases=dict(profile.get("table_header_aliases", {})),
        default_crawl_depth=int(profile.get("default_crawl_depth", 1)),
    )


def load_sources(path: str | Path) -> list[SourceConfig]:
    config_path = Path(path)
    if not config_path.exists():
        return []
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return [
        SourceConfig(
            name=item["name"],
            url=item["url"],
            tags=tuple(item.get("tags", [])),
        )
        for item in payload.get("sources", [])
    ]


class DuckDuckGoSearchProvider:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        encoded = urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(
            f"https://duckduckgo.com/html/?{encoded}",
            headers={"user-agent": "KnowledgeShard/0.1 (+local research bot)"},
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            page = response.read().decode("utf-8", errors="replace")
        return parse_duckduckgo_results(page, limit)


def parse_duckduckgo_results(page: str, limit: int = 5) -> list[SearchResult]:
    results: list[SearchResult] = []
    patterns = [
        re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r'<a[^>]+href="(?P<href>[^"]+)"[^>]+class="[^"]*result-link[^"]*"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r'<h2[^>]*>.*?<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?</h2>',
            re.IGNORECASE | re.DOTALL,
        ),
    ]
    seen: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(page):
            raw_url = html.unescape(match.group("href"))
            parsed = urllib.parse.urlparse(raw_url)
            query = urllib.parse.parse_qs(parsed.query)
            url = query.get("uddg", [raw_url])[0]
            title = strip_html(match.group("title"))
            if url and title and url not in seen:
                seen.add(url)
                results.append(SearchResult(title=title, url=url))
            if len(results) >= limit:
                return results
    return results


def strip_html(value: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(value)).split())


def generate_agenda(domain: str, obsession: str, profile: ResearchProfile, store: KnowledgeStore | None = None, limit: int = 8) -> list[str]:
    domain_label = domain.replace("-", " ")
    templates = profile.agenda_templates or (
        "{obsession} core facts",
        "{obsession} important entities",
        "{obsession} terminology",
        "{obsession} evidence sources",
    )
    agenda = [
        template.format(domain=domain_label, obsession=obsession)
        for template in templates
    ]
    if store:
        agenda.extend(f"{domain_label} {question}" for question in store.list_recent_query_questions(5))
    seen: set[str] = set()
    unique: list[str] = []
    for item in agenda:
        normalized = " ".join(item.lower().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(item)
    return unique[:limit]


class SourceDiscovery:
    def __init__(self, store: KnowledgeStore, domain: str, obsession: str | None = None, profile: ResearchProfile | None = None) -> None:
        self.store = store
        self.domain = domain
        self.obsession = obsession or domain.replace("-", " ")
        self.profile = profile or ResearchProfile(domain=domain)

    def generate_queries(self, limit: int = 10) -> list[str]:
        domain_label = self.domain.replace("-", " ")
        agenda = generate_agenda(self.domain, self.obsession, self.profile, self.store)
        queries: list[str] = [
            self.obsession,
            f"{self.obsession} guide",
            f"{self.obsession} wiki",
            f"{domain_label} guide",
            f"{domain_label} strategy mechanics",
            f"{domain_label} wiki facts",
        ]
        queries.extend(agenda)
        entities = self.store.list_entities(self.domain)[: max(limit, 10)]
        for entity in entities:
            name = entity.name.strip()
            if name and len(name) <= 80:
                queries.append(f"{domain_label} {name} guide")
        for question in self.store.list_recent_query_questions(10):
            queries.append(f"{domain_label} {question}")

        seen: set[str] = set()
        unique: list[str] = []
        for query in queries:
            normalized = " ".join(query.lower().split())
            if normalized not in seen:
                seen.add(normalized)
                unique.append(query)
        return unique[:limit]

    def fallback_results(self, query: str, limit: int = 5) -> list[SearchResult]:
        slug = "_".join(word for word in self.obsession.split() if word)
        candidates = [SearchResult(f"{self.obsession} on Wikipedia", f"https://en.wikipedia.org/wiki/{slug}")]
        candidates.extend(SearchResult(self.obsession, url) for url in self.profile.seed_urls)
        return candidates[:limit]


class SourceScorer:
    TRUST_HINTS = {
        "nintendo": 0.9,
        "mariowiki": 0.85,
        "wiki": 0.75,
        "reddit": 0.55,
        "forum": 0.5,
        "youtube": 0.45,
    }

    def __init__(self, domain: str, obsession: str | None = None, profile: ResearchProfile | None = None) -> None:
        self.domain = domain
        self.obsession = obsession or domain.replace("-", " ")
        self.profile = profile or ResearchProfile(domain=domain)
        self.domain_terms = set(tokenize(f"{domain.replace('-', ' ')} {self.obsession}"))

    def score(self, result: SearchResult, query: str, parent_url: str = "", crawl_depth: int = 0, reason: str = "search") -> SourceCandidate:
        combined = f"{result.title} {result.url} {result.snippet}"
        terms = set(tokenize(combined))
        query_terms = set(tokenize(query))
        relevance = len((self.domain_terms | query_terms) & terms) / max(len(self.domain_terms | query_terms), 1)
        source_type = self.source_type(result.url)
        trust = self.profile_trust(result.url, source_type)
        status = "rejected" if is_search_result_url(result.url) or denied_by_profile(result.url, self.profile) else "candidate"
        if self.profile.allowed_hosts and not host_allowed(result.url, self.profile):
            status = "rejected"
        if status == "rejected":
            trust = 0.0
            relevance = 0.0
        elif self.profile.allowed_hosts and host_allowed(result.url, self.profile):
            relevance = min(relevance + 0.15, 1.0)
        if status != "rejected" and self.domain.replace("-", "") in result.url.lower().replace("-", "").replace("_", ""):
            relevance = min(relevance + 0.2, 1.0)
        return SourceCandidate(
            id=uuid5(NAMESPACE_URL, f"{self.domain}:{normalize_url(result.url)}").hex,
            url=normalize_url(result.url),
            title=result.title[:200],
            domain=self.domain,
            obsession=self.obsession,
            discovery_query=query,
            source_type=source_type,
            trust_score=trust,
            relevance_score=round(relevance, 3),
            status=status,
            parent_url=parent_url,
            crawl_depth=crawl_depth,
            discovery_reason=reason,
        )

    def source_type(self, url: str) -> str:
        lowered = url.lower()
        for hint in self.TRUST_HINTS:
            if hint in lowered:
                return hint
        return "web"

    def profile_trust(self, url: str, source_type: str) -> float:
        trust_map = self.profile.source_trust or {}
        host = urllib.parse.urlparse(url).netloc.lower()
        for hint, score in trust_map.items():
            if hint.lower() in host or hint.lower() in url.lower():
                return max(0.0, min(float(score), 1.0))
        return self.TRUST_HINTS.get(source_type, 0.45)


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((scheme, netloc, path, "", parsed.query, ""))


def host_allowed(url: str, profile: ResearchProfile) -> bool:
    if not profile.allowed_hosts:
        return True
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(host == allowed.lower() or host.endswith(f".{allowed.lower()}") for allowed in profile.allowed_hosts)


def denied_by_profile(url: str, profile: ResearchProfile) -> bool:
    lowered = url.lower()
    return any(re.search(pattern, lowered) for pattern in profile.denied_url_patterns)


def required_terms_present(text: str, profile: ResearchProfile) -> bool:
    required = {term.lower() for term in profile.required_terms if term}
    if not required:
        return True
    terms = set(tokenize(text))
    return required <= terms


def is_search_result_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    if "duckduckgo.com" in host and ("/html" in path or query.startswith("q=")):
        return True
    if "google." in host and path.startswith("/search"):
        return True
    if "bing.com" in host and path.startswith("/search"):
        return True
    if "reddit.com" in host and path.startswith("/search"):
        return True
    return False


BAD_FACT_VALUES = {"", "none", "unknown", "n/a", "na", "null", "not specified", "unspecified"}
BAD_FACT_TERMS = {"duckduckgo", "google search", "bing search", "search engine", "search page", "search result"}
BAD_RELATIONS = {"search_engine", "topic_of_search", "search_result", "has_search_result", "link"}
BAD_OBJECT_PATTERNS = (
    re.compile(r"^details?\s+in\s+.+\s+section$", re.IGNORECASE),
    re.compile(r"^details?\s+in\s+.+\s+subsection\s+under\s+.+\s+section$", re.IGNORECASE),
)
GENERIC_SECTION_OBJECTS = {
    "awards",
    "development",
    "gameplay",
    "reception",
    "sales",
    "vehicles",
}


def meaningful_terms(value: str) -> set[str]:
    return {term for term in tokenize(value) if len(term) > 2}


def evidence_hash(evidence_text: str) -> str:
    return hashlib.sha256(" ".join(evidence_text.split()).encode("utf-8")).hexdigest() if evidence_text else ""


def evidence_supports_fact(fact: PendingFact, evidence_text: str) -> bool:
    evidence_terms = meaningful_terms(evidence_text)
    subject_terms = meaningful_terms(fact.subject)
    object_terms = meaningful_terms(fact.object)
    if not evidence_terms:
        return False
    return (not subject_terms or bool(subject_terms & evidence_terms)) and (not object_terms or bool(object_terms & evidence_terms))


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


def relation_allowed(relation: str, profile: ResearchProfile) -> bool:
    normalized = normalize_relation(relation, profile)
    allowed = set(profile.allowed_relations) or GENERIC_RELATIONS
    return normalized in allowed or normalized in GENERIC_RELATIONS


def fact_auto_score(fact: PendingFact, document: SourceDocument, source: SourceCandidate | None, profile: ResearchProfile) -> float:
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


def fact_quality_error(fact: PendingFact, document: SourceDocument, obsession: str, profile: ResearchProfile | None = None) -> str | None:
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
    obsession_terms = meaningful_terms(obsession)
    if subject_terms and not (subject_terms & document_terms):
        return "subject not grounded in document"
    if object_terms and not (object_terms & document_terms):
        return "object not grounded in document"
    if fact.evidence_text and not evidence_supports_fact(fact, fact.evidence_text):
        return "evidence does not support fact"
    fact_terms = meaningful_terms(f"{fact.subject} {fact.relation} {fact.object} {fact.evidence_text} {document.title}")
    if obsession_terms and not (obsession_terms & fact_terms):
        return "not relevant to obsession"
    return None


class ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)

    def text(self) -> str:
        return " ".join(self.parts)


class ResearchHTMLParser(HTMLParser):
    def __init__(self, base_url: str = "") -> None:
        super().__init__()
        self.base_url = base_url
        self._skip = False
        self.parts: list[str] = []
        self.links: list[SearchResult] = []
        self._current_link: str = ""
        self._current_link_text: list[str] = []
        self._in_table = False
        self._in_cell = False
        self._cell_text: list[str] = []
        self._current_row: list[str] = []
        self._current_table: list[list[str]] = []
        self.tables: list[list[list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag in {"script", "style", "noscript"}:
            self._skip = True
            return
        if tag == "a" and attrs_dict.get("href"):
            self._current_link = urllib.parse.urljoin(self.base_url, html.unescape(attrs_dict["href"]))
            self._current_link_text = []
        if tag == "table":
            self._in_table = True
            self._current_table = []
        if self._in_table and tag == "tr":
            self._current_row = []
        if self._in_table and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False
            return
        if tag == "a" and self._current_link:
            title = " ".join(" ".join(self._current_link_text).split()) or self._current_link
            self.links.append(SearchResult(title=title[:200], url=self._current_link))
            self._current_link = ""
            self._current_link_text = []
        if self._in_table and tag in {"td", "th"}:
            cell = " ".join(" ".join(self._cell_text).split())
            self._current_row.append(cell)
            self._in_cell = False
            self._cell_text = []
        if self._in_table and tag == "tr":
            if any(self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = []
        if tag == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False
            self._current_table = []

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = " ".join(data.split())
        if not text:
            return
        self.parts.append(text)
        if self._current_link:
            self._current_link_text.append(text)
        if self._in_cell:
            self._cell_text.append(text)

    def text(self) -> str:
        return " ".join(self.parts)


class DocumentFetcher:
    def __init__(self) -> None:
        self._crawl4ai = Crawl4AIDocumentFetcher()
        self.last_links: list[SearchResult] = []
        self.last_tables: list[list[list[str]]] = []

    def fetch(self, source: SourceCandidate, max_chars: int = 12000) -> SourceDocument:
        document = self._crawl4ai.fetch(source, max_chars)
        self.last_links = [SearchResult(link.title, link.url) for link in self._crawl4ai.last_links]
        self.last_tables = []
        return document

    def readable_text(self, page: str) -> str:
        raise OSError("DocumentFetcher now uses Crawl4AI for live pages; readable_text is no longer supported")

    def parse_html(self, page: str, base_url: str = "") -> ResearchHTMLParser:
        raise OSError("DocumentFetcher now uses Crawl4AI for live pages; parse_html is no longer supported")

    def status(self) -> dict[str, object]:
        return self._crawl4ai.status()


class FactExtractor:
    def __init__(
        self,
        store: KnowledgeStore,
        domain: str,
        obsession: str | OptionalModelRuntime | None = None,
        model_runtime: OptionalModelRuntime | None = None,
        profile: ResearchProfile | None = None,
    ) -> None:
        self.store = store
        self.domain = domain
        if isinstance(obsession, OptionalModelRuntime):
            model_runtime = obsession
            obsession = None
        self.obsession = str(obsession) if obsession else domain.replace("-", " ")
        self.model_runtime = model_runtime or OptionalModelRuntime()
        self.profile = profile or ResearchProfile(domain=domain)

    def extract(self, document: SourceDocument, limit: int = 5) -> tuple[list[PendingFact], str | None]:
        if not self.model_runtime.available:
            return [], self.model_runtime.error or "model runtime unavailable"
        prompt = (
            "Extract concise structured facts for a local knowledge graph.\n"
            f"Domain: {self.domain}\n"
            f"Obsession: {self.obsession}\n"
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
        relevance_terms = set(tokenize(f"{self.domain.replace('-', ' ')} {self.obsession}"))
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
            tags=("obsession", "extracted", *tuple(str(tag)[:40] for tag in tags)),
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
                        tags=("obsession", "extracted", "table-extracted", header[:40]),
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


class ObsessionLoop:
    def __init__(
        self,
        store: KnowledgeStore,
        domain: str = "mario-kart-wii",
        obsession: str | None = None,
        config_path: str | Path = "config/mario_kart_wii.sources.json",
        search_provider: object | None = None,
        fetcher: DocumentFetcher | None = None,
        extractor: FactExtractor | None = None,
    ) -> None:
        self.store = store
        self.domain = domain
        self.obsession = obsession or domain.replace("-", " ")
        self.config_path = Path(config_path)
        self.profile = load_research_profile(config_path, domain)
        self.discovery = SourceDiscovery(store, domain, self.obsession, self.profile)
        self.scorer = SourceScorer(domain, self.obsession, self.profile)
        self.search_provider = search_provider or DuckDuckGoSearchProvider()
        self.fetcher = fetcher or DocumentFetcher()
        self.extractor = extractor or FactExtractor(store, domain, self.obsession, profile=self.profile)
        self.structured_extractor = StructuredExtractor(domain, self.profile)
        self.document_tables: dict[str, list[list[list[str]]]] = {}

    def discover(self, query_limit: int = 5, results_per_query: int = 5, use_fallback: bool = True) -> dict:
        discovered = 0
        errors: list[str] = []
        for seed_url in self.profile.seed_urls:
            result = SearchResult(title=seed_url, url=seed_url, snippet=self.obsession)
            candidate = self.scorer.score(result, self.obsession, reason="profile-seed")
            if candidate.relevance_score > 0 and self.store.upsert_source_candidate(candidate):
                discovered += 1
        for query in self.discovery.generate_queries(query_limit):
            query_errors: list[str] = []
            try:
                results = self.search_provider.search(query, results_per_query)
            except OSError as exc:
                query_errors.append(str(exc))
                results = []
            if not results and use_fallback:
                results = self.discovery.fallback_results(query, results_per_query)
            for result in results:
                candidate = self.scorer.score(result, query)
                if candidate.relevance_score <= 0:
                    continue
                if self.store.upsert_source_candidate(candidate):
                    discovered += 1
            errors.extend(f"{query}: {error}" for error in query_errors)
            self.store.log_discovery_query(
                self.domain,
                self.obsession,
                query,
                "ok" if results and not query_errors else "fallback" if results else "error",
                len(results),
                query_errors,
            )
        return {"domain": self.domain, "obsession": self.obsession, "discovered": discovered, "errors": errors}

    def sources(self, limit: int = 20) -> list[dict]:
        return [source.__dict__ for source in self.store.list_source_candidates(self.domain, limit=limit)]

    def documents(self, limit: int = 20) -> list[dict]:
        return [document.__dict__ for document in self.store.list_source_documents(self.domain, limit)]

    def learned(self, limit: int = 20) -> list[dict]:
        return [fact.__dict__ for fact in self.store.list_learned_facts(self.domain, limit)]

    def runs(self, limit: int = 10) -> list[dict]:
        return self.store.list_obsession_runs(self.domain, limit)

    def fetch_documents(self, limit: int = 5, crawl_depth: int = 1, max_links_per_page: int = 8) -> dict:
        fetched = 0
        links_discovered = 0
        errors: list[str] = []
        sources = self.store.list_source_candidates(self.domain, statuses=("candidate", "approved"), limit=limit)
        for source in sources:
            if is_search_result_url(source.url):
                errors.append(f"{source.url}: skipped search result page")
                self.store.update_source_candidate(source.id, status="rejected", trust_delta=-0.2)
                continue
            try:
                document = self.fetcher.fetch(source)
                if not document.obsession:
                    document = replace(document, obsession=self.obsession)
            except OSError as exc:
                errors.append(f"{source.url}: {exc}")
                self.store.update_source_candidate(source.id, trust_delta=-0.02)
                continue
            if self.store.add_source_document(document):
                fetched += 1
            tables = getattr(self.fetcher, "last_tables", [])
            if tables:
                self.document_tables[document.content_hash] = tables
            if source.crawl_depth < crawl_depth:
                links = getattr(self.fetcher, "last_links", [])
                links_discovered += self._discover_links(source, links, max_links_per_page)
            self.store.update_source_candidate(source.id, fetched=True)
        return {"domain": self.domain, "fetched": fetched, "crawled": fetched, "links_discovered": links_discovered, "errors": errors}

    def extract(
        self,
        limit: int = 5,
        facts_per_document: int = 5,
        auto_approve: bool = False,
        auto_confidence_threshold: float = 0.8,
        trusted_only: bool | None = None,
        min_auto_score: float = 0.78,
    ) -> dict:
        extracted = 0
        pending_added = 0
        auto_approved = 0
        facts_rejected = 0
        errors: list[str] = []
        if trusted_only is None:
            trusted_only = self.profile.has_domain_rules
        for document in self.store.list_source_documents(self.domain, limit):
            table_facts = self.structured_extractor.extract(
                document,
                self.document_tables.get(document.content_hash, []),
                facts_per_document,
            )
            model_limit = max(facts_per_document - len(table_facts), 0)
            model_facts, error = self.extractor.extract(document, model_limit) if model_limit else ([], None)
            if error:
                errors.append(f"{document.url}: {error}")
                source = self.store.get_source_candidate(document.source_id)
                if source:
                    self.store.update_source_candidate(source.id, trust_delta=-0.02)
            for fact in [*table_facts, *model_facts]:
                source = self._source_by_url(fact.source)
                normalized_relation = normalize_relation(fact.relation, self.profile)
                fact = replace(fact, relation=normalized_relation)
                auto_score = fact_auto_score(fact, document, source, self.profile)
                will_auto_approve = (
                    auto_approve
                    and fact.confidence >= auto_confidence_threshold
                    and auto_score >= min_auto_score
                    and (not trusted_only or bool(source and source.trust_score >= 0.55))
                )
                fact = self._tag_extracted_fact(fact, "auto-approved" if will_auto_approve else "pending-review")
                quality_error = fact_quality_error(fact, document, self.obsession, self.profile)
                if (
                    not quality_error
                    and auto_approve
                    and not relation_allowed(fact.relation, self.profile)
                    and (fact.extraction_method == "table" or auto_score < min_auto_score)
                ):
                    quality_error = "relation not allowed for autonomous approval"
                if not quality_error and auto_approve and not fact.evidence_text:
                    quality_error = "missing evidence"
                if not quality_error and auto_approve and trusted_only and (not source or source.trust_score < 0.55):
                    quality_error = "source below trusted threshold"
                if quality_error:
                    errors.append(f"{document.url}: discarded {fact.subject} {fact.relation} {fact.object}: {quality_error}")
                    facts_rejected += 1
                    continue
                extracted += 1
                if will_auto_approve:
                    if self._auto_approve_fact(fact):
                        auto_approved += 1
                        if source:
                            self.store.update_source_candidate(source.id, trust_delta=0.03)
                elif self.store.add_pending_fact(fact):
                    pending_added += 1
        return {
            "domain": self.domain,
            "obsession": self.obsession,
            "extracted": extracted,
            "auto_approved": auto_approved,
            "pending_added": pending_added,
            "facts_rejected": facts_rejected,
            "pending_total": self.store.count_pending_facts(self.domain),
            "errors": errors,
        }

    def run_once(
        self,
        budget: int = 5,
        auto_approve: bool = False,
        auto_confidence_threshold: float = 0.8,
        crawl_depth: int | None = None,
        trusted_only: bool | None = None,
        max_links_per_page: int = 8,
        min_auto_score: float = 0.78,
    ) -> dict:
        started = time.perf_counter()
        tracemalloc.start()
        errors: list[str] = []
        effective_crawl_depth = self.profile.default_crawl_depth if crawl_depth is None else crawl_depth
        discovered = self.discover(query_limit=max(1, budget // 2), results_per_query=3)
        fetched = self.fetch_documents(limit=budget, crawl_depth=effective_crawl_depth, max_links_per_page=max_links_per_page)
        extracted = self.extract(
            limit=budget,
            auto_approve=auto_approve,
            auto_confidence_threshold=auto_confidence_threshold,
            trusted_only=trusted_only,
            min_auto_score=min_auto_score,
        )
        errors.extend(discovered["errors"])
        errors.extend(fetched["errors"])
        errors.extend(extracted["errors"])
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = round(time.perf_counter() - started, 3)
        memory_peak = round(peak / 1024, 1)
        status = "ok" if not errors else "partial"
        self.store.log_obsession_run(
            self.domain,
            self.obsession,
            status,
            int(discovered["discovered"]),
            int(fetched["fetched"]),
            int(extracted["extracted"]),
            int(extracted["pending_added"]) + int(extracted["auto_approved"]),
            errors,
            elapsed,
            memory_peak,
        )
        return {
            "domain": self.domain,
            "obsession": self.obsession,
            "status": status,
            "discovered": discovered["discovered"],
            "fetched": fetched["fetched"],
            "crawled": fetched["crawled"],
            "links_discovered": fetched["links_discovered"],
            "extracted": extracted["extracted"],
            "auto_approved": extracted["auto_approved"],
            "pending_added": extracted["pending_added"],
            "facts_rejected": extracted["facts_rejected"],
            "pending_total": self.store.count_pending_facts(self.domain),
            "agenda_questions": generate_agenda(self.domain, self.obsession, self.profile, self.store),
            "source_profile_hits": self._source_profile_hits(),
            "errors": errors,
            "elapsed_seconds": elapsed,
            "memory_peak_kb": memory_peak,
        }

    def run_daemon(
        self,
        budget: int = 5,
        interval_minutes: float = 30.0,
        auto_approve: bool = False,
        auto_confidence_threshold: float = 0.8,
        max_cycles: int | None = None,
        crawl_depth: int | None = None,
        trusted_only: bool | None = None,
        max_links_per_page: int = 8,
        min_auto_score: float = 0.78,
    ) -> list[dict]:
        results: list[dict] = []
        cycles = 0
        try:
            while max_cycles is None or cycles < max_cycles:
                results.append(
                    self.run_once(
                        budget=budget,
                        auto_approve=auto_approve,
                        auto_confidence_threshold=auto_confidence_threshold,
                        crawl_depth=crawl_depth,
                        trusted_only=trusted_only,
                        max_links_per_page=max_links_per_page,
                        min_auto_score=min_auto_score,
                    )
                )
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                time.sleep(max(interval_minutes, 0.01) * 60)
        except KeyboardInterrupt:
            return results
        return results

    def review(self, limit: int = 20) -> list[dict]:
        return [pending.__dict__ for pending in self.store.list_pending_facts(self.domain)[:limit]]

    def approve(self, pending_id: str) -> dict:
        fact = self.store.approve_pending_fact(pending_id)
        source = self._source_by_url(fact.source)
        if source:
            self.store.update_source_candidate(source.id, trust_delta=0.05)
        return {"approved": True, "fact_id": fact.id, "text": fact.text}

    def reject(self, pending_id: str) -> dict:
        pending = self.store.get_pending_fact(pending_id)
        self.store.reject_pending_fact(pending_id)
        if pending:
            source = self._source_by_url(pending.source)
            if source:
                self.store.update_source_candidate(source.id, trust_delta=-0.05)
        return {"rejected": True, "pending_id": pending_id}

    def _tag_extracted_fact(self, fact: PendingFact, review_tag: str) -> PendingFact:
        source = self._source_by_url(fact.source)
        source_type = source.source_type if source else "web"
        tags = tuple(dict.fromkeys((*fact.tags, review_tag, source_type)))
        return replace(fact, tags=tags)

    def _auto_approve_fact(self, pending: PendingFact) -> bool:
        exists = self.store._fact_exists(pending.subject, pending.relation, pending.object, pending.domain)
        fact = Fact(
            id=pending.id,
            subject=pending.subject,
            relation=pending.relation,
            object=pending.object,
            confidence=pending.confidence,
            source=pending.source,
            domain=pending.domain,
            tags=pending.tags,
            evidence_text=pending.evidence_text,
            evidence_hash=pending.evidence_hash,
            extraction_method=pending.extraction_method,
            created_at=pending.created_at,
        )
        self.store.upsert_fact(fact)
        return not exists

    def approve_source(self, source_id: str) -> dict:
        self.store.update_source_candidate(source_id, status="approved", trust_delta=0.1)
        return {"approved_source": True, "source_id": source_id}

    def reject_source(self, source_id: str) -> dict:
        self.store.update_source_candidate(source_id, status="rejected", trust_delta=-0.2)
        return {"rejected_source": True, "source_id": source_id}

    def _discover_links(self, source: SourceCandidate, links: list[SearchResult], limit: int) -> int:
        discovered = 0
        scored: list[SourceCandidate] = []
        for link in links:
            normalized = normalize_url(link.url)
            if normalized == source.url or is_search_result_url(normalized) or denied_by_profile(normalized, self.profile):
                continue
            if self.profile.allowed_hosts and not host_allowed(normalized, self.profile):
                continue
            candidate = self.scorer.score(
                SearchResult(link.title, normalized),
                self.obsession,
                parent_url=source.url,
                crawl_depth=source.crawl_depth + 1,
                reason="crawl-link",
            )
            link_context = f"{link.title} {normalized}"
            if candidate.relevance_score >= 0.25 and required_terms_present(link_context, self.profile):
                scored.append(candidate)
        scored.sort(key=lambda item: (item.trust_score + item.relevance_score), reverse=True)
        for candidate in scored[:limit]:
            if self.store.upsert_source_candidate(candidate):
                discovered += 1
        return discovered

    def _source_profile_hits(self) -> int:
        if not self.profile.allowed_hosts:
            return 0
        return sum(
            1
            for source in self.store.list_source_candidates(self.domain, statuses=("candidate", "approved", "rejected"), limit=200)
            if host_allowed(source.url, self.profile)
        )

    # Compatibility path for the older fixed RSS fetch command.
    def fetch(self, limit_per_source: int = 10) -> dict:
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
        return {
            "domain": self.domain,
            "fetched": fetched,
            "pending_added": added,
            "pending_total": self.store.count_pending_facts(self.domain),
            "errors": errors,
        }

    def _source_by_url(self, url: str) -> SourceCandidate | None:
        for source in self.store.list_source_candidates(self.domain, statuses=("candidate", "approved", "rejected"), limit=100):
            if source.url == normalize_url(url) or source.url == url:
                return source
        return None

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
