"""Source discovery and crawl helpers for local research ingestion."""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .ingest import Crawl4AIDocumentFetcher
from .models import SourceCandidate, SourceDocument
from .retrieval import tokenize
from .storage import KnowledgeStore


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


def generate_agenda(domain: str, topic: str, profile: ResearchProfile, store: KnowledgeStore | None = None, limit: int = 8) -> list[str]:
    domain_label = domain.replace("-", " ")
    templates = profile.agenda_templates or (
        "{obsession} core facts",
        "{obsession} important entities",
        "{obsession} terminology",
        "{obsession} evidence sources",
    )
    agenda = [template.format(domain=domain_label, obsession=topic) for template in templates]
    if store:
        agenda.extend(f"{domain_label} {question}" for question in store.list_recent_query_questions(5))
    return unique_strings(agenda)[:limit]


class SourceScorer:
    TRUST_HINTS = {
        "nintendo": 0.9,
        "mariowiki": 0.85,
        "wiki": 0.75,
        "reddit": 0.55,
        "forum": 0.5,
        "youtube": 0.45,
    }

    def __init__(self, domain: str, topic: str | None = None, profile: ResearchProfile | None = None) -> None:
        self.domain = domain
        self.topic = topic or domain.replace("-", " ")
        self.profile = profile or ResearchProfile(domain=domain)
        self.domain_terms = set(tokenize(f"{domain.replace('-', ' ')} {self.topic}"))

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
            obsession=self.topic,
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

    def status(self) -> dict[str, object]:
        return self._crawl4ai.status()


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


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = " ".join(value.lower().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output
