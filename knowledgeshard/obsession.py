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
        with urllib.request.urlopen(request, timeout=20) as response:
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


class SourceDiscovery:
    def __init__(self, store: KnowledgeStore, domain: str, obsession: str | None = None) -> None:
        self.store = store
        self.domain = domain
        self.obsession = obsession or domain.replace("-", " ")

    def generate_queries(self, limit: int = 10) -> list[str]:
        domain_label = self.domain.replace("-", " ")
        queries: list[str] = [
            self.obsession,
            f"{self.obsession} guide",
            f"{self.obsession} wiki",
            f"{domain_label} guide",
            f"{domain_label} strategy mechanics",
            f"{domain_label} wiki facts",
        ]
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
        candidates = [
            SearchResult(f"{self.obsession} on Wikipedia", f"https://en.wikipedia.org/wiki/{slug}"),
            SearchResult(f"{self.obsession} search on Super Mario Wiki", f"https://www.mariowiki.com/index.php?search={urllib.parse.quote_plus(self.obsession)}"),
        ]
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

    def __init__(self, domain: str, obsession: str | None = None) -> None:
        self.domain = domain
        self.obsession = obsession or domain.replace("-", " ")
        self.domain_terms = set(tokenize(f"{domain.replace('-', ' ')} {self.obsession}"))

    def score(self, result: SearchResult, query: str) -> SourceCandidate:
        combined = f"{result.title} {result.url} {result.snippet}"
        terms = set(tokenize(combined))
        query_terms = set(tokenize(query))
        relevance = len((self.domain_terms | query_terms) & terms) / max(len(self.domain_terms | query_terms), 1)
        source_type = self.source_type(result.url)
        trust = self.TRUST_HINTS.get(source_type, 0.45)
        status = "rejected" if is_search_result_url(result.url) else "candidate"
        if status == "rejected":
            trust = 0.0
            relevance = 0.0
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
        )

    def source_type(self, url: str) -> str:
        lowered = url.lower()
        for hint in self.TRUST_HINTS:
            if hint in lowered:
                return hint
        return "web"


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((scheme, netloc, path, "", parsed.query, ""))


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
BAD_RELATIONS = {"search_engine", "topic_of_search", "search_result", "has_search_result"}


def meaningful_terms(value: str) -> set[str]:
    return {term for term in tokenize(value) if len(term) > 2}


def fact_quality_error(fact: PendingFact, document: SourceDocument, obsession: str) -> str | None:
    subject = fact.subject.strip().lower()
    relation = fact.relation.strip().lower()
    obj = fact.object.strip().lower()
    if subject in BAD_FACT_VALUES or relation in BAD_FACT_VALUES or obj in BAD_FACT_VALUES:
        return "placeholder field"
    if relation in BAD_RELATIONS:
        return "search relation"
    combined = f"{subject} {relation} {obj}"
    if any(term in combined for term in BAD_FACT_TERMS):
        return "search-page fact"
    if is_search_result_url(fact.source) or is_search_result_url(document.url):
        return "search-result source"
    document_terms = meaningful_terms(f"{document.title} {document.text_excerpt}")
    subject_terms = meaningful_terms(fact.subject)
    object_terms = meaningful_terms(fact.object)
    obsession_terms = meaningful_terms(obsession)
    if subject_terms and not (subject_terms & document_terms):
        return "subject not grounded in document"
    if object_terms and not (object_terms & document_terms):
        return "object not grounded in document"
    fact_terms = meaningful_terms(f"{fact.subject} {fact.relation} {fact.object}")
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


class DocumentFetcher:
    def fetch(self, source: SourceCandidate, max_chars: int = 12000) -> SourceDocument:
        request = urllib.request.Request(
            source.url,
            headers={"user-agent": "KnowledgeShard/0.1 (+local research bot)"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            content_type = response.headers.get("content-type", "")
            raw = response.read(max_chars * 4)
        decoded = raw.decode("utf-8", errors="replace")
        text = self.readable_text(decoded) if "html" in content_type or "<html" in decoded[:500].lower() else decoded
        excerpt = " ".join(text.split())[:max_chars]
        content_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        return SourceDocument(
            id=uuid5(NAMESPACE_URL, f"{source.domain}:{source.url}:{content_hash}").hex,
            source_id=source.id,
            url=source.url,
            title=source.title,
            text_excerpt=excerpt,
            content_hash=content_hash,
            domain=source.domain,
        )

    def readable_text(self, page: str) -> str:
        parser = ReadableHTMLParser()
        parser.feed(page)
        return parser.text()


class FactExtractor:
    def __init__(
        self,
        store: KnowledgeStore,
        domain: str,
        obsession: str | OptionalModelRuntime | None = None,
        model_runtime: OptionalModelRuntime | None = None,
    ) -> None:
        self.store = store
        self.domain = domain
        if isinstance(obsession, OptionalModelRuntime):
            model_runtime = obsession
            obsession = None
        self.obsession = str(obsession) if obsession else domain.replace("-", " ")
        self.model_runtime = model_runtime or OptionalModelRuntime()

    def extract(self, document: SourceDocument, limit: int = 5) -> tuple[list[PendingFact], str | None]:
        if not self.model_runtime.available:
            return [], self.model_runtime.error or "model runtime unavailable"
        prompt = (
            "Extract concise structured facts for a local knowledge graph.\n"
            f"Domain: {self.domain}\n"
            f"Obsession: {self.obsession}\n"
            "Return only a JSON array. Each item must have subject, relation, object, confidence, and tags.\n"
            "Use only the document text. Do not invent facts.\n\n"
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
        if not subject or not relation or not obj:
            return None
        combined = f"{subject} {relation} {obj}"
        relevance_terms = set(tokenize(f"{self.domain.replace('-', ' ')} {self.obsession}"))
        if not (relevance_terms & set(tokenize(combined + " " + document.title))):
            return None
        confidence = max(0.0, min(float(item.get("confidence", 0.55)), 1.0))
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
            tags=("obsession", "extracted", *tuple(str(tag)[:40] for tag in tags)),
        )


def parse_json_array(text: str) -> list[object]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model response did not contain a JSON array")
    return json.loads(text[start : end + 1])


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
        self.discovery = SourceDiscovery(store, domain, self.obsession)
        self.scorer = SourceScorer(domain, self.obsession)
        self.search_provider = search_provider or DuckDuckGoSearchProvider()
        self.fetcher = fetcher or DocumentFetcher()
        self.extractor = extractor or FactExtractor(store, domain, self.obsession)

    def discover(self, query_limit: int = 5, results_per_query: int = 5, use_fallback: bool = True) -> dict:
        discovered = 0
        errors: list[str] = []
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

    def fetch_documents(self, limit: int = 5) -> dict:
        fetched = 0
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
            self.store.update_source_candidate(source.id, fetched=True)
        return {"domain": self.domain, "fetched": fetched, "errors": errors}

    def extract(
        self,
        limit: int = 5,
        facts_per_document: int = 5,
        auto_approve: bool = False,
        auto_confidence_threshold: float = 0.8,
    ) -> dict:
        extracted = 0
        pending_added = 0
        auto_approved = 0
        errors: list[str] = []
        for document in self.store.list_source_documents(self.domain, limit):
            facts, error = self.extractor.extract(document, facts_per_document)
            if error:
                errors.append(f"{document.url}: {error}")
                source = self.store.get_source_candidate(document.source_id)
                if source:
                    self.store.update_source_candidate(source.id, trust_delta=-0.02)
                continue
            for fact in facts:
                fact = self._tag_extracted_fact(fact, "auto-approved" if auto_approve and fact.confidence >= auto_confidence_threshold else "pending-review")
                quality_error = fact_quality_error(fact, document, self.obsession)
                if quality_error:
                    errors.append(f"{document.url}: discarded {fact.subject} {fact.relation} {fact.object}: {quality_error}")
                    continue
                extracted += 1
                if auto_approve and fact.confidence >= auto_confidence_threshold:
                    self._auto_approve_fact(fact)
                    auto_approved += 1
                    source = self._source_by_url(fact.source)
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
            "pending_total": self.store.count_pending_facts(self.domain),
            "errors": errors,
        }

    def run_once(
        self,
        budget: int = 5,
        auto_approve: bool = False,
        auto_confidence_threshold: float = 0.8,
    ) -> dict:
        started = time.perf_counter()
        tracemalloc.start()
        errors: list[str] = []
        discovered = self.discover(query_limit=budget, results_per_query=3)
        fetched = self.fetch_documents(limit=budget)
        extracted = self.extract(
            limit=budget,
            auto_approve=auto_approve,
            auto_confidence_threshold=auto_confidence_threshold,
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
            "extracted": extracted["extracted"],
            "auto_approved": extracted["auto_approved"],
            "pending_added": extracted["pending_added"],
            "pending_total": self.store.count_pending_facts(self.domain),
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

    def _auto_approve_fact(self, pending: PendingFact) -> None:
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
        self.store.upsert_fact(fact)

    def approve_source(self, source_id: str) -> dict:
        self.store.update_source_candidate(source_id, status="approved", trust_delta=0.1)
        return {"approved_source": True, "source_id": source_id}

    def reject_source(self, source_id: str) -> dict:
        self.store.update_source_candidate(source_id, status="rejected", trust_delta=-0.2)
        return {"rejected_source": True, "source_id": source_id}

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
