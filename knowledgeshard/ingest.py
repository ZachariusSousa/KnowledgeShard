"""Crawl4AI-backed document ingestion for local research."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse
from uuid import NAMESPACE_URL, uuid5

from .models import SourceCandidate, SourceDocument


@dataclass(frozen=True)
class CrawlLink:
    title: str
    url: str


def normalize_markdown(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def content_hash(markdown: str) -> str:
    normalized = normalize_markdown(markdown)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def result_markdown(result: Any) -> str:
    markdown = getattr(result, "markdown", "")
    if isinstance(markdown, str):
        return markdown
    if markdown is None:
        return ""
    fit = getattr(markdown, "fit_markdown", "")
    raw = getattr(markdown, "raw_markdown", "")
    return str(fit or raw or markdown)


def result_metadata(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", {}) or {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def result_url(result: Any, fallback: str) -> str:
    metadata = result_metadata(result)
    return str(
        getattr(result, "url", "")
        or metadata.get("sourceURL")
        or metadata.get("url")
        or fallback
    )


def result_title(result: Any, fallback: str) -> str:
    metadata = result_metadata(result)
    return str(metadata.get("title") or getattr(result, "title", "") or fallback)[:200]


def result_links(result: Any, base_url: str) -> list[CrawlLink]:
    raw_links = getattr(result, "links", None)
    if raw_links is None:
        raw_links = result_metadata(result).get("links", [])
    return normalize_links(raw_links, base_url)


def normalize_links(raw_links: Any, base_url: str) -> list[CrawlLink]:
    links: list[CrawlLink] = []
    if isinstance(raw_links, dict):
        iterable = []
        for value in raw_links.values():
            if isinstance(value, list):
                iterable.extend(value)
    elif isinstance(raw_links, list):
        iterable = raw_links
    else:
        iterable = []
    seen: set[str] = set()
    for item in iterable:
        title = ""
        url = ""
        if isinstance(item, str):
            url = item
        elif isinstance(item, dict):
            url = str(item.get("href") or item.get("url") or "")
            title = str(item.get("text") or item.get("title") or "")
        else:
            url = str(getattr(item, "href", "") or getattr(item, "url", ""))
            title = str(getattr(item, "text", "") or getattr(item, "title", ""))
        if not url:
            continue
        absolute = urljoin(base_url, url)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        normalized = absolute.split("#", 1)[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(CrawlLink(title=(title or normalized)[:200], url=normalized))
    return links


def source_document_from_crawl_result(
    source: SourceCandidate,
    result: Any,
    *,
    max_chars: int = 12000,
) -> SourceDocument:
    markdown = normalize_markdown(result_markdown(result))
    if not markdown:
        error = getattr(result, "error_message", "") or "crawl result did not include markdown"
        raise OSError(str(error))
    final_url = result_url(result, source.url)
    title = result_title(result, source.title)
    digest = content_hash(markdown)
    return SourceDocument(
        id=uuid5(NAMESPACE_URL, f"{source.domain}:{final_url}:{digest}").hex,
        source_id=source.id,
        url=final_url,
        title=title,
        text_excerpt=markdown[:max_chars],
        full_text=markdown,
        content_hash=digest,
        domain=source.domain,
        obsession=source.obsession,
    )


class Crawl4AIDocumentFetcher:
    """Synchronous wrapper around Crawl4AI's async crawler."""

    def __init__(self, *, check_robots_txt: bool = True) -> None:
        self.check_robots_txt = check_robots_txt
        self.last_links: list[CrawlLink] = []
        self.last_tables: list[list[list[str]]] = []
        self.last_error: str = ""

    def fetch(self, source: SourceCandidate, max_chars: int = 12000) -> SourceDocument:
        try:
            result = asyncio.run(self._crawl(source.url))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(self._crawl(source.url))
            finally:
                loop.close()
        success = bool(getattr(result, "success", True))
        if not success:
            error = str(getattr(result, "error_message", "") or "Crawl4AI crawl failed")
            self.last_error = error
            raise OSError(error)
        document = source_document_from_crawl_result(source, result, max_chars=max_chars)
        self.last_links = result_links(result, document.url)
        self.last_tables = []
        self.last_error = ""
        return document

    async def _crawl(self, url: str) -> Any:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
        except ImportError as exc:
            raise OSError("crawl4ai is required for ingestion; install requirements.txt") from exc
        run_config = CrawlerRunConfig(check_robots_txt=self.check_robots_txt)
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url, config=run_config)

    def status(self) -> dict[str, object]:
        try:
            import crawl4ai
        except ImportError as exc:
            return {"backend": "crawl4ai", "available": False, "error": str(exc)}
        version = getattr(crawl4ai, "__version__", "unknown")
        return {"backend": "crawl4ai", "available": True, "version": version, "error": self.last_error}


def markdown_links(markdown: str, base_url: str) -> list[CrawlLink]:
    raw = [{"text": text, "href": href} for text, href in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", markdown)]
    return normalize_links(raw, base_url)
