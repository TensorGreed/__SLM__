"""Website source: crawl a site and yield one row per page.

Locator format: ``web:<seed url>``, e.g. ``web:https://docs.example.com/guide/``.
Rows are ``{url, title, text}`` — ``text`` is the page's main content with
navigation stripped — so the ``text_only`` mapper can use them directly. The
crawl is the polite one in :mod:`app.services.document_sources.web_crawl`
(robots.txt, prefix-limited, rate-limited, cached); ``limit`` caps the pages
fetched. For the full document workflow (provenance, rejected-URL review) use
Data → Documents from website / repo instead.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import httpx

from app.services.dataset_import.protocols import RawRow
from app.services.dataset_import.registry import register_source
from app.services.document_sources.web_crawl import CrawlConfig, WebCrawler
from app.utils.file_parsers import DocumentParseError, html_to_text

DEFAULT_MAX_PAGES = 200
PREVIEW_MAX_PAGES = 10  # a preview / introspect fetches a handful of pages, never a whole site
_SAMPLE_PAGES = 5

# Tests swap these (no network in tests).
_client_factory: Callable[[], httpx.Client | None] = lambda: None  # noqa: E731
_sleep: Callable[[float], None] | None = None


class WebSource:
    source_id: str = "web"
    preview_limit_cap: int = PREVIEW_MAX_PAGES

    def _crawl(self, locator: str, max_pages: int):
        config = CrawlConfig(seed_urls=[locator.strip()], max_pages=max_pages)
        kwargs: dict[str, Any] = {"client": _client_factory()}
        if _sleep is not None:
            kwargs["sleep"] = _sleep
        return WebCrawler(config, **kwargs).crawl()

    def _rows(self, locator: str, max_pages: int) -> Iterable[RawRow]:
        crawl = self._crawl(locator, max_pages)
        for page in crawl.pages:
            try:
                text = html_to_text(page.read_html().decode("utf-8", errors="replace"), strip_navigation=True)
            except DocumentParseError:
                continue
            yield {"url": page.url, "title": page.title or "", "text": text}

    def load(self, locator: str, *, limit: int | None = None) -> Iterable[RawRow]:
        yield from self._rows(locator, limit or DEFAULT_MAX_PAGES)

    def describe(self, locator: str) -> dict[str, Any]:
        rows = list(self._rows(locator, _SAMPLE_PAGES))
        return {
            "source_id": self.source_id,
            "locator": locator,
            "approximate_total_rows": None,
            "sample_rows": rows,
            "columns": ["url", "title", "text"],
            "licence": None,
        }


register_source("web", WebSource)
