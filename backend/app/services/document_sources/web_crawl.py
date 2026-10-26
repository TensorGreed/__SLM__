"""Polite breadth-first website crawler (stdlib + httpx, no HTML library).

Reads a site's pages as HTML documents: stays inside allowed URL prefixes,
honours ``robots.txt`` (RFC 9309, see ``robots.py``: a 4xx → allow, an
unreachable file or a 5xx → the host is disallowed), spaces its requests, caches pages by ETag /
Last-Modified so a re-crawl only re-downloads what changed, and reports every
URL it did not keep with a stable reason code (the UI groups by reason so a
user can retry or drop whole groups).

Rejection reasons: ``http_error`` (detail = status), ``robots_disallowed``,
``robots_unreachable`` (robots.txt couldn't be read, so the host is skipped),
``outside_prefix`` (seeds, sitemap entries and redirect targets that leave the
allowed prefixes — ordinary off-site links are just not followed),
``not_html``, ``too_large``, ``excluded_by_pattern``, ``duplicate_content``,
``fetch_error``. URLs skipped because ``max_pages`` was reached are only
counted (``limit_reached``).

The HTTP client and the sleep function are injectable, so tests run on
``httpx.MockTransport`` with no network and no waiting.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

import httpx

from app.services.document_sources.robots import MAX_ROBOTS_BYTES, RobotsPolicy, parse_robots

MAX_PAGES_CAP = 5000
MAX_REDIRECTS = 5
MAX_SITEMAP_BYTES = 10 * 1024 * 1024
_HTML_TYPES = ("text/html", "application/xhtml+xml")


class CrawlConfigError(ValueError):
    """The crawl settings are unusable (message is user-facing)."""


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Absolute http(s) URL without fragment, lowercase scheme + host, default
    port dropped, empty path → ``/``. None for other schemes / unparsable."""
    try:
        joined = urljoin(base, url.strip()) if base else url.strip()
        joined, _ = urldefrag(joined)
        parts = urlsplit(joined)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    port = parts.port
    netloc = host if port in (None, 80 if scheme == "http" else 443) else f"{host}:{port}"
    if parts.username:
        return None  # credentials in a URL are never followed
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def default_prefix(url: str) -> str:
    """The directory a seed lives in: ``https://h/guide/page.html`` →
    ``https://h/guide/``. A final segment without a dot counts as a directory
    (``https://h/guide`` → ``https://h/guide/``)."""
    parts = urlsplit(url)
    path = parts.path or "/"
    if not path.endswith("/"):
        last = path.rsplit("/", 1)[-1]
        path = path + "/" if "." not in last else path.rsplit("/", 1)[0] + "/"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass
class CrawlConfig:
    seed_urls: list[str]
    allowed_prefixes: list[str] = field(default_factory=list)
    sitemap_url: str | None = None
    max_pages: int = 200
    max_depth: int = 3
    delay_seconds: float = 1.0
    user_agent: str = "BrewSLM-crawler/1.0"
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    max_page_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        seeds = [normalize_url(u) for u in self.seed_urls]
        if not self.seed_urls or any(s is None for s in seeds):
            raise CrawlConfigError("seed_urls must be one or more http(s) URLs.")
        self.seed_urls = list(dict.fromkeys(s for s in seeds if s))
        prefixes = [normalize_url(p) for p in self.allowed_prefixes]
        if any(p is None for p in prefixes):
            raise CrawlConfigError("allowed_prefixes must be http(s) URLs.")
        self.allowed_prefixes = [p for p in prefixes if p] or list(dict.fromkeys(default_prefix(s) for s in self.seed_urls))
        if self.sitemap_url:
            self.sitemap_url = normalize_url(self.sitemap_url)
            if not self.sitemap_url:
                raise CrawlConfigError("sitemap_url must be an http(s) URL.")
        self.max_pages = max(1, min(MAX_PAGES_CAP, int(self.max_pages)))
        self.max_depth = max(0, min(10, int(self.max_depth)))
        self.delay_seconds = max(0.0, min(30.0, float(self.delay_seconds)))
        for pattern in [*self.include_patterns, *self.exclude_patterns]:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise CrawlConfigError(f"Invalid regular expression {pattern!r}: {exc}") from exc


@dataclass
class CrawledPage:
    url: str
    final_url: str
    html_bytes: int
    title: str | None
    etag: str | None
    last_modified: str | None
    retrieved_at: str
    cache_path: Path
    from_cache: bool = False

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "final_url": self.final_url,
            "html_bytes": self.html_bytes,
            "title": self.title,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "retrieved_at": self.retrieved_at,
        }

    def read_html(self) -> bytes:
        return self.cache_path.read_bytes()


@dataclass
class CrawlResult:
    pages: list[CrawledPage] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    limit_reached: int = 0
    fetched: int = 0
    cache_hits: int = 0

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rejected:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
        return counts


def _is_redirect(response: httpx.Response) -> bool:
    return response.status_code in (301, 302, 303, 307, 308) and bool(response.headers.get("location"))


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.I | re.S)


def extract_links(markup: str, base_url: str) -> list[str]:
    parser = _LinkParser()
    try:
        parser.feed(markup)
    except Exception:  # noqa: BLE001 — malformed markup: keep what was found
        pass
    out: list[str] = []
    for href in parser.links:
        url = normalize_url(href, base_url)
        if url:
            out.append(url)
    return out


def extract_title(markup: str) -> str | None:
    found = _TITLE_RE.search(markup)
    if not found:
        return None
    title = " ".join(html.unescape(found.group(1)).split())
    return title or None


class CrawlCancelled(Exception):
    """The crawl was cancelled (its cancel event was set)."""


class WebCrawler:
    def __init__(
        self,
        config: CrawlConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        cache_dir: Path | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=20.0)
        self._sleep = cancel.wait if cancel is not None and sleep is time.sleep else sleep
        if cache_dir is None:
            from app.config import settings

            cache_dir = settings.DATA_DIR / "web_cache"
        self.cache_dir = cache_dir
        self._on_progress = on_progress
        self._cancel = cancel
        self._include = [re.compile(p) for p in config.include_patterns]
        self._exclude = [re.compile(p) for p in config.exclude_patterns]
        self._robots: dict[str, RobotsPolicy | bool] = {}
        self._requests = 0
        self._delay = config.delay_seconds

    # -- policy ------------------------------------------------------------

    def _in_prefix(self, url: str) -> bool:
        return any(url.startswith(prefix) for prefix in self.config.allowed_prefixes)

    def _get_capped(self, url: str, cap: int) -> tuple[httpx.Response, bytes, bool]:
        """GET (following redirects), reading at most ``cap`` bytes. Returns
        ``(response, body, truncated)``."""
        self._throttle_check()
        buffer = bytearray()
        truncated = False
        with self.client.stream(
            "GET", url, headers={"User-Agent": self.config.user_agent}, follow_redirects=True
        ) as response:
            if response.status_code == 200:
                for chunk in response.iter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > cap:
                        del buffer[cap:]
                        truncated = True
                        break
            else:
                response.read()
        return response, bytes(buffer), truncated

    def _throttle_check(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise CrawlCancelled("cancelled")

    def _robots_for(self, url: str) -> RobotsPolicy | bool:
        """The host's robots policy, True (allow all) or False (disallow all:
        robots.txt unreachable or a 5xx)."""
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host in self._robots:
            return self._robots[host]
        verdict: RobotsPolicy | bool = True
        try:
            response, body, _ = self._get_capped(f"{host}/robots.txt", MAX_ROBOTS_BYTES)
            if response.status_code == 200:
                policy = parse_robots(body.decode("utf-8", errors="replace"), self.config.user_agent)
                verdict = policy
                if policy.crawl_delay:
                    self._delay = max(self._delay, min(30.0, float(policy.crawl_delay)))
            elif response.status_code >= 500:
                verdict = False
        except httpx.HTTPError:
            verdict = False
        self._robots[host] = verdict
        return verdict

    def _robots_reject(self, url: str) -> str | None:
        """``None`` when robots.txt lets us fetch ``url``, else the reason code."""
        verdict = self._robots_for(url)
        if verdict is True:
            return None
        if verdict is False:
            return "robots_unreachable"
        return None if verdict.allows(url) else "robots_disallowed"

    def _pattern_reject(self, url: str, *, is_seed: bool) -> bool:
        if any(p.search(url) for p in self._exclude):
            return True
        return bool(self._include) and not is_seed and not any(p.search(url) for p in self._include)

    # -- cache ---------------------------------------------------------------

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json", self.cache_dir / f"{key}.body"

    def _load_cache(self, url: str) -> dict | None:
        meta_path, body_path = self._cache_paths(url)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return meta if body_path.exists() else None

    def _store_cache(self, url: str, meta: dict, body: bytes) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path, body_path = self._cache_paths(url)
        body_path.write_bytes(body)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return body_path

    # -- fetching ------------------------------------------------------------

    def _throttle(self) -> None:
        if self._requests and self._delay > 0:
            self._sleep(self._delay)
        self._requests += 1

    def _fetch_once(self, url: str, cached: dict | None) -> tuple[httpx.Response, bytes | None, bool]:
        """One GET (no redirect following). Returns (response, body, too_large)."""
        headers = {"User-Agent": self.config.user_agent, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"}
        if cached:
            if cached.get("etag"):
                headers["If-None-Match"] = cached["etag"]
            if cached.get("last_modified"):
                headers["If-Modified-Since"] = cached["last_modified"]
        self._throttle_check()
        self._throttle()
        with self.client.stream("GET", url, headers=headers, follow_redirects=False) as response:
            if response.status_code == 304 or _is_redirect(response) or response.status_code >= 400:
                response.read()
                return response, None, False
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self.config.max_page_bytes:
                return response, None, True
            ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype and ctype not in _HTML_TYPES:
                return response, None, False
            buffer = bytearray()
            for chunk in response.iter_bytes():
                buffer.extend(chunk)
                if len(buffer) > self.config.max_page_bytes:
                    return response, None, True
            return response, bytes(buffer), False

    def _fetch(self, url: str, result: CrawlResult) -> CrawledPage | None:
        """Fetch ``url`` (following in-prefix redirects). Appends a rejection
        to ``result`` and returns None when the page isn't kept."""

        def reject(reason: str, detail: str = "") -> None:
            result.rejected.append({"url": url, "reason": reason, "detail": detail})

        current = url
        for _ in range(MAX_REDIRECTS + 1):
            cached = self._load_cache(current)
            try:
                response, body, too_large = self._fetch_once(current, cached)
            except httpx.HTTPError as exc:
                reject("fetch_error", f"{type(exc).__name__}: {exc}"[:200])
                return None
            result.fetched += 1
            if self._on_progress:
                self._on_progress(result.fetched, self.config.max_pages)
            status = response.status_code
            if _is_redirect(response):
                target = normalize_url(response.headers.get("location", ""), current)
                if target is None or not self._in_prefix(target):
                    reject("outside_prefix", f"redirect to {target or response.headers.get('location', '')}")
                    return None
                blocked = self._robots_reject(target)
                if blocked:
                    reject(blocked, f"redirect to {target}")
                    return None
                current = target
                continue
            if status == 304:
                if not cached:
                    reject("http_error", "304 without a cached copy")
                    return None
                result.cache_hits += 1
                body_path = self._cache_paths(current)[1]
                return CrawledPage(
                    url=url, final_url=current, html_bytes=body_path.stat().st_size, title=cached.get("title"),
                    etag=cached.get("etag"), last_modified=cached.get("last_modified"),
                    retrieved_at=cached.get("retrieved_at") or "", cache_path=body_path, from_cache=True,
                )
            if status >= 400:
                reject("http_error", str(status))
                return None
            if too_large:
                reject("too_large", f"over {self.config.max_page_bytes} bytes")
                return None
            if body is None:
                reject("not_html", response.headers.get("content-type", "").split(";")[0].strip() or "unknown")
                return None
            encoding = response.charset_encoding or "utf-8"
            try:
                markup = body.decode(encoding, errors="replace")
            except LookupError:
                markup = body.decode("utf-8", errors="replace")
            retrieved_at = datetime.now(timezone.utc).isoformat()
            meta = {
                "url": current, "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
                "title": extract_title(markup), "retrieved_at": retrieved_at,
            }
            body_path = self._store_cache(current, meta, body)
            return CrawledPage(
                url=url, final_url=current, html_bytes=len(body), title=meta["title"], etag=meta["etag"],
                last_modified=meta["last_modified"], retrieved_at=retrieved_at, cache_path=body_path,
            )
        reject("fetch_error", "too many redirects")
        return None

    # -- sitemap -------------------------------------------------------------

    def _sitemap_urls(self, sitemap_url: str, result: CrawlResult, depth: int = 0) -> list[str]:
        try:
            response, raw, truncated = self._get_capped(sitemap_url, MAX_SITEMAP_BYTES)
        except httpx.HTTPError as exc:
            result.rejected.append({"url": sitemap_url, "reason": "fetch_error", "detail": f"sitemap: {exc}"[:200]})
            return []
        if response.status_code != 200:
            result.rejected.append({"url": sitemap_url, "reason": "http_error", "detail": f"sitemap: {response.status_code}"})
            return []
        if truncated:
            result.rejected.append({"url": sitemap_url, "reason": "too_large", "detail": f"sitemap over {MAX_SITEMAP_BYTES} bytes"})
            return []
        text = raw.decode("utf-8", errors="replace")
        locs = [html.unescape(m) for m in _LOC_RE.findall(text)]
        if "<sitemapindex" in text.lower() and depth == 0:
            out: list[str] = []
            for child in locs:
                normalized = normalize_url(child)
                if normalized:
                    out.extend(self._sitemap_urls(normalized, result, depth + 1))
            return out
        return locs

    # -- crawl ---------------------------------------------------------------

    def crawl(self) -> CrawlResult:
        result = CrawlResult()
        queue: deque[tuple[str, int, bool]] = deque()
        seen: set[str] = set()

        def consider(url: str, depth: int, *, is_seed: bool, report_outside: bool) -> None:
            if url in seen:
                return
            seen.add(url)
            if not self._in_prefix(url):
                if report_outside:
                    result.rejected.append({"url": url, "reason": "outside_prefix", "detail": ""})
                return
            if self._pattern_reject(url, is_seed=is_seed):
                result.rejected.append({"url": url, "reason": "excluded_by_pattern", "detail": ""})
                return
            queue.append((url, depth, is_seed))

        for seed in self.config.seed_urls:
            consider(seed, 0, is_seed=True, report_outside=True)
        if self.config.sitemap_url:
            for raw in self._sitemap_urls(self.config.sitemap_url, result):
                url = normalize_url(raw)
                if url:
                    consider(url, 0, is_seed=False, report_outside=True)

        content_hashes: dict[str, str] = {}
        try:
            while queue:
                if len(result.pages) >= self.config.max_pages:
                    result.limit_reached = len(queue)
                    break
                url, depth, _ = queue.popleft()
                self._throttle_check()
                blocked = self._robots_reject(url)
                if blocked:
                    result.rejected.append({"url": url, "reason": blocked, "detail": ""})
                    continue
                page = self._fetch(url, result)
                if page is None:
                    continue
                body = page.read_html()
                digest = hashlib.sha256(body).hexdigest()
                if digest in content_hashes:
                    result.rejected.append({"url": url, "reason": "duplicate_content", "detail": content_hashes[digest]})
                    continue
                content_hashes[digest] = url
                result.pages.append(page)
                if page.final_url != url:
                    seen.add(page.final_url)
                if depth < self.config.max_depth:
                    markup = body.decode("utf-8", errors="replace")
                    for link in extract_links(markup, page.final_url):
                        consider(link, depth + 1, is_seed=False, report_outside=False)
        finally:
            if self._owns_client:
                self.client.close()
        return result
