"""Website document source: crawler policy (robots, prefixes, rejects, cache),
navigation stripping, and the document-source API job.

No network: every request goes through ``httpx.MockTransport``.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from _job_wait import wait_for_jobs_idle  # noqa: E402
from app.main import app  # noqa: E402
from app.services.document_sources.web_crawl import (  # noqa: E402
    CrawlConfig,
    CrawlConfigError,
    WebCrawler,
    default_prefix,
    normalize_url,
)
from app.utils.file_parsers import parse_html  # noqa: E402

_REAL_CLIENT = httpx.Client  # the API tests patch httpx.Client itself

HOST = "https://docs.example.test"
SEED = f"{HOST}/docs/"

PAGE_TEMPLATE = """<html><head><title>{title}</title></head><body>
<header class="site-header">Site Header Words</header>
<nav><ul><li><a href="/docs/a.html">Nav Alpha</a><li><a href="/docs/b.html">Nav Beta</a></ul></nav>
<div class="breadcrumb">Home &gt; Docs &gt; {title}</div>
<div id="sidebar-menu">Sidebar Words</div>
<main><h1>{title}</h1><p>{body}</p>{links}</main>
<footer>Footer Words</footer>
</body></html>"""


def page(title: str, body: str, links: tuple[str, ...] = ()) -> str:
    anchors = "".join(f'<a href="{href}">link</a>' for href in links)
    return PAGE_TEMPLATE.format(title=title, body=body, links=anchors)


class Site:
    """A tiny mock site; records every request it served."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.robots_status = 200
        self.robots_body = "User-agent: *\nDisallow: /docs/private/\n"
        self.etags = True
        self.pages = {
            "/docs/": page("Index", "Welcome index.", ("a.html", "b.html", "private/secret.html", "/other/page.html",
                                                      "https://elsewhere.test/x", "file.pdf", "big.html", "missing.html",
                                                      "go-out", "dup.html", "skip/one.html", "#top", "mailto:x@y.test",
                                                      "a.html#section")),
            "/docs/a.html": page("Alpha", "Alpha unique content paragraph."),
            "/docs/b.html": page("Beta", "Beta unique content paragraph.", ("a.html",)),
            "/docs/dup.html": page("Alpha", "Alpha unique content paragraph."),
            "/docs/private/secret.html": page("Secret", "Hidden."),
            "/docs/skip/one.html": page("Skipped", "Skipped content."),
            "/docs/sitemap-only.html": page("From Sitemap", "Only the sitemap lists this page."),
            "/other/page.html": page("Other", "Out of prefix."),
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.url.host != "docs.example.test":
            return httpx.Response(200, text="elsewhere", headers={"content-type": "text/html"})
        if path == "/robots.txt":
            if self.robots_status == 200:
                return httpx.Response(200, text=self.robots_body)
            return httpx.Response(self.robots_status)
        if path == "/docs/sitemap.xml":
            return httpx.Response(
                200,
                text=f"<urlset><url><loc>{HOST}/docs/sitemap-only.html</loc></url>"
                f"<url><loc>{HOST}/other/out.html</loc></url></urlset>",
            )
        if path == "/docs/sitemap-index.xml":
            return httpx.Response(200, text=f"<sitemapindex><sitemap><loc>{HOST}/docs/sitemap.xml</loc></sitemap></sitemapindex>")
        if path == "/docs/go-out":
            return httpx.Response(301, headers={"location": f"{HOST}/other/page.html"})
        if path == "/docs/file.pdf":
            return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})
        if path == "/docs/big.html":
            return httpx.Response(200, content=b"<p>" + b"x" * 20000 + b"</p>", headers={"content-type": "text/html"})
        body = self.pages.get(path)
        if body is None:
            return httpx.Response(404, text="not found")
        etag = f'"{abs(hash(body)) % 10**8}"'
        if self.etags and request.headers.get("if-none-match") == etag:
            return httpx.Response(304)
        headers = {"content-type": "text/html; charset=utf-8"}
        if self.etags:
            headers["etag"] = etag
            headers["last-modified"] = "Wed, 01 Jan 2025 00:00:00 GMT"
        return httpx.Response(200, text=body, headers=headers)

    def client(self) -> httpx.Client:
        return _REAL_CLIENT(transport=httpx.MockTransport(self.handler))

    def fetched_paths(self) -> list[str]:
        return [r.url.path for r in self.requests if r.url.path != "/robots.txt"]


def crawl(site: Site, cache: Path, **overrides):
    config = CrawlConfig(seed_urls=[SEED], delay_seconds=0, max_page_bytes=10_000, **overrides)
    return WebCrawler(config, client=site.client(), sleep=lambda s: None, cache_dir=cache).crawl()


class CrawlerPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_crawl_keeps_html_pages_inside_prefix_and_reports_every_other_url(self):
        site = Site()
        result = crawl(site, self.cache, exclude_patterns=[r"/skip/"])
        kept = sorted(urlsplit_path(p.url) for p in result.pages)
        self.assertEqual(kept, ["/docs/", "/docs/a.html", "/docs/b.html"])
        reasons = {(urlsplit_path(r["url"]), r["reason"]) for r in result.rejected}
        self.assertIn(("/docs/private/secret.html", "robots_disallowed"), reasons)
        self.assertIn(("/docs/file.pdf", "not_html"), reasons)
        self.assertIn(("/docs/big.html", "too_large"), reasons)
        self.assertIn(("/docs/missing.html", "http_error"), reasons)
        self.assertIn(("/docs/skip/one.html", "excluded_by_pattern"), reasons)
        self.assertIn(("/docs/dup.html", "duplicate_content"), reasons)
        self.assertIn(("/docs/go-out", "outside_prefix"), reasons)
        http_error = next(r for r in result.rejected if r["reason"] == "http_error")
        self.assertEqual(http_error["detail"], "404")
        dup = next(r for r in result.rejected if r["reason"] == "duplicate_content")
        self.assertEqual(dup["detail"], f"{HOST}/docs/a.html")
        # Never fetched: disallowed by robots, other hosts, other prefixes, fragments, mailto.
        fetched = site.fetched_paths()
        self.assertNotIn("/docs/private/secret.html", fetched)
        self.assertNotIn("/other/page.html", fetched)
        self.assertTrue(all(r.url.host == "docs.example.test" for r in site.requests))
        self.assertEqual(fetched.count("/docs/a.html"), 1)  # #section + plain link are one URL
        self.assertEqual(result.rejection_counts["robots_disallowed"], 1)

    def test_page_records_title_etag_and_sizes(self):
        site = Site()
        result = crawl(site, self.cache, max_depth=0)
        index = result.pages[0].as_dict()
        self.assertEqual(index["url"], SEED)
        self.assertEqual(index["final_url"], SEED)
        self.assertEqual(index["title"], "Index")
        self.assertTrue(index["etag"].startswith('"'))
        self.assertTrue(index["last_modified"])
        self.assertGreater(index["html_bytes"], 100)
        self.assertTrue(index["retrieved_at"])
        self.assertEqual(len(result.pages), 1)  # depth 0: no links followed

    def test_max_pages_counts_limit_reached_without_listing_them(self):
        site = Site()
        result = crawl(site, self.cache, max_pages=2)
        self.assertEqual(len(result.pages), 2)
        self.assertGreater(result.limit_reached, 0)
        self.assertNotIn("limit_reached", {r["reason"] for r in result.rejected})

    def test_sitemap_adds_urls_and_reports_outside_prefix_entries(self):
        site = Site()
        result = crawl(site, self.cache, sitemap_url=f"{HOST}/docs/sitemap.xml", max_depth=0)
        self.assertIn("/docs/sitemap-only.html", [urlsplit_path(p.url) for p in result.pages])
        self.assertIn((f"{HOST}/other/out.html", "outside_prefix"), {(r["url"], r["reason"]) for r in result.rejected})

    def test_sitemap_index_is_followed_one_level(self):
        site = Site()
        result = crawl(site, self.cache, sitemap_url=f"{HOST}/docs/sitemap-index.xml", max_depth=0)
        self.assertIn("/docs/sitemap-only.html", [urlsplit_path(p.url) for p in result.pages])

    def test_include_patterns_limit_followed_links_but_not_the_seed(self):
        site = Site()
        result = crawl(site, self.cache, include_patterns=[r"/a\.html$"])
        self.assertEqual(sorted(urlsplit_path(p.url) for p in result.pages), ["/docs/", "/docs/a.html"])

    def test_recrawl_sends_validators_and_a_304_reuses_the_cached_body(self):
        site = Site()
        first = crawl(site, self.cache, max_depth=0)
        self.assertEqual(first.cache_hits, 0)
        site.requests.clear()
        second = crawl(site, self.cache, max_depth=0)
        sent = [r for r in site.requests if r.url.path == "/docs/"]
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].headers.get("if-none-match"))
        self.assertTrue(sent[0].headers.get("if-modified-since"))
        self.assertEqual(second.cache_hits, 1)
        self.assertEqual(second.pages[0].read_html(), first.pages[0].read_html())
        self.assertEqual(second.pages[0].etag, first.pages[0].etag)
        self.assertEqual(second.pages[0].title, "Index")

    def test_robots_unreachable_and_4xx_allow_but_5xx_disallows_the_host(self):
        site = Site()
        site.robots_status = 404
        allowed = crawl(site, self.cache, max_depth=0)
        self.assertEqual(len(allowed.pages), 1)
        site = Site()
        site.robots_status = 503
        blocked = crawl(site, Path(tempfile.mkdtemp(dir=self.cache)), max_depth=0)
        self.assertEqual(blocked.pages, [])
        self.assertEqual([r["reason"] for r in blocked.rejected], ["robots_unreachable"])

        def boom(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                raise httpx.ConnectError("no route")
            return httpx.Response(200, text=page("X", "Reachable."), headers={"content-type": "text/html"})

        client = _REAL_CLIENT(transport=httpx.MockTransport(boom))
        result = WebCrawler(
            CrawlConfig(seed_urls=[SEED], delay_seconds=0, max_depth=0), client=client, sleep=lambda s: None,
            cache_dir=Path(tempfile.mkdtemp(dir=self.cache)),
        ).crawl()
        self.assertEqual(result.pages, [])
        self.assertEqual([r["reason"] for r in result.rejected], ["robots_unreachable"])

    def test_robots_and_sitemap_fetches_are_size_capped(self):
        from app.services.document_sources import web_crawl
        from app.services.document_sources.robots import MAX_ROBOTS_BYTES

        site = Site()
        # A rule past the cap is never read; the Disallow before it still applies.
        site.robots_body = "User-agent: *\nDisallow: /docs/private/\n" + "#" * (MAX_ROBOTS_BYTES + 10) + "\nDisallow: /docs/a.html\n"
        result = crawl(site, self.cache, max_depth=1)
        fetched = {urlsplit_path(p.url) for p in result.pages}
        self.assertIn("/docs/a.html", fetched)
        self.assertNotIn("/docs/private/secret.html", fetched)

        site = Site()
        with mock.patch.object(web_crawl, "MAX_SITEMAP_BYTES", 20):
            result = crawl(site, Path(tempfile.mkdtemp(dir=self.cache)), max_depth=0, sitemap_url=f"{HOST}/docs/sitemap.xml")
        self.assertIn(("too_large", f"{HOST}/docs/sitemap.xml"), [(r["reason"], r["url"]) for r in result.rejected])
        self.assertNotIn("/docs/sitemap-only.html", [urlsplit_path(p.url) for p in result.pages])

    def test_cancel_event_stops_the_crawl_promptly(self):
        import threading

        from app.services.document_sources.web_crawl import CrawlCancelled

        site = Site()
        cancel = threading.Event()
        config = CrawlConfig(seed_urls=[SEED], delay_seconds=1.0, max_depth=2)
        crawler = WebCrawler(config, client=site.client(), sleep=lambda s: cancel.set(), cache_dir=self.cache, cancel=cancel)
        with self.assertRaises(CrawlCancelled):
            crawler.crawl()
        self.assertLessEqual(len(site.fetched_paths()), 2)

    def test_fetch_error_is_reported_not_raised(self):
        def down(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            raise httpx.ConnectTimeout("slow")

        result = WebCrawler(
            CrawlConfig(seed_urls=[SEED], delay_seconds=0), client=_REAL_CLIENT(transport=httpx.MockTransport(down)),
            sleep=lambda s: None, cache_dir=self.cache,
        ).crawl()
        self.assertEqual([r["reason"] for r in result.rejected], ["fetch_error"])

    def test_delay_is_slept_between_requests(self):
        site = Site()
        sleeps: list[float] = []
        config = CrawlConfig(seed_urls=[SEED], delay_seconds=2.0, max_depth=1, max_pages=3)
        WebCrawler(config, client=site.client(), sleep=sleeps.append, cache_dir=self.cache).crawl()
        self.assertTrue(sleeps and all(s == 2.0 for s in sleeps))

    def test_config_validation_and_helpers(self):
        with self.assertRaises(CrawlConfigError):
            CrawlConfig(seed_urls=["ftp://x.test/"])
        with self.assertRaises(CrawlConfigError):
            CrawlConfig(seed_urls=[SEED], include_patterns=["("])
        self.assertEqual(CrawlConfig(seed_urls=[SEED], max_pages=10**6).max_pages, 5000)
        self.assertEqual(default_prefix(f"{HOST}/guide/page.html"), f"{HOST}/guide/")
        self.assertEqual(default_prefix(f"{HOST}/guide"), f"{HOST}/guide/")
        self.assertEqual(CrawlConfig(seed_urls=[f"{HOST}/guide/x.html"]).allowed_prefixes, [f"{HOST}/guide/"])
        self.assertEqual(normalize_url("../b.html#frag", f"{HOST}/docs/a/c.html"), f"{HOST}/docs/b.html")
        self.assertEqual(normalize_url("HTTPS://Docs.Example.test:443"), f"{HOST}/")
        self.assertIsNone(normalize_url("mailto:a@b.test"))


class RobotsMatcherTests(unittest.TestCase):
    """RFC 9309 rules, one by one."""

    UA = "BrewSLM-crawler/1.0"

    def _allows(self, robots: str, path: str, ua: str | None = None) -> bool:
        from app.services.document_sources.robots import parse_robots

        return parse_robots(robots, ua or self.UA).allows(path)

    def test_group_selection_by_product_token_case_insensitive_with_star_fallback(self):
        robots = "User-agent: *\nDisallow: /\n\nUser-agent: BREWSLM-Crawler\nDisallow: /private/\n"
        self.assertTrue(self._allows(robots, "/docs/a"))
        self.assertFalse(self._allows(robots, "/private/a"))
        self.assertFalse(self._allows(robots, "/docs/a", ua="OtherBot/2"))  # falls back to *
        self.assertTrue(self._allows("User-agent: other\nDisallow: /\n", "/x"))  # no matching group at all
        # Consecutive User-agent lines share one group; same-agent groups merge.
        robots = "User-agent: a\nUser-agent: brewslm-crawler\nDisallow: /one\n\nUser-agent: brewslm-crawler\nDisallow: /two\n"
        self.assertFalse(self._allows(robots, "/one"))
        self.assertFalse(self._allows(robots, "/two"))
        self.assertTrue(self._allows(robots, "/three"))

    def test_longest_match_wins_and_allow_wins_ties(self):
        robots = "User-agent: *\nDisallow: /docs/\nAllow: /docs/public/\n"
        self.assertFalse(self._allows(robots, "/docs/x"))
        self.assertTrue(self._allows(robots, "/docs/public/x"))
        tie = "User-agent: *\nDisallow: /docs/a\nAllow: /docs/a\n"
        self.assertTrue(self._allows(tie, "/docs/a"))
        longer_disallow = "User-agent: *\nAllow: /docs\nDisallow: /docs/private\n"
        self.assertFalse(self._allows(longer_disallow, "/docs/private/x"))
        self.assertTrue(self._allows(longer_disallow, "/docs/other"))

    def test_wildcards_and_end_anchor(self):
        robots = "User-agent: *\nDisallow: /*.pdf$\nDisallow: /tmp*/cache\nAllow: /tmp-ok/cache\n"
        self.assertFalse(self._allows(robots, "/a/b.pdf"))
        self.assertTrue(self._allows(robots, "/a/b.pdf?x=1"))  # $ anchors the end incl. query
        self.assertTrue(self._allows(robots, "/a/b.pdfx"))
        self.assertFalse(self._allows(robots, "/tmp1/cache"))
        self.assertTrue(self._allows(robots, "/tmp-ok/cache"))
        self.assertTrue(self._allows("User-agent: *\nDisallow:\n", "/anything"))  # empty Disallow
        self.assertFalse(self._allows("User-agent: *\nDisallow: /\n", "/"))
        self.assertFalse(self._allows("User-agent: *\nDisallow: /a?b=1\n", "/a?b=1&c=2"))

    def test_percent_encoding_is_normalised_on_both_sides(self):
        self.assertFalse(self._allows("User-agent: *\nDisallow: /a~b\n", "/a%7Eb"))
        self.assertFalse(self._allows("User-agent: *\nDisallow: /a%7eb\n", "/a~b"))
        self.assertFalse(self._allows("User-agent: *\nDisallow: /a%2fb\n", "/a%2Fb"))
        self.assertTrue(self._allows("User-agent: *\nDisallow: /a%2fb\n", "/a/b"))  # %2F is not "/"
        self.assertFalse(self._allows("User-agent: *\nDisallow: /caf\u00e9\n", "/caf%C3%A9"))
        self.assertFalse(self._allows("User-agent: *\nDisallow: /caf%C3%A9\n", "/caf\u00e9"))

    def test_comments_crlf_and_crawl_delay(self):
        from app.services.document_sources.robots import parse_robots

        policy = parse_robots("# c\r\nUser-agent: *  # all\r\nCrawl-delay: 5\r\nDisallow: /x # no\r\n", self.UA)
        self.assertEqual(policy.crawl_delay, 5.0)
        self.assertFalse(policy.allows("/x/y"))


def urlsplit_path(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).path


class NavigationStrippingTests(unittest.TestCase):
    def _parse(self, markup: str, **kw) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as handle:
            handle.write(markup)
        try:
            return parse_html(Path(handle.name), **kw)
        finally:
            os.unlink(handle.name)

    def test_nav_breadcrumb_sidebar_header_footer_removed_main_kept(self):
        text = self._parse(page("Alpha", "Real body text."), strip_navigation=True)
        self.assertIn("Real body text.", text)
        for noise in ("Nav Alpha", "Home", "Sidebar Words", "Site Header Words", "Footer Words"):
            self.assertNotIn(noise, text)

    def test_default_keeps_everything_for_uploads(self):
        text = self._parse(page("Alpha", "Real body text."))
        for kept in ("Nav Alpha", "Home", "Sidebar Words", "Site Header Words", "Footer Words", "Real body text."):
            self.assertIn(kept, text)

    def test_unclosed_tags_inside_nav_do_not_swallow_the_page(self):
        markup = "<html><body><nav><ul><li>One<li>Two</ul></nav><p>After the menu.</p><hr><br></body></html>"
        text = self._parse(markup, strip_navigation=True)
        self.assertEqual(text, "After the menu.")

    def test_role_and_class_token_matching(self):
        markup = (
            '<div role="navigation">Role Nav</div><div class="main-menu">Menu</div>'
            '<div class="canvas">Canvas Stays</div><div id="toc">Contents</div><p>Body</p>'
        )
        text = self._parse(markup, strip_navigation=True)
        self.assertIn("Canvas Stays", text)
        self.assertIn("Body", text)
        for noise in ("Role Nav", "Menu", "Contents"):
            self.assertNotIn(noise, text)

    def test_article_is_preferred_when_there_is_no_main(self):
        text = self._parse("<body><p>Outside</p><article><p>Inside</p></article></body>", strip_navigation=True)
        self.assertEqual(text, "Inside")

    def test_sphinx_shape_keeps_role_main_inside_a_nav_named_wrapper(self):
        markup = (
            '<html><body class="wy-body-for-nav"><nav class="wy-nav-side">Side Links</nav>'
            '<section class="wy-nav-content-wrap"><div class="wy-nav-content"><div role="main">'
            '<h1>Guide</h1><p>Install the thing.</p></div><footer>Foot</footer></div></section></body></html>'
        )
        text = self._parse(markup, strip_navigation=True)
        self.assertEqual(text, "Guide\nInstall the thing.")

    def test_layout_wrapper_with_sidebar_class_falls_back_instead_of_emptying_the_page(self):
        markup = '<body><div class="layout with-sidebar"><h1>Title</h1><p>' + "Real content words. " * 5 + "</p></div></body>"
        text = self._parse(markup, strip_navigation=True)
        self.assertIn("Real content words.", text)
        self.assertIn("Title", text)

    def test_unclosed_p_with_nav_class_does_not_swallow_the_page(self):
        markup = '<body><p class="nav-hint">Hint text<div><p>First para.</p><p>Second para.</p></div></body>'
        text = self._parse(markup, strip_navigation=True)
        self.assertIn("First para.", text)
        self.assertIn("Second para.", text)

    def test_body_html_main_article_are_never_nav(self):
        markup = '<html class="nav"><body class="sidebar menu"><main class="toc"><p>Kept</p></main></body></html>'
        self.assertEqual(self._parse(markup, strip_navigation=True), "Kept")
        markup = '<body id="nav"><article class="menu"><p>Kept too</p></article></body>'
        self.assertEqual(self._parse(markup, strip_navigation=True), "Kept too")

    def test_mostly_nav_page_without_main_falls_back_to_tag_role_then_everything(self):
        markup = '<div class="menu"><p>' + "Only content here. " * 6 + "</p></div><nav>x</nav>"
        text = self._parse(markup, strip_navigation=True)
        self.assertIn("Only content here.", text)
        markup = "<footer>Only a footer</footer>"
        self.assertEqual(self._parse(markup, strip_navigation=True), "Only a footer")


# ── API ──────────────────────────────────────────────────────────────

_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__exit__(None, None, None)


def _poll_job(client: TestClient, job_id: int, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body.get("status") in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} still {body.get('status')}")


class WebCrawlApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _MODULE_CLIENT_CM
        self.project_id = self.client.post("/api/projects", json={"name": f"crawl-{uuid4().hex[:6]}", "description": ""}).json()["id"]
        self.site = Site()
        patcher = mock.patch("app.services.document_sources.web_crawl.httpx.Client", lambda *a, **k: self.site.client())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.url = f"/api/projects/{self.project_id}/document-sources/web-crawl"

    def tearDown(self) -> None:
        wait_for_jobs_idle(self.client)

    def _body(self, **extra) -> dict:
        return {"seed_urls": [SEED], "delay_seconds": 0, "max_page_bytes": 10_000, "exclude_patterns": ["/skip/"], **extra}

    def test_job_creates_documents_with_source_url_and_stripped_text(self):
        resp = self.client.post(self.url, json=self._body())
        self.assertEqual(resp.status_code, 202, resp.text)
        job = resp.json()
        self.assertEqual(job["kind"], "document_source_web_crawl")
        done = _poll_job(self.client, job["id"])
        self.assertEqual(done["status"], "succeeded", done)
        result = done["result"]
        self.assertEqual(result["source"], "web_crawl")
        self.assertEqual(result["documents_created"], 3)
        self.assertEqual(len(result["document_ids"]), 3)
        self.assertEqual(sorted(d["source_url"] for d in result["documents"]), [SEED, f"{SEED}a.html", f"{SEED}b.html"])
        self.assertEqual(result["rejection_counts"]["robots_disallowed"], 1)
        self.assertEqual(result["rejection_counts"]["not_html"], 1)
        self.assertIn("limit_reached", result)

        docs = self.client.get(f"/api/projects/{self.project_id}/ingestion/documents").json()
        by_url = {d["source_url"]: d for d in docs}
        self.assertEqual(set(by_url), {SEED, f"{SEED}a.html", f"{SEED}b.html"})
        alpha = by_url[f"{SEED}a.html"]
        self.assertEqual(alpha["source"], "web_crawl")
        self.assertEqual(alpha["status"], "accepted")
        self.assertTrue(alpha["filename"].endswith(".html"))
        from app.config import settings

        text = (settings.DATA_DIR / "projects" / str(self.project_id) / "raw" / alpha["filename"]).with_suffix(".extracted.txt").read_text()
        self.assertIn("Alpha unique content paragraph.", text)
        for noise in ("Nav Alpha", "Site Header Words", "Footer Words", "Sidebar Words"):
            self.assertNotIn(noise, text)

    def test_unknown_project_is_404_and_bad_body_is_422(self):
        self.assertEqual(self.client.post("/api/projects/99999999/document-sources/web-crawl", json=self._body()).status_code, 404)
        self.assertEqual(self.client.post(self.url, json={"seed_urls": ["ftp://x.test/"]}).status_code, 422)
        self.assertEqual(self.client.post(self.url, json=self._body(max_pages=0)).status_code, 422)
        self.assertEqual(self.client.post(self.url, json=self._body(max_depth=11)).status_code, 422)
        self.assertEqual(self.client.post(self.url, json=self._body(delay_seconds=31)).status_code, 422)
        self.assertEqual(self.client.post(self.url, json=self._body(exclude_patterns=["("])).status_code, 422)

    def test_inline_mode_returns_201_with_the_summary(self):
        resp = self.client.post(self.url + "?async_job=false", json=self._body(max_depth=0))
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["documents_created"], 1)

    def test_nothing_imported_fails_with_the_rejection_counts(self):
        self.site.robots_body = "User-agent: *\nDisallow: /\n"
        inline = self.client.post(self.url + "?async_job=false", json=self._body())
        self.assertEqual(inline.status_code, 422, inline.text)
        self.assertIn("robots_disallowed", inline.json()["detail"])
        job = self.client.post(self.url, json=self._body()).json()
        done = _poll_job(self.client, job["id"])
        self.assertEqual(done["status"], "failed")
        self.assertIn("1 robots_disallowed", done["error"])


class WebDatasetImportSourceTests(unittest.TestCase):
    def test_web_source_yields_url_title_text_rows_without_navigation(self):
        from app.services.dataset_import import resolve_source
        from app.services.dataset_import.sources import web as web_source

        site = Site()
        with mock.patch.object(web_source, "_client_factory", site.client), mock.patch.object(
            web_source, "_sleep", lambda s: None
        ):
            source = resolve_source("web")
            rows = list(source.load(SEED, limit=3))
            self.assertEqual(len(rows), 3)
            self.assertEqual({"url", "title", "text"}, set(rows[0]))
            index = next(r for r in rows if r["url"] == SEED)
            self.assertEqual(index["title"], "Index")
            self.assertIn("Welcome index.", index["text"])
            self.assertNotIn("Nav Alpha", index["text"])
            info = source.describe(SEED)
            self.assertEqual(info["columns"], ["url", "title", "text"])
            self.assertTrue(info["sample_rows"])

    def test_preview_caps_a_web_crawl_to_a_handful_of_pages(self):
        from app.services.dataset_import.service import preview_import
        from app.services.dataset_import.sources import web as web_source

        site = Site()
        for n in range(40):
            site.pages[f"/docs/p{n}.html"] = page(f"P{n}", f"Page number {n} content.")
        site.pages["/docs/"] = page("Index", "Index.", tuple(f"p{n}.html" for n in range(40)))
        with mock.patch.object(web_source, "_client_factory", site.client), mock.patch.object(
            web_source, "_sleep", lambda s: None
        ):
            result = preview_import(
                project_id=1, project_task_profile=None, locator=f"web:{SEED}", mapper_id="text_only",
                field_map={}, sample_cap=5, limit=500,
            )
        self.assertLessEqual(len(site.fetched_paths()), web_source.PREVIEW_MAX_PAGES)
        self.assertGreater(result.accepted_count, 0)


class _LoopProbeSource:
    """Records whether describe / load ran on a thread with a running event loop."""

    source_id = "loop_probe"
    seen: list[tuple[str, bool]] = []

    @staticmethod
    def _on_loop() -> bool:
        import asyncio

        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def load(self, locator, *, limit=None):
        _LoopProbeSource.seen.append(("load", self._on_loop()))
        yield {"text": "alpha beta gamma delta epsilon zeta eta theta"}

    def describe(self, locator):
        _LoopProbeSource.seen.append(("describe", self._on_loop()))
        return {"source_id": "loop_probe", "locator": locator, "approximate_total_rows": 1,
                "sample_rows": [{"text": "alpha beta gamma delta epsilon zeta eta theta"}], "columns": ["text"]}


class DatasetImportOffLoopTests(unittest.TestCase):
    """Sources may crawl a site / clone a repo: describe and load must not run on the event loop."""

    def setUp(self):
        from app.services.dataset_import import registry

        self.client = _MODULE_CLIENT_CM
        self.project_id = self.client.post("/api/projects", json={"name": f"offloop-{uuid4().hex[:6]}", "description": ""}).json()["id"]
        _LoopProbeSource.seen = []
        registry.register_source("loop_probe", _LoopProbeSource)
        self.addCleanup(lambda: registry._SOURCES.pop("loop_probe", None))

    def tearDown(self):
        wait_for_jobs_idle(self.client)

    def test_introspect_preview_and_run_use_a_worker_thread(self):
        base = f"/api/projects/{self.project_id}/dataset-import"
        resp = self.client.post("/api/dataset-import/introspect", json={"locator": "loop_probe:x"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = {"locator": "loop_probe:x", "mapper_id": "text_only", "field_map": {}}
        resp = self.client.post(f"{base}/preview", json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        resp = self.client.post(f"{base}/run", json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        calls = dict.fromkeys(name for name, _ in _LoopProbeSource.seen)
        self.assertEqual(set(calls), {"describe", "load"})
        self.assertFalse(any(on_loop for _, on_loop in _LoopProbeSource.seen), _LoopProbeSource.seen)
        self.assertEqual(sum(1 for name, _ in _LoopProbeSource.seen if name == "load"), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
