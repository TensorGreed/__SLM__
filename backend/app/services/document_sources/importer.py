"""Turn crawled pages / repo files into ``RawDocument`` rows.

Shared by the website and git-repo sources: each file goes through the same
``ingest_file`` → ``process_document`` path an upload takes (so cleaning, the
passage index and everything after treat them alike), with the provenance a
reader needs to cite it stored in ``RawDocument.metadata_``. A page / file
whose processing fails is reported as rejected ``parse_error`` and its
document is removed — the import never leaves a half-made document behind.

The result dict is the Job result of both sources::

    {source, documents_created, document_ids, documents: [{id, filename,
     source_url, path?}] (cap 500), rejected: [{url|path, reason, detail}]
     (cap 500), rejection_counts, limit_reached, commit?, ref?, licence?}

Zero documents created raises ``ValueError`` with a diagnostic (a silent
"Done" with nothing imported misleads).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
from pathlib import PurePosixPath
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from app.database import async_session_factory
from app.models.dataset import DocumentStatus
from app.services import provenance_service
from app.services.provenance_service import _parse_iso
from app.services.code_chunker import detect_language
from app.services.document_sources.git_repo import GitCancelled, GitRepoConfig, RepoSnapshot, clone_repo
from app.services.document_sources.web_crawl import CrawlCancelled, CrawlConfig, CrawlResult, WebCrawler
from app.services.ingestion_service import delete_document, ingest_file, process_document
from app.utils.file_parsers import SUPPORTED_EXTENSIONS

RESULT_LIST_CAP = 500
_PROGRESS_INTERVAL_SECONDS = 0.5
_PREFERRED_EXTENSION = {
    "python": ".py", "shell": ".sh", "javascript": ".js", "typescript": ".ts", "dockerfile": ".dockerfile",
    "java": ".java", "csharp": ".cs", "go": ".go", "c": ".c", "cpp": ".cpp", "hcl": ".tf", "yaml": ".yaml",
    "json": ".json", "sql": ".sql", "xml": ".xml", "properties": ".properties", "ini": ".ini",
}

ProgressFn = Callable[[float | None, str | None], Awaitable[None]]


def page_filename(url: str) -> str:
    """A flat, readable file name for a crawled page: host + path as a slug,
    ``.html``. A query string adds a short hash so ``?page=2`` doesn't collide."""
    parts = urlsplit(url)
    path = re.sub(r"\.(?:html?|php|aspx?)$", "", parts.path.strip("/"), flags=re.I)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{parts.hostname or 'page'}-{path}" if path else (parts.hostname or "page"))
    slug = re.sub(r"-{2,}", "-", slug).strip("-.") or "page"
    slug = slug[:100]
    if parts.query:
        slug += "-" + hashlib.sha256(parts.query.encode("utf-8")).hexdigest()[:8]
    return slug + ".html"


def repo_filename(path: str, language: str | None) -> str:
    """A flat file name for a repo file: ``/`` → ``__``. A name with no
    extension ingestion accepts (``Dockerfile``, a shebang script) gets the
    language's usual one appended."""
    flat = path.replace("/", "__")
    if PurePosixPath(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
        flat += _PREFERRED_EXTENSION.get(language or "", ".txt")
    return flat


def _cap(rows: list[dict]) -> list[dict]:
    return rows[:RESULT_LIST_CAP]


def _diagnostic(source: str, counts: dict[str, int], limit_reached: int, extra: str = "") -> str:
    parts = [f"{n} {reason}" for reason, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    if limit_reached:
        parts.append(f"{limit_reached} skipped by the limit")
    detail = ", ".join(parts) if parts else "nothing was found"
    return f"No documents were imported from the {source} source ({detail}).{(' ' + extra) if extra else ''}"


def _counts(rejected: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rejected:
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    return counts


async def _ingest_one(
    db: Any,
    project_id: int,
    *,
    filename: str,
    content: bytes,
    source: str,
    license_info: str,
    metadata: dict[str, Any],
    data_source: Any = None,
    source_url: str | None = None,
    source_ref: str | None = None,
):
    """Ingest + process one document. Returns ``(doc, error)``; on a
    processing error the document is deleted and ``error`` is its message.
    ``data_source`` is the provenance-ledger entry the document joins."""
    doc = await ingest_file(db, project_id, filename, content, source=source, license_info=license_info)
    doc.metadata_ = {**(doc.metadata_ or {}), **metadata}
    if data_source is not None:
        await provenance_service.attach_document(
            doc, data_source, source_url=source_url, source_ref=source_ref,
            retrieved_at=_parse_iso(metadata.get("retrieved_at")),
        )
    await db.flush()
    doc = await process_document(db, project_id, doc.id)
    if doc.status == DocumentStatus.ERROR:
        error = str((doc.metadata_ or {}).get("error") or "could not be processed")
        doc_id = doc.id
        await delete_document(db, project_id, doc_id)
        await db.commit()
        return None, error
    await db.commit()
    return doc, None


async def import_crawl(
    project_id: int, crawl: CrawlResult, *, job_id: int | None = None, progress: ProgressFn | None = None,
    is_cancelled: Callable[[], Awaitable[bool]] | None = None, licence: str | None = None,
    locator: str | None = None,
) -> dict[str, Any]:
    """Create documents for the crawl's accepted pages. One provenance source
    covers the whole crawl (``locator`` = the first seed; ``licence`` as the
    caller declared it, ``None`` = unknown)."""
    rejected = [dict(row) for row in crawl.rejected]
    documents: list[dict] = []
    ids: list[int] = []
    async with async_session_factory() as db:
        seed = locator or (crawl.pages[0].url if crawl.pages else "")
        ledger = await provenance_service.create_source(
            db, project_id, kind=provenance_service.KIND_WEB_CRAWL, title=f"Website {seed}", locator=seed,
            licence=licence, metadata={"import_job_id": job_id, "pages": len(crawl.pages)},
        )
        for index, page in enumerate(crawl.pages, start=1):
            if is_cancelled and index % 10 == 0 and await is_cancelled():
                break
            metadata = {
                "source_url": page.url, "final_url": page.final_url, "retrieved_at": page.retrieved_at,
                "etag": page.etag, "title": page.title, "strip_navigation": True, "import_job_id": job_id,
            }
            doc, error = await _ingest_one(
                db, project_id, filename=page_filename(page.url), content=page.read_html(),
                source="web_crawl", license_info=ledger.licence or "", metadata=metadata,
                data_source=ledger, source_url=page.url,
            )
            if doc is None:
                rejected.append({"url": page.url, "reason": "parse_error", "detail": error[:300]})
            else:
                ids.append(doc.id)
                documents.append({"id": doc.id, "filename": doc.filename, "source_url": page.url})
            if progress:
                await progress(0.6 + 0.4 * index / max(1, len(crawl.pages)), f"Imported {index}/{len(crawl.pages)} pages")
        if not ids:
            await db.delete(ledger)
            await db.commit()
    counts = _counts(rejected)
    if not ids:
        raise ValueError(_diagnostic("web crawl", counts, crawl.limit_reached, "Check the seed URL, the allowed prefixes and robots.txt."))
    return {
        "source": "web_crawl",
        "documents_created": len(ids),
        "document_ids": ids,
        "documents": _cap(documents),
        "rejected": _cap(rejected),
        "rejection_counts": counts,
        "limit_reached": crawl.limit_reached,
        "pages_fetched": crawl.fetched,
        "cache_hits": crawl.cache_hits,
        "data_source_id": int(ledger.id),
        "licence": ledger.licence,
    }


async def import_repo(
    project_id: int, snapshot: RepoSnapshot, *, job_id: int | None = None, progress: ProgressFn | None = None,
    is_cancelled: Callable[[], Awaitable[bool]] | None = None,
) -> dict[str, Any]:
    """Create documents for the snapshot's accepted files."""
    rejected = [dict(row) for row in snapshot.rejected]
    documents: list[dict] = []
    ids: list[int] = []
    from datetime import datetime, timezone

    retrieved_at = datetime.now(timezone.utc).isoformat()
    async with async_session_factory() as db:
        ledger = await provenance_service.create_source(
            db, project_id, kind=provenance_service.KIND_GIT_REPO, title=f"Repository {snapshot.url}",
            locator=snapshot.url, ref=snapshot.commit, licence=snapshot.licence,
            metadata={"ref_requested": snapshot.ref, "licence_file": snapshot.licence_file, "import_job_id": job_id},
        )
        for index, repo_file in enumerate(snapshot.files, start=1):
            if is_cancelled and index % 10 == 0 and await is_cancelled():
                break
            try:
                content = repo_file.read_bytes()
            except OSError as exc:
                rejected.append({"path": repo_file.path, "reason": "parse_error", "detail": str(exc)[:300]})
                continue
            language = repo_file.language or detect_language(repo_file.path)
            metadata = {
                "source_url": snapshot.url, "ref": snapshot.ref, "commit": snapshot.commit, "path": repo_file.path,
                "language": language, "content_kind": repo_file.content_kind, "licence_file": snapshot.licence_file,
                "retrieved_at": retrieved_at, "import_job_id": job_id,
            }
            doc, error = await _ingest_one(
                db, project_id, filename=repo_filename(repo_file.path, language), content=content,
                source="git_repo", license_info=snapshot.licence, metadata=metadata,
                data_source=ledger, source_url=snapshot.url, source_ref=snapshot.commit,
            )
            if doc is None:
                rejected.append({"path": repo_file.path, "reason": "parse_error", "detail": error[:300]})
            else:
                ids.append(doc.id)
                documents.append({"id": doc.id, "filename": doc.filename, "source_url": snapshot.url, "path": repo_file.path})
            if progress:
                await progress(0.5 + 0.5 * index / max(1, len(snapshot.files)), f"Imported {index}/{len(snapshot.files)} files")
        if not ids:
            await db.delete(ledger)
            await db.commit()
    counts = _counts(rejected)
    if not ids:
        raise ValueError(_diagnostic("git repo", counts, snapshot.limit_reached, f"Commit {snapshot.commit[:7]}."))
    return {
        "source": "git_repo",
        "documents_created": len(ids),
        "document_ids": ids,
        "documents": _cap(documents),
        "rejected": _cap(rejected),
        "rejection_counts": counts,
        "limit_reached": snapshot.limit_reached,
        "commit": snapshot.commit,
        "ref": snapshot.ref,
        "licence": snapshot.licence,
        "data_source_id": int(ledger.id),
    }


class ImportCancelled(Exception):
    """The Job was cancelled while the crawl / clone was running."""


async def _with_progress(
    work: Callable[[], Any],
    report: Callable[[], tuple[float, str]] | None,
    progress: ProgressFn | None,
    *,
    cancel: threading.Event | None = None,
    is_cancelled: Callable[[], Awaitable[bool]] | None = None,
) -> Any:
    """Run blocking ``work`` in a thread; while it runs, publish ``report()``
    from the event loop (the worker only updates plain shared state). ``cancel``
    is set when the Job's cancellation lands (``is_cancelled`` polled each
    tick) or the awaiting task itself is cancelled, so the worker stops promptly;
    the worker's own stop is reported as :class:`ImportCancelled`."""
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        while not task.done():
            if progress and report:
                fraction, message = report()
                await progress(fraction, message)
            if cancel is not None and is_cancelled is not None and not cancel.is_set() and await is_cancelled():
                cancel.set()
            await asyncio.wait({task}, timeout=_PROGRESS_INTERVAL_SECONDS)
    except BaseException:
        if cancel is not None:
            cancel.set()
        raise
    result = task.result()
    if cancel is not None and cancel.is_set():
        raise ImportCancelled("cancelled")
    return result


async def run_web_crawl(
    project_id: int,
    config: CrawlConfig,
    *,
    job_id: int | None = None,
    progress: ProgressFn | None = None,
    is_cancelled: Callable[[], Awaitable[bool]] | None = None,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] | None = None,
    cache_dir: Any = None,
    licence: str | None = None,
) -> dict[str, Any]:
    """Crawl, then import. The crawl runs in a worker thread. ``licence`` is
    the licence the caller declares for the site (recorded on the source)."""
    state = {"fetched": 0}

    def on_progress(fetched: int, max_pages: int) -> None:
        state["fetched"] = fetched

    kwargs: dict[str, Any] = {"client": client, "cache_dir": cache_dir, "on_progress": on_progress}
    if sleep is not None:
        kwargs["sleep"] = sleep
    cancel = threading.Event()
    kwargs["cancel"] = cancel
    crawler = WebCrawler(config, **kwargs)

    def report() -> tuple[float, str]:
        done = state["fetched"]
        return min(0.6, 0.6 * done / config.max_pages), f"Fetched {done}/{config.max_pages} pages"

    try:
        crawl = await _with_progress(crawler.crawl, report, progress, cancel=cancel, is_cancelled=is_cancelled)
    except CrawlCancelled as exc:
        raise ImportCancelled("cancelled") from exc
    if progress:
        await progress(0.6, f"Fetched {crawl.fetched} pages; importing {len(crawl.pages)}")
    return await import_crawl(
        project_id, crawl, job_id=job_id, progress=progress, is_cancelled=is_cancelled,
        licence=licence, locator=config.seed_urls[0],
    )


async def run_git_repo(
    project_id: int,
    config: GitRepoConfig,
    *,
    job_id: int | None = None,
    progress: ProgressFn | None = None,
    is_cancelled: Callable[[], Awaitable[bool]] | None = None,
    runner: Any = None,
) -> dict[str, Any]:
    """Clone, classify, then import. The clone runs in a worker thread."""
    if progress:
        await progress(0.05, "Cloning repository")
    cancel = threading.Event()
    try:
        snapshot = await _with_progress(
            lambda: clone_repo(config, runner=runner, cancel=cancel), None, None, cancel=cancel, is_cancelled=is_cancelled,
        )
    except GitCancelled as exc:
        raise ImportCancelled("cancelled") from exc
    try:
        if progress:
            await progress(0.5, f"Cloned {snapshot.commit[:7]}; importing {len(snapshot.files)} files")
        return await import_repo(project_id, snapshot, job_id=job_id, progress=progress, is_cancelled=is_cancelled)
    finally:
        snapshot.cleanup()
