"""Document-source API — pull documents from a website or a git repository.

- ``POST /api/projects/{id}/document-sources/web-crawl``
- ``POST /api/projects/{id}/document-sources/git-repo``

``?async_job=true`` (the default) starts a background Job
(``document_source_web_crawl`` / ``document_source_git_repo``) and returns the
serialized Job with 202; the Job's ``result`` is the import summary (see
``services/document_sources/importer``). ``?async_job=false`` runs inline and
returns the summary with 201. Nothing imported is a failure (the Job fails /
inline answers 422) with the rejection counts in the message.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.exceptions import ProjectNotFoundError, SLMError
from app.models.project import Project
from app.services.document_sources.git_repo import (
    GitRepoConfig,
    GitSourceError,
    validate_repo_url,
)
from app.services.document_sources.importer import ImportCancelled, run_git_repo, run_web_crawl
from app.services.document_sources.web_crawl import CrawlConfig, CrawlConfigError, normalize_url
from app.services.jobs_service import JobProgressHandle, serialize_job, start_job

router = APIRouter(prefix="/projects/{project_id}/document-sources", tags=["Document Sources"])


def _http_url_list(values: list[str], field: str) -> list[str]:
    out = []
    for value in values:
        if normalize_url(value) is None:
            raise ValueError(f"{field}: {value!r} is not an http(s) URL")
        out.append(value.strip())
    return out


class WebCrawlRequest(BaseModel):
    seed_urls: list[str] = Field(..., min_length=1, max_length=50)
    allowed_prefixes: list[str] = Field(default_factory=list, max_length=50)
    sitemap_url: str | None = None
    max_pages: int = Field(200, ge=1, le=5000)
    max_depth: int = Field(3, ge=0, le=10)
    delay_seconds: float = Field(1.0, ge=0, le=30)
    user_agent: str = Field("BrewSLM-crawler/1.0", min_length=1, max_length=200)
    include_patterns: list[str] = Field(default_factory=list, max_length=50)
    exclude_patterns: list[str] = Field(default_factory=list, max_length=50)
    max_page_bytes: int = Field(2_000_000, ge=1_000, le=50_000_000)
    # The licence you declare for the site (recorded on the import's
    # provenance source; empty / "unknown" = unknown). Not a crawl setting.
    licence: str | None = Field(default=None, max_length=255)

    @field_validator("seed_urls")
    @classmethod
    def _seeds(cls, value: list[str]) -> list[str]:
        return _http_url_list(value, "seed_urls")

    @field_validator("allowed_prefixes")
    @classmethod
    def _prefixes(cls, value: list[str]) -> list[str]:
        return _http_url_list(value, "allowed_prefixes")

    @field_validator("sitemap_url")
    @classmethod
    def _sitemap(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        if not value:
            return None
        return _http_url_list([value], "sitemap_url")[0]

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def _patterns(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from exc
        return value

    def to_config(self) -> CrawlConfig:
        return CrawlConfig(**self.model_dump(exclude={"licence"}))


class GitRepoRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)
    ref: str = Field("HEAD", min_length=1, max_length=200)
    include_globs: list[str] = Field(default_factory=list, max_length=100)
    exclude_globs: list[str] = Field(default_factory=list, max_length=100)
    default_excludes: bool = True
    max_file_bytes: int = Field(200_000, ge=1, le=5_000_000)
    max_files: int = Field(2000, ge=1, le=20000)

    def to_config(self) -> GitRepoConfig:
        return GitRepoConfig(**self.model_dump())


async def _require_project(db: AsyncSession, project_id: int) -> None:
    found = await db.execute(select(Project.id).where(Project.id == project_id))
    if found.scalar_one_or_none() is None:
        raise ProjectNotFoundError(project_id)


def _bad_request(exc: Exception) -> SLMError:
    return SLMError(
        status_code=422,
        error_code="DOCUMENT_SOURCE_INVALID",
        stage="ingestion",
        message=str(exc),
        actionable_fix="Fix the source settings and try again.",
    )


def _nothing_imported(exc: ValueError) -> SLMError:
    return SLMError(
        status_code=422,
        error_code="DOCUMENT_SOURCE_EMPTY",
        stage="ingestion",
        message=str(exc),
        actionable_fix="Loosen the filters or check the address, then retry.",
    )


async def _start(db: AsyncSession, project_id: int, kind: str, title: str, params: dict[str, Any], work) -> JSONResponse:
    async def _runner(handle: JobProgressHandle) -> dict:
        async def progress(fraction: float | None, message: str | None) -> None:
            await handle.set_progress(fraction=fraction, message=message)

        return await work(handle.job_id, progress, handle.check_cancelled)

    job = await start_job(db, kind=kind, title=title, runner=_runner, project_id=project_id, params=params)
    return JSONResponse(status_code=202, content=serialize_job(job))


@router.post("/web-crawl", status_code=201)
async def start_web_crawl(
    project_id: int,
    data: WebCrawlRequest,
    async_job: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """Crawl a website into HTML documents (navigation stripped, provenance
    in ``metadata.source_url``). 202 + Job by default; 201 + summary inline."""
    await _require_project(db, project_id)
    try:
        config = data.to_config()
    except (CrawlConfigError, ValueError) as exc:
        raise _bad_request(exc) from exc
    if not async_job:
        try:
            return await run_web_crawl(project_id, config, licence=data.licence)
        except ValueError as exc:
            raise _nothing_imported(exc) from exc

    seed = config.seed_urls[0]

    async def work(job_id: int, progress, is_cancelled) -> dict:
        try:
            return await run_web_crawl(
                project_id, config, job_id=job_id, progress=progress, is_cancelled=is_cancelled, licence=data.licence,
            )
        except ImportCancelled:
            return {"cancelled": True, "documents_created": 0}

    return await _start(
        db, project_id, "document_source_web_crawl", f"Crawl {seed} · project #{project_id}",
        {"project_id": project_id, **data.model_dump()}, work,
    )


@router.post("/git-repo", status_code=201)
async def start_git_repo_import(
    project_id: int,
    data: GitRepoRequest,
    async_job: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """Import a git repository's docs + source files at a pinned commit
    (licence + commit recorded per document). 202 + Job by default; 201 +
    summary inline."""
    await _require_project(db, project_id)
    try:
        validate_repo_url(data.url)
        config = data.to_config()
    except (GitSourceError, ValueError) as exc:
        raise _bad_request(exc) from exc
    if not async_job:
        try:
            return await run_git_repo(project_id, config)
        except GitSourceError as exc:
            raise _bad_request(exc) from exc
        except ValueError as exc:
            raise _nothing_imported(exc) from exc

    async def work(job_id: int, progress, is_cancelled) -> dict:
        try:
            return await run_git_repo(project_id, config, job_id=job_id, progress=progress, is_cancelled=is_cancelled)
        except ImportCancelled:
            return {"cancelled": True, "documents_created": 0}

    return await _start(
        db, project_id, "document_source_git_repo", f"Import {data.url}@{config.ref} · project #{project_id}",
        {"project_id": project_id, **data.model_dump()}, work,
    )
