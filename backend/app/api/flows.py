"""Guided flows — several pipeline steps chained as one background Job.

Today: ``documents-to-qa`` (documents → generated Q&A → answer key → split →
train → lift check). See ``documents_qa_flow_service``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.documents_qa_flow_service import (
    DEFAULT_MAX_PASSAGES,
    DEFAULT_PAIRS_PER_PASSAGE,
    preview_documents_qa_flow,
    run_documents_qa_flow,
)
from app.services.synth_backends.base import SynthBackendError
from app.services.seed_documents_service import (
    DEFAULT_GENRES,
    MAX_COUNT,
    MAX_WORDS,
    MIN_COUNT,
    MIN_WORDS,
    run_seed_documents,
)

router = APIRouter(prefix="/projects/{project_id}/flows", tags=["Flows"])


class DocumentsQaFlowRequest(BaseModel):
    # Unset → sized to the decision: enough test examples that a 0.15
    # judge-score gap clears noise (documents_qa_flow_service.size_for_decision).
    max_passages: int | None = Field(None, ge=1, le=500)
    pairs_per_passage: int = Field(DEFAULT_PAIRS_PER_PASSAGE, ge=1, le=10)
    backend: str | None = None
    # False: generate + split only, so the rows can be reviewed before training.
    train: bool = True
    # True: skip generation and re-split + train the pairs already generated
    # (e.g. the same data on another base model).
    reuse_existing: bool = False
    # Before training, score the base model with and without the document
    # passages (judged). When retrieval alone already answers well the flow
    # stops there unless train_if_retrieval_ready.
    passages_check: bool = True
    train_if_retrieval_ready: bool = False


@router.get("/documents-to-qa/preview")
async def documents_to_qa_preview(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Can this project run the flow (enough cleaned passages, a reachable
    generation model), and what would it produce?"""
    try:
        return await preview_documents_qa_flow(db, project_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/documents-to-qa", status_code=202)
async def start_documents_to_qa(
    project_id: int,
    data: DocumentsQaFlowRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Start the flow as a background Job (the bell tracks it; the training
    run it starts gets its own watcher + lift check). 409 while one is
    already running for the project."""
    from sqlalchemy import select

    from app.models.job import Job, JobStatus
    from app.services.jobs_service import JobProgressHandle, serialize_job, start_job

    req = data or DocumentsQaFlowRequest()
    preview = await documents_to_qa_preview(project_id, db)
    if not preview["eligible"]:
        raise HTTPException(400, {"error_code": "FLOW_NOT_ELIGIBLE", "message": " ".join(preview["blockers"])})
    in_flight = (
        await db.execute(
            select(Job)
            .where(Job.kind == "documents_qa_flow", Job.project_id == project_id)
            .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
            .limit(1)
        )
    ).scalar_one_or_none()
    if in_flight is not None:
        raise HTTPException(
            409,
            {
                "error_code": "FLOW_ALREADY_RUNNING",
                "message": "The documents → Q&A flow is already running for this project. Watch the bell.",
                "metadata": {"existing_job_id": in_flight.id},
            },
        )

    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        from app.database import async_session_factory

        async def _progress(fraction: float, message: str) -> None:
            await handle.set_progress(fraction=fraction, message=message)

        async with async_session_factory() as runner_db:
            try:
                summary = await run_documents_qa_flow(
                    runner_db,
                    project_id,
                    max_passages=req.max_passages,
                    pairs_per_passage=req.pairs_per_passage,
                    backend=req.backend,
                    train=req.train,
                    reuse_existing=req.reuse_existing,
                    passages_check=req.passages_check,
                    train_if_retrieval_ready=req.train_if_retrieval_ready,
                    progress=_progress,
                )
            except SynthBackendError as exc:
                raise RuntimeError(f"Generation model failed: {exc}") from exc
            return summary.as_dict()

    job = await start_job(
        db,
        kind="documents_qa_flow",
        title=f"Documents → Q&A assistant · project #{project_id}",
        runner=_runner,
        project_id=project_id,
        params={
            "project_id": project_id,
            "max_passages": req.max_passages,
            "pairs_per_passage": req.pairs_per_passage,
            "train": req.train,
            "backend": req.backend,
        },
    )
    return serialize_job(job)


# ── Seed documents: write sample documents for a domain ─────────────────


class SeedDocumentsRequest(BaseModel):
    """What the sample documents must contain (``kinds``), which genres to
    spread them over (default ``DEFAULT_GENRES``), how many and how long.
    ``backend``: a synth backend name / pin token; unset → the first
    reachable local model."""

    kinds: str = Field("", max_length=2000)
    genres: list[str] = Field(default_factory=list, max_length=40)
    count: int = Field(60, ge=MIN_COUNT, le=MAX_COUNT)
    words: int = Field(160, ge=MIN_WORDS, le=MAX_WORDS)
    backend: str | None = None
    domain_hint: str = Field("", max_length=500)
    chunk_size: int = Field(900, ge=200, le=4000)
    chunk_overlap: int = Field(60, ge=0, le=1000)


@router.get("/seed-documents/defaults")
async def seed_documents_defaults() -> dict[str, Any]:
    return {"genres": list(DEFAULT_GENRES), "count": 60, "words": 160}


@router.post("/seed-documents", status_code=202)
async def start_seed_documents(
    project_id: int,
    data: SeedDocumentsRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Write sample documents with a generation model, ingest them as one
    text document and clean it into passages — as a background Job (the
    bell tracks it). 409 while one is already running for the project."""
    from sqlalchemy import select

    from app.models.job import Job, JobStatus
    from app.services.jobs_service import JobProgressHandle, serialize_job, start_job

    req = data or SeedDocumentsRequest()
    if req.chunk_overlap >= req.chunk_size:
        raise HTTPException(400, {"error_code": "BAD_CHUNKING", "message": "chunk_overlap must be smaller than chunk_size"})
    in_flight = (
        await db.execute(
            select(Job)
            .where(Job.kind == "seed_documents", Job.project_id == project_id)
            .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
            .limit(1)
        )
    ).scalar_one_or_none()
    if in_flight is not None:
        raise HTTPException(
            409,
            {
                "error_code": "FLOW_ALREADY_RUNNING",
                "message": "Sample documents are already being written for this project. Watch the bell.",
                "metadata": {"existing_job_id": in_flight.id},
            },
        )

    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        from app.database import async_session_factory

        async def _progress(fraction: float, message: str) -> None:
            await handle.set_progress(fraction=fraction, message=message)

        async with async_session_factory() as runner_db:
            try:
                summary = await run_seed_documents(
                    runner_db,
                    project_id,
                    kinds=req.kinds,
                    genres=req.genres,
                    count=req.count,
                    words=req.words,
                    backend=req.backend,
                    domain_hint=req.domain_hint,
                    chunk_size=req.chunk_size,
                    chunk_overlap=req.chunk_overlap,
                    progress=_progress,
                )
            except SynthBackendError as exc:
                raise RuntimeError(f"Generation model failed: {exc}") from exc
            return summary.as_dict()

    job = await start_job(
        db,
        kind="seed_documents",
        title=f"Sample documents · project #{project_id}",
        runner=_runner,
        project_id=project_id,
        params={"project_id": project_id, "count": req.count, "words": req.words, "genres": req.genres, "backend": req.backend},
    )
    return serialize_job(job)
