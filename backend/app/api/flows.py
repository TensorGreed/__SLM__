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

router = APIRouter(prefix="/projects/{project_id}/flows", tags=["Flows"])


class DocumentsQaFlowRequest(BaseModel):
    max_passages: int = Field(DEFAULT_MAX_PASSAGES, ge=1, le=500)
    pairs_per_passage: int = Field(DEFAULT_PAIRS_PER_PASSAGE, ge=1, le=10)
    backend: str | None = None
    # False: generate + split only, so the rows can be reviewed before training.
    train: bool = True
    # True: skip generation and re-split + train the pairs already generated
    # (e.g. the same data on another base model).
    reuse_existing: bool = False


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
