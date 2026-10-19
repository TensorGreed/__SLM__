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
from app.services.cloud_backend_pins import CloudPinError, resolve_cloud_backend
from app.services.documents_qa_flow_service import (
    DEFAULT_MAX_PASSAGES,
    DEFAULT_PAIRS_PER_PASSAGE,
    preview_documents_qa_flow,
    run_documents_qa_flow,
)
from app.services.prompt_contamination_service import (
    PromptCarriesBenchmarkValues,
    http_detail,
    refuse_if_contaminated,
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


async def _cloud_backend_or_http(db: AsyncSession, project_id: int, pin: str | None) -> Any:
    """A ``cloud:<provider>:<model>`` pin → a cloud backend with the project's
    saved key (None for local backend names — ``pick_backend`` resolves
    those). Same 400 / 402 as the Synthetic tab's span routes."""
    try:
        return await resolve_cloud_backend(db, project_id, pin)
    except CloudPinError as exc:
        error_code = "CLOUD_KEY_MISSING" if exc.status_code == 402 else "BAD_BACKEND_PIN"
        raise HTTPException(exc.status_code, {"error_code": error_code, "message": exc.message}) from exc


class DocumentsQaFlowRequest(BaseModel):
    # Unset → sized to the decision: enough test examples that a 0.15
    # judge-score gap clears noise (documents_qa_flow_service.size_for_decision).
    max_passages: int | None = Field(None, ge=1, le=2000)
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
    # Train the model to answer FROM the retrieved passages (the served
    # retrieval prompt; passage_prompts_service). Scored with the passages
    # against the base model + the same passages.
    train_with_passages: bool = False
    passages_withheld_fraction: float | None = Field(None, ge=0.0, le=0.5)
    # The judge-score difference the test examples must resolve: sizes the
    # test (and, with keep_test, validation) set — 0.15 → 45 rows, 0.08 → 157.
    decision_gap: float = Field(0.15, ge=0.01, le=1.0)
    # Keep the active test examples verbatim and grow them to the sized
    # target with this flow's answer-key questions (spread over the
    # documents); train / val questions matching a test question are dropped.
    keep_test: bool = False
    # The rows the pre-training retrieval sweep picks on: "val" (then the
    # chosen retrieval is scored once on test) or "test" (older behaviour).
    sweep_split: str = Field("val", pattern="^(val|test)$")
    # Train on a teacher model's answers to exactly the served passages
    # (implies train_with_passages, no withheld rows).
    answers_from_teacher: bool = False
    # The teacher: a cloud pin (the project's saved key) or a synth backend
    # name; unset → the generation backend.
    teacher_backend: str | None = None
    # Drop training answers the local answer judge grades wrong against the
    # pair's reference.
    teacher_filter: bool = True
    # Anthropic Message Batches (half price, slower) for the generation and
    # the teacher calls.
    use_batch: bool = False
    # Direct generation / teacher calls in flight at once (a local server
    # with parallel slots — OLLAMA_NUM_PARALLEL — or a cloud API).
    concurrency: int = Field(1, ge=1, le=8)


@router.get("/documents-to-qa/preview")
async def documents_to_qa_preview(
    project_id: int,
    backend: str | None = None,
    decision_gap: float = 0.15,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Can this project run the flow (enough cleaned passages, a reachable
    generation model), and what would it produce? ``backend``: the model the
    user picked (a ``cloud:<provider>:<model>`` pin counts when the project
    has its key saved)."""
    try:
        if not (0.01 <= float(decision_gap) <= 1.0):
            raise ValueError("decision_gap must be between 0.01 and 1")
        return await preview_documents_qa_flow(db, project_id, backend=backend, decision_gap=float(decision_gap))
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
    # A cloud pin needs the project's saved key: 400 malformed, 402 no key.
    cloud_backend = await _cloud_backend_or_http(db, project_id, req.backend)
    teacher_backend = await _cloud_backend_or_http(db, project_id, req.teacher_backend) if req.answers_from_teacher else None
    if req.use_batch:
        pins = [req.backend] if not req.reuse_existing else []
        if req.answers_from_teacher:
            pins.append(req.teacher_backend or req.backend)
        if not pins or any(not str(pin or "").startswith("cloud:anthropic:") for pin in pins):
            raise HTTPException(400, {
                "error_code": "BATCH_NEEDS_ANTHROPIC",
                "message": "use_batch needs an Anthropic cloud model (cloud:anthropic:<model>) for every model it sends calls to.",
            })
    preview = await documents_to_qa_preview(project_id, req.backend, req.decision_gap, db)
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
                    backend_override=cloud_backend,
                    train=req.train,
                    reuse_existing=req.reuse_existing,
                    passages_check=req.passages_check,
                    train_if_retrieval_ready=req.train_if_retrieval_ready,
                    train_with_passages=req.train_with_passages,
                    passages_withheld_fraction=req.passages_withheld_fraction,
                    decision_gap=req.decision_gap,
                    keep_test=req.keep_test,
                    sweep_split=req.sweep_split,
                    answers_from_teacher=req.answers_from_teacher,
                    teacher_backend=req.teacher_backend,
                    teacher_backend_override=teacher_backend,
                    teacher_filter=req.teacher_filter,
                    use_batch=req.use_batch,
                    concurrency=req.concurrency,
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
            "train_with_passages": req.train_with_passages,
            "decision_gap": req.decision_gap,
            "keep_test": req.keep_test,
            "sweep_split": req.sweep_split,
            "answers_from_teacher": req.answers_from_teacher,
            "teacher_backend": req.teacher_backend,
            "use_batch": req.use_batch,
            "concurrency": req.concurrency,
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
    layout: str = Field("plain", pattern="^(plain|paged|scanned)$")
    words_per_page: int = Field(110, ge=40, le=600)
    chunk_size: int = Field(900, ge=200, le=4000)
    chunk_overlap: int = Field(60, ge=0, le=1000)
    # The prompt (kinds / genres / domain hint) is refused when it carries a
    # value from a benchmark's gold rows (PROMPT_CARRIES_BENCHMARK_VALUES);
    # true runs anyway — the user has seen the list.
    ignore_benchmark_values: bool = False


def seed_prompt_fields(req: "SeedDocumentsRequest") -> dict[str, Any]:
    """The user-written parts of the seed-documents prompt."""
    return {"kinds": req.kinds, "genres": req.genres, "domain_hint": req.domain_hint}


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
    cloud_backend = await _cloud_backend_or_http(db, project_id, req.backend)
    try:
        prompt_check = await refuse_if_contaminated(
            db, project_id, seed_prompt_fields(req), override=req.ignore_benchmark_values,
        )
    except PromptCarriesBenchmarkValues as exc:
        raise HTTPException(400, http_detail(exc)) from exc
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
                    layout=req.layout,
                    words_per_page=req.words_per_page,
                    backend_override=cloud_backend,
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
        params={"project_id": project_id, "count": req.count, "words": req.words, "genres": req.genres, "backend": req.backend, "layout": req.layout,
                "prompt_check": {"clean": prompt_check["clean"], "offending_total": prompt_check["offending_total"],
                                 "overridden": bool(req.ignore_benchmark_values and not prompt_check["clean"])}},
    )
    return serialize_job(job)
