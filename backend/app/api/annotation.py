"""Annotation foundation API (Story 1.1).

Endpoints (mounted under ``/api/projects/{project_id}/label-jobs``):

- ``POST   /``                       — create a label job.
- ``GET    /``                       — list jobs for the project.
- ``GET    /{job_id}``               — job detail + stats.
- ``PATCH  /{job_id}``               — update mutable fields.
- ``DELETE /{job_id}``               — drop job + cascade rows.
- ``POST   /{job_id}/seed-from-dataset`` — seed N work units from a dataset.
- ``POST   /{job_id}/next-row``      — assign one unlabeled row to a user.
- ``POST   /{job_id}/rows/{row_id}/submit`` — persist the reviewer label.

The router is project-scoped so list / detail queries can be unioned
with the project's own RBAC layer (the global ``API_DEPENDENCIES``
wrapper already authenticates).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import DatasetType
from app.models.label_job import (
    KNOWN_JOB_STATUSES,
    KNOWN_LABEL_TYPES,
)
from app.services.annotation.promotion import promote_labeled_rows
from app.services.annotation_service import (
    assign_next,
    create_job,
    delete_job,
    get_job,
    job_stats,
    job_to_dict,
    list_jobs,
    row_to_dict,
    seed_rows_from_dataset,
    skip_row,
    submit_label,
    update_job_fields,
)


router = APIRouter(
    prefix="/projects/{project_id}/label-jobs", tags=["Annotation"]
)


# ── Request models ───────────────────────────────────────────────────


class JobCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    label_type: str = Field(..., min_length=1, max_length=32)
    label_schema: dict[str, Any] = Field(default_factory=dict)
    instructions: str | None = Field(default=None, max_length=10_000)
    target_rows: int | None = Field(default=None, ge=1, le=1_000_000)


class JobUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    instructions: str | None = Field(default=None, max_length=10_000)
    status: str | None = Field(default=None, min_length=1, max_length=32)
    target_rows: int | None = Field(default=None, ge=0, le=1_000_000)


class SeedFromDatasetRequest(BaseModel):
    dataset_id: int = Field(..., ge=1)
    n: int = Field(..., ge=1, le=100_000)


class NextRowRequest(BaseModel):
    user_id: int | None = Field(
        default=None,
        description="Reviewer being assigned the row. Optional in "
        "auth-disabled local-dev setups; required when AUTH_ENABLED.",
    )


class SubmitLabelRequest(BaseModel):
    label_payload: dict[str, Any] = Field(...)
    reviewer_notes: str | None = Field(default=None, max_length=10_000)


class PromoteLabelsRequest(BaseModel):
    target_dataset_type: str = Field(
        default="synthetic",
        description=(
            "Where promoted rows land. ``synthetic`` (default) routes "
            "classification + span rows into the project's synthetic "
            "dataset and preference-pair rows into the alignment "
            "preference file. ``gold_dev`` treats the labels as gold "
            "eval ground truth."
        ),
    )


# ── Error translation ────────────────────────────────────────────────


_ERROR_CODES: dict[str, tuple[int, str]] = {
    "project_not_found": (404, "Project not found."),
    "label_job_not_found": (404, "Label job not found."),
    "label_row_not_found": (404, "Label row not found."),
    "dataset_not_found": (404, "Dataset not found."),
    "dataset_project_mismatch": (
        400,
        "Dataset belongs to a different project.",
    ),
    "dataset_file_missing": (
        400,
        "Dataset file is missing on disk; cannot seed rows.",
    ),
    "job_name_required": (400, "Job name is required."),
    "job_name_too_long": (400, "Job name is too long (max 120 chars)."),
    "label_payload_required": (
        400,
        "label_payload must be a non-empty object.",
    ),
    "label_row_already_labeled": (
        409,
        "Row already labeled; cannot skip after submit.",
    ),
    "seed_n_must_be_positive": (400, "n must be >= 1."),
    "invalid_target_dataset_type": (
        400,
        "target_dataset_type must be 'synthetic' or 'gold_dev'.",
    ),
}


def _translate_value_error(exc: ValueError) -> HTTPException:
    code = str(exc)
    if code in _ERROR_CODES:
        status, detail = _ERROR_CODES[code]
        return HTTPException(status, detail)
    if code.startswith("invalid_label_type:"):
        bad = code.split(":", 1)[1]
        return HTTPException(
            400,
            f"Unknown label_type {bad!r}. Allowed: "
            f"{sorted(KNOWN_LABEL_TYPES)}",
        )
    if code.startswith("invalid_job_status:"):
        bad = code.split(":", 1)[1]
        return HTTPException(
            400,
            f"Unknown status {bad!r}. Allowed: {sorted(KNOWN_JOB_STATUSES)}",
        )
    if code.startswith("invalid_target_dataset_type:"):
        bad = code.split(":", 1)[1]
        return HTTPException(
            400,
            f"Unknown target_dataset_type {bad!r}. Allowed: synthetic, gold_dev.",
        )
    return HTTPException(400, code)


# ── Endpoints ────────────────────────────────────────────────────────


@router.post("/", status_code=201)
async def create_label_job(
    project_id: int,
    req: JobCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        job = await create_job(
            db,
            project_id=project_id,
            name=req.name,
            label_type=req.label_type,
            label_schema=req.label_schema,
            instructions=req.instructions,
            target_rows=req.target_rows,
        )
    except ValueError as exc:
        raise _translate_value_error(exc) from exc
    await db.commit()
    return job_to_dict(job)


@router.get("/")
async def list_label_jobs(
    project_id: int, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    rows = await list_jobs(db, project_id=project_id)
    return {"jobs": [job_to_dict(job) for job in rows]}


@router.get("/{job_id}")
async def get_label_job(
    project_id: int,
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        stats = await job_stats(db, project_id=project_id, job_id=job_id)
    except ValueError as exc:
        raise _translate_value_error(exc) from exc
    job = await get_job(db, project_id=project_id, job_id=job_id)
    # job is non-None — stats above would have raised otherwise.
    payload = job_to_dict(job)
    payload["stats"] = stats
    return payload


@router.patch("/{job_id}")
async def patch_label_job(
    project_id: int,
    job_id: int,
    req: JobUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise HTTPException(404, "Label job not found.")
    try:
        update_job_fields(
            job,
            name=req.name,
            instructions=req.instructions,
            status=req.status,
            target_rows=req.target_rows,
        )
    except ValueError as exc:
        raise _translate_value_error(exc) from exc
    await db.flush()
    await db.commit()
    return job_to_dict(job)


@router.delete("/{job_id}", status_code=204)
async def delete_label_job(
    project_id: int,
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    deleted = await delete_job(db, project_id=project_id, job_id=job_id)
    if not deleted:
        raise HTTPException(404, "Label job not found.")
    await db.commit()


@router.post("/{job_id}/seed-from-dataset")
async def seed_from_dataset(
    project_id: int,
    job_id: int,
    req: SeedFromDatasetRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        added = await seed_rows_from_dataset(
            db,
            project_id=project_id,
            job_id=job_id,
            dataset_id=req.dataset_id,
            n=req.n,
        )
    except ValueError as exc:
        raise _translate_value_error(exc) from exc
    await db.commit()
    return {"seeded": added}


@router.post("/{job_id}/next-row")
async def next_row(
    project_id: int,
    job_id: int,
    req: NextRowRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        row = await assign_next(
            db,
            project_id=project_id,
            job_id=job_id,
            user_id=req.user_id,
        )
    except ValueError as exc:
        raise _translate_value_error(exc) from exc
    await db.commit()
    if row is None:
        return {"row": None, "queue_empty": True}
    return {"row": row_to_dict(row), "queue_empty": False}


@router.post("/{job_id}/rows/{row_id}/skip")
async def skip_row_endpoint(
    project_id: int,
    job_id: int,
    row_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        row = await skip_row(
            db, project_id=project_id, job_id=job_id, row_id=row_id
        )
    except ValueError as exc:
        raise _translate_value_error(exc) from exc
    await db.commit()
    return row_to_dict(row)


@router.post("/{job_id}/rows/{row_id}/submit")
async def submit_row_label(
    project_id: int,
    job_id: int,
    row_id: int,
    req: SubmitLabelRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        row = await submit_label(
            db,
            project_id=project_id,
            job_id=job_id,
            row_id=row_id,
            label_payload=req.label_payload,
            reviewer_notes=req.reviewer_notes,
        )
    except ValueError as exc:
        raise _translate_value_error(exc) from exc
    await db.commit()
    return row_to_dict(row)


@router.post("/{job_id}/promote")
async def promote_job_labels(
    project_id: int,
    job_id: int,
    req: PromoteLabelsRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Story 1.6 — promote every labeled, unpromoted row in ``job_id``
    into the project's synthetic / gold / alignment dataset. Idempotent."""
    try:
        target_type = DatasetType(req.target_dataset_type)
    except ValueError as exc:
        raise _translate_value_error(
            ValueError("invalid_target_dataset_type")
        ) from exc
    try:
        result = await promote_labeled_rows(
            db,
            project_id=project_id,
            job_id=job_id,
            target_dataset_type=target_type,
        )
    except ValueError as exc:
        raise _translate_value_error(exc) from exc
    await db.commit()
    return result
