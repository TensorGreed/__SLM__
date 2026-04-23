"""Gold-set annotation workbench API routes (priority.md P10).

Three endpoints:
- `POST /gold-sets/{gold_set_id}/rows/sample` — stratified or random sampling
  from a source dataset into the active draft version, deduping by row hash.
- `PATCH /gold-sets/{gold_set_id}/rows/{row_id}` — update row annotation state.
- `GET /gold-sets/{gold_set_id}/queue` — reviewer queue listing with filters.

The `{gold_set_id}` path param is a `Dataset.id` whose `dataset_type` must be
one of `GOLD_DEV` / `GOLD_TEST`; the service enforces this and returns 400
otherwise.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.gold_set_annotation import GoldSetReviewerQueueStatus
from app.services.gold_workbench_service import (
    list_reviewer_queue,
    sample_rows_from_source,
    update_row,
    _serialize_row,
)


router = APIRouter(prefix="/gold-sets", tags=["Gold Set Workbench"])


_SERVICE_ERROR_STATUS: dict[str, int] = {
    "gold_set_not_found": 404,
    "row_not_found": 404,
    "source_dataset_not_found": 404,
    "not_a_gold_set": 400,
    "source_dataset_wrong_project": 400,
    "target_count_must_be_positive": 400,
    "invalid_strategy": 400,
    "stratify_by_required": 400,
    "status_required": 400,
    "invalid_status": 400,
    "input_must_be_object": 400,
    "expected_must_be_object": 400,
    "labels_must_be_object": 400,
}


def _reraise(reason: str) -> HTTPException:
    status = _SERVICE_ERROR_STATUS.get(reason, 400)
    return HTTPException(status, reason)


class SampleRowsRequest(BaseModel):
    source_dataset_id: int = Field(..., ge=1)
    target_count: int = Field(..., ge=1, le=5000)
    strategy: Literal["random", "stratified"] = "random"
    stratify_by: str | None = Field(default=None, max_length=128)
    seed: int | None = Field(default=None)
    reviewer_id: int | None = Field(default=None, ge=1)
    user_id: int | None = Field(default=None, ge=1)


class RowPatchRequest(BaseModel):
    # Pydantic's `exclude_unset` discrimination lets us tell "field omitted" from
    # "field sent as null" — explicit nulls clear reviewer / labels / etc.
    input: dict[str, Any] | None = None
    expected: dict[str, Any] | None = None
    rationale: str | None = None
    labels: dict[str, Any] | None = None
    status: str | None = None
    reviewer_id: int | None = None


@router.post("/{gold_set_id}/rows/sample", status_code=201)
async def sample_gold_set_rows(
    gold_set_id: int,
    req: SampleRowsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Sample rows from a source dataset into the active draft version."""
    try:
        return await sample_rows_from_source(
            db,
            gold_set_id=gold_set_id,
            source_dataset_id=req.source_dataset_id,
            target_count=req.target_count,
            strategy=req.strategy,
            stratify_by=req.stratify_by,
            seed=req.seed,
            reviewer_id=req.reviewer_id,
            user_id=req.user_id,
        )
    except ValueError as exc:
        raise _reraise(str(exc)) from exc


@router.patch("/{gold_set_id}/rows/{row_id}")
async def patch_gold_set_row(
    gold_set_id: int,
    row_id: int,
    req: RowPatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update row annotation state and sync the reviewer-queue projection."""
    patch_payload = req.model_dump(exclude_unset=True)
    if not patch_payload:
        raise HTTPException(400, "empty_patch")
    try:
        row = await update_row(
            db,
            gold_set_id=gold_set_id,
            row_id=row_id,
            patch=patch_payload,
        )
    except ValueError as exc:
        raise _reraise(str(exc)) from exc
    return _serialize_row(row)


@router.get("/{gold_set_id}/queue")
async def get_gold_set_queue(
    gold_set_id: int,
    reviewer_id: int | None = Query(default=None, ge=1),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List reviewer-queue entries for this gold set, with optional filters."""
    queue_status: GoldSetReviewerQueueStatus | None = None
    if status is not None:
        try:
            queue_status = GoldSetReviewerQueueStatus(str(status).lower())
        except ValueError as exc:
            raise HTTPException(400, "invalid_queue_status") from exc

    try:
        return await list_reviewer_queue(
            db,
            gold_set_id=gold_set_id,
            reviewer_id=reviewer_id,
            status=queue_status,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise _reraise(str(exc)) from exc
