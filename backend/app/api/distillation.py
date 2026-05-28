"""Knowledge-distillation API routes (Track 1, Epic A).

Slice 1: teacher logit capture. ``POST .../distillation/capture`` validates
the source dataset, starts a background capture job, and returns a task
envelope (202). The frontend polls ``GET .../distillation/tasks/{task_id}``
for live progress + final results — identical contract to the cleaning
background-task endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import Dataset
from app.services.distillation import (
    get_capture_task_status,
    start_capture_task,
)

router = APIRouter(
    prefix="/projects/{project_id}/distillation",
    tags=["Distillation"],
)


class CaptureRequest(BaseModel):
    dataset_id: int
    top_k: int = Field(10, ge=1, le=20)
    teacher_model_name: str | None = None
    limit: int | None = Field(None, ge=1, le=100_000)


@router.post("/capture", status_code=202)
async def start_capture(
    project_id: int,
    req: CaptureRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start a teacher logit-capture job over a source dataset.

    Validates the dataset belongs to the project (404 otherwise) before
    detaching the per-row teacher calls onto the event loop. Returns the
    task envelope immediately.
    """
    result = await db.execute(
        select(Dataset).where(
            Dataset.id == req.dataset_id,
            Dataset.project_id == project_id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            404,
            f"Dataset {req.dataset_id} not found in project {project_id}.",
        )

    task = start_capture_task(
        project_id=project_id,
        dataset_id=req.dataset_id,
        top_k=req.top_k,
        teacher_model_name=req.teacher_model_name,
        limit=req.limit,
    )
    return task.to_dict()


@router.get("/tasks/{task_id}")
async def capture_task_status(
    project_id: int,
    task_id: str,
):
    """Poll a capture job for progress + results.

    404 when the id is unknown (process restart / evicted / never created)
    or belongs to a different project — don't leak cross-project state.
    """
    payload = get_capture_task_status(task_id)
    if payload is None:
        raise HTTPException(404, f"Capture task '{task_id}' not found.")
    if payload.get("project_id") != project_id:
        raise HTTPException(
            404, f"Capture task '{task_id}' not found in this project."
        )
    return payload
