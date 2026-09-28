"""Task shape — detect what kind of task the project's data is, and confirm it.

- ``GET  /api/projects/{id}/task-shape?intent=…`` — rank task shapes for the
  project's current data (plus what's confirmed today).
- ``POST /api/projects/{id}/task-shape/confirm`` — persist the choice where
  dataset prep / training / eval read it and snapshot the matching recipe.

The import wizard gets the same detector result inline from
``/dataset-import/introspect`` (``task_shape``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services.task_shape_service import (
    TASK_SHAPES,
    confirm_task_shape,
    detect_task_shape,
    sample_project_rows,
)

router = APIRouter(prefix="/projects/{project_id}/task-shape", tags=["Task Shape"])


class ConfirmTaskShapeRequest(BaseModel):
    task_profile: str = Field(..., min_length=1, max_length=64)
    field_mapping: dict[str, str] | None = None
    adapter_config: dict[str, Any] | None = None


def _catalog() -> list[dict[str, Any]]:
    return [
        {"task_profile": profile, "label": spec["label"], "description": spec["description"]}
        for profile, spec in TASK_SHAPES.items()
    ]


def _confirmed(project: Project) -> dict[str, Any] | None:
    preset = project.dataset_adapter_preset if isinstance(project.dataset_adapter_preset, dict) else {}
    profile = preset.get("task_profile")
    if not profile:
        return None
    spec = TASK_SHAPES.get(profile)
    return {
        "task_profile": profile,
        "label": spec["label"] if spec else profile,
        "adapter_id": preset.get("adapter_id"),
    }


@router.get("")
async def get_task_shape(
    project_id: int,
    intent: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    rows = await sample_project_rows(db, project_id)
    detection = detect_task_shape(rows, intent=intent) if rows else None
    return {
        "project_id": project_id,
        "confirmed": _confirmed(project),
        "detection": detection,
        "catalog": _catalog(),
    }


@router.post("/confirm")
async def post_confirm_task_shape(
    project_id: int,
    req: ConfirmTaskShapeRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    try:
        return await confirm_task_shape(
            db,
            project_id,
            req.task_profile,
            adapter_config=req.adapter_config,
            field_mapping=req.field_mapping,
        )
    except ValueError as exc:
        message = str(exc)
        raise HTTPException(404 if "not found" in message.lower() else 400, message) from exc
