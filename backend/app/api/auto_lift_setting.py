"""Per-project switch for the automatic base-vs-fine-tuned lift check.

``GET/PUT /api/projects/{id}/settings/auto-lift-eval`` read / write
``Project.runtime_config["auto_lift_eval"]`` (``true`` / ``false``; ``null`` clears it so the
project follows the default, which is on). The global ``AUTO_LIFT_EVAL_ENABLED`` setting still wins
when it is false: the response says so (``global_enabled`` / ``effective`` / ``reason``). The check
a user starts by hand is never affected.

``GET/PUT /api/projects/{id}/settings/null-as-absent`` read / write the opt-in "null counts as absent"
JSON scoring setting (``null_scoring``; ``runtime_config["json_scoring"]["null_as_absent"]``, default off).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services.post_training_eval_service import auto_lift_setting

router = APIRouter(prefix="/projects/{project_id}/settings", tags=["Project settings"])


class AutoLiftEvalBody(BaseModel):
    enabled: bool | None = None


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


@router.get("/auto-lift-eval")
async def get_auto_lift_eval(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return auto_lift_setting(await _project(db, project_id))


@router.put("/auto-lift-eval")
async def put_auto_lift_eval(
    project_id: int, body: AutoLiftEvalBody, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """``{enabled: true|false|null}``; null removes the project's own setting (follow the default)."""
    project = await _project(db, project_id)
    cfg = dict(project.runtime_config or {})
    if body.enabled is None:
        cfg.pop("auto_lift_eval", None)
    else:
        cfg["auto_lift_eval"] = body.enabled
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return auto_lift_setting(project)


class NullAsAbsentBody(BaseModel):
    enabled: bool | None = None


@router.get("/null-as-absent")
async def get_null_as_absent(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    from app.services import null_scoring

    return {"enabled": null_scoring.project_setting((await _project(db, project_id)).runtime_config)}


@router.put("/null-as-absent")
async def put_null_as_absent(
    project_id: int, body: NullAsAbsentBody, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """``{enabled: true|false|null}``: true turns the rule on for this project's evals (a run may still
    override it); false / null remove it (the default, off)."""
    from app.services import null_scoring

    project = await _project(db, project_id)
    project.runtime_config = null_scoring.with_project_setting(project.runtime_config, body.enabled)
    await db.commit()
    await db.refresh(project)
    return {"enabled": null_scoring.project_setting(project.runtime_config)}
