"""Demo projects API — pre-loaded showcases for new ML engineers.

Two routes:

- ``GET  /api/demo-projects`` — catalog of available demo archetypes
  read from ``backend/data/demo_samples/``.
- ``POST /api/demo-projects/{slug}`` — seed (or fetch existing) the
  demo project for that slug. Idempotent — re-posting returns the same
  project record so the "Try a demo" tile click is safe to repeat.

Stable reason codes:

- ``demo_slug_unknown`` (404)
- ``demo_manifest_invalid`` (400)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.demo_project_service import (
    list_demo_archetypes,
    seed_demo_project,
)


router = APIRouter(prefix="/demo-projects", tags=["DemoProjects"])


_NOT_FOUND_CODES = {"demo_slug_unknown"}


def _raise_for(exc: ValueError) -> None:
    detail = str(exc) or "demo_project_error"
    head = detail.split(":", 1)[0]
    if head in _NOT_FOUND_CODES:
        raise HTTPException(404, detail=detail) from exc
    raise HTTPException(400, detail=detail) from exc


@router.get("")
async def list_demos():
    return {"archetypes": list_demo_archetypes()}


@router.post("/{slug}")
async def seed_demo(slug: str, db: AsyncSession = Depends(get_db)):
    try:
        project, summary = await seed_demo_project(db, slug)
    except ValueError as exc:
        _raise_for(exc)
        return  # unreachable; pacifies type-checker
    await db.commit()
    return {
        "summary": summary,
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "status": project.status.value if project.status else None,
            "beginner_mode": project.beginner_mode,
            "target_profile_id": project.target_profile_id,
            "training_preferred_plan_profile": project.training_preferred_plan_profile,
            "evaluation_preferred_pack_id": project.evaluation_preferred_pack_id,
        },
    }
