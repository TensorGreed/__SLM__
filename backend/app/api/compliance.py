"""Compliance bundles API: named groups of entity types tagged with the
regimes that care about them (``compliance_bundles_service``)."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services import compliance_bundles_service as bundles
from app.services.span_type_definitions_service import project_definitions

router = APIRouter(prefix="/projects/{project_id}/compliance", tags=["Compliance"])


class ApplyBundlesRequest(BaseModel):
    bundle_ids: list[str] = Field(default_factory=list)
    remove_bundle_ids: list[str] = Field(default_factory=list)


class TypeRegimesRequest(BaseModel):
    """``{type: [regime tags]}`` — replaces the project's map. Tags outside
    the built-in regimes are the project's custom tags."""

    type_regimes: dict[str, list[str]] = Field(default_factory=dict)


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _payload(project: Project) -> dict[str, Any]:
    state = bundles.compliance_state(project.runtime_config)
    return {
        **state,
        "entity_types": bundles.entity_types(state),
        "definitions": project_definitions(project),
        "regimes": [
            {"regime": tag, "label": bundles.regime_label(tag), "custom": tag in state["custom"]}
            for tag in list(bundles.BUILTIN_REGIME_LABELS) + state["custom"]
        ],
    }


@router.get("/bundles")
async def get_bundles(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Built-in bundles (each with its types' starting definitions) and the
    project's applied state."""
    project = await _project(db, project_id)
    payload = _payload(project)
    applied = set(payload["bundles"])
    return {
        "builtin": [{**b, "applied": b["id"] in applied} for b in bundles.list_bundles()],
        **payload,
    }


@router.post("/apply")
async def apply_bundles(project_id: int, data: ApplyBundlesRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Apply (and/or remove) bundles. Adds definitions only for types the
    project has not defined; returns the resulting entity-type list."""
    project = await _project(db, project_id)
    try:
        cfg, result = bundles.apply_bundles(project.runtime_config, data.bundle_ids, data.remove_bundle_ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project.runtime_config = cfg
    await db.commit()
    return {
        **_payload(project),
        "definitions_added": result["definitions_added"],
        "definitions_kept": result["definitions_kept"],
    }


@router.put("/type-regimes")
async def put_type_regimes(project_id: int, data: TypeRegimesRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Replace the type→regimes map (the user's edits, custom tags included)."""
    project = await _project(db, project_id)
    cfg, _ = bundles.set_type_regimes(project.runtime_config, data.type_regimes)
    project.runtime_config = cfg
    await db.commit()
    return _payload(project)
