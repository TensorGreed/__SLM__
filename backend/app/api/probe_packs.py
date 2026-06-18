"""Probe Pack API — Coach-stage-2 phase 8.

Mounts at ``/api/projects/{project_id}/probe-pack``:

- ``GET /`` — the platform-authored, recipe-keyed adversarial probe pack
  for the project (the held-out ruler the user did not author). Returns
  an ``applicable=False`` payload when the project has no recipe or no
  pack exists for its task shape yet.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.probe_pack_service import (
    PROBE_GATE_DEFAULT_THRESHOLD,
    get_probe_pack_for_project,
    set_probe_gate,
    set_probe_kind_weights,
)

router = APIRouter(prefix="/projects/{project_id}/probe-pack", tags=["Probe Pack"])


class ProbeGateConfig(BaseModel):
    """Phase 13 — optional probe gate config. Off by default; when
    enabled, ``probe_pass_rate ≥ min_pass_rate`` becomes an eval gate."""
    enabled: bool = False
    min_pass_rate: float = Field(default=PROBE_GATE_DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    required: bool = True


class ProbeKindWeightsBody(BaseModel):
    """Phase 22 — per-project probe-kind weight overrides. Unknown kinds
    and out-of-range values are dropped server-side."""
    weights: dict[str, float] = Field(default_factory=dict)


@router.get("")
async def get_probe_pack(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the platform-authored probe pack for a project's recipe."""
    try:
        return await get_probe_pack_for_project(db, project_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.put("/gate")
async def put_probe_gate(
    project_id: int,
    body: ProbeGateConfig,
    db: AsyncSession = Depends(get_db),
):
    """Enable/disable + configure the optional probe gate for a project."""
    try:
        return await set_probe_gate(
            db,
            project_id,
            enabled=body.enabled,
            min_pass_rate=body.min_pass_rate,
            required=body.required,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.put("/kind-weights")
async def put_probe_kind_weights(
    project_id: int,
    body: ProbeKindWeightsBody,
    db: AsyncSession = Depends(get_db),
):
    """Set per-kind weight overrides; returns the effective (merged) map."""
    try:
        return await set_probe_kind_weights(db, project_id, body.weights)
    except ValueError as e:
        raise HTTPException(404, str(e))
