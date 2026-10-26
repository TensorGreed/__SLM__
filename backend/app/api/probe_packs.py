"""Probe Pack API — Coach-stage-2 phase 8.

Mounts at ``/api/projects/{project_id}/probe-pack``:

- ``GET /`` — the platform-authored, recipe-keyed adversarial probe pack
  for the project (the held-out ruler the user did not author). Returns
  an ``applicable=False`` payload when the project has no recipe or no
  pack exists for its task shape yet.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services.probe_pack_service import (
    PROBE_GATE_DEFAULT_THRESHOLD,
    UserProbePackError,
    get_project_user_probe_packs,
    parse_user_probe_pack_text,
    restore_user_probe_packs_version,
    set_project_user_probe_packs,
    user_probe_packs_status,
    validate_user_probe_pack_set,
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


# ── User probe packs (your own checks) ──────────────────────────────────


class UserProbePacksBody(BaseModel):
    content: str | None = Field(default=None, description="One pack as YAML or JSON text.")
    packs: list[dict[str, Any]] | None = Field(default=None, description="Pack documents as JSON; send either this or content.")
    mode: Literal["replace", "merge"] | None = Field(
        default=None,
        description="replace = the sent packs become the whole set; merge = packs with the same id are replaced, "
        "other packs are kept. Default: merge for 'content', replace for 'packs'.",
    )


class UserProbeRestoreBody(BaseModel):
    version: str


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _invalid(errors: list[dict[str, Any]]) -> HTTPException:
    first = errors[0] if errors else {"message": "Invalid probe pack."}
    suffix = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    return HTTPException(422, {
        "error_code": "INVALID_PROBE_PACK",
        "message": f"{first['message']}{suffix}",
        "metadata": {"errors": errors},
    })


def _docs_from_body(body: UserProbePacksBody) -> tuple[list[Any], list[dict[str, Any]]]:
    if (body.content is None) == (body.packs is None):
        return [], [{"code": "bad_request", "path": "", "message": "Send exactly one of 'content' or 'packs'."}]
    if body.packs is not None:
        return list(body.packs), []
    try:
        return [parse_user_probe_pack_text(body.content or "")], []
    except ValueError as exc:
        return [], [{"code": "parse_error", "path": "", "message": str(exc)}]


@router.get("/user")
async def get_user_probe_packs(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return user_probe_packs_status(await _project(db, project_id))


@router.put("/user")
async def put_user_probe_packs(project_id: int, body: UserProbePacksBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Save user probe packs. ``mode`` merge (default for ``content``) replaces the
    pack with the same id and keeps the others; replace swaps the whole set. 422 ``INVALID_PROBE_PACK`` with
    ``metadata.errors`` when anything fails -- nothing is saved."""
    project = await _project(db, project_id)
    docs, errors = _docs_from_body(body)
    if errors:
        raise _invalid(errors)
    mode = body.mode or ("merge" if body.content is not None else "replace")
    if mode == "merge":
        incoming_ids = {d.get("id") for d in docs if isinstance(d, dict)}
        kept = [p for p in get_project_user_probe_packs(project) if p.get("id") not in incoming_ids]
        docs = [*kept, *docs]
    try:
        set_project_user_probe_packs(project, docs, source={"kind": "manual"})
    except UserProbePackError as exc:
        raise _invalid(exc.errors)
    await db.commit()
    return user_probe_packs_status(project)


@router.post("/user/restore")
async def restore_user_probe_packs(project_id: int, body: UserProbeRestoreBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Restore a remembered full snapshot (history entries with ``restorable``)."""
    project = await _project(db, project_id)
    try:
        restore_user_probe_packs_version(project, body.version)
    except LookupError as exc:
        raise HTTPException(404, {"error_code": "PROBE_PACK_VERSION_NOT_FOUND", "message": str(exc)})
    except UserProbePackError as exc:
        raise _invalid(exc.errors)
    await db.commit()
    return user_probe_packs_status(project)


@router.post("/user/validate")
async def validate_user_probe_packs(project_id: int, body: UserProbePacksBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    await _project(db, project_id)
    docs, errors = _docs_from_body(body)
    packs: list[dict[str, Any]] = []
    if not errors:
        packs, errors = validate_user_probe_pack_set(docs)
    return {
        "valid": not errors,
        "errors": errors,
        "pack_count": len(packs),
        "probe_count": sum(len(p.get("probes") or []) for p in packs),
        "critical_count": sum(1 for p in packs for x in p.get("probes") or [] if x.get("critical")),
        "packs": packs if not errors else [],
    }


@router.delete("/user")
async def delete_user_probe_packs(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _project(db, project_id)
    set_project_user_probe_packs(project, [], source={"kind": "manual"})
    await db.commit()
    return user_probe_packs_status(project)
