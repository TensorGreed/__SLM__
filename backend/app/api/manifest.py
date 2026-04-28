"""Pipeline-as-code manifest API (priority.md P21 + P22).

Read side (P21):
- ``GET /api/manifest/schema`` — JSON Schema for tooling / IDE integration.
- ``GET /api/projects/{id}/manifest/export`` — render the project's
  current state as a ``brewslm.yaml`` body (``format=yaml`` by default,
  ``format=json`` for tooling that prefers the parsed shape).
- ``POST /api/projects/{id}/manifest/apply-plan`` — diff preview against
  the project; identical contract to the new ``manifest/diff`` route.

Write / validate side (P22):
- ``POST /api/manifest/validate`` — Pydantic + cross-reference validation
  with structured ``ManifestValidationIssue`` rows (`code`, `severity`,
  `field`, `message`, `actionable_fix`).
- ``POST /api/projects/{id}/manifest/diff`` — explicit alias for the
  apply-plan deserializer.
- ``POST /api/projects/{id}/manifest/apply`` and
  ``POST /api/manifest/apply`` — apply the manifest against an existing
  project (``project_id``) or create a new one (no path id), with
  ``plan_only=true|false`` to short-circuit writes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.brewslm_manifest import BrewslmManifest, ManifestApplyPlan
from app.services.brewslm_manifest_service import (
    deserialize_manifest_to_apply_plan,
    manifest_from_yaml,
    manifest_to_yaml,
    serialize_project_to_manifest,
)
from app.services.manifest_apply_service import (
    ManifestApplyResult,
    ManifestValidationResult,
    apply_manifest,
    diff_manifest_against_project,
    validate_manifest,
)


router = APIRouter(prefix="/manifest", tags=["Manifest"])
project_router = APIRouter(prefix="/projects/{project_id}/manifest", tags=["Manifest"])


@router.get("/schema")
async def get_manifest_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``brewslm.yaml``.

    Pydantic generates this off the ``BrewslmManifest`` model, so the
    schema and the validator stay in lockstep — there's no separately
    maintained schema document to drift.
    """
    return BrewslmManifest.model_json_schema()


@project_router.get("/export")
async def export_project_manifest(
    project_id: int,
    format: str = Query("yaml", pattern="^(yaml|json)$"),
    db: AsyncSession = Depends(get_db),
):
    try:
        manifest = await serialize_project_to_manifest(db, project_id=project_id)
    except ValueError as exc:
        if str(exc) == "project_not_found":
            raise HTTPException(404, detail="project_not_found") from exc
        raise

    if format == "json":
        return manifest.model_dump(mode="json")

    body = manifest_to_yaml(manifest)
    return PlainTextResponse(content=body, media_type="application/x-yaml")


class ManifestApplyPlanRequest(BaseModel):
    """POST body for the apply-plan preview.

    Callers may submit the manifest as a parsed JSON object
    (``manifest``) or as a raw YAML body (``manifest_yaml``). Tests +
    the CLI use the YAML form; the frontend uses the JSON form.
    """

    manifest: dict[str, Any] | None = None
    manifest_yaml: str | None = None


@project_router.post("/apply-plan", response_model=ManifestApplyPlan)
async def post_manifest_apply_plan(
    project_id: int,
    payload: ManifestApplyPlanRequest,
    db: AsyncSession = Depends(get_db),
) -> ManifestApplyPlan:
    if payload.manifest is None and not payload.manifest_yaml:
        raise HTTPException(400, detail="manifest_required")

    try:
        if payload.manifest_yaml is not None:
            manifest = manifest_from_yaml(payload.manifest_yaml)
        else:
            manifest = BrewslmManifest.model_validate(payload.manifest)
    except ValueError as exc:
        raise HTTPException(400, detail=f"manifest_invalid:{exc}") from exc

    try:
        plan = await deserialize_manifest_to_apply_plan(
            db, manifest=manifest, project_id=project_id
        )
    except ValueError as exc:
        if str(exc) == "project_not_found":
            raise HTTPException(404, detail="project_not_found") from exc
        raise
    return plan


class ManifestApplyPlanNewProjectRequest(BaseModel):
    manifest: dict[str, Any] | None = None
    manifest_yaml: str | None = Field(default=None)


@router.post("/apply-plan", response_model=ManifestApplyPlan)
async def post_new_project_apply_plan(
    payload: ManifestApplyPlanNewProjectRequest,
    db: AsyncSession = Depends(get_db),
) -> ManifestApplyPlan:
    """Preview the apply-plan for a brand-new project (no existing project_id)."""
    if payload.manifest is None and not payload.manifest_yaml:
        raise HTTPException(400, detail="manifest_required")
    try:
        if payload.manifest_yaml is not None:
            manifest = manifest_from_yaml(payload.manifest_yaml)
        else:
            manifest = BrewslmManifest.model_validate(payload.manifest)
    except ValueError as exc:
        raise HTTPException(400, detail=f"manifest_invalid:{exc}") from exc

    return await deserialize_manifest_to_apply_plan(db, manifest=manifest, project_id=None)


# -- P22 surfaces -----------------------------------------------------------


def _parse_manifest_payload(
    *, manifest: dict[str, Any] | None, manifest_yaml: str | None
) -> BrewslmManifest:
    """Shared body-parsing helper for the P22 endpoints.

    Raises :class:`HTTPException` so the API layer doesn't have to
    repeat the same try/except dance in every route. ``ValueError`` from
    Pydantic / PyYAML is mapped to 400 ``manifest_invalid:...`` so the
    CLI / frontend can show the exception text verbatim.
    """
    if manifest is None and not manifest_yaml:
        raise HTTPException(400, detail="manifest_required")
    try:
        if manifest_yaml is not None:
            return manifest_from_yaml(manifest_yaml)
        return BrewslmManifest.model_validate(manifest)
    except ValueError as exc:
        raise HTTPException(400, detail=f"manifest_invalid:{exc}") from exc


class ManifestValidateRequest(BaseModel):
    manifest: dict[str, Any] | None = None
    manifest_yaml: str | None = None


@router.post("/validate", response_model=ManifestValidationResult)
async def post_manifest_validate(
    payload: ManifestValidateRequest,
    db: AsyncSession = Depends(get_db),
) -> ManifestValidationResult:
    """Validate a manifest against the schema + live catalogs.

    Schema-level errors (Pydantic ``ValidationError``) come back as a
    400 ``manifest_invalid:...`` so callers can distinguish "this isn't
    a valid manifest at all" from "this manifest references resources
    that don't exist in the catalog" (200 with errors[]).
    """
    manifest = _parse_manifest_payload(
        manifest=payload.manifest, manifest_yaml=payload.manifest_yaml
    )
    return await validate_manifest(db, manifest=manifest)


class ManifestDiffRequest(BaseModel):
    manifest: dict[str, Any] | None = None
    manifest_yaml: str | None = None


@project_router.post("/diff", response_model=ManifestApplyPlan)
async def post_manifest_diff(
    project_id: int,
    payload: ManifestDiffRequest,
    db: AsyncSession = Depends(get_db),
) -> ManifestApplyPlan:
    manifest = _parse_manifest_payload(
        manifest=payload.manifest, manifest_yaml=payload.manifest_yaml
    )
    try:
        return await diff_manifest_against_project(
            db, manifest=manifest, project_id=project_id
        )
    except ValueError as exc:
        if str(exc) == "project_not_found":
            raise HTTPException(404, detail="project_not_found") from exc
        raise


class ManifestApplyRequest(BaseModel):
    manifest: dict[str, Any] | None = None
    manifest_yaml: str | None = None
    plan_only: bool = False


@project_router.post("/apply", response_model=ManifestApplyResult)
async def post_manifest_apply(
    project_id: int,
    payload: ManifestApplyRequest,
    db: AsyncSession = Depends(get_db),
) -> ManifestApplyResult:
    manifest = _parse_manifest_payload(
        manifest=payload.manifest, manifest_yaml=payload.manifest_yaml
    )
    try:
        return await apply_manifest(
            db,
            manifest=manifest,
            project_id=project_id,
            plan_only=bool(payload.plan_only),
        )
    except ValueError as exc:
        if str(exc) == "project_not_found":
            raise HTTPException(404, detail="project_not_found") from exc
        raise


@router.post("/apply", response_model=ManifestApplyResult)
async def post_new_project_apply(
    payload: ManifestApplyRequest,
    db: AsyncSession = Depends(get_db),
) -> ManifestApplyResult:
    """Apply a manifest to create a new project (no existing project_id)."""
    manifest = _parse_manifest_payload(
        manifest=payload.manifest, manifest_yaml=payload.manifest_yaml
    )
    return await apply_manifest(
        db,
        manifest=manifest,
        project_id=None,
        plan_only=bool(payload.plan_only),
    )
