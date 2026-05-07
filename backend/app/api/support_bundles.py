"""Support-bundle API (priority.md P34, Wave G).

Routes:

- ``POST /api/projects/{id}/support-bundle`` — generate a redacted
  support bundle (zip of sectioned JSON). Returns metadata + a
  ``download_url`` containing a single-use-style ``token`` query.
- ``GET  /api/projects/{id}/support-bundles`` — list bundles created
  for a project (newest first).
- ``GET  /api/support-bundles/{bundle_uid}/download?token=...`` —
  stream the zip. Validates token + expiry.

This is a thin substitute for true signed URLs (no object storage in
this codebase). The ``bundle_uid`` is unguessable hex and the token is
checked constant-time.

Stable reason codes:
- ``project_not_found`` (404)
- ``support_bundle_not_found`` (404)
- ``support_bundle_invalid_token`` (403)
- ``support_bundle_expired`` (410)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.support_bundle_service import (
    create_support_bundle,
    list_support_bundles,
    resolve_bundle_for_download,
)


router = APIRouter(prefix="/support-bundles", tags=["SupportBundles"])
project_router = APIRouter(
    prefix="/projects/{project_id}", tags=["SupportBundles"]
)


_NOT_FOUND_CODES = {"project_not_found", "support_bundle_not_found"}


def _raise_for(exc: ValueError) -> None:
    detail = str(exc) or "support_bundle_error"
    head = detail.split(":", 1)[0]
    if head in _NOT_FOUND_CODES:
        raise HTTPException(404, detail=detail) from exc
    if head == "support_bundle_invalid_token":
        raise HTTPException(403, detail=detail) from exc
    if head == "support_bundle_expired":
        raise HTTPException(410, detail=detail) from exc
    raise HTTPException(400, detail=detail) from exc


class CreateBundleRequest(BaseModel):
    actor: str | None = Field(default=None, max_length=128)
    ttl_seconds: int | None = Field(default=None, ge=60, le=30 * 24 * 3600)


@project_router.post("/support-bundle")
async def create_bundle(
    project_id: int,
    req: CreateBundleRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    payload = req or CreateBundleRequest()
    try:
        return await create_support_bundle(
            db,
            project_id=project_id,
            actor=payload.actor,
            ttl_seconds=payload.ttl_seconds or 24 * 3600,
        )
    except ValueError as exc:
        _raise_for(exc)


@project_router.get("/support-bundles")
async def list_bundles_for_project(
    project_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_support_bundles(
            db, project_id=project_id, limit=limit
        )
    except ValueError as exc:
        _raise_for(exc)


@router.get("/{bundle_uid}/download")
async def download_bundle(
    bundle_uid: str,
    token: str = Query(..., min_length=8, max_length=128),
    db: AsyncSession = Depends(get_db),
):
    try:
        row, path = await resolve_bundle_for_download(
            db, bundle_uid=bundle_uid, token=token
        )
    except ValueError as exc:
        _raise_for(exc)
    return FileResponse(
        path=str(path),
        media_type="application/zip",
        filename=f"brewslm-support-bundle-{row.project_id}-{row.bundle_uid[:8]}.zip",
    )
