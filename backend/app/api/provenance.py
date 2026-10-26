"""Licence & provenance ledger API (see ``services/provenance_service``).

- ``GET   /api/projects/{id}/provenance/sources`` - every source with its
  licence, usage, document / row counts and what reached the prepared splits.
- ``PATCH /api/projects/{id}/provenance/sources`` - bulk edit licence /
  licence verification / usage of the listed sources.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.exceptions import ProjectNotFoundError, SLMError
from app.models.project import Project
from app.security import get_request_principal
from app.services import provenance_service as prov

router = APIRouter(prefix="/projects/{project_id}/provenance", tags=["Provenance"])


class SourceUpdateRequest(BaseModel):
    source_ids: list[int] = Field(..., min_length=1, max_length=500)
    # Absent = leave alone; null / "" / "unknown" = clear (licence unknown).
    licence: str | None = Field(default=None, max_length=255)
    licence_verified: bool | None = None
    usage: str | None = Field(default=None, max_length=8)


async def _require_project(db: AsyncSession, project_id: int) -> None:
    found = await db.execute(select(Project.id).where(Project.id == project_id))
    if found.scalar_one_or_none() is None:
        raise ProjectNotFoundError(project_id)


@router.get("/sources")
async def list_provenance_sources(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    await _require_project(db, project_id)
    return await prov.list_sources(db, project_id)


@router.patch("/sources")
async def update_provenance_sources(
    project_id: int,
    data: SourceUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    await _require_project(db, project_id)
    given = data.model_fields_set
    kwargs: dict[str, Any] = {}
    if "licence" in given:
        kwargs["licence"] = data.licence
    if "licence_verified" in given and data.licence_verified is not None:
        kwargs["licence_verified"] = data.licence_verified
    if "usage" in given and data.usage is not None:
        kwargs["usage"] = data.usage
    principal = get_request_principal(request)
    try:
        await prov.update_sources(
            db, project_id, data.source_ids, user=principal.username if principal else None, **kwargs
        )
    except prov.ProvenanceError as exc:
        raise SLMError(
            status_code=422, error_code="PROVENANCE_INVALID", stage="ingestion", message=str(exc),
            actionable_fix="Use usage any, train, eval or rag.",
        ) from exc
    except prov.SourceNotFoundError as exc:
        raise SLMError(
            status_code=404, error_code="PROVENANCE_SOURCE_NOT_FOUND", stage="ingestion", message=str(exc),
            actionable_fix="Reload the source list and try again.",
        ) from exc
    except prov.UsageLockedError as exc:
        raise SLMError(
            status_code=409, error_code="PROVENANCE_USAGE_LOCKED", stage="ingestion", message=str(exc),
            actionable_fix="Leave the usage of predefined-split sources alone; edit the licence only.",
            metadata={"source_ids": exc.source_ids},
        ) from exc
    listing = await prov.list_sources(db, project_id)
    wanted = set(data.source_ids)
    return {**listing, "sources": [s for s in listing["sources"] if s["id"] in wanted]}
