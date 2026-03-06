"""Typed artifact registry API routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services.artifact_registry_service import (
    get_latest_artifact,
    list_artifacts,
    list_latest_artifact_keys,
    publish_artifact,
    publish_artifact_batch,
    serialize_artifact,
)

router = APIRouter(prefix="/projects/{project_id}/artifacts", tags=["Artifacts"])


class ArtifactPublishRequest(BaseModel):
    artifact_key: str = Field(..., min_length=1, max_length=255)
    uri: str | None = Field(default=None, max_length=2048)
    schema_ref: str | None = Field(default=None, max_length=255)
    producer_stage: str | None = Field(default=None, max_length=64)
    producer_run_id: str | None = Field(default=None, max_length=128)
    producer_step_id: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["materialized", "failed"] = "materialized"


class ArtifactBatchPublishRequest(BaseModel):
    artifacts: list[ArtifactPublishRequest] = Field(default_factory=list)


async def _ensure_project_exists(db: AsyncSession, project_id: int) -> None:
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")


@router.post("/publish", status_code=201)
async def publish_single_artifact(
    project_id: int,
    req: ArtifactPublishRequest,
    db: AsyncSession = Depends(get_db),
):
    """Publish one new artifact version for a project."""
    await _ensure_project_exists(db, project_id)
    try:
        record = await publish_artifact(
            db=db,
            project_id=project_id,
            artifact_key=req.artifact_key,
            uri=req.uri,
            schema_ref=req.schema_ref,
            producer_stage=req.producer_stage,
            producer_run_id=req.producer_run_id,
            producer_step_id=req.producer_step_id,
            metadata=req.metadata,
            status=req.status,
        )
        return serialize_artifact(record)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/publish-batch", status_code=201)
async def publish_artifact_batch_route(
    project_id: int,
    req: ArtifactBatchPublishRequest,
    db: AsyncSession = Depends(get_db),
):
    """Publish multiple artifact versions in order."""
    await _ensure_project_exists(db, project_id)
    if not req.artifacts:
        raise HTTPException(400, "artifacts must contain at least one item")

    published = []
    try:
        for item in req.artifacts:
            rows = await publish_artifact_batch(
                db=db,
                project_id=project_id,
                artifact_keys=[item.artifact_key],
                schema_ref=item.schema_ref,
                producer_stage=item.producer_stage,
                producer_run_id=item.producer_run_id,
                producer_step_id=item.producer_step_id,
                metadata=item.metadata,
                status=item.status,
            )
            published.extend(rows)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "project_id": project_id,
        "count": len(published),
        "artifacts": [serialize_artifact(row) for row in published],
    }


@router.get("")
async def list_project_artifacts(
    project_id: int,
    artifact_key: str | None = None,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    """List artifact versions for a project, optionally filtered by key."""
    await _ensure_project_exists(db, project_id)
    try:
        rows = await list_artifacts(
            db=db,
            project_id=project_id,
            artifact_key=artifact_key,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "project_id": project_id,
        "count": len(rows),
        "artifacts": [serialize_artifact(row) for row in rows],
    }


@router.get("/keys")
async def list_project_artifact_keys(
    project_id: int,
    only_materialized: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """List latest artifact keys for a project."""
    await _ensure_project_exists(db, project_id)
    keys = await list_latest_artifact_keys(
        db=db,
        project_id=project_id,
        only_materialized=only_materialized,
    )
    return {
        "project_id": project_id,
        "count": len(keys),
        "keys": keys,
    }


@router.get("/latest/{artifact_key:path}")
async def get_latest_project_artifact(
    project_id: int,
    artifact_key: str,
    db: AsyncSession = Depends(get_db),
):
    """Return latest version for an artifact key."""
    await _ensure_project_exists(db, project_id)
    try:
        latest = await get_latest_artifact(db=db, project_id=project_id, artifact_key=artifact_key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if latest is None:
        raise HTTPException(404, f"Artifact '{artifact_key}' not found")
    return serialize_artifact(latest)
