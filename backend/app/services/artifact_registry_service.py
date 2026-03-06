"""Typed artifact registry service."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact import ArtifactRecord, ArtifactStatus


DEFAULT_ARTIFACT_SCHEMA_REF = "slm.artifact/v1"


def normalize_artifact_key(value: str) -> str:
    normalized = ".".join(part.strip() for part in value.split(".") if part and part.strip())
    if not normalized:
        raise ValueError("artifact_key is required")
    return normalized


def infer_artifact_type(artifact_key: str) -> str:
    key = normalize_artifact_key(artifact_key)
    prefix, _, _ = key.partition(".")
    return prefix or key


def _normalize_status(status: ArtifactStatus | str | None) -> ArtifactStatus:
    if status is None:
        return ArtifactStatus.MATERIALIZED
    if isinstance(status, ArtifactStatus):
        return status
    normalized = str(status).strip().lower()
    return ArtifactStatus(normalized)


async def _next_version(db: AsyncSession, project_id: int, artifact_key: str) -> int:
    result = await db.execute(
        select(ArtifactRecord.version)
        .where(
            ArtifactRecord.project_id == project_id,
            ArtifactRecord.artifact_key == artifact_key,
        )
        .order_by(ArtifactRecord.version.desc(), ArtifactRecord.id.desc())
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    return int(latest or 0) + 1


async def publish_artifact(
    db: AsyncSession,
    project_id: int,
    artifact_key: str,
    *,
    uri: str | None = None,
    schema_ref: str | None = None,
    producer_stage: str | None = None,
    producer_run_id: str | None = None,
    producer_step_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    status: ArtifactStatus | str | None = None,
) -> ArtifactRecord:
    """Create a new immutable version for an artifact key."""
    key = normalize_artifact_key(artifact_key)
    record = ArtifactRecord(
        project_id=project_id,
        artifact_key=key,
        artifact_type=infer_artifact_type(key),
        version=await _next_version(db, project_id, key),
        status=_normalize_status(status),
        uri=(uri or "").strip(),
        schema_ref=(schema_ref or DEFAULT_ARTIFACT_SCHEMA_REF).strip() or DEFAULT_ARTIFACT_SCHEMA_REF,
        producer_stage=(producer_stage or "").strip() or None,
        producer_run_id=(producer_run_id or "").strip() or None,
        producer_step_id=(producer_step_id or "").strip() or None,
        metadata_=dict(metadata or {}),
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)
    return record


async def publish_artifact_batch(
    db: AsyncSession,
    project_id: int,
    artifact_keys: Iterable[str],
    *,
    schema_ref: str | None = None,
    producer_stage: str | None = None,
    producer_run_id: str | None = None,
    producer_step_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    status: ArtifactStatus | str | None = None,
) -> list[ArtifactRecord]:
    published: list[ArtifactRecord] = []
    for key in artifact_keys:
        cleaned = str(key).strip()
        if not cleaned:
            continue
        published.append(
            await publish_artifact(
                db=db,
                project_id=project_id,
                artifact_key=cleaned,
                schema_ref=schema_ref,
                producer_stage=producer_stage,
                producer_run_id=producer_run_id,
                producer_step_id=producer_step_id,
                metadata=metadata,
                status=status,
            )
        )
    return published


async def get_latest_artifact(
    db: AsyncSession,
    project_id: int,
    artifact_key: str,
) -> ArtifactRecord | None:
    key = normalize_artifact_key(artifact_key)
    result = await db.execute(
        select(ArtifactRecord)
        .where(
            ArtifactRecord.project_id == project_id,
            ArtifactRecord.artifact_key == key,
        )
        .order_by(ArtifactRecord.version.desc(), ArtifactRecord.id.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_artifacts(
    db: AsyncSession,
    project_id: int,
    *,
    artifact_key: str | None = None,
    limit: int = 200,
) -> list[ArtifactRecord]:
    query = select(ArtifactRecord).where(ArtifactRecord.project_id == project_id)
    if artifact_key:
        query = query.where(ArtifactRecord.artifact_key == normalize_artifact_key(artifact_key))
    query = query.order_by(ArtifactRecord.created_at.desc(), ArtifactRecord.id.desc()).limit(max(1, min(limit, 500)))
    result = await db.execute(query)
    return list(result.scalars().all())


async def list_latest_artifact_keys(
    db: AsyncSession,
    project_id: int,
    *,
    only_materialized: bool = True,
) -> list[str]:
    result = await db.execute(
        select(ArtifactRecord)
        .where(ArtifactRecord.project_id == project_id)
        .order_by(
            ArtifactRecord.artifact_key.asc(),
            ArtifactRecord.version.desc(),
            ArtifactRecord.id.desc(),
        )
    )
    latest_by_key: dict[str, ArtifactRecord] = {}
    for record in result.scalars().all():
        key = record.artifact_key
        if key not in latest_by_key:
            latest_by_key[key] = record

    keys: list[str] = []
    for key, record in latest_by_key.items():
        if only_materialized and record.status != ArtifactStatus.MATERIALIZED:
            continue
        keys.append(key)
    keys.sort()
    return keys


def serialize_artifact(record: ArtifactRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "project_id": record.project_id,
        "artifact_key": record.artifact_key,
        "artifact_type": record.artifact_type,
        "version": record.version,
        "status": record.status.value,
        "uri": record.uri,
        "schema_ref": record.schema_ref,
        "producer_stage": record.producer_stage,
        "producer_run_id": record.producer_run_id,
        "producer_step_id": record.producer_step_id,
        "metadata": record.metadata_ or {},
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }
