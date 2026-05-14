"""Saved-config CRUD + ``run_from_config`` orchestration.

Phase G of DATASET_IMPORT_PLAN.md. Lets users persist the
mapping they confirmed in the wizard (or CLI) and re-run it later
against a refreshed source with one click. Re-runs share the same
``run_import`` code path as a fresh import, including the audit-log
hook — the only thing different is that ``config_id`` lands in the
RunEvent payload so the timeline can show "re-imported from saved
mapping #12".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset_import_config import DatasetImportConfig
from app.services.dataset_import.protocols import ImportResult
from app.services.dataset_import.service import run_import


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def list_configs(
    db: AsyncSession, project_id: int
) -> list[DatasetImportConfig]:
    """Return saved configs for a project, newest-updated first."""

    result = await db.execute(
        select(DatasetImportConfig)
        .where(DatasetImportConfig.project_id == project_id)
        .order_by(
            DatasetImportConfig.updated_at.desc(), DatasetImportConfig.id.desc()
        )
    )
    return list(result.scalars().all())


async def get_config(
    db: AsyncSession, project_id: int, config_id: int
) -> DatasetImportConfig | None:
    """Return a single config or None when it doesn't belong to this
    project. Callers translate None to a 404."""

    result = await db.execute(
        select(DatasetImportConfig).where(
            DatasetImportConfig.id == config_id,
            DatasetImportConfig.project_id == project_id,
        )
    )
    return result.scalar_one_or_none()


async def save_config(
    db: AsyncSession,
    *,
    project_id: int,
    name: str,
    locator: str,
    mapper_id: str,
    field_map: dict[str, Any] | None = None,
    drop_reasons: list[str] | None = None,
    description: str | None = None,
    limit: int | None = None,
) -> DatasetImportConfig:
    """Insert a new saved config. Raises ``ValueError`` with stable
    codes for caller-side validation (empty name, name collides with an
    existing config in the project)."""

    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("config_name_required")
    if len(clean_name) > 120:
        raise ValueError("config_name_too_long")
    if not (locator or "").strip():
        raise ValueError("config_locator_required")
    if not (mapper_id or "").strip():
        raise ValueError("config_mapper_id_required")

    row = DatasetImportConfig(
        project_id=project_id,
        name=clean_name,
        description=(description or None),
        locator=locator.strip(),
        mapper_id=mapper_id.strip(),
        field_map=dict(field_map or {}),
        drop_reasons=list(drop_reasons or []),
        limit=limit,
    )
    db.add(row)
    try:
        await db.flush()
    except IntegrityError as exc:
        # Surface the unique-name violation as a recognizable code so
        # the API layer can return 409 instead of a 500.
        await db.rollback()
        raise ValueError("config_name_taken") from exc
    return row


async def delete_config(
    db: AsyncSession, project_id: int, config_id: int
) -> bool:
    """Delete a saved config. Returns False when nothing was deleted
    (config doesn't exist or belongs to a different project)."""

    row = await get_config(db, project_id, config_id)
    if row is None:
        return False
    await db.delete(row)
    await db.flush()
    return True


async def run_from_config(
    db: AsyncSession,
    *,
    project_id: int,
    config: DatasetImportConfig,
    project_task_profile: str | None = None,
) -> ImportResult:
    """Re-import using a saved config. Bumps ``last_run_at`` /
    ``last_run_accepted`` on success — keeps the "Saved mappings" UI
    accurate without joining against the RunEvent stream.

    Forwards ``config_id`` to ``run_import`` so the audit-log RunEvent
    payload carries the link back to this config row.
    """

    result = await run_import(
        db,
        project_id=project_id,
        project_task_profile=project_task_profile,
        locator=config.locator,
        mapper_id=config.mapper_id,
        field_map=dict(config.field_map or {}),
        limit=config.limit,
        drop_reasons=set(config.drop_reasons or []),
        config_id=config.id,
    )

    config.last_run_at = _utcnow()
    config.last_run_accepted = result.accepted_count
    await db.flush()
    return result


def config_to_dict(config: DatasetImportConfig) -> dict[str, Any]:
    """Serialize a saved config for the API."""

    return {
        "id": config.id,
        "project_id": config.project_id,
        "name": config.name,
        "description": config.description,
        "locator": config.locator,
        "mapper_id": config.mapper_id,
        "field_map": dict(config.field_map or {}),
        "drop_reasons": list(config.drop_reasons or []),
        "limit": config.limit,
        "created_at": (
            config.created_at.isoformat() if config.created_at else None
        ),
        "updated_at": (
            config.updated_at.isoformat() if config.updated_at else None
        ),
        "last_run_at": (
            config.last_run_at.isoformat() if config.last_run_at else None
        ),
        "last_run_accepted": config.last_run_accepted,
    }
