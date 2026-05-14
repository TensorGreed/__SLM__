"""Annotation foundation service (Story 1.1).

Backend half of the in-product labeling flow. Five public entry points:

- :func:`create_job` — register a labeling pass + emit
  ``annotation_job_created`` audit event.
- :func:`seed_rows_from_dataset` — read a project's dataset JSONL and
  copy up to N rows into ``label_rows`` for review.
- :func:`assign_next` — hand one unassigned row to a reviewer. Uses a
  compare-and-set update so two concurrent callers never see the same
  row.
- :func:`submit_label` — persist the reviewer's label_payload + emit
  ``annotation_label_submitted``.
- :func:`job_stats` — total / labeled / assigned / unlabeled counts +
  basic job metadata.

Audit hook follows the same best-effort pattern as
``dataset_import.service._emit_import_audit_event``: observability bugs
log and continue; they never break the data write path.

Per the [[keep-brewslm-general]] memory, this service is shape-aware
(classification / span / preference_pair) but never domain-aware — the
``label_type`` discriminator is the only place we branch, and adding a
new shape means extending KNOWN_LABEL_TYPES + a frontend renderer, not
a domain-specific service.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset
from app.models.label_job import (
    JOB_STATUS_ACTIVE,
    KNOWN_JOB_STATUSES,
    KNOWN_LABEL_TYPES,
    LabelJob,
    LabelRow,
)
from app.models.project import Project


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Audit hook (best-effort, mirrors dataset_import pattern)
# ---------------------------------------------------------------------------


async def _emit_annotation_audit_event(
    db: AsyncSession,
    *,
    project_id: int,
    reason_code: str,
    summary: str,
    payload: dict[str, Any],
) -> None:
    """Best-effort RunEvent emission for annotation actions.

    Failures log + swallow — the data write path must never break
    because the audit row didn't land. Pattern mirrors
    :func:`app.services.dataset_import.service._emit_import_audit_event`.
    """
    try:
        from app.models.run_event import SEVERITY_INFO, STAGE_INGESTION
        from app.services.run_event_service import emit_event

        run_id = (
            f"annotation-{int(_utcnow().timestamp() * 1000)}"
        )
        await emit_event(
            db,
            project_id=project_id,
            run_id=run_id,
            stage=STAGE_INGESTION,
            severity=SEVERITY_INFO,
            reason_code=reason_code,
            summary=summary,
            payload=payload,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[run_event] emit_failed annotation project={project_id} "
            f"reason={reason_code!r} err={exc!r}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------


async def _ensure_project(db: AsyncSession, project_id: int) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError("project_not_found")
    return project


async def create_job(
    db: AsyncSession,
    *,
    project_id: int,
    name: str,
    label_type: str,
    label_schema: dict[str, Any] | None = None,
    instructions: str | None = None,
    target_rows: int | None = None,
) -> LabelJob:
    """Create a label job. Raises ``ValueError`` with stable codes for
    validation failures so the API layer can translate to 400/422."""

    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("job_name_required")
    if len(clean_name) > 120:
        raise ValueError("job_name_too_long")
    if label_type not in KNOWN_LABEL_TYPES:
        raise ValueError(f"invalid_label_type:{label_type}")

    await _ensure_project(db, project_id)

    job = LabelJob(
        project_id=project_id,
        name=clean_name,
        label_type=label_type,
        label_schema=dict(label_schema or {}),
        instructions=(instructions.strip() if instructions else None) or None,
        status=JOB_STATUS_ACTIVE,
        target_rows=target_rows,
    )
    db.add(job)
    await db.flush()

    await _emit_annotation_audit_event(
        db,
        project_id=project_id,
        reason_code="annotation_job_created",
        summary=(
            f"Annotation job {job.id!r} created "
            f"({label_type}, target={target_rows or 'unbounded'})"
        ),
        payload={
            "job_id": job.id,
            "name": clean_name,
            "label_type": label_type,
            "target_rows": target_rows,
        },
    )

    return job


async def get_job(
    db: AsyncSession, *, project_id: int, job_id: int
) -> LabelJob | None:
    """Return a job scoped to its project, or None when missing or
    cross-project."""
    result = await db.execute(
        select(LabelJob).where(
            LabelJob.id == job_id, LabelJob.project_id == project_id
        )
    )
    return result.scalar_one_or_none()


async def list_jobs(
    db: AsyncSession, *, project_id: int
) -> list[LabelJob]:
    """All jobs for a project, newest-updated first."""
    result = await db.execute(
        select(LabelJob)
        .where(LabelJob.project_id == project_id)
        .order_by(LabelJob.updated_at.desc(), LabelJob.id.desc())
    )
    return list(result.scalars().all())


async def delete_job(
    db: AsyncSession, *, project_id: int, job_id: int
) -> bool:
    """Delete a job + cascade rows. Returns False on 404."""
    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        return False
    # Manual cascade — no SQLAlchemy ORM relationship is declared so
    # delete-orphan isn't wired up. Explicit DELETE is clearer anyway.
    await db.execute(
        LabelRow.__table__.delete().where(LabelRow.job_id == job_id)
    )
    await db.delete(job)
    await db.flush()
    return True


# ---------------------------------------------------------------------------
# Row seeding
# ---------------------------------------------------------------------------


async def seed_rows_from_dataset(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    dataset_id: int,
    n: int,
) -> int:
    """Read up to ``n`` rows from ``dataset_id``'s JSONL file and
    materialize them as :class:`LabelRow` work units.

    Returns the actual count seeded — capped by the file's length and
    by any malformed lines (silently skipped).
    """

    if n <= 0:
        raise ValueError("seed_n_must_be_positive")

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise ValueError("dataset_not_found")
    if dataset.project_id != project_id:
        raise ValueError("dataset_project_mismatch")

    file_path = Path(dataset.file_path or "")
    if not file_path.exists():
        raise ValueError("dataset_file_missing")

    added = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            source_id = row.get("id")
            source_row_id = (
                str(source_id) if source_id is not None else str(idx)
            )
            db.add(
                LabelRow(
                    job_id=job_id,
                    source_row_id=source_row_id[:128],
                    raw_payload=row,
                )
            )
            added += 1
            if added >= n:
                break

    if added:
        # Bump the job's updated_at so list_jobs orders correctly.
        job.updated_at = _utcnow()
    await db.flush()
    return added


# ---------------------------------------------------------------------------
# Assignment + submission
# ---------------------------------------------------------------------------


async def assign_next(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    user_id: int | None,
) -> LabelRow | None:
    """Atomically hand one unlabeled, unassigned row to ``user_id``.

    Returns the assigned row or ``None`` when the job has no remaining
    work. Concurrent callers will not see the same row because the
    UPDATE checks ``assigned_to IS NULL`` again — a SQLite-safe
    compare-and-set that retries on lost-update races.
    """

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    for _ in range(10):
        result = await db.execute(
            select(LabelRow)
            .where(
                LabelRow.job_id == job_id,
                LabelRow.assigned_to.is_(None),
                LabelRow.labeled_at.is_(None),
            )
            .order_by(LabelRow.id.asc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None

        now = _utcnow()
        upd = await db.execute(
            update(LabelRow)
            .where(
                LabelRow.id == row.id,
                LabelRow.assigned_to.is_(None),
            )
            .values(assigned_to=user_id, assigned_at=now)
        )
        await db.flush()
        if upd.rowcount and upd.rowcount > 0:
            await db.refresh(row)
            return row

    return None


async def submit_label(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    row_id: int,
    label_payload: dict[str, Any],
    reviewer_notes: str | None = None,
) -> LabelRow:
    """Persist a reviewer's label for one row and emit the audit event.

    Raises ``ValueError`` with stable codes:
      - ``label_job_not_found`` — job missing or cross-project.
      - ``label_row_not_found`` — row missing or doesn't belong to job.
      - ``label_payload_required`` — label_payload is empty / None.
    """

    if not isinstance(label_payload, dict) or not label_payload:
        raise ValueError("label_payload_required")

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    result = await db.execute(
        select(LabelRow).where(
            LabelRow.id == row_id, LabelRow.job_id == job_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("label_row_not_found")

    now = _utcnow()
    row.label_payload = dict(label_payload)
    row.labeled_at = now
    if reviewer_notes is not None:
        cleaned = reviewer_notes.strip()
        row.reviewer_notes = cleaned or None
    job.updated_at = now
    await db.flush()

    await _emit_annotation_audit_event(
        db,
        project_id=project_id,
        reason_code="annotation_label_submitted",
        summary=(
            f"Label submitted for job {job_id} row {row_id} "
            f"({job.label_type})"
        ),
        payload={
            "job_id": job_id,
            "row_id": row_id,
            "user_id": row.assigned_to,
            "label_type": job.label_type,
            "label_payload": dict(label_payload),
        },
    )

    return row


# ---------------------------------------------------------------------------
# Stats + serialization
# ---------------------------------------------------------------------------


async def job_stats(
    db: AsyncSession, *, project_id: int, job_id: int
) -> dict[str, Any]:
    """Return total / labeled / assigned / unlabeled counts for a job."""

    job = await get_job(db, project_id=project_id, job_id=job_id)
    if job is None:
        raise ValueError("label_job_not_found")

    total_q = await db.execute(
        select(func.count(LabelRow.id)).where(LabelRow.job_id == job_id)
    )
    total = int(total_q.scalar() or 0)

    labeled_q = await db.execute(
        select(func.count(LabelRow.id)).where(
            LabelRow.job_id == job_id,
            LabelRow.labeled_at.is_not(None),
        )
    )
    labeled = int(labeled_q.scalar() or 0)

    assigned_q = await db.execute(
        select(func.count(LabelRow.id)).where(
            LabelRow.job_id == job_id,
            LabelRow.assigned_to.is_not(None),
            LabelRow.labeled_at.is_(None),
        )
    )
    assigned = int(assigned_q.scalar() or 0)

    unlabeled = total - labeled - assigned

    return {
        "job_id": job.id,
        "name": job.name,
        "label_type": job.label_type,
        "status": job.status,
        "target_rows": job.target_rows,
        "total": total,
        "labeled": labeled,
        "assigned": assigned,
        "unlabeled": unlabeled,
    }


def job_to_dict(job: LabelJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "project_id": job.project_id,
        "name": job.name,
        "label_type": job.label_type,
        "label_schema": dict(job.label_schema or {}),
        "instructions": job.instructions,
        "status": job.status,
        "target_rows": job.target_rows,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


def row_to_dict(row: LabelRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "job_id": row.job_id,
        "source_row_id": row.source_row_id,
        "raw_payload": dict(row.raw_payload or {}),
        "assigned_to": row.assigned_to,
        "assigned_at": (
            row.assigned_at.isoformat() if row.assigned_at else None
        ),
        "label_payload": (
            dict(row.label_payload) if row.label_payload else None
        ),
        "labeled_at": (
            row.labeled_at.isoformat() if row.labeled_at else None
        ),
        "reviewer_notes": row.reviewer_notes,
    }


def update_job_fields(
    job: LabelJob,
    *,
    name: str | None = None,
    instructions: str | None = None,
    status: str | None = None,
    target_rows: int | None = None,
) -> LabelJob:
    """In-place update of mutable job fields. Validates ``status`` against
    :data:`KNOWN_JOB_STATUSES`."""
    if name is not None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("job_name_required")
        if len(clean_name) > 120:
            raise ValueError("job_name_too_long")
        job.name = clean_name
    if instructions is not None:
        cleaned = instructions.strip()
        job.instructions = cleaned or None
    if status is not None:
        if status not in KNOWN_JOB_STATUSES:
            raise ValueError(f"invalid_job_status:{status}")
        job.status = status
    if target_rows is not None:
        job.target_rows = max(0, int(target_rows))
    job.updated_at = _utcnow()
    return job
