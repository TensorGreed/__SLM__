"""Canonical RunEvent emission + read service (priority.md P31, Wave G).

Single emission entry point used by every stage hook in the codebase.
Designed to be **best-effort**: callers wrap ``emit_event`` in their own
``try/except`` so that observability bugs never break the stage they
report on. Mirrors the P14 ``capture_training_manifest`` posture —
emit failures log to stdout but don't propagate.

Public surface:

- ``emit_event(...)`` — append a single row to ``run_events``. Validates
  the stage / severity strings against the canonical enums in
  :mod:`app.models.run_event` so a typo doesn't slip a bad value into
  the timeline.
- ``list_run_events(...)`` — read path for the API, with stable filters
  for ``run_id`` / ``stage`` / ``severity`` / ``since`` / ``until`` /
  ``limit``.
- ``list_run_events_for_run(...)`` — convenience: every event for a
  single ``run_id``, ordered by ``ts`` ascending. Used by the
  per-experiment / per-deployment drill-in surfaces.

Stable reason codes (raised as ``ValueError`` from the API path):

- ``project_not_found`` (404)
- ``invalid_stage`` (400)
- ``invalid_severity`` (400)
- ``invalid_window`` (400 — ``since >= until``)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.models.run_event import (
    KNOWN_SEVERITIES,
    KNOWN_STAGES,
    SEVERITY_INFO,
    RunEvent,
)


_MAX_EVENTS_PER_QUERY = 500


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalise_actor(actor: str | None) -> str:
    cleaned = (actor or "").strip()
    return cleaned[:128] if cleaned else "system"


def _coerce_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


async def emit_event(
    db: AsyncSession,
    *,
    project_id: int,
    run_id: str,
    stage: str,
    summary: str | None = None,
    severity: str = SEVERITY_INFO,
    reason_code: str | None = None,
    actor: str | None = None,
    parent_run_id: str | None = None,
    payload: dict[str, Any] | None = None,
    ts: datetime | str | None = None,
) -> RunEvent:
    """Persist a single ``RunEvent`` row.

    Raises ``ValueError`` with stable codes for caller-side mistakes
    (unknown stage / severity), but leaves transient DB errors to
    propagate so the calling stage knows its observability didn't
    land. Production callers wrap this in try/except — see the hook
    sites in ``training_service``, ``export_service``, etc.
    """
    if stage not in KNOWN_STAGES:
        raise ValueError(f"invalid_stage:{stage}")
    if severity not in KNOWN_SEVERITIES:
        raise ValueError(f"invalid_severity:{severity}")

    ts_value: datetime | None = (
        ts if isinstance(ts, datetime) else _coerce_ts(ts)
    )
    if ts_value is None:
        ts_value = _utcnow()
    elif ts_value.tzinfo is None:
        ts_value = ts_value.replace(tzinfo=timezone.utc)

    row = RunEvent(
        project_id=project_id,
        run_id=str(run_id)[:128] if run_id is not None else "",
        parent_run_id=(
            str(parent_run_id)[:128] if parent_run_id else None
        ),
        stage=stage,
        severity=severity,
        reason_code=(str(reason_code)[:128] if reason_code else None),
        actor=_normalise_actor(actor),
        summary=summary,
        payload=dict(payload or {}),
        ts=ts_value,
    )
    db.add(row)
    await db.flush()
    return row


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


async def _ensure_project(db: AsyncSession, project_id: int) -> None:
    result = await db.execute(select(Project).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise ValueError("project_not_found")


def _serialize(row: RunEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "run_id": row.run_id,
        "parent_run_id": row.parent_run_id,
        "stage": row.stage,
        "severity": row.severity,
        "reason_code": row.reason_code,
        "actor": row.actor,
        "summary": row.summary,
        "payload": dict(row.payload or {}),
        "ts": row.ts.isoformat() if row.ts else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _serialize_many(rows: Iterable[RunEvent]) -> list[dict[str, Any]]:
    return [_serialize(r) for r in rows]


async def list_run_events(
    db: AsyncSession,
    *,
    project_id: int,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    stage: str | None = None,
    severity: str | None = None,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    await _ensure_project(db, project_id)

    if stage is not None and stage not in KNOWN_STAGES:
        raise ValueError(f"invalid_stage:{stage}")
    if severity is not None and severity not in KNOWN_SEVERITIES:
        raise ValueError(f"invalid_severity:{severity}")

    since_dt = since if isinstance(since, datetime) else _coerce_ts(since)
    until_dt = until if isinstance(until, datetime) else _coerce_ts(until)
    if since_dt and until_dt and since_dt >= until_dt:
        raise ValueError("invalid_window")

    bounded = max(1, min(int(limit), _MAX_EVENTS_PER_QUERY))
    stmt = select(RunEvent).where(RunEvent.project_id == project_id)
    if run_id is not None:
        stmt = stmt.where(RunEvent.run_id == str(run_id))
    if parent_run_id is not None:
        stmt = stmt.where(RunEvent.parent_run_id == str(parent_run_id))
    if stage is not None:
        stmt = stmt.where(RunEvent.stage == stage)
    if severity is not None:
        stmt = stmt.where(RunEvent.severity == severity)
    if since_dt is not None:
        stmt = stmt.where(RunEvent.ts >= since_dt)
    if until_dt is not None:
        stmt = stmt.where(RunEvent.ts <= until_dt)
    stmt = stmt.order_by(RunEvent.ts.desc(), RunEvent.id.desc()).limit(bounded)

    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    return {
        "project_id": project_id,
        "limit": bounded,
        "events": _serialize_many(rows),
    }


async def list_run_events_for_run(
    db: AsyncSession, *, run_id: str, limit: int = 200
) -> dict[str, Any]:
    """Every event for a given run_id, oldest first."""
    bounded = max(1, min(int(limit), _MAX_EVENTS_PER_QUERY))
    result = await db.execute(
        select(RunEvent)
        .where(RunEvent.run_id == str(run_id))
        .order_by(RunEvent.ts.asc(), RunEvent.id.asc())
        .limit(bounded)
    )
    rows = list(result.scalars().all())
    return {
        "run_id": str(run_id),
        "limit": bounded,
        "events": _serialize_many(rows),
    }
