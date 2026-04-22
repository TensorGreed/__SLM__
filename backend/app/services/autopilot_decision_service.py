"""Autopilot decision-log persistence and query service.

Maps the in-memory decision-log entries that autopilot orchestration already
produces (see `_append_autopilot_decision` in `app.api.training`) into rows in
the `autopilot_decisions` table, and exposes filtered reads for the
`/autopilot/decisions` API.

Writes use their own short-lived session so that persistence is committed
independently of the caller's transaction. This matters because an autopilot
orchestration can raise an HTTPException (rolling back the outer session)
while we still want the decision-log trail preserved for post-mortem.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.autopilot_decision import AutopilotDecision


_STATUS_ACTION_MAP: dict[str, str] = {
    "applied": "applied",
    "completed": "applied",
    "blocked": "blocked",
    "failed": "blocked",
    "rolled_back": "rolled_back",
    "warning": "warned",
    "warn": "warned",
    "skipped": "skipped",
    "active": "info",
    "inactive": "info",
    "info": "info",
}


def _derive_action(entry: dict[str, Any]) -> str:
    status = str(entry.get("status") or "").strip().lower()
    blocker = bool(entry.get("blocker"))
    changed = bool(entry.get("changed"))
    if blocker:
        return "blocked"
    if changed:
        return _STATUS_ACTION_MAP.get(status, "applied")
    return _STATUS_ACTION_MAP.get(status, "info")


def _derive_reason_code(entry: dict[str, Any]) -> str | None:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        for key in ("reason_code", "error_code", "code"):
            value = metadata.get(key)
            if value:
                token = str(value).strip()
                if token:
                    return token
    return None


def _derive_confidence(entry: dict[str, Any]) -> float | None:
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return None
    for key in ("confidence", "confidence_score"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _compose_rationale(entry: dict[str, Any]) -> str | None:
    summary = str(entry.get("summary") or "").strip()
    why = str(entry.get("why") or "").strip()
    if summary and why and summary != why:
        return f"{summary} — {why}"
    return summary or why or None


def _extract_payload(entry: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("before", "after", "fixes", "metadata", "why"):
        if key in entry and entry[key] is not None:
            payload[key] = entry[key]
    return payload


def _build_row(
    entry: dict[str, Any],
    *,
    run_id: str,
    project_id: int | None,
    dry_run: bool,
    intent: str,
    sequence: int,
) -> AutopilotDecision:
    return AutopilotDecision(
        run_id=str(run_id),
        project_id=int(project_id) if project_id is not None else None,
        sequence=int(sequence),
        stage=str(entry.get("step") or "unknown").strip() or "unknown",
        status=str(entry.get("status") or "info").strip() or "info",
        action=_derive_action(entry),
        reason_code=_derive_reason_code(entry),
        confidence=_derive_confidence(entry),
        rationale=_compose_rationale(entry),
        summary=(str(entry.get("summary") or "").strip() or None),
        actor="autopilot",
        changed=bool(entry.get("changed")),
        safe=bool(entry.get("safe", True)),
        blocker=bool(entry.get("blocker")),
        dry_run=bool(dry_run),
        intent=(str(intent or "").strip() or None),
        payload=_extract_payload(entry),
    )


async def persist_decision_log(
    *,
    run_id: str,
    project_id: int | None,
    dry_run: bool,
    intent: str,
    entries: Iterable[dict[str, Any]],
) -> int:
    """Persist a full decision-log to the autopilot_decisions table.

    Uses its own DB session to guarantee the log survives even if the caller's
    transaction rolls back (e.g. an HTTPException from an orchestration step).
    Silently returns 0 on persistence error — the in-memory decision-log is
    still returned to the caller and the API response remains intact.
    """
    run_token = str(run_id or "").strip()
    if not run_token:
        return 0

    rows: list[AutopilotDecision] = []
    for sequence, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            continue
        rows.append(
            _build_row(
                entry,
                run_id=run_token,
                project_id=project_id,
                dry_run=dry_run,
                intent=intent,
                sequence=sequence,
            )
        )

    if not rows:
        return 0

    try:
        async with async_session_factory() as db:
            db.add_all(rows)
            await db.commit()
    except Exception:
        # Decision-log persistence is best-effort. Do not break orchestration.
        return 0
    return len(rows)


def _serialize_decision(row: AutopilotDecision) -> dict[str, Any]:
    return {
        "id": int(row.id),
        "run_id": row.run_id,
        "project_id": row.project_id,
        "sequence": int(row.sequence or 0),
        "stage": row.stage,
        "status": row.status,
        "action": row.action,
        "reason_code": row.reason_code,
        "confidence": row.confidence,
        "rationale": row.rationale,
        "summary": row.summary,
        "actor": row.actor,
        "changed": bool(row.changed),
        "safe": bool(row.safe),
        "blocker": bool(row.blocker),
        "dry_run": bool(row.dry_run),
        "intent": row.intent,
        "payload": dict(row.payload or {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def list_decisions(
    db: AsyncSession,
    *,
    project_id: int | None = None,
    run_id: str | None = None,
    stage: str | None = None,
    status: str | None = None,
    action: str | None = None,
    reason_code: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Return a filtered page of decision-log entries.

    Ordered by `(created_at desc, run_id, sequence)` so the most recent run
    appears first but the within-run sequence is preserved on filter-free
    queries scoped to a single run.
    """
    stmt = select(AutopilotDecision)

    conditions = []
    if project_id is not None:
        conditions.append(AutopilotDecision.project_id == int(project_id))
    if run_id:
        conditions.append(AutopilotDecision.run_id == str(run_id))
    if stage:
        conditions.append(AutopilotDecision.stage == str(stage))
    if status:
        conditions.append(AutopilotDecision.status == str(status))
    if action:
        conditions.append(AutopilotDecision.action == str(action))
    if reason_code:
        conditions.append(AutopilotDecision.reason_code == str(reason_code))
    if since is not None:
        conditions.append(AutopilotDecision.created_at >= since)
    if until is not None:
        conditions.append(AutopilotDecision.created_at <= until)
    if conditions:
        stmt = stmt.where(and_(*conditions))

    stmt = stmt.order_by(
        AutopilotDecision.created_at.desc(),
        AutopilotDecision.run_id,
        AutopilotDecision.sequence,
    )

    clamped_limit = max(1, min(int(limit or 100), 1000))
    clamped_offset = max(0, int(offset or 0))
    stmt = stmt.limit(clamped_limit).offset(clamped_offset)

    result = await db.execute(stmt)
    rows = list(result.scalars().all())

    filters = {
        "project_id": int(project_id) if project_id is not None else None,
        "run_id": str(run_id) if run_id else None,
        "stage": str(stage) if stage else None,
        "status": str(status) if status else None,
        "action": str(action) if action else None,
        "reason_code": str(reason_code) if reason_code else None,
        "since": since.isoformat() if since is not None else None,
        "until": until.isoformat() if until is not None else None,
    }
    return {
        "items": [_serialize_decision(row) for row in rows],
        "limit": clamped_limit,
        "offset": clamped_offset,
        "returned": len(rows),
        "filters": filters,
    }


async def get_run_decisions(
    db: AsyncSession,
    run_id: str,
) -> dict[str, Any]:
    """Return every decision-log entry for one autopilot run, ordered by sequence."""
    run_token = str(run_id or "").strip()
    stmt = (
        select(AutopilotDecision)
        .where(AutopilotDecision.run_id == run_token)
        .order_by(AutopilotDecision.sequence.asc())
    )
    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    project_id: int | None = None
    if rows:
        project_id = rows[0].project_id

    return {
        "run_id": run_token,
        "project_id": project_id,
        "items": [_serialize_decision(row) for row in rows],
        "count": len(rows),
    }
