"""Autopilot snapshot capture and rollback execution.

Flow:

- `capture_snapshot(...)` writes a pre-change snapshot row in its own session
  so persistence survives even if the caller's transaction later rolls back.
- `update_snapshot_post_state(...)` annotates the row with what the action
  actually produced (experiment id, training task id, etc.) so rollback knows
  precisely what to unwind.
- `rollback_decision(db, decision_id, ...)` performs the reversal atomically
  inside the caller's transaction — experiment cancel, training cancel,
  project-config restore — then writes a new autopilot_decisions row
  (`action="rolled_back"`) and marks the snapshot restored.
- `preview_rollback(db, decision_id)` returns the same plan without mutating.

The decision <-> snapshot link is keyed by (`run_id`, `decision_sequence`) so
callers can pass just a numeric `decision_id` from the P1 decision-log API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.autopilot_decision import AutopilotDecision
from app.models.autopilot_snapshot import AutopilotSnapshot
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project


# Project-level fields that autopilot may mutate. Order is not meaningful,
# but the set is authoritative for restore operations.
PROJECT_CONFIG_FIELDS: tuple[str, ...] = (
    "base_model_name",
    "target_profile_id",
    "training_preferred_plan_profile",
    "evaluation_preferred_pack_id",
    "dataset_adapter_preset",
    "active_domain_blueprint_version",
    "domain_pack_id",
    "domain_profile_id",
    "beginner_mode",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(value: datetime | None) -> datetime | None:
    """Return `value` as a timezone-aware UTC datetime, or None.

    SQLite drops tzinfo when round-tripping through a `DateTime(timezone=True)`
    column, so rows read back can be naive even though we wrote aware values.
    Normalize before any comparison.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _clone_jsonable(value: Any) -> Any:
    """Return a JSON-safe deep copy of `value`, tolerating ORM types."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _clone_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone_jsonable(v) for v in value]
    # SQLAlchemy enums expose .value.
    return getattr(value, "value", str(value))


async def _load_project_config(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Snapshot the mutable project-config fields as a plain dict."""
    result = await db.execute(select(Project).where(Project.id == int(project_id)))
    project = result.scalar_one_or_none()
    if project is None:
        return {}
    snapshot: dict[str, Any] = {}
    for field in PROJECT_CONFIG_FIELDS:
        value = getattr(project, field, None)
        snapshot[field] = _clone_jsonable(value)
    return snapshot


async def capture_project_snapshot(
    db: AsyncSession,
    *,
    project_id: int,
) -> dict[str, Any]:
    """Read-only helper that returns the current project config as JSON."""
    return await _load_project_config(db, project_id)


async def capture_snapshot(
    *,
    run_id: str,
    project_id: int,
    decision_sequence: int,
    snapshot_type: str = "autopilot_generic",
    pre_state: dict[str, Any] | None = None,
    rollback_actions: Iterable[dict[str, Any]] | None = None,
) -> int | None:
    """Persist a snapshot row in its own session; return the new id or None.

    `decision_sequence` must match the sequence the associated decision will be
    given inside the in-memory decision_log (i.e. `len(decision_log)` at the
    moment of capture, before the corresponding `_append_autopilot_decision`).
    """
    run_token = str(run_id or "").strip()
    if not run_token:
        return None

    snapshot = AutopilotSnapshot(
        run_id=run_token,
        decision_sequence=int(decision_sequence),
        project_id=int(project_id) if project_id is not None else None,
        snapshot_type=str(snapshot_type or "autopilot_generic"),
        pre_state=_clone_jsonable(pre_state or {}),
        post_state={},
        rollback_actions=[_clone_jsonable(action) for action in (rollback_actions or [])],
    )
    try:
        async with async_session_factory() as db:
            db.add(snapshot)
            await db.commit()
            await db.refresh(snapshot)
            return int(snapshot.id)
    except Exception:
        return None


async def update_snapshot_post_state(
    snapshot_id: int | None,
    *,
    post_state: dict[str, Any] | None = None,
    extra_rollback_actions: Iterable[dict[str, Any]] | None = None,
) -> None:
    """Attach the post-change state + any additional rollback actions.

    Best-effort: swallows exceptions so the orchestration response is never
    blocked by a snapshot-update failure.
    """
    if not snapshot_id:
        return
    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(AutopilotSnapshot).where(AutopilotSnapshot.id == int(snapshot_id))
            )
            row = result.scalar_one_or_none()
            if row is None:
                return
            if post_state is not None:
                merged = dict(row.post_state or {})
                merged.update(_clone_jsonable(post_state))
                row.post_state = merged
            if extra_rollback_actions:
                existing = list(row.rollback_actions or [])
                existing.extend(_clone_jsonable(a) for a in extra_rollback_actions)
                row.rollback_actions = existing
            await db.commit()
    except Exception:
        return


def _serialize_snapshot(row: AutopilotSnapshot) -> dict[str, Any]:
    return {
        "id": int(row.id),
        "run_id": row.run_id,
        "decision_sequence": int(row.decision_sequence or 0),
        "project_id": row.project_id,
        "snapshot_type": row.snapshot_type,
        "pre_state": dict(row.pre_state or {}),
        "post_state": dict(row.post_state or {}),
        "rollback_actions": list(row.rollback_actions or []),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "restored_at": row.restored_at.isoformat() if row.restored_at else None,
        "restored_by": row.restored_by,
        "restored_reason": row.restored_reason,
        "restored_decision_id": row.restored_decision_id,
    }


async def _fetch_decision(db: AsyncSession, decision_id: int) -> AutopilotDecision | None:
    result = await db.execute(
        select(AutopilotDecision).where(AutopilotDecision.id == int(decision_id))
    )
    return result.scalar_one_or_none()


async def _fetch_snapshot_for_decision(
    db: AsyncSession,
    decision: AutopilotDecision,
) -> AutopilotSnapshot | None:
    stmt = select(AutopilotSnapshot).where(
        and_(
            AutopilotSnapshot.run_id == decision.run_id,
            AutopilotSnapshot.decision_sequence == int(decision.sequence),
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_snapshot_for_decision(
    db: AsyncSession,
    decision_id: int,
) -> dict[str, Any] | None:
    """Return the serialized snapshot matching a decision id, or None."""
    decision = await _fetch_decision(db, decision_id)
    if decision is None:
        return None
    snapshot = await _fetch_snapshot_for_decision(db, decision)
    if snapshot is None:
        return None
    return _serialize_snapshot(snapshot)


async def list_snapshots(
    db: AsyncSession,
    *,
    project_id: int | None = None,
    run_id: str | None = None,
    include_restored: bool = True,
    include_expired: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    stmt = select(AutopilotSnapshot)
    conditions = []
    if project_id is not None:
        conditions.append(AutopilotSnapshot.project_id == int(project_id))
    if run_id:
        conditions.append(AutopilotSnapshot.run_id == str(run_id))
    if not include_restored:
        conditions.append(AutopilotSnapshot.restored_at.is_(None))
    if not include_expired:
        conditions.append(AutopilotSnapshot.expires_at > _utcnow())
    if conditions:
        stmt = stmt.where(and_(*conditions))
    stmt = stmt.order_by(AutopilotSnapshot.created_at.desc())
    clamped_limit = max(1, min(int(limit or 100), 1000))
    clamped_offset = max(0, int(offset or 0))
    stmt = stmt.limit(clamped_limit).offset(clamped_offset)
    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    return {
        "items": [_serialize_snapshot(row) for row in rows],
        "limit": clamped_limit,
        "offset": clamped_offset,
        "returned": len(rows),
    }


def _plan_rollback_steps(
    snapshot: AutopilotSnapshot,
) -> list[dict[str, Any]]:
    """Turn stored rollback_actions + post_state into concrete executable steps.

    Each step is a dict like {"kind": "cancel_experiment", "experiment_id": 42}
    that `_execute_rollback_steps` can dispatch on.
    """
    post_state = dict(snapshot.post_state or {})
    pre_state = dict(snapshot.pre_state or {})
    actions = list(snapshot.rollback_actions or [])
    steps: list[dict[str, Any]] = []

    for action in actions:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("kind") or "").strip().lower()
        if kind == "cancel_experiment":
            experiment_id = post_state.get("experiment_id")
            if experiment_id:
                steps.append(
                    {
                        "kind": "cancel_experiment",
                        "experiment_id": int(experiment_id),
                        "training_started": bool(post_state.get("training_started")),
                    }
                )
        elif kind == "restore_project_config":
            fields = [
                str(field)
                for field in list(action.get("fields") or PROJECT_CONFIG_FIELDS)
                if str(field) in PROJECT_CONFIG_FIELDS
            ]
            project_pre = dict(pre_state.get("project") or {})
            restore_payload = {
                field: project_pre.get(field)
                for field in fields
                if field in project_pre
            }
            if restore_payload:
                steps.append(
                    {
                        "kind": "restore_project_config",
                        "fields": restore_payload,
                    }
                )
        else:
            # Unknown action kind — keep it visible so the preview UI can show it.
            steps.append({"kind": kind or "unknown", "raw": action})
    return steps


async def preview_rollback(
    db: AsyncSession,
    decision_id: int,
) -> dict[str, Any]:
    """Return a non-mutating preview of the rollback.

    Returns a dict with `reversible`, `reason` (if not), planned `steps`, the
    matched snapshot, and the originating decision. Safe to call on already
    rolled-back or expired snapshots — will report the reason.
    """
    decision = await _fetch_decision(db, decision_id)
    if decision is None:
        return {
            "decision_id": int(decision_id),
            "reversible": False,
            "reason": "decision_not_found",
            "message": f"Autopilot decision {decision_id} not found.",
        }

    snapshot = await _fetch_snapshot_for_decision(db, decision)
    if snapshot is None:
        return {
            "decision_id": int(decision.id),
            "decision": _serialize_decision_summary(decision),
            "reversible": False,
            "reason": "no_snapshot",
            "message": (
                "This decision has no snapshot — either it was informational "
                "only, or snapshot capture failed at the time of the change."
            ),
        }

    if snapshot.restored_at is not None:
        return {
            "decision_id": int(decision.id),
            "decision": _serialize_decision_summary(decision),
            "snapshot": _serialize_snapshot(snapshot),
            "reversible": False,
            "reason": "already_rolled_back",
            "message": "This decision has already been rolled back.",
        }

    expires_at = _as_aware_utc(snapshot.expires_at)
    if expires_at is not None and expires_at <= _utcnow():
        return {
            "decision_id": int(decision.id),
            "decision": _serialize_decision_summary(decision),
            "snapshot": _serialize_snapshot(snapshot),
            "reversible": False,
            "reason": "snapshot_expired",
            "message": (
                "Rollback is unavailable: snapshot exceeded its retention "
                f"window (expired at {expires_at.isoformat()})."
            ),
        }

    steps = _plan_rollback_steps(snapshot)
    return {
        "decision_id": int(decision.id),
        "decision": _serialize_decision_summary(decision),
        "snapshot": _serialize_snapshot(snapshot),
        "reversible": True,
        "steps": steps,
    }


def _serialize_decision_summary(decision: AutopilotDecision) -> dict[str, Any]:
    return {
        "id": int(decision.id),
        "run_id": decision.run_id,
        "sequence": int(decision.sequence or 0),
        "stage": decision.stage,
        "status": decision.status,
        "action": decision.action,
        "project_id": decision.project_id,
        "summary": decision.summary,
    }


async def _execute_rollback_steps(
    db: AsyncSession,
    steps: list[dict[str, Any]],
    *,
    project_id: int | None,
) -> list[dict[str, Any]]:
    """Run the planned steps in sequence, returning per-step outcomes."""
    outcomes: list[dict[str, Any]] = []
    for step in steps:
        kind = str(step.get("kind") or "").strip().lower()
        if kind == "cancel_experiment":
            experiment_id = int(step.get("experiment_id") or 0)
            outcomes.append(await _cancel_experiment(db, project_id, experiment_id))
        elif kind == "restore_project_config":
            fields = dict(step.get("fields") or {})
            outcomes.append(await _restore_project_config(db, project_id, fields))
        else:
            outcomes.append(
                {"kind": kind or "unknown", "status": "skipped", "message": "Unknown rollback step."}
            )
    return outcomes


async def _cancel_experiment(
    db: AsyncSession,
    project_id: int | None,
    experiment_id: int,
) -> dict[str, Any]:
    if not experiment_id:
        return {"kind": "cancel_experiment", "status": "skipped", "message": "No experiment id in snapshot."}
    stmt = select(Experiment).where(Experiment.id == int(experiment_id))
    if project_id is not None:
        stmt = stmt.where(Experiment.project_id == int(project_id))
    exp = (await db.execute(stmt)).scalar_one_or_none()
    if exp is None:
        return {
            "kind": "cancel_experiment",
            "status": "skipped",
            "experiment_id": int(experiment_id),
            "message": "Experiment not found; nothing to cancel.",
        }

    previous_status = exp.status.value if hasattr(exp.status, "value") else str(exp.status)
    if exp.status == ExperimentStatus.CANCELLED:
        return {
            "kind": "cancel_experiment",
            "status": "noop",
            "experiment_id": int(experiment_id),
            "previous_status": previous_status,
            "message": "Experiment already cancelled.",
        }

    if exp.status == ExperimentStatus.RUNNING:
        cfg = dict(exp.config or {})
        runtime = dict(cfg.get("_runtime") or {})
        task_id = str(runtime.get("task_id", "")).strip()
        if task_id:
            try:
                from app.services.job_service import cancel_task

                cancel_task(task_id, terminate=True)
                runtime["cancel_status"] = "cancel_requested"
            except Exception as exc:
                runtime["cancel_status"] = f"cancel_error:{exc}"
        else:
            runtime["cancel_status"] = "cancel_requested_without_task_id"
        runtime["cancel_requested_at"] = _utcnow().isoformat()
        cfg["_runtime"] = runtime
        exp.config = cfg

    exp.status = ExperimentStatus.CANCELLED
    if exp.completed_at is None:
        exp.completed_at = _utcnow()
    await db.flush()
    return {
        "kind": "cancel_experiment",
        "status": "applied",
        "experiment_id": int(experiment_id),
        "previous_status": previous_status,
    }


async def _restore_project_config(
    db: AsyncSession,
    project_id: int | None,
    fields: dict[str, Any],
) -> dict[str, Any]:
    if not project_id or not fields:
        return {"kind": "restore_project_config", "status": "skipped", "fields": {}}
    result = await db.execute(select(Project).where(Project.id == int(project_id)))
    project = result.scalar_one_or_none()
    if project is None:
        return {
            "kind": "restore_project_config",
            "status": "skipped",
            "message": "Project not found.",
        }
    restored: dict[str, Any] = {}
    for field, value in fields.items():
        if field not in PROJECT_CONFIG_FIELDS:
            continue
        current = getattr(project, field, None)
        current_clone = _clone_jsonable(current)
        if current_clone != value:
            setattr(project, field, value)
            restored[field] = {"from": current_clone, "to": value}
    if restored:
        await db.flush()
    return {
        "kind": "restore_project_config",
        "status": "applied" if restored else "noop",
        "fields": restored,
    }


async def rollback_decision(
    db: AsyncSession,
    decision_id: int,
    *,
    actor: str = "api",
    reason: str | None = None,
) -> dict[str, Any]:
    """Execute the rollback atomically inside the provided session.

    Writes a new autopilot_decisions row describing the rollback and marks the
    snapshot restored. Raises no exceptions for expected user errors — returns
    a structured response with `ok=False` and a machine-readable `reason`.
    """
    preview = await preview_rollback(db, decision_id)
    if not preview.get("reversible", False):
        return {
            "ok": False,
            "decision_id": int(decision_id),
            "reason": preview.get("reason"),
            "message": preview.get("message"),
            "decision": preview.get("decision"),
            "snapshot": preview.get("snapshot"),
        }

    # Re-fetch bound ORM objects in this session (preview fetched them but
    # serialized, so we need fresh instances here).
    decision = await _fetch_decision(db, decision_id)
    assert decision is not None  # preview would have returned not-found
    snapshot = await _fetch_snapshot_for_decision(db, decision)
    assert snapshot is not None

    steps = _plan_rollback_steps(snapshot)
    outcomes = await _execute_rollback_steps(
        db, steps, project_id=snapshot.project_id
    )

    now = _utcnow()
    snapshot.restored_at = now
    snapshot.restored_by = str(actor or "api")
    snapshot.restored_reason = (str(reason).strip() if reason else None)

    # Sequence of the new rollback decision: append to the end of the run.
    seq_stmt = select(AutopilotDecision).where(
        AutopilotDecision.run_id == decision.run_id
    )
    existing_rows = list((await db.execute(seq_stmt)).scalars().all())
    next_sequence = max((int(r.sequence or 0) for r in existing_rows), default=-1) + 1

    summary = (
        f"Rolled back decision {decision.id} (stage={decision.stage}) by {actor}."
    )
    rationale = reason if reason else "Autopilot rollback requested."
    rollback_row = AutopilotDecision(
        run_id=decision.run_id,
        project_id=decision.project_id,
        sequence=next_sequence,
        stage="rollback",
        status="completed",
        action="rolled_back",
        reason_code="AUTOPILOT_ROLLBACK",
        confidence=None,
        rationale=str(rationale).strip() or None,
        summary=summary,
        actor=str(actor or "api"),
        changed=True,
        safe=True,
        blocker=False,
        dry_run=False,
        intent=decision.intent,
        payload={
            "reverted_decision_id": int(decision.id),
            "reverted_decision_sequence": int(decision.sequence or 0),
            "reverted_stage": decision.stage,
            "snapshot_id": int(snapshot.id),
            "steps": steps,
            "outcomes": outcomes,
            "reason": reason or None,
        },
    )
    db.add(rollback_row)
    await db.flush()

    snapshot.restored_decision_id = int(rollback_row.id)
    await db.flush()

    return {
        "ok": True,
        "decision_id": int(decision.id),
        "snapshot": _serialize_snapshot(snapshot),
        "rollback_decision": {
            "id": int(rollback_row.id),
            "run_id": rollback_row.run_id,
            "sequence": int(rollback_row.sequence),
            "stage": rollback_row.stage,
            "status": rollback_row.status,
            "action": rollback_row.action,
            "created_at": rollback_row.created_at.isoformat() if rollback_row.created_at else None,
        },
        "outcomes": outcomes,
    }


async def purge_expired_snapshots() -> int:
    """Delete snapshots past their TTL. Returns count removed."""
    try:
        async with async_session_factory() as db:
            # Load all snapshots and filter in Python so the comparison handles
            # rows that come back as naive datetimes on SQLite.
            rows = list((await db.execute(select(AutopilotSnapshot))).scalars().all())
            now = _utcnow()
            expired = [
                row
                for row in rows
                if _as_aware_utc(row.expires_at) is not None
                and _as_aware_utc(row.expires_at) < now
            ]
            for row in expired:
                await db.delete(row)
            await db.commit()
            return len(expired)
    except Exception:
        return 0
