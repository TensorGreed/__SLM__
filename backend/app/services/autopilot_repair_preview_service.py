"""Autopilot repair-preview + apply separation.

`create_preview_record(...)` persists a row capturing the dry-run orchestration
result and a `plan_token`. `apply_preview(...)` looks the row up, re-validates
state (via a `state_hash` drift check), replays the recorded request body with
`dry_run=False`, and marks the row applied so a second call is rejected.

The actual orchestration work is delegated to
`_orchestrate_newbie_autopilot_v2` in `app.api.training`; this module owns the
persistence, diff computation, drift detection, and guardrails around it.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.autopilot_repair_preview import AutopilotRepairPreview
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.project import Project


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(value: datetime | None) -> datetime | None:
    """Normalize a datetime for comparison — SQLite strips tzinfo on round-trip."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# ---------------------------------------------------------------------------
# State hashing — captures the project shape an apply must still match.
# ---------------------------------------------------------------------------

_PROJECT_HASHED_FIELDS: tuple[str, ...] = (
    "name",
    "base_model_name",
    "target_profile_id",
    "training_preferred_plan_profile",
    "evaluation_preferred_pack_id",
    "active_domain_blueprint_version",
    "domain_pack_id",
    "domain_profile_id",
    "beginner_mode",
)


def _clone_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _clone_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone_jsonable(v) for v in value]
    return getattr(value, "value", str(value))


async def compute_state_hash(db: AsyncSession, project_id: int) -> str:
    """Return a stable hex digest of the project shape that affects planning.

    Inputs: selected project fields + sorted dataset ids + sorted non-cancelled
    experiment ids + project.updated_at. Any mutation that matters for an
    autopilot plan will flip the digest.
    """
    project_result = await db.execute(select(Project).where(Project.id == int(project_id)))
    project = project_result.scalar_one_or_none()
    if project is None:
        return ""

    project_part: dict[str, Any] = {
        field: _clone_jsonable(getattr(project, field, None))
        for field in _PROJECT_HASHED_FIELDS
    }
    project_part["updated_at"] = (
        project.updated_at.isoformat() if project.updated_at else None
    )
    project_part["dataset_adapter_preset"] = _clone_jsonable(project.dataset_adapter_preset)

    dataset_rows = (
        await db.execute(
            select(Dataset.id).where(Dataset.project_id == int(project_id))
        )
    ).scalars().all()
    experiment_rows = (
        await db.execute(
            select(Experiment.id).where(Experiment.project_id == int(project_id))
        )
    ).scalars().all()

    payload = {
        "project": project_part,
        "dataset_ids": sorted(int(x) for x in dataset_rows),
        "experiment_ids": sorted(int(x) for x in experiment_rows),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Config diff — structured summary of what the apply would do.
# ---------------------------------------------------------------------------

def _summarize_repairs(repairs: dict[str, Any]) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for key, value in (repairs or {}).items():
        if value is None:
            continue
        if not isinstance(value, dict):
            planned.append({"kind": key, "raw": value})
            continue
        entry: dict[str, Any] = {"kind": key}
        # Preserve commonly-used keys for UI rendering.
        for field in (
            "applied",
            "succeeded",
            "required",
            "original_intent",
            "rewritten_intent",
            "source",
            "from_target_profile_id",
            "to_target_profile_id",
            "from_profile",
            "to_profile",
            "reason",
        ):
            if field in value:
                entry[field] = value[field]
        planned.append(entry)
    return planned


def compute_config_diff(
    *,
    response_payload: dict[str, Any],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return a structured diff from a dry-run orchestrate response.

    `response_payload` must already be a plain dict (e.g. from
    `AutopilotV2OrchestrationResponse.model_dump()`).
    """
    guardrails = dict(response_payload.get("guardrails") or {})
    can_run = bool(guardrails.get("can_run", False))
    repairs = dict(response_payload.get("repairs") or {})
    plan_v2 = dict(response_payload.get("plan_v2") or {})
    plan_options = list(plan_v2.get("plans") or [])
    selected_profile = response_payload.get("selected_profile") or plan_v2.get(
        "recommended_profile"
    )

    selected_plan = next(
        (
            plan
            for plan in plan_options
            if isinstance(plan, dict)
            and str(plan.get("profile") or "") == str(selected_profile or "")
        ),
        plan_options[0] if plan_options else {},
    )
    safe_config_preview = dict(selected_plan.get("config") or {}) if isinstance(selected_plan, dict) else {}
    preflight_ok = bool(dict(selected_plan.get("preflight") or {}).get("ok", False)) if selected_plan else False

    repairs_planned = _summarize_repairs(repairs)
    decision_log_preview = [
        {
            "step": row.get("step"),
            "status": row.get("status"),
            "summary": row.get("summary"),
            "changed": bool(row.get("changed")),
            "blocker": bool(row.get("blocker")),
        }
        for row in list(response_payload.get("decision_log") or [])
        if isinstance(row, dict)
    ]

    summary_parts: list[str] = []
    if can_run:
        base_model = str(safe_config_preview.get("base_model") or request_payload.get("base_model") or "") or "selected base model"
        profile_label = str(selected_profile or "balanced")
        summary_parts.append(
            f"Will create a new training experiment on {base_model} using the "
            f"'{profile_label}' profile and start it."
        )
    else:
        blockers = [
            str(item).strip()
            for item in list(guardrails.get("blockers") or [])
            if str(item).strip()
        ]
        head = blockers[0] if blockers else "unresolved guardrails."
        summary_parts.append(f"Apply will refuse until blockers clear: {head}")

    applied_repairs = [r for r in repairs_planned if r.get("applied") or r.get("succeeded")]
    if applied_repairs:
        summary_parts.append(
            "Auto-repairs to be applied: "
            + ", ".join(str(r.get("kind")) for r in applied_repairs)
            + "."
        )

    return {
        "summary": " ".join(summary_parts),
        "would_create_experiment": can_run,
        "would_start_training": can_run,
        "selected_profile": selected_profile,
        "effective_target_profile_id": response_payload.get("effective_target_profile_id"),
        "resolved_target_device": response_payload.get("resolved_target_device"),
        "repairs_planned": repairs_planned,
        "safe_config_preview": safe_config_preview,
        "preflight_ok": preflight_ok,
        "guardrails": guardrails,
        "decision_log_preview": decision_log_preview,
        "strict_mode": bool(response_payload.get("strict_mode")),
    }


# ---------------------------------------------------------------------------
# Persistence + apply guardrails.
# ---------------------------------------------------------------------------

def _serialize_preview(row: AutopilotRepairPreview) -> dict[str, Any]:
    return {
        "id": int(row.id),
        "plan_token": row.plan_token,
        "project_id": row.project_id,
        "intent": row.intent,
        "config_diff": dict(row.config_diff or {}),
        "state_hash": row.state_hash,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "applied_at": row.applied_at.isoformat() if row.applied_at else None,
        "applied_run_id": row.applied_run_id,
        "applied_by": row.applied_by,
        "applied_reason": row.applied_reason,
    }


def _new_plan_token() -> str:
    return uuid.uuid4().hex


async def create_preview_record(
    db: AsyncSession,
    *,
    project_id: int,
    intent: str,
    request_payload: dict[str, Any],
    dry_run_response: dict[str, Any],
) -> dict[str, Any]:
    """Persist the preview row and return its serialized form + plan_token."""
    state_hash = await compute_state_hash(db, project_id)
    config_diff = compute_config_diff(
        response_payload=dry_run_response,
        request_payload=request_payload,
    )
    row = AutopilotRepairPreview(
        plan_token=_new_plan_token(),
        project_id=int(project_id),
        intent=str(intent or "").strip() or None,
        request_payload=_clone_jsonable(request_payload or {}),
        config_diff=config_diff,
        dry_run_response=_clone_jsonable(dry_run_response or {}),
        state_hash=state_hash,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)

    serialized = _serialize_preview(row)
    return {
        "preview": serialized,
        "config_diff": config_diff,
        "dry_run_response": dry_run_response,
        "state_hash": state_hash,
    }


async def fetch_preview_by_token(
    db: AsyncSession,
    plan_token: str,
) -> AutopilotRepairPreview | None:
    token = str(plan_token or "").strip()
    if not token:
        return None
    result = await db.execute(
        select(AutopilotRepairPreview).where(AutopilotRepairPreview.plan_token == token)
    )
    return result.scalar_one_or_none()


async def get_preview_by_token(
    db: AsyncSession,
    plan_token: str,
) -> dict[str, Any] | None:
    row = await fetch_preview_by_token(db, plan_token)
    if row is None:
        return None
    return _serialize_preview(row)


async def assert_apply_allowed(
    db: AsyncSession,
    preview: AutopilotRepairPreview,
    *,
    expected_state_hash: str | None,
    force: bool,
) -> tuple[bool, dict[str, Any] | None]:
    """Return (ok, failure_payload). On failure the payload has `reason`/`message`."""
    if preview.applied_at is not None:
        return False, {
            "reason": "already_applied",
            "message": "This preview has already been applied.",
            "preview": _serialize_preview(preview),
        }

    expires_at = _as_aware_utc(preview.expires_at)
    if expires_at is not None and expires_at <= _utcnow():
        return False, {
            "reason": "preview_expired",
            "message": (
                f"Preview exceeded its retention window "
                f"(expired at {expires_at.isoformat()})."
            ),
            "preview": _serialize_preview(preview),
        }

    current_state_hash = await compute_state_hash(db, preview.project_id)
    if expected_state_hash is not None and expected_state_hash != preview.state_hash:
        return False, {
            "reason": "state_hash_mismatch",
            "message": (
                "Supplied expected_state_hash does not match the preview; "
                "the caller may be looking at a stale plan."
            ),
            "expected_state_hash": expected_state_hash,
            "preview_state_hash": preview.state_hash,
            "preview": _serialize_preview(preview),
        }
    if not force and current_state_hash != preview.state_hash:
        return False, {
            "reason": "state_drift",
            "message": (
                "Project state changed since the preview was captured. "
                "Re-run the preview or retry apply with force=true."
            ),
            "preview_state_hash": preview.state_hash,
            "current_state_hash": current_state_hash,
            "preview": _serialize_preview(preview),
        }
    return True, None


async def mark_preview_applied(
    db: AsyncSession,
    preview: AutopilotRepairPreview,
    *,
    run_id: str,
    actor: str,
    reason: str | None,
) -> dict[str, Any]:
    preview.applied_at = _utcnow()
    preview.applied_run_id = str(run_id or "") or None
    preview.applied_by = str(actor or "api") or "api"
    preview.applied_reason = (str(reason).strip() if reason else None)
    await db.flush()
    return _serialize_preview(preview)
