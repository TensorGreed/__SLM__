"""Autopilot API — decision-log query + rollback + repair preview/apply surface.

This router owns the `/autopilot/*` path space. Priorities covered:

- P1 — decision-log reads (`GET /autopilot/decisions`, run-scoped reads).
- P2 — snapshot + rollback (`GET /autopilot/snapshots`,
  `POST /autopilot/rollback/{decision_id}` and its `/preview` sibling).
- P3 — repair preview/apply separation
  (`POST /autopilot/repair-preview`, `POST /autopilot/repair-apply`,
  `GET /autopilot/repair-previews/{plan_token}`).

Follow-up priority P4 (CLI parity) will add CLI-facing helpers here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.autopilot_decision_service import (
    get_run_decisions,
    list_decisions,
)
from app.services.autopilot_repair_preview_service import (
    assert_apply_allowed,
    create_preview_record,
    fetch_preview_by_token,
    get_preview_by_token,
    mark_preview_applied,
)
from app.services.autopilot_snapshot_service import (
    get_snapshot_for_decision,
    list_snapshots,
    preview_rollback,
    purge_expired_snapshots,
    rollback_decision,
)


_ROLLBACK_FAILURE_STATUS: dict[str, int] = {
    "decision_not_found": 404,
    "no_snapshot": 409,
    "already_rolled_back": 409,
    "snapshot_expired": 410,
}


_APPLY_FAILURE_STATUS: dict[str, int] = {
    "preview_not_found": 404,
    "already_applied": 409,
    "preview_expired": 410,
    "state_drift": 409,
    "state_hash_mismatch": 409,
}


class AutopilotRollbackRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=1024)
    actor: Optional[str] = Field(default=None, max_length=64)


class AutopilotRepairApplyRequest(BaseModel):
    plan_token: str = Field(..., min_length=8, max_length=64)
    actor: Optional[str] = Field(default=None, max_length=64)
    reason: Optional[str] = Field(default=None, max_length=1024)
    expected_state_hash: Optional[str] = Field(default=None, max_length=64)
    force: bool = False


router = APIRouter(prefix="/autopilot", tags=["Autopilot"])


@router.get("/decisions")
async def list_autopilot_decisions(
    project_id: Optional[int] = Query(default=None, description="Filter by project."),
    run_id: Optional[str] = Query(default=None, description="Filter by autopilot run id."),
    stage: Optional[str] = Query(default=None, description="Filter by pipeline stage / step."),
    status: Optional[str] = Query(default=None, description="Filter by decision status."),
    action: Optional[str] = Query(default=None, description="Filter by derived action."),
    reason_code: Optional[str] = Query(default=None, description="Filter by reason code."),
    since: Optional[datetime] = Query(default=None, description="Created at >= (ISO-8601)."),
    until: Optional[datetime] = Query(default=None, description="Created at <= (ISO-8601)."),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List autopilot decision-log entries with optional filters."""
    return await list_decisions(
        db,
        project_id=project_id,
        run_id=run_id,
        stage=stage,
        status=status,
        action=action,
        reason_code=reason_code,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}/decisions")
async def list_decisions_for_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return every decision-log entry for one autopilot run, ordered by sequence."""
    payload = await get_run_decisions(db, run_id)
    if not payload.get("items"):
        raise HTTPException(404, f"No autopilot decisions recorded for run_id {run_id!r}.")
    return payload


@router.get("/snapshots")
async def list_autopilot_snapshots(
    project_id: Optional[int] = Query(default=None, description="Filter by project."),
    run_id: Optional[str] = Query(default=None, description="Filter by autopilot run id."),
    include_restored: bool = Query(default=True, description="Include already-rolled-back snapshots."),
    include_expired: bool = Query(default=True, description="Include expired snapshots."),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List captured autopilot snapshots (rollback candidates)."""
    return await list_snapshots(
        db,
        project_id=project_id,
        run_id=run_id,
        include_restored=include_restored,
        include_expired=include_expired,
        limit=limit,
        offset=offset,
    )


@router.get("/decisions/{decision_id}/snapshot")
async def get_decision_snapshot(
    decision_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the snapshot associated with a decision, or 404 if none exists."""
    snapshot = await get_snapshot_for_decision(db, decision_id)
    if snapshot is None:
        raise HTTPException(
            404, f"No snapshot recorded for decision {decision_id}."
        )
    return snapshot


@router.post("/rollback/{decision_id}/preview")
async def preview_autopilot_rollback(
    decision_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a non-mutating preview of what rolling back `decision_id` would do."""
    return await preview_rollback(db, decision_id)


@router.post("/rollback/{decision_id}")
async def execute_autopilot_rollback(
    decision_id: int,
    req: Optional[AutopilotRollbackRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """Roll back the given autopilot decision if a valid snapshot exists.

    Returns a structured error response with HTTP 404/409/410 when the decision
    is not reversible (see `_ROLLBACK_FAILURE_STATUS` for the mapping). On
    success, returns the updated snapshot, the new rollback-decision row, and
    per-step outcomes so the caller can render a confirmation UI.
    """
    actor = (req.actor if req and req.actor else "api").strip() or "api"
    reason = (req.reason if req else None)

    result = await rollback_decision(
        db,
        decision_id,
        actor=actor,
        reason=reason,
    )

    if not result.get("ok"):
        status_code = _ROLLBACK_FAILURE_STATUS.get(
            str(result.get("reason") or ""), 409
        )
        raise HTTPException(status_code=status_code, detail=result)

    return result


@router.post("/snapshots/purge-expired")
async def purge_expired_autopilot_snapshots():
    """Admin operation: delete snapshots past their TTL.

    Returns the number of rows removed. Intended for cron/scheduled use; safe
    to call ad-hoc because purge logic swallows its own errors.
    """
    removed = await purge_expired_snapshots()
    return {"removed": int(removed)}


# ---------------------------------------------------------------------------
# P3 — repair preview/apply separation.
#
# The two endpoints below wrap the existing orchestrator (from app.api.training)
# so we can persist the dry-run plan, hand the caller a plan_token, and require
# an explicit `repair-apply` before any mutation happens. This gives the UI a
# safe diff-review step and lets a CI/agent check the plan before committing.
# ---------------------------------------------------------------------------


@router.post("/repair-preview")
async def autopilot_repair_preview(
    payload: dict,
    db: AsyncSession = Depends(get_db),
):
    """Dry-run the autopilot plan, persist the result, and return a plan token.

    The body accepts every field of `AutopilotV2OrchestrationRequest` plus a
    top-level `project_id`. Any `dry_run` value in the body is ignored — the
    preview is always dry. Use the returned `plan_token` with
    `POST /autopilot/repair-apply` to actually execute the plan.
    """
    # Imported lazily to avoid a module-level circular import with training.py.
    from app.api.training import (
        AutopilotV2OrchestrationRequest,
        _get_project_or_404,
        _orchestrate_newbie_autopilot_v2,
    )

    body = dict(payload or {})
    project_id = body.pop("project_id", None)
    if project_id is None:
        raise HTTPException(
            status_code=422, detail="`project_id` is required in the request body."
        )

    try:
        request_model = AutopilotV2OrchestrationRequest(**body)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"reason": "invalid_request", "message": str(exc)},
        )

    dry_request = request_model.model_copy(update={"dry_run": True})
    await _get_project_or_404(db, int(project_id))

    try:
        response = await _orchestrate_newbie_autopilot_v2(
            db=db,
            project_id=int(project_id),
            req=dry_request,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    dry_run_response = response.model_dump()
    persisted = await create_preview_record(
        db,
        project_id=int(project_id),
        intent=str(request_model.intent or ""),
        request_payload=request_model.model_dump(),
        dry_run_response=dry_run_response,
    )
    return persisted


@router.get("/repair-previews/{plan_token}")
async def get_autopilot_repair_preview(
    plan_token: str,
    db: AsyncSession = Depends(get_db),
):
    """Return a previously created preview by token."""
    row = await get_preview_by_token(db, plan_token)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"reason": "preview_not_found", "message": f"Unknown plan_token {plan_token!r}."},
        )
    return row


@router.post("/repair-apply")
async def autopilot_repair_apply(
    req: AutopilotRepairApplyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Execute a previously previewed plan, after state-drift re-validation.

    Refuses with 404/409/410 when the plan is unknown, already applied,
    expired, or the project state has changed since the preview was captured
    (see `_APPLY_FAILURE_STATUS` for the mapping). Pass `force=true` to skip
    state-drift detection.
    """
    from app.api.training import (
        AutopilotV2OrchestrationRequest,
        _get_project_or_404,
        _orchestrate_newbie_autopilot_v2,
    )

    preview = await fetch_preview_by_token(db, req.plan_token)
    if preview is None:
        raise HTTPException(
            status_code=404,
            detail={
                "reason": "preview_not_found",
                "message": f"Unknown plan_token {req.plan_token!r}.",
            },
        )

    ok, failure = await assert_apply_allowed(
        db,
        preview,
        expected_state_hash=req.expected_state_hash,
        force=bool(req.force),
    )
    if not ok and failure:
        status_code = _APPLY_FAILURE_STATUS.get(str(failure.get("reason") or ""), 409)
        raise HTTPException(status_code=status_code, detail=failure)

    request_payload = dict(preview.request_payload or {})
    try:
        request_model = AutopilotV2OrchestrationRequest(**request_payload)
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "invalid_request_replay",
                "message": f"Persisted request failed validation: {exc}",
            },
        )

    apply_request = request_model.model_copy(update={"dry_run": False})
    await _get_project_or_404(db, int(preview.project_id))
    try:
        response = await _orchestrate_newbie_autopilot_v2(
            db=db,
            project_id=int(preview.project_id),
            req=apply_request,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    actor = (req.actor or "api").strip() or "api"
    serialized_preview = await mark_preview_applied(
        db,
        preview,
        run_id=response.run_id,
        actor=actor,
        reason=req.reason,
    )
    return {
        "ok": True,
        "preview": serialized_preview,
        "response": response.model_dump(),
    }
