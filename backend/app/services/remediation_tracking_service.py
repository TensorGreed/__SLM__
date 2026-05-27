"""Remediation tracking service (E2).

Records user clicks on suggested-action buttons (forecast panel +
failure-cluster cards) and pairs each click with the lift in pass_rate
between consecutive evals so we can answer "did this fix help?" per
action kind.

Three public functions:
  * ``record_action_event`` — called from
    ``POST /api/projects/{id}/remediation/events`` when the user
    clicks a button.
  * ``stamp_evaluation_lift`` — called from
    ``evaluate_experiment_auto_gates`` after gates resolve; stamps
    every pending event between the previous eval and now with the
    pass-rate delta.
  * ``aggregate_outcomes_by_kind`` — used by the admin endpoint to
    return per-kind counts + median/mean lift + positive-lift rate.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import EvalResult, Experiment
from app.models.remediation_action_event import (
    RemediationActionEvent,
    RemediationOutcome,
)


def compute_params_hash(kind: str, params: Any) -> str:
    """Stable 16-hex-char hash of (kind + canonicalised params).
    Re-clicks of the same suggestion collapse to the same hash so the
    aggregation doesn't double-count.

    Params are JSON-serialised with sorted keys; ``None``/missing
    params hash differently from an empty dict so the caller can tell
    "no params" from "explicit empty params" downstream if needed.
    """
    try:
        payload = json.dumps(params, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(params)
    raw = f"{kind}|{payload}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def record_action_event(
    db: AsyncSession,
    *,
    project_id: int,
    kind: str,
    params: Any,
    outcome: RemediationOutcome = RemediationOutcome.CLICKED,
) -> RemediationActionEvent:
    """Insert one event row. The caller (the API layer) is responsible
    for project-membership / auth checks — this function trusts its
    inputs.
    """
    event = RemediationActionEvent(
        project_id=project_id,
        action_kind=kind,
        params_hash=compute_params_hash(kind, params),
        outcome=outcome,
    )
    db.add(event)
    await db.flush()
    await db.refresh(event)
    return event


async def stamp_evaluation_lift(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    current_pass_rate: float | None,
) -> int:
    """Resolve pending action events with the lift between the
    just-completed eval and the previous eval for the same project.

    Returns the number of events stamped. Silent no-op when either:
      - the current eval has no pass_rate (nothing to compare),
      - there's no previous eval to compare against (first run),
      - no pending events exist between the two evals.

    Lift is computed in *percentage points* (current - previous) × 100
    so a pass_rate move from 0.40 to 0.55 stamps as +15.0 rather than
    +0.15 — easier to interpret in the admin payload.
    """
    if current_pass_rate is None:
        return 0

    # Find the most-recent eval result before this experiment's evals
    # that belongs to the SAME project. We walk experiments because
    # eval_results don't carry project_id directly.
    prev_pass_rate, prev_created_at = await _previous_pass_rate(
        db, project_id=project_id, current_experiment_id=experiment_id
    )
    if prev_pass_rate is None or prev_created_at is None:
        return 0

    lift_pct = (current_pass_rate - prev_pass_rate) * 100.0

    # Pending events between the previous eval and now get stamped.
    # We don't include resolved events — re-running eval shouldn't
    # overwrite an existing lift number with a different one.
    pending_q = await db.execute(
        select(RemediationActionEvent).where(
            RemediationActionEvent.project_id == project_id,
            RemediationActionEvent.resolved_at.is_(None),
            RemediationActionEvent.observed_at >= prev_created_at,
        )
    )
    pending = list(pending_q.scalars())
    now = datetime.now(timezone.utc)
    for event in pending:
        event.evaluation_lift_pct = round(lift_pct, 4)
        event.experiment_id = experiment_id
        event.resolved_at = now
    if pending:
        await db.flush()
    return len(pending)


async def _previous_pass_rate(
    db: AsyncSession,
    *,
    project_id: int,
    current_experiment_id: int,
) -> tuple[float | None, datetime | None]:
    """Look up the most recent eval (with a non-null pass_rate) for
    the project that landed BEFORE the current experiment started.

    Returns (pass_rate, created_at) or (None, None) when no prior
    eval exists. The pair lets the caller scope the pending-event
    window without a separate query."""
    current_exp = await db.get(Experiment, current_experiment_id)
    if current_exp is None:
        return None, None
    cutoff = current_exp.started_at or current_exp.created_at

    result = await db.execute(
        select(EvalResult, Experiment.created_at, Experiment.started_at)
        .join(Experiment, Experiment.id == EvalResult.experiment_id)
        .where(
            Experiment.project_id == project_id,
            EvalResult.pass_rate.is_not(None),
            EvalResult.created_at < cutoff,
        )
        .order_by(EvalResult.created_at.desc())
        .limit(1)
    )
    row = result.first()
    if row is None:
        return None, None
    eval_result, exp_created, exp_started = row
    return eval_result.pass_rate, eval_result.created_at


async def aggregate_outcomes_by_kind(
    db: AsyncSession,
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    """Bucket event rows by action_kind and compute counts + lift
    summary. Returns one bucket when ``kind`` is given; otherwise
    returns ``by_kind`` with one bucket per distinct kind seen.

    Median + mean lift are computed in Python over the resolved
    subset (SQLite has no median aggregate). Positive-lift rate is
    the fraction of resolved events whose lift was strictly > 0 —
    a "did the fix correlate with an improvement?" headline number.
    """
    query = select(RemediationActionEvent)
    if kind:
        query = query.where(RemediationActionEvent.action_kind == kind)
    rows = (await db.execute(query)).scalars().all()

    if kind is not None:
        return {
            "kind": kind,
            "total_events": len(rows),
            **_bucket_stats(rows),
        }

    # No filter → group by kind and emit one bucket per kind seen,
    # plus an overall roll-up.
    by_kind: dict[str, list[RemediationActionEvent]] = {}
    for ev in rows:
        by_kind.setdefault(ev.action_kind, []).append(ev)
    return {
        "kind": None,
        "total_events": len(rows),
        **_bucket_stats(rows),
        "by_kind": [
            {
                "kind": k,
                "total_events": len(evs),
                **_bucket_stats(evs),
            }
            for k, evs in sorted(by_kind.items())
        ],
    }


def _bucket_stats(events: list[RemediationActionEvent]) -> dict[str, Any]:
    """Internal: per-bucket numbers (counts + lift summary). Pulled
    out of the aggregator so the same calc covers the filtered and
    grouped paths."""
    by_outcome: dict[str, int] = {}
    for ev in events:
        by_outcome[ev.outcome.value] = by_outcome.get(ev.outcome.value, 0) + 1

    resolved = [
        ev for ev in events if ev.evaluation_lift_pct is not None
    ]
    lifts = [ev.evaluation_lift_pct for ev in resolved if ev.evaluation_lift_pct is not None]
    positive = [v for v in lifts if v > 0]

    return {
        "by_outcome": by_outcome,
        "resolved_count": len(resolved),
        "positive_lift_count": len(positive),
        "positive_lift_rate": (len(positive) / len(resolved)) if resolved else None,
        "median_lift_pct": round(statistics.median(lifts), 4) if lifts else None,
        "mean_lift_pct": round(statistics.mean(lifts), 4) if lifts else None,
    }
