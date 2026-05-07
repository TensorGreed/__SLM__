"""Failure clustering over RunEvents (priority.md P33, Wave G).

Groups ``severity in {error, critical}`` :class:`RunEvent` rows by
``(project_id, stage, reason_code, signature)`` into
:class:`FailureCluster` rows that the failure-analysis surface (P36)
and support bundle (P34) can read directly.

This is a **separate abstraction** from
:mod:`app.services.failure_cluster_service` (P12), which clusters
**per-row** evaluation failures within a single ``EvalResult``. P12
operates on prediction-vs-reference pairs; this service operates on
the cross-stage RunEvent log. Both ship a ``failure-clusters``
surface, on different routes (``/evaluation/{eval_result_id}/
failure-clusters`` vs ``/projects/{id}/failure-clusters``).

Public surface:

- ``compute_failure_clusters(...)`` — recompute clusters for a project
  from the event log. **Idempotent** — running twice on the same data
  produces the same persisted state. The signature dimension makes
  cluster identity stable across recomputes, and we upsert by
  ``(project_id, stage, reason_code, signature)``.
- ``list_failure_clusters(...)`` — read path for the API + UI.

The signature is the first 12 hex chars of
``sha1(stage|reason_code|normalised_summary)``. Normalisation strips
ids, timestamps, and digit runs so "OOM at step 1200" and "OOM at
step 4500" collapse into a single signature.

Spec'd as a "nightly job" but this codebase has no scheduler — the API
exposes a recompute endpoint so an operator (or future cron) can drive
it explicitly.

Stable reason codes (raised as ``ValueError``):
- ``project_not_found`` (404)
- ``invalid_window`` (400)
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.failure_cluster import FailureCluster
from app.models.project import Project
from app.models.run_event import (
    SEVERITY_CRITICAL,
    SEVERITY_ERROR,
    RunEvent,
)


_MAX_EXEMPLARS_PER_CLUSTER = 5
_MAX_EVENTS_PER_COMPUTE = 10_000
_FAILURE_SEVERITIES = (SEVERITY_ERROR, SEVERITY_CRITICAL)

# Patterns stripped from the summary before hashing — these are the
# things that vary between otherwise-identical failures and would
# otherwise prevent clustering. Order matters: more specific first.
_NORMALISE_PATTERNS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"),  # ISO ts
    re.compile(r"\b[0-9a-f]{8,}\b"),  # hex tokens / hashes
    re.compile(r"\b\d+\b"),  # numbers (step, batch, line counts)
    re.compile(r"\s+"),  # whitespace runs
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


async def _ensure_project(db: AsyncSession, project_id: int) -> None:
    result = await db.execute(select(Project).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise ValueError("project_not_found")


def _normalise_summary(summary: str | None) -> str:
    """Strip variable bits (timestamps, hex tokens, digits, whitespace)
    so two errors that differ only by step / line / hash cluster together."""
    raw = (summary or "").strip().lower()
    if not raw:
        return ""
    for pattern in _NORMALISE_PATTERNS:
        raw = pattern.sub(" ", raw)
    return raw.strip()


def compute_signature(*, stage: str, reason_code: str, summary: str | None) -> str:
    """12-char hex digest of (stage|reason_code|normalised_summary).

    Public so callers (tests, ad-hoc analysis) can predict cluster
    identity without going through the DB.
    """
    body = f"{stage}|{reason_code}|{_normalise_summary(summary)}"
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


async def compute_failure_clusters(
    db: AsyncSession,
    *,
    project_id: int,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
) -> dict[str, Any]:
    """Recompute persisted failure clusters from the event log.

    Idempotent — re-running on the same window produces the same set
    of cluster rows with the same counts. New events between runs are
    folded into existing clusters (or new ones); clusters whose
    underlying events have been pruned simply stop being touched.
    """
    await _ensure_project(db, project_id)

    since_dt = since if isinstance(since, datetime) else _coerce_ts(since)
    until_dt = until if isinstance(until, datetime) else _coerce_ts(until)
    if since_dt and until_dt and since_dt >= until_dt:
        raise ValueError("invalid_window")

    stmt = select(RunEvent).where(
        RunEvent.project_id == project_id,
        RunEvent.severity.in_(list(_FAILURE_SEVERITIES)),
    )
    if since_dt is not None:
        stmt = stmt.where(RunEvent.ts >= since_dt)
    if until_dt is not None:
        stmt = stmt.where(RunEvent.ts <= until_dt)
    stmt = stmt.order_by(RunEvent.ts.asc(), RunEvent.id.asc()).limit(
        _MAX_EVENTS_PER_COMPUTE
    )

    rows = list((await db.execute(stmt)).scalars().all())

    # Group events by (stage, reason_code, signature). Skip events that
    # somehow lack a reason_code — ``emit_event`` rejects this at write
    # time but old rows might exist if the taxonomy was added later.
    grouped: dict[tuple[str, str, str], list[RunEvent]] = defaultdict(list)
    skipped_no_reason = 0
    for ev in rows:
        if not ev.reason_code:
            skipped_no_reason += 1
            continue
        sig = compute_signature(
            stage=ev.stage,
            reason_code=ev.reason_code,
            summary=ev.summary,
        )
        grouped[(ev.stage, ev.reason_code, sig)].append(ev)

    # Load existing clusters for this project so the upsert can update
    # in place without an INSERT-OR-IGNORE dance (portable across
    # SQLite and Postgres).
    existing_stmt = select(FailureCluster).where(
        FailureCluster.project_id == project_id
    )
    existing_rows = list((await db.execute(existing_stmt)).scalars().all())
    existing_by_key: dict[tuple[str, str, str], FailureCluster] = {
        (row.stage, row.reason_code, row.signature): row
        for row in existing_rows
    }

    now = _utcnow()
    touched: set[tuple[str, str, str]] = set()
    created = 0
    updated = 0

    for key, events in grouped.items():
        events_sorted = sorted(events, key=lambda e: e.ts or now)
        first_ts = events_sorted[0].ts or now
        last_ts = events_sorted[-1].ts or now
        # Newest-first exemplar list, capped.
        exemplars = list(reversed(events_sorted))[
            :_MAX_EXEMPLARS_PER_CLUSTER
        ]
        exemplar_ids = [e.id for e in exemplars]
        exemplar_summaries = [e.summary or "" for e in exemplars]

        existing = existing_by_key.get(key)
        if existing is None:
            row = FailureCluster(
                project_id=project_id,
                stage=key[0],
                reason_code=key[1],
                signature=key[2],
                failure_count=len(events_sorted),
                first_seen_at=first_ts,
                last_seen_at=last_ts,
                exemplar_event_ids=exemplar_ids,
                exemplar_summaries=exemplar_summaries,
                last_computed_at=now,
            )
            db.add(row)
            created += 1
        else:
            existing.failure_count = len(events_sorted)
            # Preserve the original first_seen_at if older than the
            # newly-computed one.
            if existing.first_seen_at and first_ts:
                if first_ts < existing.first_seen_at:
                    existing.first_seen_at = first_ts
            else:
                existing.first_seen_at = first_ts
            existing.last_seen_at = last_ts
            existing.exemplar_event_ids = exemplar_ids
            existing.exemplar_summaries = exemplar_summaries
            existing.last_computed_at = now
            updated += 1
        touched.add(key)

    await db.flush()

    return {
        "project_id": project_id,
        "window_start": since_dt.isoformat() if since_dt else None,
        "window_end": until_dt.isoformat() if until_dt else None,
        "events_considered": len(rows),
        "events_skipped_no_reason_code": skipped_no_reason,
        "clusters_total": len(touched),
        "clusters_created": created,
        "clusters_updated": updated,
        "computed_at": now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


def _serialize_cluster(row: FailureCluster) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "stage": row.stage,
        "reason_code": row.reason_code,
        "signature": row.signature,
        "failure_count": row.failure_count,
        "first_seen_at": (
            row.first_seen_at.isoformat() if row.first_seen_at else None
        ),
        "last_seen_at": (
            row.last_seen_at.isoformat() if row.last_seen_at else None
        ),
        "exemplar_event_ids": list(row.exemplar_event_ids or []),
        "exemplar_summaries": list(row.exemplar_summaries or []),
        "last_computed_at": (
            row.last_computed_at.isoformat()
            if row.last_computed_at
            else None
        ),
    }


async def list_failure_clusters(
    db: AsyncSession,
    *,
    project_id: int,
    stage: str | None = None,
    reason_code: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    await _ensure_project(db, project_id)

    bounded = max(1, min(int(limit), 500))
    stmt = select(FailureCluster).where(
        FailureCluster.project_id == project_id
    )
    if stage is not None:
        stmt = stmt.where(FailureCluster.stage == stage)
    if reason_code is not None:
        stmt = stmt.where(FailureCluster.reason_code == reason_code)
    # Surface the most painful clusters first: highest count, then
    # most recent.
    stmt = stmt.order_by(
        FailureCluster.failure_count.desc(),
        FailureCluster.last_seen_at.desc(),
    ).limit(bounded)

    rows = list((await db.execute(stmt)).scalars().all())
    return {
        "project_id": project_id,
        "limit": bounded,
        "clusters": [_serialize_cluster(r) for r in rows],
    }
