"""Unified timeline service (priority.md P32, Wave G).

Joins :class:`RunEvent` rows by ``project_id`` and the
``parent_run_id`` pointer into a tree-ordered timeline. One node per
``run_id``; children link upward via the parent pointer that's
populated in the P31 stage hooks (eval → exp, export → exp, etc).

The node payload is intentionally **compact** — per-node summary +
roll-ups, no full event list. Clients drill into a single run via the
existing ``GET /api/run-events/run/{run_id}`` endpoint to fetch the
ordered event stream.

Public surface:

- ``build_timeline(...)`` — windowed tree assembly, with the filter
  set spec'd in priority.md (``since`` / ``stage`` / ``severity``)
  plus convenience ``until`` and ``run_id`` (anchor on a subtree).

Stable reason codes (raised as ``ValueError``):

- ``project_not_found`` (404)
- ``invalid_stage`` / ``invalid_severity`` / ``invalid_window`` (400)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.models.run_event import (
    KNOWN_SEVERITIES,
    KNOWN_STAGES,
    RunEvent,
)


_MAX_EVENTS_PER_TIMELINE = 2000

# Severity ranking for "highest_severity" rollup. Higher number wins.
_SEVERITY_RANK: dict[str, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
    "critical": 3,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Validation helpers
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


# ---------------------------------------------------------------------------
# Per-run summarisation
# ---------------------------------------------------------------------------


def _summarise_run(
    run_id: str, events: list[RunEvent]
) -> dict[str, Any]:
    """Compact per-run rollup. ``events`` is sorted by ``ts`` ASC."""
    first = events[0]
    last = events[-1]

    severity_counts: dict[str, int] = defaultdict(int)
    stages_present: list[str] = []
    seen_stages: set[str] = set()
    parent_candidates: list[str] = []
    for ev in events:
        severity_counts[ev.severity] += 1
        if ev.stage not in seen_stages:
            seen_stages.add(ev.stage)
            stages_present.append(ev.stage)
        if ev.parent_run_id and ev.parent_run_id not in parent_candidates:
            parent_candidates.append(ev.parent_run_id)

    highest_rank = max(_SEVERITY_RANK.get(s, 0) for s in severity_counts)
    highest_severity = next(
        (sev for sev, rank in _SEVERITY_RANK.items() if rank == highest_rank),
        "info",
    )

    duration_seconds: float | None = None
    if first.ts and last.ts:
        duration_seconds = max(
            (last.ts - first.ts).total_seconds(), 0.0
        )

    # Pick the most recent reason_code (errors are the actionable part).
    latest_reason: str | None = None
    for ev in reversed(events):
        if ev.reason_code:
            latest_reason = ev.reason_code
            break

    # Pick a single parent. Multiple events emitting different parents
    # for the same run_id is a service bug; pick the first one we saw
    # so the tree is deterministic.
    parent_run_id = parent_candidates[0] if parent_candidates else None

    return {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "is_orphan": False,  # filled in during tree assembly
        "stage": first.stage,
        "stages_present": stages_present,
        "summary": last.summary,
        "actor": last.actor,
        "first_ts": first.ts.isoformat() if first.ts else None,
        "last_ts": last.ts.isoformat() if last.ts else None,
        "duration_seconds": duration_seconds,
        "event_count": len(events),
        "severity_counts": dict(severity_counts),
        "highest_severity": highest_severity,
        "latest_reason_code": latest_reason,
        "children": [],  # filled during tree assembly
    }


# ---------------------------------------------------------------------------
# Tree assembly
# ---------------------------------------------------------------------------


def _restrict_to_subtree(
    *,
    anchor_run_id: str,
    by_run: dict[str, list[RunEvent]],
    parent_index: dict[str, str | None],
) -> dict[str, list[RunEvent]]:
    """Keep only ``anchor`` + every transitive descendant.

    ``parent_index`` maps run_id → parent_run_id (or None). Walk
    children-of-anchor breadth-first.
    """
    children_index: dict[str, list[str]] = defaultdict(list)
    for run_id, parent in parent_index.items():
        if parent is not None:
            children_index[parent].append(run_id)

    keep: set[str] = set()
    queue: list[str] = [anchor_run_id]
    while queue:
        head = queue.pop(0)
        if head in keep:
            continue
        keep.add(head)
        queue.extend(children_index.get(head, []))

    return {rid: ev for rid, ev in by_run.items() if rid in keep}


def _assemble_tree(
    nodes_by_run: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Hook each node onto its parent. Returns (roots, orphaned_count).

    A node is "orphaned" when its ``parent_run_id`` is set but the parent
    isn't in ``nodes_by_run`` — i.e. the parent was filtered out or
    never emitted an event in the window. Orphans become roots, with
    ``is_orphan=True`` so the UI can label them.
    """
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    orphans = 0

    for rid, node in nodes_by_run.items():
        parent = node.get("parent_run_id")
        if parent and parent in nodes_by_run:
            children_by_parent[parent].append(rid)
        else:
            if parent:
                node["is_orphan"] = True
            roots.append(rid)

    # Sort children + roots by first_ts ASC (chronological readout).
    def _ts_key(rid: str) -> str:
        return nodes_by_run[rid].get("first_ts") or ""

    for parent_rid, child_ids in children_by_parent.items():
        child_ids.sort(key=_ts_key)
    roots.sort(key=_ts_key)

    def _attach(rid: str) -> dict[str, Any]:
        node = dict(nodes_by_run[rid])
        node["children"] = [
            _attach(child_rid)
            for child_rid in children_by_parent.get(rid, [])
        ]
        return node

    tree = [_attach(rid) for rid in roots]
    orphans = sum(1 for r in tree if r.get("is_orphan"))
    return tree, orphans


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def build_timeline(
    db: AsyncSession,
    *,
    project_id: int,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    stage: str | None = None,
    severity: str | None = None,
    run_id: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Build the tree-ordered timeline for a project.

    ``limit`` caps the **event count** loaded (not the run count). When
    the cap is hit, a ``truncated`` flag is set on the result so the UI
    can prompt the user to narrow the window.
    """
    await _ensure_project(db, project_id)

    if stage is not None and stage not in KNOWN_STAGES:
        raise ValueError(f"invalid_stage:{stage}")
    if severity is not None and severity not in KNOWN_SEVERITIES:
        raise ValueError(f"invalid_severity:{severity}")

    since_dt = since if isinstance(since, datetime) else _coerce_ts(since)
    until_dt = until if isinstance(until, datetime) else _coerce_ts(until)
    if since_dt and until_dt and since_dt >= until_dt:
        raise ValueError("invalid_window")

    bounded_limit = max(1, min(int(limit), _MAX_EVENTS_PER_TIMELINE))

    stmt = select(RunEvent).where(RunEvent.project_id == project_id)
    if stage is not None:
        stmt = stmt.where(RunEvent.stage == stage)
    if severity is not None:
        stmt = stmt.where(RunEvent.severity == severity)
    if since_dt is not None:
        stmt = stmt.where(RunEvent.ts >= since_dt)
    if until_dt is not None:
        stmt = stmt.where(RunEvent.ts <= until_dt)
    # Order by ts ASC so per-run summarisation reads chronologically;
    # the tree assembler re-sorts roots at the end.
    stmt = stmt.order_by(RunEvent.ts.asc(), RunEvent.id.asc()).limit(
        bounded_limit
    )

    result = await db.execute(stmt)
    events = list(result.scalars().all())
    truncated = len(events) >= bounded_limit

    # Group by run_id.
    by_run: dict[str, list[RunEvent]] = defaultdict(list)
    for ev in events:
        by_run[ev.run_id].append(ev)

    # Build a parent index so the subtree restriction can walk children.
    parent_index: dict[str, str | None] = {}
    for rid, run_events in by_run.items():
        parent_index[rid] = next(
            (e.parent_run_id for e in run_events if e.parent_run_id),
            None,
        )

    if run_id is not None:
        anchor = str(run_id)
        if anchor not in by_run:
            return {
                "project_id": project_id,
                "window_start": since_dt.isoformat() if since_dt else None,
                "window_end": until_dt.isoformat() if until_dt else None,
                "total_events": 0,
                "total_runs": 0,
                "orphaned_count": 0,
                "truncated": False,
                "anchor_run_id": anchor,
                "anchor_present": False,
                "tree": [],
            }
        by_run = _restrict_to_subtree(
            anchor_run_id=anchor,
            by_run=by_run,
            parent_index=parent_index,
        )

    # Summarise per run.
    nodes_by_run: dict[str, dict[str, Any]] = {
        rid: _summarise_run(rid, run_events)
        for rid, run_events in by_run.items()
    }

    # Sub-tree mode: when an anchor is set, force the anchor to be the
    # only root by clearing the parent pointer (otherwise the anchor's
    # parent — possibly outside the subtree — would mark it orphaned).
    if run_id is not None and run_id in nodes_by_run:
        nodes_by_run[run_id]["parent_run_id"] = None
        nodes_by_run[run_id]["is_orphan"] = False

    tree, orphan_count = _assemble_tree(nodes_by_run)

    total_events = sum(
        node["event_count"] for node in nodes_by_run.values()
    )

    payload: dict[str, Any] = {
        "project_id": project_id,
        "window_start": since_dt.isoformat() if since_dt else None,
        "window_end": until_dt.isoformat() if until_dt else None,
        "total_events": total_events,
        "total_runs": len(nodes_by_run),
        "orphaned_count": orphan_count,
        "truncated": truncated,
        "tree": tree,
    }
    if run_id is not None:
        payload["anchor_run_id"] = str(run_id)
        payload["anchor_present"] = run_id in nodes_by_run
    return payload
