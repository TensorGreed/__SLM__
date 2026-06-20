"""Synth review queue service (USER-SUCCESS Epic 2b).

Lists pending synthetic rows grouped by ``synth_source``, supports
bulk accept / reject, and rewrites the project's synthetic.jsonl
file so accepted rows enter training and rejected rows are
permanently removed.

The Epic 2a write path always stamps ``review_status="pending"`` so
no synth row enters dataset prep until the user (or an auto-accept
flow) flips it. The dataset_service's JSONL loader skips pending
rows by default — this service is what un-blocks them.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType


ReviewAction = Literal["accept", "reject"]


def _synthetic_jsonl_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"


def _read_all_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _atomic_write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the full row list back to disk via a temp-file + rename
    so a crash mid-write can't truncate the synthetic dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.move(str(tmp), str(path))


def _row_preview(row: dict[str, Any], *, limit: int = 280) -> str:
    """Single-line preview string used by the queue list. Drops the
    provenance fields and serializes only the payload-shaped keys."""
    skip = {"id", "synth_source", "synth_confidence", "review_status", "status", "generated_at", "source", "model"}
    payload = {k: v for k, v in row.items() if k not in skip}
    if not payload:
        payload = row
    raw = json.dumps(payload, ensure_ascii=False, default=str)
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1] + "…"


def _resolve_source_label(row: dict[str, Any]) -> str:
    """Pick a human-meaningful source for a synth row.

    Epic-2 playbook rows carry an explicit ``synth_source`` like
    ``playbook:classification:hard_negatives:vs=billing``. Legacy
    rows (from the pre-Epic-2 ``/synthetic/generate`` flow) don't
    have ``synth_source`` but do have a legacy ``source`` field
    (``teacher_model`` / ``demo_heuristic``). We surface that
    instead so the user sees ``legacy:teacher_model`` rather than
    ``playbook:unknown``.
    """
    explicit = row.get("synth_source")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    legacy = row.get("source")
    if isinstance(legacy, str) and legacy.strip():
        return f"legacy:{legacy}"
    return "legacy:manual"


# Hard cap on rows-per-group included in the response. A 1000-row
# legacy bucket would otherwise blow up the payload — and the UI
# can't usefully render 1000 rows inline anyway. The frontend gets a
# `truncated` flag + a `total_in_group` count so it can show
# "showing 25 of 1000" footers.
ACCEPTED_ROWS_PER_GROUP_CAP = 25


def _confidence_sort_key(row: dict[str, Any]) -> tuple[float, int]:
    """Sort key for confidence-ascending order: (confidence, id). Rows with a
    missing / non-numeric ``synth_confidence`` sort to the end (``inf``) — they
    carry no uncertainty signal to rank, so they shouldn't crowd out the
    genuinely-low-confidence rows a reviewer should see first."""
    conf = row.get("synth_confidence")
    conf_val = float(conf) if isinstance(conf, (int, float)) else float("inf")
    rid = row.get("id")
    return (conf_val, int(rid) if isinstance(rid, int) else 0)


def _resolve_class_label(row: dict[str, Any]) -> str:
    """Group key when grouping by class — the row's ``label`` (classification
    target). Unlabeled / non-classification rows fall into one bucket so the
    grouping never silently drops them."""
    label = row.get("label")
    if label is not None and str(label).strip():
        return str(label)
    return "(unlabeled)"


def _bucket_rows_by_source(
    rows: list[dict[str, Any]],
    *,
    rows_per_group_cap: int | None = None,
    key_fn: Any = None,
) -> list[dict[str, Any]]:
    """Group rows by a key (``synth_source`` by default, or the class label
    when ``key_fn`` is ``_resolve_class_label``), sort, project to the UI shape.

    The output group still carries its key in ``synth_source`` regardless of the
    grouping dimension, so the review UI renders either grouping transparently
    (it just shows the group key + its rows).

    When ``rows_per_group_cap`` is set, each group's ``rows`` array
    is truncated to that length + a ``truncated`` flag is emitted.
    The pending queue passes None (we want every row in the UI for
    accept/reject); the accepted view passes the cap because
    legacy buckets may have thousands of rows.
    """
    resolve = key_fn or _resolve_source_label
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = resolve(row)
        groups[source].append(row)
    grouped: list[dict[str, Any]] = []
    for source in sorted(groups):
        # Sort by confidence ASCENDING so the most uncertain rows — the ones a
        # reviewer's attention is worth most on — surface first. Rows with no
        # ``synth_confidence`` trail (we can't rank them as uncertain); ``id``
        # breaks ties for a stable order.
        entries = sorted(groups[source], key=_confidence_sort_key)
        total_in_group = len(entries)
        if rows_per_group_cap is not None and total_in_group > rows_per_group_cap:
            visible_entries = entries[:rows_per_group_cap]
            truncated = True
        else:
            visible_entries = entries
            truncated = False
        grouped.append({
            "synth_source": source,
            "count": total_in_group,
            "truncated": truncated,
            "rows": [
                {
                    "id": row.get("id"),
                    "synth_confidence": row.get("synth_confidence"),
                    "preview": _row_preview(row),
                    "payload": {
                        k: v for k, v in row.items()
                        if k not in {"id", "synth_source", "synth_confidence", "review_status", "status"}
                    },
                }
                for row in visible_entries
            ],
        })
    return grouped


async def list_review_queue(
    db: AsyncSession,
    project_id: int,
    *,
    group_by: str = "source",
) -> dict[str, Any]:
    """Return synth rows for a project, split into pending + accepted
    groups (keyed by ``synth_source``, or by class label when
    ``group_by="class"`` — Epic E).

    The pending list is the active review queue. The accepted list
    shows the rows that already passed review and will enter the
    next dataset prep — answers the user's question of *where do
    approved synth rows show up?*"""
    # Confirm the Synthetic dataset exists for the project — if it
    # doesn't, the queue is empty, return an empty payload.
    ds_result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.SYNTHETIC,
        )
    )
    dataset = ds_result.scalar_one_or_none()
    path = _synthetic_jsonl_path(project_id)
    all_rows = _read_all_rows(path)
    pending = [r for r in all_rows if r.get("review_status") == "pending"]
    # "Accepted" = explicitly accepted, OR no review_status field
    # (legacy rows from the pre-Epic-2a flow). Both will enter
    # training. Anything else (rejected etc.) is excluded.
    accepted = [
        r for r in all_rows
        if r.get("review_status") in (None, "accepted")
    ]
    # Arc 5 — soft-reject. Rejected rows used to be physically
    # deleted; now they stay on disk with ``review_status=rejected``
    # so the user can review what was rejected, recover an
    # accidentally-rejected row, and bulk-purge with reason
    # filtering. Matches the project preference: rejected rows
    # are selectable + bulk-droppable rather than vanishing
    # immediately.
    rejected = [r for r in all_rows if r.get("review_status") == "rejected"]

    key_fn = _resolve_class_label if group_by == "class" else _resolve_source_label

    return {
        "project_id": project_id,
        "dataset_id": dataset.id if dataset else None,
        "group_by": "class" if group_by == "class" else "source",
        "total_rows": len(all_rows),
        "total_pending": len(pending),
        "total_accepted": len(accepted),
        "total_rejected": len(rejected),
        "groups": _bucket_rows_by_source(pending, key_fn=key_fn),
        "accepted_groups": _bucket_rows_by_source(
            accepted, rows_per_group_cap=ACCEPTED_ROWS_PER_GROUP_CAP, key_fn=key_fn,
        ),
        "rejected_groups": _bucket_rows_by_source(
            rejected, rows_per_group_cap=ACCEPTED_ROWS_PER_GROUP_CAP, key_fn=key_fn,
        ),
    }


async def bulk_update_review_queue(
    db: AsyncSession,
    project_id: int,
    *,
    row_ids: list[int],
    action: ReviewAction,
    reject_reason: str | None = None,
) -> dict[str, Any]:
    """Bulk apply an accept / reject to a set of pending rows.

    - ``accept`` flips ``review_status`` from ``"pending"`` to
      ``"accepted"`` for matching rows. Accepted rows are picked up
      by the next dataset prep run.
    - ``reject`` flips ``review_status`` to ``"rejected"`` AND
      stamps ``reject_reason`` (when supplied) on the row. Rows
      stay on disk so the user can:
        * review what was rejected (Rejected section in the UI);
        * recover an accidentally-rejected row by re-marking it
          ``pending`` (future feature);
        * bulk-purge a reason cohort via ``purge_rejected_rows``.
      Prior behavior (physical delete) ran against the project
      preference "rejected rows are selectable + bulk-droppable" —
      vanishing rows aren't selectable.

    Returns a summary dict with counts (`accepted` / `rejected` /
    `not_found` / `not_pending`) so the UI can show what landed.
    """
    if action not in ("accept", "reject"):
        raise ValueError("action must be 'accept' or 'reject'")
    if reject_reason is not None:
        reject_reason = str(reject_reason).strip() or None
    if not row_ids:
        return {
            "accepted": 0,
            "rejected": 0,
            "not_found": 0,
            "not_pending": 0,
            "total_remaining_pending": 0,
        }

    target_ids = set(int(rid) for rid in row_ids)
    path = _synthetic_jsonl_path(project_id)
    rows = _read_all_rows(path)

    accepted_count = 0
    rejected_count = 0
    not_pending = 0
    new_rows: list[dict[str, Any]] = []
    matched_ids: set[int] = set()

    for row in rows:
        rid = row.get("id")
        if isinstance(rid, int) and rid in target_ids:
            matched_ids.add(rid)
            status = row.get("review_status")
            if status != "pending":
                # Already accepted (or rejected, in which case it
                # shouldn't be on disk; defensive). Keep as-is.
                not_pending += 1
                new_rows.append(row)
                continue
            if action == "accept":
                row["review_status"] = "accepted"
                accepted_count += 1
                new_rows.append(row)
            else:  # reject — soft (mark, don't delete)
                row["review_status"] = "rejected"
                if reject_reason:
                    row["reject_reason"] = reject_reason
                rejected_count += 1
                new_rows.append(row)
        else:
            new_rows.append(row)

    not_found = len(target_ids - matched_ids)

    if accepted_count + rejected_count > 0:
        _atomic_write_rows(path, new_rows)
        # Update the Dataset row's record_count to match the file.
        ds_result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.SYNTHETIC,
            )
        )
        dataset = ds_result.scalar_one_or_none()
        if dataset is not None:
            dataset.record_count = len(new_rows)
            await db.flush()

    remaining_pending = sum(1 for r in new_rows if r.get("review_status") == "pending")

    return {
        "accepted": accepted_count,
        "rejected": rejected_count,
        "not_found": not_found,
        "not_pending": not_pending,
        "total_remaining_pending": remaining_pending,
    }


async def bulk_update_by_source(
    db: AsyncSession,
    project_id: int,
    *,
    source: str,
    action: ReviewAction,
    reject_reason: str | None = None,
) -> dict[str, Any]:
    """Accept / reject *every pending row* in one ``synth_source`` group by
    key — no row-id enumeration needed.

    This is what makes the Data Studio review-queue panel actionable: its
    cross-workflow summary surfaces pending groups by source + count, so a
    one-click "Accept all (N)" / "Reject all (N)" on a group is the natural
    bulk. We resolve the matching pending row ids (via the same
    ``_resolve_source_label`` the grouping uses, so a group's key always maps
    back to its rows) and delegate to the tested ``bulk_update_review_queue``
    — no second mutation path to keep in sync.

    Returns the same summary as ``bulk_update_review_queue`` plus the echoed
    ``source`` and ``matched`` (pending rows in the group at call time)."""
    if action not in ("accept", "reject"):
        raise ValueError("action must be 'accept' or 'reject'")
    source = str(source or "").strip()
    if not source:
        raise ValueError("source must be a non-empty synth_source key")

    path = _synthetic_jsonl_path(project_id)
    rows = _read_all_rows(path)
    matched_ids = [
        int(row["id"])
        for row in rows
        if row.get("review_status") == "pending"
        and isinstance(row.get("id"), int)
        and _resolve_source_label(row) == source
    ]

    summary = await bulk_update_review_queue(
        db,
        project_id,
        row_ids=matched_ids,
        action=action,
        reject_reason=reject_reason,
    )
    if not matched_ids:
        # bulk_update_review_queue early-returns a zero summary (with a
        # hardcoded total_remaining_pending=0) when handed no ids — it never
        # reads the file. The group simply had no pending rows; report the
        # project's *actual* remaining pending so the panel doesn't falsely
        # show "0 left to review".
        summary["total_remaining_pending"] = sum(
            1 for row in rows if row.get("review_status") == "pending"
        )
    summary["source"] = source
    summary["matched"] = len(matched_ids)
    return summary


async def purge_rejected_rows(
    db: AsyncSession,
    project_id: int,
    *,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Physically remove rejected rows from synthetic.jsonl. The
    soft-reject in ``bulk_update_review_queue`` keeps rejected
    rows around so the user can review them; this is the explicit
    "I've reviewed the rejected pile, now delete it" step that
    drops them for good.

    ``reasons`` filters by ``reject_reason`` — when set, only
    rejected rows whose reason is in the list (or who have a
    matching empty-reason entry when ``""`` is passed) get
    purged. When omitted, every rejected row is removed.

    Returns a summary dict ``{purged: <count>, retained: <count>,
    total_rows: <count>}`` so the UI can confirm what landed.
    """
    path = _synthetic_jsonl_path(project_id)
    rows = _read_all_rows(path)
    if not rows:
        return {"purged": 0, "retained": 0, "total_rows": 0}

    reason_filter: set[str] | None = None
    if reasons is not None:
        reason_filter = {
            str(r).strip() for r in reasons if isinstance(r, str)
        }
        # Empty-set filter would purge nothing — treat as "no
        # filter" for caller convenience.
        if not reason_filter:
            reason_filter = None

    purged_count = 0
    retained_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("review_status") != "rejected":
            retained_rows.append(row)
            continue
        if reason_filter is not None:
            row_reason = str(row.get("reject_reason") or "").strip()
            if row_reason not in reason_filter:
                retained_rows.append(row)
                continue
        purged_count += 1
        # Don't append — row is physically dropped here.

    if purged_count > 0:
        _atomic_write_rows(path, retained_rows)
        # Keep the Dataset's record_count in sync with the file
        # the same way bulk_update_review_queue does — the
        # next dataset-prep read counts rows + would silently
        # mismatch otherwise.
        ds_result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.SYNTHETIC,
            )
        )
        dataset = ds_result.scalar_one_or_none()
        if dataset is not None:
            dataset.record_count = len(retained_rows)
            await db.flush()

    return {
        "purged": purged_count,
        "retained": len(retained_rows),
        "total_rows": len(retained_rows),
    }
