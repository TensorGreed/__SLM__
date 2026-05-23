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


def _bucket_rows_by_source(
    rows: list[dict[str, Any]],
    *,
    rows_per_group_cap: int | None = None,
) -> list[dict[str, Any]]:
    """Group rows by synth_source, sort, project to the UI shape.

    When ``rows_per_group_cap`` is set, each group's ``rows`` array
    is truncated to that length + a ``truncated`` flag is emitted.
    The pending queue passes None (we want every row in the UI for
    accept/reject); the accepted view passes the cap because
    legacy buckets may have thousands of rows.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = _resolve_source_label(row)
        groups[source].append(row)
    grouped: list[dict[str, Any]] = []
    for source in sorted(groups):
        entries = sorted(groups[source], key=lambda r: r.get("id") or 0)
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
) -> dict[str, Any]:
    """Return synth rows for a project, split into pending + accepted
    groups (both keyed by ``synth_source``).

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

    return {
        "project_id": project_id,
        "dataset_id": dataset.id if dataset else None,
        "total_rows": len(all_rows),
        "total_pending": len(pending),
        "total_accepted": len(accepted),
        "groups": _bucket_rows_by_source(pending),
        "accepted_groups": _bucket_rows_by_source(
            accepted, rows_per_group_cap=ACCEPTED_ROWS_PER_GROUP_CAP,
        ),
    }


async def bulk_update_review_queue(
    db: AsyncSession,
    project_id: int,
    *,
    row_ids: list[int],
    action: ReviewAction,
) -> dict[str, Any]:
    """Bulk apply an accept / reject to a set of pending rows.

    - ``accept`` flips ``review_status`` from ``"pending"`` to
      ``"accepted"`` for matching rows. Accepted rows are picked up
      by the next dataset prep run.
    - ``reject`` REMOVES the rows from synthetic.jsonl entirely.
      Rejected rows can't be recovered — they're considered bad
      synthetic data and should not enter the training corpus, the
      review queue, or any future eval.

    Returns a summary dict with counts (`accepted` / `rejected` /
    `not_found` / `not_pending`) so the UI can show what landed.
    """
    if action not in ("accept", "reject"):
        raise ValueError("action must be 'accept' or 'reject'")
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
            else:  # reject
                rejected_count += 1
                # Don't append — row is dropped from disk.
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
