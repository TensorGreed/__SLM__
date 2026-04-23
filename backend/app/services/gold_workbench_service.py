"""Gold-set annotation workbench service (priority.md P10).

Responsibilities:
- Sample rows from a source Dataset (random or stratified) into the active draft
  version of a gold set, skipping duplicates by stable source-row key.
- Update individual row annotation state (expected / rationale / labels / status /
  reviewer) and keep the per-reviewer queue table in lockstep with row state.
- List the reviewer queue with optional filtering.

Design notes:
- "gold set" is represented by a Dataset row with `dataset_type ∈ {GOLD_DEV, GOLD_TEST}`.
  We validate this shape on every call; no new gold-set identity is introduced.
- Rows always belong to a `GoldSetVersion`. We auto-create a draft version on the
  first sample call. Locking is not yet exposed (the workbench frontend / P11
  will gate that).
- The reviewer queue is a *projection* of row state. Assigning a reviewer via
  PATCH creates/updates the queue row; clearing it deletes the queue row.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.gold_set_annotation import (
    GoldSetReviewerQueue,
    GoldSetReviewerQueueStatus,
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)


_GOLD_DATASET_TYPES = {DatasetType.GOLD_DEV, DatasetType.GOLD_TEST}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _stable_row_key(payload: Any) -> str:
    """Stable, short key used to dedup sampled rows within a version."""
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(row if isinstance(row, dict) else {"value": row})
    return records


def _stratified_indices(
    rows: list[dict[str, Any]],
    *,
    stratify_by: str,
    target_count: int,
    rng: random.Random,
) -> list[int]:
    """Return a list of row indices proportionally sampled across stratify_by buckets.

    Ties in the proportional allocation are broken by rounding down and
    topping up from the largest buckets.
    """
    buckets: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        key = row.get(stratify_by)
        bucket_key = str(key) if key is not None else "__missing__"
        buckets.setdefault(bucket_key, []).append(idx)

    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return []

    target = min(target_count, total)
    initial: dict[str, int] = {}
    for key, members in buckets.items():
        share = int(target * len(members) / total)
        initial[key] = min(share, len(members))

    picked = sum(initial.values())
    leftover = target - picked
    if leftover > 0:
        ordered = sorted(
            buckets.keys(),
            key=lambda key: len(buckets[key]) - initial[key],
            reverse=True,
        )
        for key in ordered:
            if leftover == 0:
                break
            capacity = len(buckets[key]) - initial[key]
            if capacity <= 0:
                continue
            take = min(capacity, leftover)
            initial[key] += take
            leftover -= take

    chosen: list[int] = []
    for key, members in buckets.items():
        count = initial.get(key, 0)
        if count <= 0:
            continue
        shuffled = list(members)
        rng.shuffle(shuffled)
        chosen.extend(shuffled[:count])
    rng.shuffle(chosen)
    return chosen


async def _require_gold_set(db: AsyncSession, gold_set_id: int) -> Dataset:
    dataset = await db.get(Dataset, gold_set_id)
    if dataset is None:
        raise ValueError("gold_set_not_found")
    if dataset.dataset_type not in _GOLD_DATASET_TYPES:
        raise ValueError("not_a_gold_set")
    return dataset


async def _get_source_dataset(
    db: AsyncSession,
    *,
    project_id: int,
    source_dataset_id: int,
) -> Dataset:
    source = await db.get(Dataset, source_dataset_id)
    if source is None:
        raise ValueError("source_dataset_not_found")
    if source.project_id != project_id:
        raise ValueError("source_dataset_wrong_project")
    return source


async def ensure_draft_version(
    db: AsyncSession,
    *,
    gold_set_id: int,
    user_id: int | None = None,
) -> GoldSetVersion:
    """Return the existing draft version or create a new one at the next revision."""
    existing = await db.execute(
        select(GoldSetVersion)
        .where(
            GoldSetVersion.gold_set_id == gold_set_id,
            GoldSetVersion.status == GoldSetVersionStatus.DRAFT,
        )
        .order_by(GoldSetVersion.version.desc())
        .limit(1)
    )
    draft = existing.scalar_one_or_none()
    if draft is not None:
        return draft

    max_version = await db.execute(
        select(func.max(GoldSetVersion.version)).where(
            GoldSetVersion.gold_set_id == gold_set_id
        )
    )
    next_version = int(max_version.scalar_one_or_none() or 0) + 1

    draft = GoldSetVersion(
        gold_set_id=gold_set_id,
        version=next_version,
        status=GoldSetVersionStatus.DRAFT,
        notes="",
        created_by_user_id=user_id,
    )
    db.add(draft)
    await db.flush()
    await db.refresh(draft)
    return draft


async def _existing_row_keys_in_version(
    db: AsyncSession, *, version_id: int
) -> set[str]:
    rows = await db.execute(
        select(GoldSetRow.source_row_key).where(GoldSetRow.version_id == version_id)
    )
    return {key for (key,) in rows.all() if key is not None}


async def sample_rows_from_source(
    db: AsyncSession,
    *,
    gold_set_id: int,
    source_dataset_id: int,
    target_count: int,
    strategy: str = "random",
    stratify_by: str | None = None,
    seed: int | None = None,
    reviewer_id: int | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    if target_count <= 0:
        raise ValueError("target_count_must_be_positive")

    normalized_strategy = str(strategy or "random").strip().lower()
    if normalized_strategy not in {"random", "stratified"}:
        raise ValueError("invalid_strategy")
    if normalized_strategy == "stratified" and not (stratify_by or "").strip():
        raise ValueError("stratify_by_required")

    gold_set = await _require_gold_set(db, gold_set_id)
    source = await _get_source_dataset(
        db, project_id=gold_set.project_id, source_dataset_id=source_dataset_id
    )

    file_path = Path(source.file_path) if source.file_path else None
    records = _load_jsonl(file_path) if file_path else []

    rng = random.Random(seed)

    if not records:
        chosen_rows: list[dict[str, Any]] = []
    elif normalized_strategy == "stratified":
        indices = _stratified_indices(
            records,
            stratify_by=stratify_by or "",
            target_count=target_count,
            rng=rng,
        )
        chosen_rows = [records[i] for i in indices]
    else:
        pool = list(range(len(records)))
        rng.shuffle(pool)
        chosen_rows = [records[i] for i in pool[:target_count]]

    version = await ensure_draft_version(db, gold_set_id=gold_set_id, user_id=user_id)
    existing_keys = await _existing_row_keys_in_version(db, version_id=version.id)

    created_rows: list[GoldSetRow] = []
    skipped_duplicates = 0
    for record in chosen_rows:
        key = _stable_row_key(record)
        if key in existing_keys:
            skipped_duplicates += 1
            continue
        existing_keys.add(key)

        row = GoldSetRow(
            gold_set_id=gold_set_id,
            version_id=version.id,
            source_row_key=key,
            source_dataset_id=source.id,
            input=record,
            expected={},
            rationale="",
            labels={},
            reviewer_id=reviewer_id,
            status=GoldSetRowStatus.PENDING,
        )
        db.add(row)
        created_rows.append(row)

    if created_rows:
        await db.flush()

    # Queue-side mirror for auto-assigned reviewer.
    if reviewer_id is not None and created_rows:
        for row in created_rows:
            db.add(
                GoldSetReviewerQueue(
                    gold_set_id=gold_set_id,
                    row_id=row.id,
                    reviewer_id=reviewer_id,
                    priority=0,
                    status=GoldSetReviewerQueueStatus.PENDING,
                )
            )
        await db.flush()

    await db.commit()

    return {
        "gold_set_id": gold_set_id,
        "version_id": version.id,
        "version": version.version,
        "requested": target_count,
        "created": len(created_rows),
        "skipped_duplicates": skipped_duplicates,
        "strategy": normalized_strategy,
        "stratify_by": stratify_by if normalized_strategy == "stratified" else None,
        "rows": [_serialize_row(row) for row in created_rows],
    }


def _row_status_to_queue_status(status: GoldSetRowStatus) -> GoldSetReviewerQueueStatus:
    if status == GoldSetRowStatus.PENDING:
        return GoldSetReviewerQueueStatus.PENDING
    if status == GoldSetRowStatus.IN_REVIEW:
        return GoldSetReviewerQueueStatus.IN_PROGRESS
    if status in (GoldSetRowStatus.APPROVED, GoldSetRowStatus.REJECTED):
        return GoldSetReviewerQueueStatus.COMPLETED
    return GoldSetReviewerQueueStatus.PENDING


def _terminal_row_status(status: GoldSetRowStatus) -> bool:
    return status in (GoldSetRowStatus.APPROVED, GoldSetRowStatus.REJECTED)


async def _sync_queue_for_row(
    db: AsyncSession,
    *,
    row: GoldSetRow,
    previous_reviewer_id: int | None,
) -> None:
    """Keep the `gold_set_reviewer_queue` row in sync with assignment state."""
    if previous_reviewer_id is not None and previous_reviewer_id != row.reviewer_id:
        stale = await db.execute(
            select(GoldSetReviewerQueue).where(
                GoldSetReviewerQueue.row_id == row.id,
                GoldSetReviewerQueue.reviewer_id == previous_reviewer_id,
            )
        )
        stale_entry = stale.scalar_one_or_none()
        if stale_entry is not None:
            await db.delete(stale_entry)

    if row.reviewer_id is None:
        return

    existing = await db.execute(
        select(GoldSetReviewerQueue).where(
            GoldSetReviewerQueue.row_id == row.id,
            GoldSetReviewerQueue.reviewer_id == row.reviewer_id,
        )
    )
    entry = existing.scalar_one_or_none()
    next_status = _row_status_to_queue_status(row.status)
    if entry is None:
        db.add(
            GoldSetReviewerQueue(
                gold_set_id=row.gold_set_id,
                row_id=row.id,
                reviewer_id=row.reviewer_id,
                priority=0,
                status=next_status,
                completed_at=_utcnow() if next_status == GoldSetReviewerQueueStatus.COMPLETED else None,
            )
        )
    else:
        entry.status = next_status
        entry.completed_at = _utcnow() if next_status == GoldSetReviewerQueueStatus.COMPLETED else None


async def update_row(
    db: AsyncSession,
    *,
    gold_set_id: int,
    row_id: int,
    patch: dict[str, Any],
) -> GoldSetRow:
    """Apply a PATCH-style update to a gold-set row and sync queue state."""
    await _require_gold_set(db, gold_set_id)

    row = await db.get(GoldSetRow, row_id)
    if row is None or row.gold_set_id != gold_set_id:
        raise ValueError("row_not_found")

    previous_reviewer_id = row.reviewer_id

    if "input" in patch:
        value = patch["input"]
        if value is not None and not isinstance(value, dict):
            raise ValueError("input_must_be_object")
        row.input = value or {}

    if "expected" in patch:
        value = patch["expected"]
        if value is not None and not isinstance(value, dict):
            raise ValueError("expected_must_be_object")
        row.expected = value or {}

    if "rationale" in patch:
        row.rationale = str(patch["rationale"] or "")

    if "labels" in patch:
        value = patch["labels"]
        if value is not None and not isinstance(value, dict):
            raise ValueError("labels_must_be_object")
        row.labels = value or {}

    if "status" in patch:
        raw_status = patch["status"]
        if raw_status is None:
            raise ValueError("status_required")
        try:
            row.status = GoldSetRowStatus(str(raw_status).lower())
        except ValueError as exc:
            raise ValueError("invalid_status") from exc

    if "reviewer_id" in patch:
        new_reviewer = patch["reviewer_id"]
        row.reviewer_id = int(new_reviewer) if new_reviewer is not None else None

    if _terminal_row_status(row.status):
        row.reviewed_at = _utcnow()

    await _sync_queue_for_row(db, row=row, previous_reviewer_id=previous_reviewer_id)

    await db.flush()
    await db.refresh(row)
    await db.commit()
    return row


async def list_reviewer_queue(
    db: AsyncSession,
    *,
    gold_set_id: int,
    reviewer_id: int | None = None,
    status: GoldSetReviewerQueueStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    await _require_gold_set(db, gold_set_id)

    clamped_limit = max(1, min(int(limit or 50), 500))
    clamped_offset = max(0, int(offset or 0))

    filters = [GoldSetReviewerQueue.gold_set_id == gold_set_id]
    if reviewer_id is not None:
        filters.append(GoldSetReviewerQueue.reviewer_id == int(reviewer_id))
    if status is not None:
        filters.append(GoldSetReviewerQueue.status == status)

    count_result = await db.execute(
        select(func.count(GoldSetReviewerQueue.id)).where(*filters)
    )
    total = int(count_result.scalar_one() or 0)

    listing = await db.execute(
        select(GoldSetReviewerQueue, GoldSetRow)
        .join(GoldSetRow, GoldSetRow.id == GoldSetReviewerQueue.row_id)
        .where(*filters)
        .order_by(
            GoldSetReviewerQueue.priority.desc(),
            GoldSetReviewerQueue.assigned_at.asc(),
            GoldSetReviewerQueue.id.asc(),
        )
        .limit(clamped_limit)
        .offset(clamped_offset)
    )
    items = [_serialize_queue_entry(entry, row) for entry, row in listing.all()]

    return {
        "gold_set_id": gold_set_id,
        "count": total,
        "limit": clamped_limit,
        "offset": clamped_offset,
        "items": items,
    }


def _serialize_row(row: GoldSetRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "gold_set_id": row.gold_set_id,
        "version_id": row.version_id,
        "source_row_key": row.source_row_key,
        "source_dataset_id": row.source_dataset_id,
        "input": row.input or {},
        "expected": row.expected or {},
        "rationale": row.rationale or "",
        "labels": row.labels or {},
        "reviewer_id": row.reviewer_id,
        "status": row.status.value if isinstance(row.status, GoldSetRowStatus) else str(row.status),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
    }


def _snippet(payload: Any, limit: int = 160) -> str:
    if payload is None:
        return ""
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str) if not isinstance(payload, str) else payload
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _serialize_queue_entry(entry: GoldSetReviewerQueue, row: GoldSetRow) -> dict[str, Any]:
    return {
        "queue_id": entry.id,
        "row_id": entry.row_id,
        "reviewer_id": entry.reviewer_id,
        "priority": entry.priority,
        "status": entry.status.value if isinstance(entry.status, GoldSetReviewerQueueStatus) else str(entry.status),
        "assigned_at": entry.assigned_at.isoformat() if entry.assigned_at else None,
        "completed_at": entry.completed_at.isoformat() if entry.completed_at else None,
        "row_preview": {
            "status": row.status.value if isinstance(row.status, GoldSetRowStatus) else str(row.status),
            "labels": row.labels or {},
            "input_snippet": _snippet(row.input),
            "expected_snippet": _snippet(row.expected),
        },
    }


def iter_valid_row_statuses() -> Iterable[str]:
    return [status.value for status in GoldSetRowStatus]
