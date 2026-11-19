"""Gold evaluation dataset service — create, import, and manage gold Q&A sets."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DatasetVersion

GOLD_TYPES = (DatasetType.GOLD_DEV, DatasetType.GOLD_TEST)


class GoldLockedError(ValueError):
    """The gold dataset is locked (immutable). A ``ValueError`` so the older
    callers that catch it keep working."""


def _gold_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "gold"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def get_or_create_gold_dataset(
    db: AsyncSession, project_id: int, dataset_type: DatasetType
) -> Dataset:
    """Get or create a gold dataset (dev or test).

    Uses `.first()` rather than `.scalar_one_or_none()` so we tolerate historical
    duplicates (e.g., from a double-click race on "Add Q&A pair" before a unique
    constraint existed). The lowest-id match wins, which keeps user data stable
    across retries.
    """
    result = await db.execute(
        select(Dataset)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == dataset_type,
        )
        .order_by(Dataset.id.asc())
    )
    ds = result.scalars().first()
    if ds:
        return ds

    name = "Gold Dev Set" if dataset_type == DatasetType.GOLD_DEV else "Gold Test Set"
    ds = Dataset(
        project_id=project_id,
        name=name,
        dataset_type=dataset_type,
        description=f"Gold evaluation dataset ({dataset_type.value})",
    )
    db.add(ds)
    await db.flush()
    await db.refresh(ds)
    return ds


def _read_rows(path: str | Path | None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)


def _next_row_id(ds: Dataset) -> int:
    """First free row id. Row ids are stable (deleting a row never renumbers the rest, so a past
    result's row id still names the same row); with no deletions this equals ``record_count + 1``."""
    top = 0
    for row in _read_rows(ds.file_path):
        rid = row.get("id")
        if isinstance(rid, int) and not isinstance(rid, bool):
            top = max(top, rid)
    return max(top, int(ds.record_count or 0)) + 1


def _with_caller_id(entry: dict, source: dict) -> None:
    """Keep a caller-supplied ``id`` as ``source_row_id`` (``id`` itself is system-owned)."""
    if "id" in source and "source_row_id" not in source and source["id"] not in (None, ""):
        entry["source_row_id"] = source["id"]


async def add_gold_row(
    db: AsyncSession,
    project_id: int,
    row: dict,
    dataset_type: DatasetType = DatasetType.GOLD_DEV,
) -> dict:
    """Add a single gold row in any recipe shape.

    Mirrors ``import_qa_pairs`` for single-row inserts: preserves
    every caller-supplied field (recipe-shaped or otherwise), overlays
    system-owned ``id`` + ``created_at``, and backfills QA-era
    defaults (``difficulty`` / ``criticality`` / ``is_hallucination_trap`` /
    ``metadata``) so existing eval surfaces that read them on qa-sft
    rows keep working.

    Used by ``POST /api/projects/{id}/gold/add`` for both manual
    inline additions (any recipe) and the legacy ``add_qa_pair``
    wrapper below.
    """
    ds = await get_or_create_gold_dataset(db, project_id, dataset_type)
    if ds.is_locked:
        raise GoldLockedError("Gold dataset is locked. Cannot add new entries.")

    gold_dir = _gold_dir(project_id)
    filename = f"{dataset_type.value}.jsonl"
    file_path = gold_dir / filename

    entry: dict = dict(row)
    entry.setdefault("difficulty", "medium")
    entry.setdefault("criticality", "normal")
    entry.setdefault("is_hallucination_trap", False)
    entry.setdefault("metadata", {})
    # System-owned fields ALWAYS win — never let the caller override
    # id / created_at.
    _with_caller_id(entry, row)
    entry["id"] = _next_row_id(ds)
    entry["created_at"] = datetime.now(timezone.utc).isoformat()

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    ds.record_count += 1
    ds.file_path = str(file_path)
    await db.flush()

    return entry


async def add_qa_pair(
    db: AsyncSession,
    project_id: int,
    question: str,
    answer: str,
    dataset_type: DatasetType = DatasetType.GOLD_DEV,
    difficulty: str = "medium",
    criticality: str = "normal",
    is_hallucination_trap: bool = False,
    metadata: dict | None = None,
) -> dict:
    """Add a single qa-sft pair. Thin wrapper over ``add_gold_row`` —
    kept for backward compat with callers (tests, scripts) that pass
    QA fields by keyword argument."""
    return await add_gold_row(
        db,
        project_id,
        {
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "criticality": criticality,
            "is_hallucination_trap": is_hallucination_trap,
            "metadata": metadata or {},
        },
        dataset_type,
    )


async def import_qa_pairs(
    db: AsyncSession,
    project_id: int,
    pairs: list[dict],
    dataset_type: DatasetType = DatasetType.GOLD_DEV,
) -> dict:
    """Import gold rows in any recipe shape.

    Despite the historic ``import_qa_pairs`` name, this is the
    generic gold-import path used by all recipes. The function:
      * preserves every caller-supplied field (spread ``**pair``)
      * overlays system fields (``id``, ``created_at``) on top
      * defaults QA-era flags (``difficulty`` / ``criticality`` /
        ``is_hallucination_trap``) so existing eval surfaces that
        read them on qa-sft rows keep working

    Non-QA recipes (classification, span-extraction, summarization)
    pass their own shapes (``text``+``label``, ``text``+``entities``,
    ``document``+``summary``) and those fields round-trip cleanly
    into the JSONL file. The evaluator's field-alias lookup picks
    them up downstream.
    """
    ds = await get_or_create_gold_dataset(db, project_id, dataset_type)
    if ds.is_locked:
        raise GoldLockedError("Gold dataset is locked. Cannot add new entries.")

    gold_dir = _gold_dir(project_id)
    filename = f"{dataset_type.value}.jsonl"
    file_path = gold_dir / filename

    count = 0
    next_id = _next_row_id(ds)
    with open(file_path, "a", encoding="utf-8") as f:
        for pair in pairs:
            # Preserve everything the caller sent (recipe-shape fields
            # like text/label/entities/document/summary survive).
            entry: dict = dict(pair)
            # Backfill QA-era defaults only when the caller didn't
            # supply them — eval handlers for non-QA recipes ignore
            # these keys, so the cost of keeping them as defaults is
            # zero.
            entry.setdefault("difficulty", "medium")
            entry.setdefault("criticality", "normal")
            entry.setdefault("is_hallucination_trap", False)
            entry.setdefault("metadata", {})
            # System-owned fields ALWAYS win — never let the caller
            # override id / created_at.
            _with_caller_id(entry, pair)
            entry["id"] = next_id + count
            entry["created_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(entry) + "\n")
            count += 1

    ds.record_count += count
    ds.file_path = str(file_path)
    await db.flush()

    return {"imported": count, "total": ds.record_count}


async def get_gold_entries(
    db: AsyncSession, project_id: int, dataset_type: DatasetType
) -> list[dict]:
    """Read all entries from a gold dataset."""
    ds = await get_or_create_gold_dataset(db, project_id, dataset_type)
    if not ds.file_path or not Path(ds.file_path).exists():
        return []

    entries = []
    with open(ds.file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


async def lock_gold_dataset(
    db: AsyncSession, project_id: int, dataset_type: DatasetType
) -> Dataset:
    """Lock a gold dataset (make immutable)."""
    ds = await get_or_create_gold_dataset(db, project_id, dataset_type)
    ds.is_locked = True
    await db.flush()
    await db.refresh(ds)
    return ds


# --- file import + row management (Phase 1 hardening) ------------------------------------------
#
# Recipe-agnostic: rows are stored as given (any task shape), exactly like ``import_qa_pairs``.
# Every rewrite snapshots the previous file as a ``DatasetVersion`` so an old result's
# ``answer_key_stamp`` (file sha256) still resolves to a retained file.


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def _snapshot(db: AsyncSession, ds: Dataset, reason: str) -> int | None:
    """Copy the current gold file to ``gold/versions/`` and record a ``DatasetVersion``
    (``manifest.sha256`` = the file hash). Returns the version number, or None when there was
    nothing to keep (no file / no rows)."""
    if not ds.file_path or not Path(ds.file_path).exists() or not (ds.record_count or 0):
        return None
    from sqlalchemy import func

    top = (
        await db.execute(select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == ds.id))
    ).scalar_one_or_none()
    version = int(top or 0) + 1
    vdir = _gold_dir(ds.project_id) / "versions"
    vdir.mkdir(parents=True, exist_ok=True)
    dest = vdir / f"{ds.dataset_type.value}_v{version}.jsonl"
    shutil.copyfile(ds.file_path, dest)
    db.add(DatasetVersion(
        dataset_id=ds.id,
        version=version,
        file_path=str(dest),
        record_count=int(ds.record_count or 0),
        manifest={"source": "gold_edit", "reason": reason, "sha256": _sha256(dest)},
    ))
    await db.flush()
    return version


async def _writable_gold(db: AsyncSession, project_id: int, dataset_type: DatasetType) -> Dataset:
    ds = await get_or_create_gold_dataset(db, project_id, dataset_type)
    if ds.is_locked:
        raise GoldLockedError("Gold dataset is locked. Cannot change its entries.")
    return ds


def gold_dataset_summary(ds: Dataset) -> dict:
    """Label for a gold set: which dataset, which type, how many rows (picked-data provenance)."""
    return {
        "id": ds.id,
        "name": ds.name,
        "dataset_type": ds.dataset_type.value,
        "record_count": int(ds.record_count or 0),
        "is_locked": bool(ds.is_locked),
    }


def _stamp_rows(rows: list[dict], first_id: int) -> list[dict]:
    """Backfill the QA-era defaults and system fields on rows the caller supplied."""
    out: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    for offset, pair in enumerate(rows):
        entry: dict = dict(pair)
        entry.setdefault("difficulty", "medium")
        entry.setdefault("criticality", "normal")
        entry.setdefault("is_hallucination_trap", False)
        entry.setdefault("metadata", {})
        _with_caller_id(entry, pair)
        entry["id"] = first_id + offset
        entry["created_at"] = now
        out.append(entry)
    return out


async def replace_gold_rows(
    db: AsyncSession, project_id: int, rows: list[dict], dataset_type: DatasetType,
    *, reason: str = "replace",
) -> dict:
    """Replace the whole set with ``rows`` (ids restart at 1; the previous file is snapshotted)."""
    ds = await _writable_gold(db, project_id, dataset_type)
    version = await _snapshot(db, ds, reason)
    path = _gold_dir(project_id) / f"{dataset_type.value}.jsonl"
    stamped = _stamp_rows(rows, 1)
    _write_rows(path, stamped)
    ds.record_count = len(stamped)
    ds.file_path = str(path)
    await db.flush()
    return {"imported": len(stamped), "total": ds.record_count, "snapshot_version": version}


async def delete_gold_rows(
    db: AsyncSession, project_id: int, dataset_type: DatasetType, row_ids: list[int]
) -> dict:
    """Delete the rows with these ids. Ids of the remaining rows do not change. ``missing`` lists
    requested ids that matched no row; nothing is written when none matched."""
    ds = await _writable_gold(db, project_id, dataset_type)
    wanted = {int(i) for i in row_ids}
    rows = _read_rows(ds.file_path)
    present = {r.get("id") for r in rows}
    missing = sorted(i for i in wanted if i not in present)
    kept = [r for r in rows if r.get("id") not in wanted]
    deleted = len(rows) - len(kept)
    version = None
    if deleted:
        version = await _snapshot(db, ds, "delete_rows")
        _write_rows(Path(ds.file_path), kept)
        ds.record_count = len(kept)
        await db.flush()
    return {"deleted": deleted, "remaining": len(kept), "missing": missing, "snapshot_version": version}


async def clear_gold_rows(db: AsyncSession, project_id: int, dataset_type: DatasetType) -> dict:
    """Delete every row (the dataset and its retained versions stay)."""
    ds = await _writable_gold(db, project_id, dataset_type)
    deleted = int(ds.record_count or 0)
    version = await _snapshot(db, ds, "clear")
    path = _gold_dir(project_id) / f"{dataset_type.value}.jsonl"
    _write_rows(path, [])
    ds.record_count = 0
    ds.file_path = str(path)
    await db.flush()
    return {"deleted": deleted, "remaining": 0, "snapshot_version": version}

