"""Gold evaluation dataset service — create, import, and manage gold Q&A sets."""

import asyncio
import hashlib
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DatasetVersion

GOLD_TYPES = (DatasetType.GOLD_DEV, DatasetType.GOLD_TEST)


class GoldLockedError(ValueError):
    """The gold dataset is locked (immutable). A ``ValueError`` so the older
    callers that catch it keep working."""


class GoldDuplicateIdsError(ValueError):
    """A requested row id is shared by several rows (possible in older files), so deleting by id
    would remove more than the caller picked."""

    def __init__(self, ids: list[int]):
        self.ids = sorted(ids)
        super().__init__(f"Row id(s) {self.ids} appear on more than one row, so they cannot be deleted by id.")


# One lock per (project, gold type): a delete and an append (or two imports) never interleave.
_WRITE_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


@asynccontextmanager
async def gold_write_lock(project_id: int, dataset_type: DatasetType):
    lock = _WRITE_LOCKS.setdefault((int(project_id), dataset_type.value), asyncio.Lock())
    async with lock:
        yield


def unique_tmp_path(path: Path) -> Path:
    """A temp name beside ``path`` (same directory, so the final ``replace`` is atomic) that no other
    writer, in this process or another, shares: ``<name>.<uuid4 hex>.tmp``."""
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    """Replace ``path`` with ``rows`` via a uniquely named temp file; the temp is removed on failure.
    Callers hold ``gold_write_lock``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _next_row_id(ds: Dataset) -> int:
    """First free row id. Row ids are stable (deleting a row never renumbers the rest, so a past
    result's row id still names the same row). A running maximum is kept in
    ``Dataset.metadata_["next_row_id"]`` so deleting the LAST row never frees its id; when missing it
    is seeded from the largest id in the file and ``record_count``."""
    top = 0
    for row in _read_rows(ds.file_path):
        rid = row.get("id")
        if isinstance(rid, int) and not isinstance(rid, bool):
            top = max(top, rid)
    seeded = max(top, int(ds.record_count or 0)) + 1
    kept = (ds.metadata_ or {}).get("next_row_id")
    if isinstance(kept, int) and not isinstance(kept, bool):
        return max(kept, seeded)
    return seeded


def _set_next_row_id(ds: Dataset, value: int, *, exact: bool = False) -> None:
    """Record the next free id (a running maximum unless ``exact``); reassigns the JSON column."""
    meta = dict(ds.metadata_ or {})
    old = meta.get("next_row_id")
    if not exact and isinstance(old, int) and not isinstance(old, bool):
        value = max(value, old)
    meta["next_row_id"] = int(value)
    ds.metadata_ = meta


async def _commit_then_rename(db: AsyncSession, tmp: Path, final: Path) -> None:
    """Commit the DB changes, then move the temp file into place. A failed commit rolls back and
    removes the temp file, so the DB and the file never disagree."""
    try:
        await db.commit()
    except BaseException:
        await db.rollback()
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(final)


def _with_caller_id(entry: dict, source: dict) -> None:
    """Keep a caller-supplied ``id`` as ``source_row_id`` (``id`` itself is system-owned)."""
    if "id" in source and "source_row_id" not in source and source["id"] not in (None, ""):
        entry["source_row_id"] = source["id"]


async def add_gold_row(
    db: AsyncSession,
    project_id: int,
    row: dict,
    dataset_type: DatasetType = DatasetType.GOLD_DEV,
    *,
    atomic: bool = False,
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

    line = json.dumps(entry) + "\n"
    ds.record_count += 1
    _set_next_row_id(ds, entry["id"] + 1)
    if atomic:
        await _atomic_append(db, ds, file_path, [line])
    else:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line)
        ds.file_path = str(file_path)
        await db.flush()

    return entry


async def _atomic_append(
    db: AsyncSession, ds: Dataset, file_path: Path, lines: list[str], *, commit: bool = True
) -> None:
    """Append ``lines`` via a temp copy: DB commit first, then the rename (see ``_commit_then_rename``).
    ``commit=False`` (a caller that owns a larger all-or-nothing transaction) only flushes, then
    renames: readers still never see a half-written file."""
    tmp = unique_tmp_path(file_path)
    try:
        if file_path.exists():
            shutil.copyfile(file_path, tmp)
        else:
            tmp.write_text("", encoding="utf-8")
        with open(tmp, "a", encoding="utf-8") as f:
            f.writelines(lines)
        ds.file_path = str(file_path)
        await db.flush()
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if not commit:
        tmp.replace(file_path)
        return
    await _commit_then_rename(db, tmp, file_path)


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
    *,
    atomic: bool = False,
    commit: bool = True,
) -> dict:
    """Import gold rows in any recipe shape.

    Writers hold ``gold_write_lock`` and pass ``atomic=True`` (temp copy + rename; the DB is committed
    first unless ``commit=False``, for a caller whose own transaction must stay all-or-nothing).

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
    lines: list[str] = []
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
        lines.append(json.dumps(entry) + "\n")
        count += 1

    ds.record_count += count
    _set_next_row_id(ds, next_id + count)
    if atomic:
        await _atomic_append(db, ds, file_path, lines, commit=commit)
    else:
        with open(file_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
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
# Every rewrite records TWO ``DatasetVersion`` rows so ``answer_key_stamp`` (file sha256) resolves for
# evals scored before AND after it: a ``role: "previous"`` snapshot of the file being replaced, and a
# ``role: "current"`` version of the new content.


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _active_count(rows: list[dict]) -> int:
    """Rows that count toward ``record_count``: review-rejected rows do not (``gold_review_service``)."""
    from app.services.gold_review_service import is_rejected

    return sum(1 for r in rows if not is_rejected(r))


async def _latest_version(db: AsyncSession, ds: Dataset) -> DatasetVersion | None:
    return (
        await db.execute(
            select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id).order_by(DatasetVersion.version.desc()).limit(1)
        )
    ).scalars().first()


async def _add_version(db: AsyncSession, ds: Dataset, source: Path, reason: str, role: str, rows: int) -> int:
    latest = await _latest_version(db, ds)
    version = int(latest.version if latest else 0) + 1
    vdir = _gold_dir(ds.project_id) / "versions"
    vdir.mkdir(parents=True, exist_ok=True)
    dest = vdir / f"{ds.dataset_type.value}_v{version}.jsonl"
    shutil.copyfile(source, dest)
    db.add(DatasetVersion(
        dataset_id=ds.id,
        version=version,
        file_path=str(dest),
        record_count=rows,
        manifest={"source": "gold_edit", "reason": reason, "role": role, "sha256": _sha256(dest)},
    ))
    await db.flush()
    return version


async def _snapshot(db: AsyncSession, ds: Dataset, reason: str) -> int | None:
    """Keep the file about to be replaced as a ``role: "previous"`` version (``manifest.sha256`` = its
    hash). Returns the version number that holds those bytes, or None when there was nothing to keep
    (no file / no rows IN THE FILE, whatever ``record_count`` says: review-rejected rows are rows).
    Skipped (returning the existing number) when the latest version already holds identical bytes."""
    if not ds.file_path or not Path(ds.file_path).exists():
        return None
    rows = _read_rows(ds.file_path)
    if not rows:
        return None
    latest = await _latest_version(db, ds)
    if latest is not None and isinstance(latest.manifest, dict) and latest.manifest.get("sha256") == _sha256(Path(ds.file_path)):
        return int(latest.version)
    return await _add_version(db, ds, Path(ds.file_path), reason, "previous", _active_count(rows))


async def _record_current(db: AsyncSession, ds: Dataset, new_file: Path, reason: str, rows: list[dict]) -> int | None:
    """Record the NEW content as a ``role: "current"`` version (skipped when the latest version already
    has those bytes, e.g. clearing an already empty set)."""
    latest = await _latest_version(db, ds)
    if latest is not None and isinstance(latest.manifest, dict) and latest.manifest.get("sha256") == _sha256(new_file):
        return None
    return await _add_version(db, ds, new_file, reason, "current", _active_count(rows))


async def _rewrite(
    db: AsyncSession, ds: Dataset, rows: list[dict], reason: str, *, next_id: int, exact_next: bool = False
) -> int | None:
    """Replace the file with ``rows``: write a temp file, record versions, commit the DB, then rename.
    Returns the "previous" snapshot version. Callers hold ``gold_write_lock``."""
    path = _gold_dir(ds.project_id) / f"{ds.dataset_type.value}.jsonl"
    tmp = unique_tmp_path(path)
    try:
        _write_rows(tmp, rows)
        previous = await _snapshot(db, ds, reason)
        await _record_current(db, ds, tmp, reason, rows)
        ds.record_count = _active_count(rows)
        ds.file_path = str(path)
        _set_next_row_id(ds, next_id, exact=exact_next)
        await db.flush()
    except BaseException:
        await db.rollback()
        tmp.unlink(missing_ok=True)
        raise
    await _commit_then_rename(db, tmp, path)
    return previous


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
    """Replace the whole set with ``rows`` (ids restart at 1; the previous file is snapshotted).
    Commits the session (file rename follows the commit)."""
    ds = await _writable_gold(db, project_id, dataset_type)
    stamped = _stamp_rows(rows, 1)
    version = await _rewrite(db, ds, stamped, reason, next_id=len(stamped) + 1, exact_next=True)
    return {"imported": len(stamped), "total": ds.record_count, "snapshot_version": version}


async def delete_gold_rows(
    db: AsyncSession, project_id: int, dataset_type: DatasetType, row_ids: list[int]
) -> dict:
    """Delete the rows with these ids. Ids of the remaining rows do not change. ``missing`` lists
    requested ids that matched no row; nothing is written when none matched. Raises
    ``GoldDuplicateIdsError`` when a requested id is on several rows (older files)."""
    ds = await _writable_gold(db, project_id, dataset_type)
    wanted = {int(i) for i in row_ids}
    rows = _read_rows(ds.file_path)
    seen: dict[Any, int] = {}
    for r in rows:
        seen[r.get("id")] = seen.get(r.get("id"), 0) + 1
    dupes = [i for i in wanted if seen.get(i, 0) > 1]
    if dupes:
        raise GoldDuplicateIdsError(dupes)
    present = set(seen)
    missing = sorted(i for i in wanted if i not in present)
    kept = [r for r in rows if r.get("id") not in wanted]
    deleted = len(rows) - len(kept)
    version = None
    if deleted:
        version = await _rewrite(db, ds, kept, "delete_rows", next_id=_next_row_id(ds))
    return {"deleted": deleted, "remaining": len(kept), "missing": missing, "snapshot_version": version}


async def clear_gold_rows(db: AsyncSession, project_id: int, dataset_type: DatasetType) -> dict:
    """Delete every row (the dataset and its retained versions stay; row ids keep counting up)."""
    ds = await _writable_gold(db, project_id, dataset_type)
    deleted = len(_read_rows(ds.file_path))
    version = await _rewrite(db, ds, [], "clear", next_id=_next_row_id(ds))
    return {"deleted": deleted, "remaining": 0, "snapshot_version": version}
