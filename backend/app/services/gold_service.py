"""Gold evaluation dataset service — create, import, and manage gold Q&A sets."""

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType


def _gold_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "gold"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def get_or_create_gold_dataset(
    db: AsyncSession, project_id: int, dataset_type: DatasetType
) -> Dataset:
    """Get or create a gold dataset (dev or test)."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == dataset_type,
        )
    )
    ds = result.scalar_one_or_none()
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
    """Add a single Q&A pair to the gold dataset."""
    ds = await get_or_create_gold_dataset(db, project_id, dataset_type)
    if ds.is_locked:
        raise ValueError("Gold dataset is locked. Cannot add new entries.")

    gold_dir = _gold_dir(project_id)
    filename = f"{dataset_type.value}.jsonl"
    file_path = gold_dir / filename

    entry = {
        "id": ds.record_count + 1,
        "question": question,
        "answer": answer,
        "difficulty": difficulty,
        "criticality": criticality,
        "is_hallucination_trap": is_hallucination_trap,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    ds.record_count += 1
    ds.file_path = str(file_path)
    await db.flush()

    return entry


async def import_qa_pairs(
    db: AsyncSession,
    project_id: int,
    pairs: list[dict],
    dataset_type: DatasetType = DatasetType.GOLD_DEV,
) -> dict:
    """Import multiple Q&A pairs at once."""
    ds = await get_or_create_gold_dataset(db, project_id, dataset_type)
    if ds.is_locked:
        raise ValueError("Gold dataset is locked. Cannot add new entries.")

    gold_dir = _gold_dir(project_id)
    filename = f"{dataset_type.value}.jsonl"
    file_path = gold_dir / filename

    count = 0
    with open(file_path, "a", encoding="utf-8") as f:
        for pair in pairs:
            entry = {
                "id": ds.record_count + count + 1,
                "question": pair.get("question", ""),
                "answer": pair.get("answer", ""),
                "difficulty": pair.get("difficulty", "medium"),
                "criticality": pair.get("criticality", "normal"),
                "is_hallucination_trap": pair.get("is_hallucination_trap", False),
                "metadata": pair.get("metadata", {}),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
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
