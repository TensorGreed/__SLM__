"""Dataset preparation service — combine, split, and freeze datasets."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DatasetVersion


def _prep_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def combine_datasets(
    db: AsyncSession,
    project_id: int,
    include_types: list[DatasetType] | None = None,
) -> list[dict]:
    """Combine entries from cleaned, synthetic, and gold datasets."""
    if include_types is None:
        include_types = [DatasetType.CLEANED, DatasetType.SYNTHETIC, DatasetType.GOLD_DEV]

    all_entries = []
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(include_types),
        )
    )
    datasets = result.scalars().all()

    for ds in datasets:
        if ds.file_path and Path(ds.file_path).exists():
            with open(ds.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        entry["_source_dataset"] = ds.dataset_type.value
                        all_entries.append(entry)

    return all_entries


async def split_dataset(
    db: AsyncSession,
    project_id: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    include_types: list[str] | None = None,
) -> dict:
    """Split combined data into train/val/test and save as JSONL."""
    # Resolve types
    types = None
    if include_types:
        types = [DatasetType(t) for t in include_types]

    entries = await combine_datasets(db, project_id, types)
    if not entries:
        raise ValueError("No data available to split. Ingest and process documents first.")

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(entries)

    total = len(entries)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": entries[:train_end],
        "val": entries[train_end:val_end],
        "test": entries[val_end:],
    }

    prep_dir = _prep_dir(project_id)
    file_paths = {}

    for split_name, split_data in splits.items():
        file_path = prep_dir / f"{split_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in split_data:
                f.write(json.dumps(entry) + "\n")
        file_paths[split_name] = str(file_path)

        # Create or update dataset record
        ds_type = {
            "train": DatasetType.TRAIN,
            "val": DatasetType.VALIDATION,
            "test": DatasetType.TEST,
        }[split_name]

        result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == ds_type,
            )
        )
        ds = result.scalar_one_or_none()
        if not ds:
            ds = Dataset(
                project_id=project_id,
                name=f"{split_name.title()} Set",
                dataset_type=ds_type,
            )
            db.add(ds)
            await db.flush()

        ds.record_count = len(split_data)
        ds.file_path = str(file_path)
        await db.flush()

    # Generate manifest
    manifest = {
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_entries": total,
        "splits": {k: len(v) for k, v in splits.items()},
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "file_paths": file_paths,
    }
    manifest_path = prep_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
