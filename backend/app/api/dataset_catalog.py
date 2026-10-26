"""Dataset catalog routes: list a project's datasets and peek at a dataset's columns.

The column peek feeds the column pickers of the UI (e.g. the data-carrying context of
the code-findings synth playbooks). It reads only the first rows of the file.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import Dataset, DatasetType
from app.services.dataset_service import _load_records_from_file

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["Datasets"])

COLUMN_SAMPLE_ROWS = 200
_ANSWER_KEY_TYPES = (DatasetType.GOLD_DEV, DatasetType.GOLD_TEST)


def _type_value(dataset: Dataset) -> str:
    return getattr(dataset.dataset_type, "value", str(dataset.dataset_type))


@router.get("")
async def list_datasets(project_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.id)
    )
    return {
        "datasets": [
            {
                "id": d.id,
                "name": d.name,
                "dataset_type": _type_value(d),
                "record_count": d.record_count or 0,
                "is_locked": bool(d.is_locked),
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "file_path": d.file_path or None,
            }
            for d in result.scalars().all()
        ]
    }


@router.get("/{dataset_id}/columns")
async def dataset_columns(project_id: int, dataset_id: int, db: AsyncSession = Depends(get_db)):
    """Distinct top-level keys of the dataset's first rows, in order of first appearance."""
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None or dataset.project_id != project_id:
        raise HTTPException(404, f"Dataset {dataset_id} not found in this project.")
    if dataset.dataset_type in _ANSWER_KEY_TYPES:
        raise HTTPException(400, {
            "error_code": "context_dataset_not_allowed",
            "message": "The answer key is held out for evaluation; its columns are not offered here.",
            "metadata": {"dataset_id": dataset.id, "dataset_type": _type_value(dataset)},
        })
    columns: dict[str, None] = {}
    sampled = 0
    if dataset.file_path and Path(dataset.file_path).exists():
        for record in _load_records_from_file(Path(dataset.file_path))[:COLUMN_SAMPLE_ROWS]:
            if isinstance(record, dict):
                sampled += 1
                for key in record:
                    columns.setdefault(str(key), None)
    return {"dataset_id": dataset.id, "columns": list(columns), "sampled_rows": sampled}
