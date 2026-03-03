"""Gold evaluation dataset API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import DatasetType
from app.services.gold_service import (
    add_qa_pair,
    get_gold_entries,
    import_qa_pairs,
    lock_gold_dataset,
)

router = APIRouter(prefix="/projects/{project_id}/gold", tags=["Gold Dataset"])


class QAPairCreate(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    dataset_type: str = "gold_dev"
    difficulty: str = "medium"
    criticality: str = "normal"
    is_hallucination_trap: bool = False


class QAPairBatchImport(BaseModel):
    pairs: list[dict]
    dataset_type: str = "gold_dev"


@router.post("/add", status_code=201)
async def add_pair(
    project_id: int,
    data: QAPairCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a Q&A pair to the gold dataset."""
    ds_type = DatasetType.GOLD_DEV if data.dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    try:
        entry = await add_qa_pair(
            db, project_id, data.question, data.answer,
            ds_type, data.difficulty, data.criticality, data.is_hallucination_trap,
        )
        return entry
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/import")
async def import_pairs(
    project_id: int,
    data: QAPairBatchImport,
    db: AsyncSession = Depends(get_db),
):
    """Import multiple Q&A pairs."""
    ds_type = DatasetType.GOLD_DEV if data.dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    try:
        result = await import_qa_pairs(db, project_id, data.pairs, ds_type)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/entries")
async def list_entries(
    project_id: int,
    dataset_type: str = "gold_dev",
    db: AsyncSession = Depends(get_db),
):
    """List all gold dataset entries."""
    ds_type = DatasetType.GOLD_DEV if dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    entries = await get_gold_entries(db, project_id, ds_type)
    return {"entries": entries, "total": len(entries)}


@router.post("/lock")
async def lock_dataset(
    project_id: int,
    dataset_type: str = "gold_dev",
    db: AsyncSession = Depends(get_db),
):
    """Lock a gold dataset (make immutable)."""
    ds_type = DatasetType.GOLD_DEV if dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    ds = await lock_gold_dataset(db, project_id, ds_type)
    return {"id": ds.id, "name": ds.name, "is_locked": ds.is_locked, "record_count": ds.record_count}
