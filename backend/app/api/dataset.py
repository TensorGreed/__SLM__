"""Dataset preparation API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import DatasetType
from app.services.dataset_service import combine_datasets, profile_project_dataset, split_dataset

router = APIRouter(prefix="/projects/{project_id}/dataset", tags=["Dataset Prep"])


class SplitRequest(BaseModel):
    train_ratio: float = Field(0.8, gt=0, lt=1)
    val_ratio: float = Field(0.1, ge=0, lt=1)
    test_ratio: float = Field(0.1, ge=0, lt=1)
    seed: int = 42
    include_types: list[DatasetType] | None = None
    chat_template: str = "llama3"

    @model_validator(mode="after")
    def validate_ratios(self):
        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        return self


class ProfileRequest(BaseModel):
    dataset_type: DatasetType = DatasetType.CLEANED
    sample_size: int = Field(default=500, ge=10, le=5000)
    document_id: int | None = None
    field_mapping: dict[str, str] | None = None


@router.post("/split")
async def split(
    project_id: int,
    req: SplitRequest,
    db: AsyncSession = Depends(get_db),
):
    """Split combined data into train/validation/test JSONL datasets."""
    try:
        manifest = await split_dataset(
            db=db,
            project_id=project_id,
            train_ratio=req.train_ratio,
            val_ratio=req.val_ratio,
            test_ratio=req.test_ratio,
            seed=req.seed,
            include_types=[t.value for t in req.include_types] if req.include_types else None,
            chat_template=req.chat_template,
        )
        return manifest
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/preview")
async def preview(
    project_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    chat_template: str = "llama3",
    include_types: list[DatasetType] | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Preview combined dataset entries before splitting."""
    entries = await combine_datasets(db, project_id, include_types, chat_template)
    return {
        "total": len(entries),
        "preview": entries[:limit],
        "chat_template": chat_template,
        "included_types": [t.value for t in include_types] if include_types else None,
    }


@router.post("/profile")
async def profile(
    project_id: int,
    req: ProfileRequest,
    db: AsyncSession = Depends(get_db),
):
    """Inspect schema/normalization coverage for a project dataset."""
    try:
        return await profile_project_dataset(
            db=db,
            project_id=project_id,
            dataset_type=req.dataset_type,
            sample_size=req.sample_size,
            document_id=req.document_id,
            field_mapping=req.field_mapping,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
