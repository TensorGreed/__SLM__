"""Tokenization analysis API routes."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import Dataset, DatasetType
from app.services.tokenization_service import analyze_dataset_tokens, get_vocab_sample

router = APIRouter(prefix="/projects/{project_id}/tokenization", tags=["Tokenization"])

SPLIT_TO_DATASET_TYPE = {
    "train": DatasetType.TRAIN,
    "validation": DatasetType.VALIDATION,
    "test": DatasetType.TEST,
}


class AnalyzeRequest(BaseModel):
    model_name: str = Field(..., min_length=1)
    split: str = Field("train", pattern="^(train|validation|test)$")
    max_seq_length: int = Field(2048, ge=128, le=32768)
    text_field: str = "text"
    question_field: str = "question"
    answer_field: str = "answer"


@router.post("/analyze")
async def analyze(
    project_id: int,
    req: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Analyze token statistics for a prepared split."""
    dataset_type = SPLIT_TO_DATASET_TYPE[req.split]
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == dataset_type,
        )
    )
    dataset = result.scalar_one_or_none()
    if not dataset or not dataset.file_path:
        raise HTTPException(404, f"No {req.split} dataset found. Run dataset split first.")

    dataset_path = Path(dataset.file_path)
    if not dataset_path.exists():
        raise HTTPException(404, f"Dataset file missing: {dataset_path}")

    try:
        return analyze_dataset_tokens(
            dataset_path=str(dataset_path),
            model_name=req.model_name,
            max_seq_length=req.max_seq_length,
            text_field=req.text_field,
            question_field=req.question_field,
            answer_field=req.answer_field,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/vocab-sample")
async def vocab_sample(
    project_id: int,
    model_name: str = Query(..., min_length=1),
    sample_size: int = Query(100, ge=1, le=1000),
):
    """Get tokenizer vocabulary sample for a model."""
    try:
        return get_vocab_sample(model_name, sample_size)
    except ValueError as e:
        raise HTTPException(400, str(e))
