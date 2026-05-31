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


class AnalyzeSplitsRequest(BaseModel):
    """Same shape as AnalyzeRequest minus the `split` field — the
    endpoint walks every available split itself and returns a payload
    keyed by split name. Splits that don't exist (e.g. test missing
    because the project hasn't been prepped yet) are reported via
    `missing_splits`, not failed with 404, so the UI can render
    partial overlays as soon as train is ready."""
    model_name: str = Field(..., min_length=1)
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


@router.post("/analyze-splits")
async def analyze_splits(
    project_id: int,
    req: AnalyzeSplitsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run token-length analysis across every prepared split that exists.

    Returns ``{splits: {train: <result>, validation: <result>, test:
    <result>}, missing_splits: [...]}``. Each split's payload is the
    same shape ``analyze_dataset_tokens`` returns for a single dataset.
    Missing splits are reported, not 404'd — the panel can render an
    overlay as soon as one split is ready (train is the first to
    materialise during dataset prep).
    """
    splits_payload: dict[str, dict] = {}
    missing: list[str] = []
    errors: dict[str, str] = {}

    for split_name, dataset_type in SPLIT_TO_DATASET_TYPE.items():
        result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == dataset_type,
            )
        )
        dataset = result.scalar_one_or_none()
        if not dataset or not dataset.file_path:
            missing.append(split_name)
            continue
        dataset_path = Path(dataset.file_path)
        if not dataset_path.exists():
            missing.append(split_name)
            continue
        try:
            splits_payload[split_name] = analyze_dataset_tokens(
                dataset_path=str(dataset_path),
                model_name=req.model_name,
                max_seq_length=req.max_seq_length,
                text_field=req.text_field,
                question_field=req.question_field,
                answer_field=req.answer_field,
            )
        except ValueError as e:
            # One bad split doesn't fail the whole call — the UI can
            # still render the splits that succeeded and surface the
            # error inline for the one that didn't.
            errors[split_name] = str(e)

    if not splits_payload and not errors:
        raise HTTPException(
            404,
            f"No prepared splits found for project {project_id}. "
            f"Run dataset split first.",
        )

    return {
        "model_name": req.model_name,
        "max_seq_length": req.max_seq_length,
        "splits": splits_payload,
        "missing_splits": missing,
        "errors": errors,
    }


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
