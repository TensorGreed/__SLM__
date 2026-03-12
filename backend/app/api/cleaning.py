"""Data Cleaning API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.cleaning_service import clean_document

router = APIRouter(prefix="/projects/{project_id}/cleaning", tags=["Cleaning"])


class CleanRequest(BaseModel):
    document_id: int
    chunk_size: int = Field(1000, ge=100, le=10000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    redact_pii: bool = True
    redact_toxicity: bool = False

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class CleanBatchRequest(BaseModel):
    document_ids: list[int]
    chunk_size: int = Field(1000, ge=100, le=10000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    redact_pii: bool = True
    redact_toxicity: bool = False

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@router.post("/clean")
async def clean_single(
    project_id: int,
    req: CleanRequest,
    db: AsyncSession = Depends(get_db),
):
    """Clean a single document."""
    try:
        result = await clean_document(
            db,
            project_id,
            req.document_id,
            req.chunk_size,
            req.chunk_overlap,
            req.redact_pii,
            req.redact_toxicity,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/clean-batch")
async def clean_batch(
    project_id: int,
    req: CleanBatchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Clean multiple documents."""
    results = []
    errors = []
    for doc_id in req.document_ids:
        try:
            result = await clean_document(
                db,
                project_id,
                doc_id,
                req.chunk_size,
                req.chunk_overlap,
                req.redact_pii,
                req.redact_toxicity,
            )
            results.append(result)
        except Exception as e:
            errors.append({"document_id": doc_id, "error": str(e)})

    return {"cleaned": len(results), "errors": errors, "results": results}


@router.get("/chunks")
async def get_cleaned_chunks(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return all cleaned text chunks for a project (from .chunks.jsonl files)."""
    import json as _json
    from pathlib import Path
    from sqlalchemy import select
    from app.models.dataset import Dataset, RawDocument

    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
    )
    docs = result.scalars().all()

    all_chunks: list[dict] = []
    for doc in docs:
        if not doc.file_path:
            continue
        chunks_path = Path(doc.file_path).with_suffix(".chunks.jsonl")
        if chunks_path.exists():
            for line in chunks_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        chunk = _json.loads(line)
                        chunk["document_id"] = doc.id
                        all_chunks.append(chunk)
                    except _json.JSONDecodeError:
                        pass

    return {"chunks": all_chunks, "total": len(all_chunks)}
