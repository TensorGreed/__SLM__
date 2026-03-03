"""Data Cleaning API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.cleaning_service import clean_document

router = APIRouter(prefix="/projects/{project_id}/cleaning", tags=["Cleaning"])


class CleanRequest(BaseModel):
    document_id: int
    chunk_size: int = Field(1000, ge=100, le=10000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    redact_pii: bool = True


class CleanBatchRequest(BaseModel):
    document_ids: list[int]
    chunk_size: int = Field(1000, ge=100, le=10000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    redact_pii: bool = True


@router.post("/clean")
async def clean_single(
    project_id: int,
    req: CleanRequest,
    db: AsyncSession = Depends(get_db),
):
    """Clean a single document."""
    try:
        result = await clean_document(
            db, req.document_id, req.chunk_size, req.chunk_overlap, req.redact_pii
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
                db, doc_id, req.chunk_size, req.chunk_overlap, req.redact_pii
            )
            results.append(result)
        except Exception as e:
            errors.append({"document_id": doc_id, "error": str(e)})

    return {"cleaned": len(results), "errors": errors, "results": results}
