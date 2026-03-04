"""Data Ingestion API routes — file upload and document management."""

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import DocumentStatus
from app.schemas.dataset import DocumentResponse, DocumentUploadResponse
from app.services.ingestion_service import (
    delete_document,
    ingest_file,
    ingest_remote_dataset,
    list_documents,
    process_document,
)

router = APIRouter(prefix="/projects/{project_id}/ingestion", tags=["Ingestion"])


class RemoteImportRequest(BaseModel):
    source_type: Literal["huggingface", "kaggle", "url"]
    identifier: str = Field(..., min_length=1)   # dataset ID, slug, or URL
    split: str = "train"
    max_samples: int | None = Field(default=None, ge=1, le=100000)
    config_name: str | None = None
    field_mapping: dict[str, str] | None = None
    normalize_for_training: bool = True


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_file(
    project_id: int,
    file: UploadFile = File(...),
    source: str = Form(default="upload"),
    sensitivity: str = Form(default="internal"),
    license_info: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document for ingestion."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file")

    try:
        doc = await ingest_file(
            db=db,
            project_id=project_id,
            filename=file.filename,
            file_content=content,
            source=source,
            sensitivity=sensitivity,
            license_info=license_info,
        )
        return DocumentUploadResponse.model_validate(doc)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/upload-batch", status_code=201)
async def upload_batch(
    project_id: int,
    files: list[UploadFile] = File(...),
    source: str = Form(default="upload"),
    sensitivity: str = Form(default="internal"),
    db: AsyncSession = Depends(get_db),
):
    """Upload multiple documents at once."""
    results = []
    errors = []
    for f in files:
        if not f.filename:
            continue
        try:
            content = await f.read()
            doc = await ingest_file(
                db=db,
                project_id=project_id,
                filename=f.filename,
                file_content=content,
                source=source,
                sensitivity=sensitivity,
            )
            results.append(DocumentUploadResponse.model_validate(doc))
        except Exception as e:
            errors.append({"filename": f.filename, "error": str(e)})

    return {"uploaded": len(results), "errors": errors, "documents": results}


@router.post("/import-remote", status_code=201)
async def import_remote_dataset(
    project_id: int,
    req: RemoteImportRequest,
    db: AsyncSession = Depends(get_db),
):
    """Import a dataset from HuggingFace Hub, Kaggle, or a direct URL."""
    try:
        result = await ingest_remote_dataset(
            db=db,
            project_id=project_id,
            source_type=req.source_type,
            identifier=req.identifier,
            split=req.split,
            max_samples=req.max_samples,
            config_name=req.config_name,
            field_mapping=req.field_mapping,
            normalize_for_training=req.normalize_for_training,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents(
    project_id: int,
    status: DocumentStatus | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all ingested documents for a project."""
    try:
        docs = await list_documents(db, project_id, status)
        return [DocumentResponse.model_validate(d) for d in docs]
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/documents/{document_id}/process", response_model=DocumentResponse)
async def process_doc(
    project_id: int,
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Process (parse) an ingested document to extract text."""
    try:
        doc = await process_document(db, project_id, document_id)
        return DocumentResponse.model_validate(doc)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/documents/{document_id}", status_code=204)
async def remove_document(
    project_id: int,
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document."""
    try:
        await delete_document(db, project_id, document_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
