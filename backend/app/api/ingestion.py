"""Data Ingestion API routes — file upload and document management."""
import asyncio

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.dataset import DocumentStatus
from app.schemas.dataset import DocumentResponse, DocumentUploadResponse
from app.services.job_service import cancel_task, get_task_status
from app.services.ingestion_service import (
    delete_document,
    get_import_job_status,
    ingest_file,
    ingest_remote_dataset,
    list_documents,
    process_document,
    queue_remote_import,
    inspect_remote_dataset,
)
from app.services.dataset_intelligence_service import get_project_eda_stats
from app.services.dataset_intelligence_service import remove_project_outliers

router = APIRouter(prefix="/projects/{project_id}/ingestion", tags=["Ingestion"])


class RemoteImportRequest(BaseModel):
    source_type: str
    identifier: str = Field(..., min_length=1)   # dataset ID, slug, or URL
    split: str = "train"
    max_samples: int | None = None
    config_name: str | None = None
    field_mapping: dict[str, str] | None = None
    adapter_id: str = "default-canonical"
    task_profile: str | None = None
    adapter_config: dict[str, object] | None = None
    normalize_for_training: bool = True
    hf_token: str | None = Field(default=None, max_length=4096)
    kaggle_username: str | None = Field(default=None, max_length=255)
    kaggle_key: str | None = Field(default=None, max_length=4096)
    use_saved_secrets: bool = True

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type(cls, value: object) -> str:
        token = str(value or "").strip().lower()
        compact = token.replace("_", "").replace("-", "").replace(" ", "")
        alias_map = {
            "hf": "huggingface",
            "huggingface": "huggingface",
            "kaggle": "kaggle",
            "url": "url",
        }
        normalized = alias_map.get(compact)
        if normalized:
            return normalized
        raise ValueError("source_type must be one of: huggingface, kaggle, url")

    @field_validator("identifier")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        token = value.strip()
        if not token:
            raise ValueError("identifier cannot be empty")
        return token

    @field_validator("split", mode="before")
    @classmethod
    def normalize_split(cls, value: object) -> str:
        token = str(value or "").strip()
        return token or "train"

    @field_validator("adapter_id", mode="before")
    @classmethod
    def normalize_adapter_id(cls, value: object) -> str:
        token = str(value or "").strip().lower()
        normalized = token.replace("_", "-").replace(" ", "-")
        return normalized or "default-canonical"

    @field_validator("task_profile", mode="before")
    @classmethod
    def normalize_task_profile(cls, value: object) -> str | None:
        if value is None:
            return None
        token = str(value).strip().lower().replace("_", "-").replace(" ", "-")
        return token or None

    @field_validator("max_samples", mode="before")
    @classmethod
    def normalize_max_samples(cls, value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError("max_samples must be an integer")
        if parsed <= 0:
            return None
        if parsed > 100000:
            raise ValueError("max_samples must be <= 100000")
        return parsed

    @field_validator("hf_token", "kaggle_username", "kaggle_key", mode="before")
    @classmethod
    def normalize_optional_strings(cls, value: object) -> str | None:
        if value is None:
            return None
        token = str(value).strip()
        return token or None


class EDAOutlierRemovalRequest(BaseModel):
    min_tokens: int = Field(default=5, ge=1, le=4096)
    max_tokens: int = Field(default=4096, ge=1, le=200000)
    toxicity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_rows: int = Field(default=100000, ge=1, le=2000000)


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


@router.get("/import-remote/inspect")
async def inspect_remote(
    project_id: int,
    source_type: str,
    identifier: str,
    hf_token: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Inspect a remote dataset's configs and splits before importing."""
    try:
        return await inspect_remote_dataset(
            project_id=project_id,
            source_type=source_type,
            identifier=identifier,
            hf_token=hf_token,
            db=db,
        )
    except Exception as e:
        raise HTTPException(400, str(e))


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
            adapter_id=req.adapter_id,
            task_profile=req.task_profile,
            adapter_config=req.adapter_config,
            normalize_for_training=req.normalize_for_training,
            hf_token=req.hf_token,
            kaggle_username=req.kaggle_username,
            kaggle_key=req.kaggle_key,
            use_saved_secrets=req.use_saved_secrets,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/import-remote/queue", status_code=202)
async def import_remote_dataset_queue(
    project_id: int,
    req: RemoteImportRequest,
):
    """Queue a remote import job and stream progress via websocket logs."""
    try:
        return await queue_remote_import(
            project_id=project_id,
            source_type=req.source_type,
            identifier=req.identifier,
            split=req.split,
            max_samples=req.max_samples,
            config_name=req.config_name,
            field_mapping=req.field_mapping,
            adapter_id=req.adapter_id,
            task_profile=req.task_profile,
            adapter_config=req.adapter_config,
            normalize_for_training=req.normalize_for_training,
            hf_token=req.hf_token,
            kaggle_username=req.kaggle_username,
            kaggle_key=req.kaggle_key,
            use_saved_secrets=req.use_saved_secrets,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/imports/status")
async def remote_import_job_status(
    project_id: int,
    report_path: str = Query(..., min_length=1, max_length=4096),
):
    """Poll a queued remote import job status by report path."""
    try:
        return get_import_job_status(project_id, report_path)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/imports/tasks/{task_id}")
async def remote_import_task_status(
    project_id: int,
    task_id: str,
):
    """Read Celery task state for an ingestion import job."""
    try:
        return get_task_status(task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/imports/tasks/{task_id}/cancel")
async def cancel_remote_import_task(
    project_id: int,
    task_id: str,
):
    """Request cancellation for a queued/running ingestion import task."""
    try:
        return cancel_task(task_id, terminate=True)
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


@router.get("/eda")
async def get_project_eda(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get Exploratory Data Analysis (EDA) stats for the project's raw ingestion data."""
    try:
        return await get_project_eda_stats(db, project_id)
    except Exception as e:
        raise HTTPException(500, f"Error calculating EDA: {e}")


@router.post("/eda/remove-outliers")
async def remove_eda_outliers(
    project_id: int,
    req: EDAOutlierRemovalRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Filter out obvious short/long/toxic rows from prepared train split."""
    payload = req or EDAOutlierRemovalRequest()
    try:
        return await remove_project_outliers(
            db,
            project_id=project_id,
            min_tokens=payload.min_tokens,
            max_tokens=payload.max_tokens,
            toxicity_threshold=payload.toxicity_threshold,
            max_rows=payload.max_rows,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


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


@router.websocket("/ws/logs")
async def ingestion_logs(websocket: WebSocket, project_id: int):
    """Stream ingestion worker log lines for a project."""
    await websocket.accept()
    redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()
    channel = f"log:ingestion:project:{project_id}"
    await pubsub.subscribe(channel)
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("type") == "message":
                await websocket.send_json({"type": "log", "text": str(message.get("data", ""))})
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
            except TimeoutError:
                pass
            except WebSocketDisconnect:
                break
    finally:
        await pubsub.unsubscribe(channel)
        await redis_client.aclose()
