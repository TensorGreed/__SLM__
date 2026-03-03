"""Data Ingestion service — handles file uploads, parsing, and storage."""

import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.utils.file_parsers import (
    SUPPORTED_EXTENSIONS,
    compute_file_hash,
    get_file_type,
    parse_file,
)


def _project_data_dir(project_id: int) -> Path:
    """Return the data directory for a specific project."""
    d = settings.DATA_DIR / "projects" / str(project_id) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def get_or_create_raw_dataset(
    db: AsyncSession, project_id: int
) -> Dataset:
    """Get or create the 'raw' dataset for a project."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset:
        return dataset

    dataset = Dataset(
        project_id=project_id,
        name="Raw Documents",
        dataset_type=DatasetType.RAW,
        description="Ingested raw documents",
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)
    return dataset


async def ingest_file(
    db: AsyncSession,
    project_id: int,
    filename: str,
    file_content: bytes,
    source: str = "upload",
    sensitivity: str = "internal",
    license_info: str = "",
) -> RawDocument:
    """Ingest a single file: save to disk, parse, create DB record."""

    # Validate extension
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Get or create raw dataset
    dataset = await get_or_create_raw_dataset(db, project_id)

    # Save file to disk
    data_dir = _project_data_dir(project_id)
    file_path = data_dir / filename
    # Handle duplicate filenames
    counter = 1
    while file_path.exists():
        stem = Path(filename).stem
        file_path = data_dir / f"{stem}_{counter}{ext}"
        counter += 1

    file_path.write_bytes(file_content)

    # Compute hash
    file_hash = compute_file_hash(file_path)

    # Create document record
    doc = RawDocument(
        dataset_id=dataset.id,
        filename=file_path.name,
        file_type=get_file_type(filename),
        file_path=str(file_path),
        file_size_bytes=len(file_content),
        source=source,
        sensitivity=sensitivity,
        license_info=license_info,
        status=DocumentStatus.PENDING,
        metadata_={"hash": file_hash, "original_name": filename},
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Update dataset record count
    dataset.record_count += 1
    await db.flush()

    return doc


async def process_document(db: AsyncSession, document_id: int) -> RawDocument:
    """Process (parse) an ingested document and extract text content."""
    result = await db.execute(
        select(RawDocument).where(RawDocument.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise ValueError(f"Document {document_id} not found")

    try:
        doc.status = DocumentStatus.PROCESSING
        await db.flush()

        file_path = Path(doc.file_path)
        text = parse_file(file_path)

        # Save extracted text
        text_path = file_path.with_suffix(".extracted.txt")
        text_path.write_text(text, encoding="utf-8")

        doc.status = DocumentStatus.ACCEPTED
        doc.metadata_ = {
            **(doc.metadata_ or {}),
            "extracted_text_path": str(text_path),
            "char_count": len(text),
            "word_count": len(text.split()),
        }
        await db.flush()
        await db.refresh(doc)

    except Exception as e:
        doc.status = DocumentStatus.ERROR
        doc.metadata_ = {**(doc.metadata_ or {}), "error": str(e)}
        await db.flush()
        await db.refresh(doc)

    return doc


async def list_documents(
    db: AsyncSession, project_id: int, status: DocumentStatus | None = None
) -> list[RawDocument]:
    """List all documents for a project, optionally filtered by status."""
    dataset = await get_or_create_raw_dataset(db, project_id)
    query = select(RawDocument).where(RawDocument.dataset_id == dataset.id)
    if status:
        query = query.where(RawDocument.status == status)
    query = query.order_by(RawDocument.ingested_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all())


async def delete_document(db: AsyncSession, document_id: int) -> None:
    """Delete a document from disk and database."""
    result = await db.execute(
        select(RawDocument).where(RawDocument.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise ValueError(f"Document {document_id} not found")

    # Remove file from disk
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()
    extracted = file_path.with_suffix(".extracted.txt")
    if extracted.exists():
        extracted.unlink()

    await db.delete(doc)
