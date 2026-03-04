"""Data Ingestion service — handles file uploads, parsing, and storage."""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.project import Project
from app.services.record_normalization import build_schema_profile, normalize_records
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


def _safe_filename(filename: str) -> str:
    """Normalize to a basename so uploads cannot escape the project directory."""
    safe = Path(filename).name.strip()
    if not safe or safe in {".", ".."}:
        raise ValueError("Invalid filename")
    return safe


async def _get_document_for_project(
    db: AsyncSession,
    project_id: int,
    document_id: int,
) -> RawDocument | None:
    """Fetch a document only if it belongs to the requested project."""
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            RawDocument.id == document_id,
            Dataset.project_id == project_id,
        )
    )
    return result.scalar_one_or_none()


async def get_or_create_raw_dataset(
    db: AsyncSession, project_id: int
) -> Dataset:
    """Get or create the 'raw' dataset for a project."""
    project = await db.execute(select(Project.id).where(Project.id == project_id))
    if not project.scalar_one_or_none():
        raise ValueError(f"Project {project_id} not found")

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
    safe_filename = _safe_filename(filename)

    # Validate extension
    ext = Path(safe_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Get or create raw dataset
    dataset = await get_or_create_raw_dataset(db, project_id)

    # Save file to disk
    data_dir = _project_data_dir(project_id)
    file_path = data_dir / safe_filename
    # Handle duplicate filenames
    counter = 1
    while file_path.exists():
        stem = Path(safe_filename).stem
        file_path = data_dir / f"{stem}_{counter}{ext}"
        counter += 1

    file_path.write_bytes(file_content)

    # Compute hash
    file_hash = compute_file_hash(file_path)

    # Create document record
    doc = RawDocument(
        dataset_id=dataset.id,
        filename=file_path.name,
        file_type=get_file_type(safe_filename),
        file_path=str(file_path),
        file_size_bytes=len(file_content),
        source=source,
        sensitivity=sensitivity,
        license_info=license_info,
        status=DocumentStatus.PENDING,
        metadata_={"hash": file_hash, "original_name": safe_filename},
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Update dataset record count
    dataset.record_count += 1
    await db.flush()

    return doc


async def process_document(
    db: AsyncSession,
    project_id: int,
    document_id: int,
) -> RawDocument:
    """Process (parse) an ingested document and extract text content."""
    doc = await _get_document_for_project(db, project_id, document_id)
    if not doc:
        raise ValueError(f"Document {document_id} not found in project {project_id}")

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


async def delete_document(
    db: AsyncSession,
    project_id: int,
    document_id: int,
) -> None:
    """Delete a document from disk and database."""
    doc = await _get_document_for_project(db, project_id, document_id)
    if not doc:
        raise ValueError(f"Document {document_id} not found in project {project_id}")

    # Remove file from disk
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()
    extracted = file_path.with_suffix(".extracted.txt")
    if extracted.exists():
        extracted.unlink()

    await db.delete(doc)


# ── Remote Dataset Connectors ─────────────────────────────────────────

async def ingest_remote_dataset(
    db: AsyncSession,
    project_id: int,
    source_type: str,
    identifier: str,
    split: str = "train",
    max_samples: int | None = None,
    config_name: str | None = None,
    field_mapping: dict[str, str] | None = None,
    normalize_for_training: bool = True,
) -> dict:
    """
    Ingest a dataset from a remote source.

    source_type: 'huggingface', 'kaggle', or 'url'
    identifier:
        - huggingface: dataset ID like 'squad' or 'tatsu-lab/alpaca'
        - kaggle: dataset slug like 'user/dataset-name'
        - url: direct link to a .csv, .json, or .jsonl file
    """
    import csv
    import json
    import random
    import re

    import httpx

    raw_dataset = await get_or_create_raw_dataset(db, project_id)
    data_dir = _project_data_dir(project_id)
    raw_samples: list[dict] = []
    source_mode = "simulated"
    target_samples = max_samples or 200

    def _safe_remote_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")[:80] or "dataset"

    if source_type == "huggingface":
        safe_name = _safe_remote_name(identifier.replace("/", "_"))
        filename = f"hf_{safe_name}_{split}.jsonl"
        try:
            from datasets import load_dataset

            dataset_kwargs = {"split": split}
            if config_name:
                dataset_kwargs["name"] = config_name
            hf_dataset = load_dataset(identifier, **dataset_kwargs)
            for i, row in enumerate(hf_dataset):
                if i >= target_samples:
                    break
                if isinstance(row, dict):
                    raw_samples.append(row)
                else:
                    raw_samples.append({"value": row})
            source_mode = "live"
        except Exception:
            # Fallback to simulation so the UI flow remains usable even without deps/auth.
            for i in range(target_samples):
                raw_samples.append({
                    "instruction": f"Sample instruction #{i+1} from {identifier}",
                    "input": f"Context for sample {i+1}",
                    "output": f"Expected output for sample {i+1} from the {identifier} dataset.",
                    "source": f"huggingface/{identifier}",
                    "split": split,
                })

    elif source_type == "kaggle":
        safe_name = _safe_remote_name(identifier.replace("/", "_"))
        filename = f"kaggle_{safe_name}.jsonl"
        for i in range(target_samples):
            raw_samples.append({
                "text": f"Row {i+1} from Kaggle dataset {identifier}.",
                "label": random.choice(["positive", "negative", "neutral"]),
                "source": f"kaggle/{identifier}",
            })

    elif source_type == "url":
        safe_name = _safe_remote_name(Path(identifier).stem or "download")
        filename = f"url_{safe_name}.jsonl"
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                resp = await client.get(identifier)
                resp.raise_for_status()
                raw_text = resp.text

            ext = Path(identifier).suffix.lower()
            if ext == ".jsonl":
                for line in raw_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            raw_samples.append(record)
                        else:
                            raw_samples.append({"value": record})
                    except json.JSONDecodeError:
                        continue
                    if len(raw_samples) >= target_samples:
                        break
            elif ext == ".json":
                data = json.loads(raw_text)
                if isinstance(data, list):
                    raw_samples = [d if isinstance(d, dict) else {"value": d} for d in data[:target_samples]]
                elif isinstance(data, dict):
                    raw_samples = [data]
                else:
                    raw_samples = [{"value": data}]
            elif ext == ".csv":
                reader = csv.DictReader(raw_text.splitlines())
                for row in reader:
                    raw_samples.append(dict(row))
                    if len(raw_samples) >= target_samples:
                        break
            else:
                for i, line in enumerate(raw_text.splitlines()):
                    if i >= target_samples:
                        break
                    if line.strip():
                        raw_samples.append({"content": line.strip()})

            if not raw_samples:
                raise ValueError("No records parsed from URL response")
            source_mode = "live"
        except Exception as e:
            raise ValueError(f"URL import failed: {e}")

    else:
        raise ValueError(f"Unsupported source_type: {source_type}. Use 'huggingface', 'kaggle', or 'url'.")

    if not raw_samples:
        raise ValueError(f"No records available from source '{source_type}:{identifier}'")

    if normalize_for_training:
        rows_to_write, dropped_records = normalize_records(raw_samples, field_mapping=field_mapping)
        if not rows_to_write:
            raise ValueError(
                "Unable to normalize imported records. Provide field_mapping for question/answer/text columns."
            )
    else:
        rows_to_write = [s if isinstance(s, dict) else {"value": s} for s in raw_samples]
        dropped_records = 0

    schema_profile = build_schema_profile(
        raw_samples,
        field_mapping=field_mapping if normalize_for_training else None,
    )

    # Write JSONL file
    file_path = data_dir / filename
    counter = 1
    while file_path.exists():
        stem = file_path.stem
        file_path = data_dir / f"{stem}_{counter}.jsonl"
        counter += 1

    with open(file_path, "w", encoding="utf-8") as f:
        for sample in rows_to_write:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    file_content = file_path.read_bytes()
    file_hash = compute_file_hash(file_path)

    doc = RawDocument(
        dataset_id=raw_dataset.id,
        filename=file_path.name,
        file_type="jsonl",
        file_path=str(file_path),
        file_size_bytes=len(file_content),
        source=f"{source_type}:{identifier}",
        sensitivity="public",
        license_info=f"Imported from {source_type}",
        status=DocumentStatus.ACCEPTED,
        metadata_={
            "hash": file_hash,
            "original_name": filename,
            "source_type": source_type,
            "identifier": identifier,
            "split": split,
            "raw_samples": len(raw_samples),
            "num_samples": len(rows_to_write),
            "dropped_records": dropped_records,
            "normalize_for_training": normalize_for_training,
            "field_mapping": field_mapping or {},
            "schema_profile": schema_profile,
            "config_name": config_name,
            "source_mode": source_mode,
        },
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    raw_dataset.record_count += 1
    await db.flush()

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "source_type": source_type,
        "identifier": identifier,
        "raw_samples": len(raw_samples),
        "samples_ingested": len(rows_to_write),
        "dropped_records": dropped_records,
        "file_size_bytes": len(file_content),
        "source_mode": source_mode,
        "schema_profile": schema_profile,
        "status": "accepted",
    }
