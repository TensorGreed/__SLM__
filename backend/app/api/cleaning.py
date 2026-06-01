"""Data Cleaning API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.cleaning_service import (
    clean_document,
    get_clean_task_status,
    start_clean_batch_task,
)

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


@router.post("/clean-batch-async", status_code=202)
async def clean_batch_async(
    project_id: int,
    req: CleanBatchRequest,
):
    """Start a cleaning batch as a background task; return a task_id.

    The synchronous ``clean-batch`` endpoint holds the HTTP request
    open for the entire job. With large documents (100K-row HF
    imports), that exceeds the dev proxy's 10-minute timeout and the
    frontend sees a "network error" while the worker is still
    cleaning. This variant detaches the work from the request
    lifetime — the response returns within milliseconds with a
    ``task_id`` the frontend polls via :func:`task_status`.

    The job itself runs on the same event loop as the API, against a
    fresh DB session per the lifecycle pattern used elsewhere
    (``cloud_burst_service``).
    """

    task = start_clean_batch_task(
        project_id=project_id,
        document_ids=req.document_ids,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        redact_pii=req.redact_pii,
        redact_toxicity=req.redact_toxicity,
    )
    return task.to_dict()


@router.get("/tasks/{task_id}")
async def task_status(
    project_id: int,
    task_id: str,
):
    """Poll a backgrounded cleaning job for progress + results.

    Returns 404 when the id is unknown (process restarted, registry
    evicted the record, or the id was never created). The frontend
    treats 404 the same as a fatal failure — the task can't be
    resumed.
    """

    payload = get_clean_task_status(task_id)
    if payload is None:
        raise HTTPException(404, f"Cleaning task '{task_id}' not found.")
    if payload.get("project_id") != project_id:
        raise HTTPException(
            404, f"Cleaning task '{task_id}' not found in this project."
        )
    return payload


@router.get("/chunks")
async def get_cleaned_chunks(
    project_id: int,
    limit: int = 200,
    offset: int = 0,
    random_sample: bool = True,
    seed: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Return cleaned text chunks for a project.

    Streams ``.chunks.jsonl`` files for the project line-by-line so a
    74k-chunk project doesn't stream 37MB of JSON to the browser on
    every "Load from Cleaned Data" click. Three modes:

    - **random_sample=true** (default) — reservoir sample ``limit``
      chunks across the whole pool. Each call returns a different
      sample unless ``seed`` is provided.
    - **random_sample=false, offset=N** — paginated: skip the first
      N chunks, return the next ``limit``. Useful for "Load more".
    - ``limit=0`` returns just the total count + no rows. Cheapest
      way to ask "how many chunks does this project have?"

    Response:
      - ``chunks``: list of chunk dicts (``document_id`` is injected)
      - ``total``: full chunk count across the project
      - ``returned``: len(chunks)
      - ``limit`` / ``offset`` / ``random_sample`` / ``seed``: echoed
    """
    import json as _json
    import random as _random
    from pathlib import Path
    from sqlalchemy import select
    from app.models.dataset import Dataset, DatasetType, RawDocument

    if limit < 0:
        raise HTTPException(400, "limit must be >= 0")
    if limit > 5000:
        raise HTTPException(400, "limit must be <= 5000")
    if offset < 0:
        raise HTTPException(400, "offset must be >= 0")

    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
    )
    docs = list(result.scalars().all())

    # Two ingest paths feed a project (see data_health_service for the
    # full explanation): document-cleaning produces .chunks.jsonl
    # files; dataset-import writes labelled rows directly to a
    # SYNTHETIC / CLEANED Dataset. Synth panel's "Load from Cleaned
    # Data" must surface BOTH so a classification project's 30K
    # imported rows aren't invisible.
    dataset_result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_([DatasetType.SYNTHETIC, DatasetType.CLEANED]),
        )
    )
    labelled_datasets = list(dataset_result.scalars().all())

    def _iter_chunk_lines():
        """Yield parsed chunks across both ingest paths:

        - ``.chunks.jsonl`` next to each RawDocument (doc pipeline)
        - rows of each labelled SYNTHETIC/CLEANED ``.jsonl`` file
          (dataset-import pipeline). Each row is shaped into a
          chunk-compatible dict so the picker UI doesn't need a
          branch.
        """
        for doc in docs:
            if not doc.file_path:
                continue
            chunks_path = Path(doc.file_path).with_suffix(".chunks.jsonl")
            if not chunks_path.exists():
                continue
            try:
                with chunks_path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            chunk = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        chunk["document_id"] = doc.id
                        chunk["ingest_path"] = "document"
                        yield chunk
            except OSError:
                continue

        for ds in labelled_datasets:
            if not ds.file_path:
                continue
            ds_path = Path(ds.file_path)
            if not ds_path.exists():
                continue
            try:
                with ds_path.open("r", encoding="utf-8") as handle:
                    for row_idx, raw_line in enumerate(handle):
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            row = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        # Map the row's text field into the
                        # cleaning-chunk shape so the picker renders
                        # uniformly. Falls back across common field
                        # names so we don't lose rows on adapter
                        # variants.
                        text = (
                            row.get("text")
                            or row.get("input")
                            or row.get("prompt")
                            or row.get("question")
                            or row.get("source")
                            or ""
                        )
                        if not isinstance(text, str) or not text.strip():
                            continue
                        yield {
                            # Negative document_id to flag the
                            # dataset-import origin without colliding
                            # with real RawDocument ids.
                            "document_id": -int(ds.id),
                            "chunk_id": row_idx,
                            "text": text,
                            "label": row.get("label"),
                            "ingest_path": "dataset_import",
                            "dataset_type": ds.dataset_type.value if hasattr(ds.dataset_type, "value") else str(ds.dataset_type),
                        }
            except OSError:
                continue

    rng = _random.Random(seed) if seed is not None else _random.Random()
    selected: list[dict] = []
    total = 0

    if random_sample:
        # Reservoir sample: O(total) time, O(limit) memory.
        for chunk in _iter_chunk_lines():
            total += 1
            if limit == 0:
                continue
            if len(selected) < limit:
                selected.append(chunk)
            else:
                idx = rng.randint(0, total - 1)
                if idx < limit:
                    selected[idx] = chunk
    else:
        # Paginated: skip ``offset``, take ``limit``, then keep
        # counting (cheap) so the response can show the full total.
        taken = 0
        skipped = 0
        for chunk in _iter_chunk_lines():
            total += 1
            if skipped < offset:
                skipped += 1
                continue
            if taken < limit:
                selected.append(chunk)
                taken += 1

    return {
        "chunks": selected,
        "returned": len(selected),
        "total": total,
        "limit": limit,
        "offset": offset,
        "random_sample": random_sample,
        "seed": seed,
    }
