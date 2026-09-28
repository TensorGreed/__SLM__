"""Data Cleaning service — deduplication, PII detection, quality scoring, chunking."""

import asyncio
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.utils.tabular_io import is_tabular, load_rows, read_text_file
from app.database import async_session_factory
from app.models.dataset import Dataset, DatasetType, RawDocument


# ── PII / Secret Patterns ──────────────────────────────────────────────

PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    "api_key": re.compile(r'(?:api[_-]?key|apikey|token|secret)["\s:=]+["\']?([A-Za-z0-9_\-]{20,})["\']?', re.IGNORECASE),
    "aws_key": re.compile(r'AKIA[0-9A-Z]{16}'),
}

TOXIC_PATTERNS = {
    "abusive_language": re.compile(r"\b(idiot|moron|stupid|dumb)\b", re.IGNORECASE),
    "hate_speech": re.compile(r"\b(nazi|white\s+supremac|racist|ethnic\s+cleans)\b", re.IGNORECASE),
    "violent_threat": re.compile(r"\b(kill|murder|bomb|shoot|burn\s+down)\b", re.IGNORECASE),
}


def detect_pii(text: str) -> list[dict]:
    """Detect PII patterns in text. Returns list of {type, match, position}."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({
                "type": pii_type,
                "match": match.group()[:20] + "..." if len(match.group()) > 20 else match.group(),
                "position": match.start(),
            })
    return findings


def redact_pii(text: str) -> str:
    """Replace PII patterns with [REDACTED] placeholders."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
    return text


def detect_toxicity(text: str) -> list[dict]:
    """Detect simple toxicity patterns in text for safety filtering."""
    findings: list[dict] = []
    for category, pattern in TOXIC_PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group().strip()
            findings.append(
                {
                    "type": category,
                    "match": value[:30] + ("..." if len(value) > 30 else ""),
                    "position": match.start(),
                }
            )
    return findings


def redact_toxicity(text: str) -> str:
    """Replace matched toxic fragments with category-aware placeholders."""
    for category, pattern in TOXIC_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{category.upper()}]", text)
    return text


# ── Quality Scoring ────────────────────────────────────────────────────

def compute_quality_score(text: str) -> float:
    """Score text quality 0.0–1.0 based on length, coherence, and structure."""
    if not text.strip():
        return 0.0

    score = 0.0

    # Length score (0–0.3) — prefer 200-5000 chars
    char_count = len(text)
    if char_count < 50:
        score += 0.0
    elif char_count < 200:
        score += 0.1
    elif char_count < 5000:
        score += 0.3
    else:
        score += 0.2

    # Word diversity (0–0.2)
    words = text.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        score += min(0.2, unique_ratio * 0.25)

    # Sentence structure (0–0.2)
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    if sentences:
        sentence_ratio = len(valid_sentences) / max(len(sentences), 1)
        score += min(0.2, sentence_ratio * 0.25)

    # Not mostly boilerplate (0–0.15)
    boilerplate_indicators = ['cookie', 'privacy policy', 'terms of service', 'subscribe', 'click here']
    boilerplate_count = sum(1 for bp in boilerplate_indicators if bp in text.lower())
    score += max(0, 0.15 - boilerplate_count * 0.03)

    # Encoding quality (0–0.15) — penalize garbled text
    non_ascii = sum(1 for c in text if ord(c) > 127 and ord(c) < 256)
    if char_count > 0:
        garble_ratio = non_ascii / char_count
        score += max(0, 0.15 - garble_ratio * 1.5)

    return round(min(1.0, score), 3)


# ── Deduplication ───────────────────────────────────────────────────────

def compute_text_hash(text: str) -> str:
    """Hash normalized text for dedup."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


# ── Chunking ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start + chunk_size // 2, end)
            if last_period > start:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


# ── Boilerplate Removal ────────────────────────────────────────────────

BOILERPLATE_PATTERNS = [
    re.compile(r'©\s*\d{4}.*?(?:\n|$)', re.IGNORECASE),
    re.compile(r'all rights reserved.*?(?:\n|$)', re.IGNORECASE),
    re.compile(r'cookie\s*(?:policy|notice|settings).*?(?:\n|$)', re.IGNORECASE),
    re.compile(r'subscribe\s*(?:to|for)?\s*(?:our)?\s*newsletter.*?(?:\n|$)', re.IGNORECASE),
    re.compile(r'follow\s*us\s*on.*?(?:\n|$)', re.IGNORECASE),
]


def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate text patterns."""
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub('', text)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _cleaned_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "cleaned"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _render_record_text(record: object) -> str:
    """Render a structured row into a plain-text snippet for cleaning."""
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return str(record).strip()

    text = str(record.get("text") or "").strip()
    if text:
        return text

    question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    if question and answer:
        return f"Q: {question}\nA: {answer}"
    if question:
        return question
    if answer:
        return answer

    input_text = str(record.get("input_text") or "").strip()
    target_text = str(record.get("target_text") or "").strip()
    if input_text and target_text:
        return f"Input: {input_text}\nTarget: {target_text}"
    if input_text:
        return input_text
    if target_text:
        return target_text

    prompt = str(record.get("prompt") or record.get("instruction") or "").strip()
    completion = str(record.get("completion") or record.get("output") or "").strip()
    if prompt and completion:
        return f"Prompt: {prompt}\nCompletion: {completion}"
    if prompt:
        return prompt
    if completion:
        return completion

    for key in ("document", "content", "body", "value"):
        value = str(record.get(key) or "").strip()
        if value:
            return value

    parts: list[str] = []
    for value in record.values():
        if isinstance(value, str):
            token = value.strip()
            if token:
                parts.append(token)
        if len(parts) >= 4:
            break
    return "\n".join(parts).strip()


def _load_rows_from_source_file(file_path: Path) -> list[object]:
    if not file_path.exists():
        return []
    if is_tabular(file_path):
        try:
            return list(load_rows(file_path))
        except ValueError:
            pass
    return [read_text_file(file_path)]


def _materialize_extracted_text(doc: RawDocument) -> Path:
    """
    Ensure `.extracted.txt` exists for a document.

    Remote imports write structured JSONL directly and skip the explicit "process document"
    step; for those docs we synthesize extracted text from structured rows on demand.
    """
    source_path = Path(doc.file_path)
    extracted_path = source_path.with_suffix(".extracted.txt")
    if extracted_path.exists():
        return extracted_path

    rows = _load_rows_from_source_file(source_path)
    snippets: list[str] = []
    for row in rows:
        snippet = _render_record_text(row)
        if snippet:
            snippets.append(snippet)

    synthesized_text = "\n\n".join(snippets).strip()
    if synthesized_text:
        extracted_path.write_text(synthesized_text, encoding="utf-8")
    return extracted_path


async def get_or_create_cleaned_dataset(
    db: AsyncSession,
    project_id: int,
) -> Dataset:
    """Get or create the cleaned dataset for a project."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.CLEANED,
        )
    )
    ds = result.scalar_one_or_none()
    if ds:
        return ds

    ds = Dataset(
        project_id=project_id,
        name="Cleaned Dataset",
        dataset_type=DatasetType.CLEANED,
        description="Cleaned and chunked text data",
    )
    db.add(ds)
    await db.flush()
    await db.refresh(ds)
    return ds


async def _replace_document_entries(
    db: AsyncSession,
    project_id: int,
    document_id: int,
    new_entries: list[dict],
) -> Path:
    """Swap this document's rows in the project's cleaned.jsonl (other
    documents' rows are kept) and update the CLEANED dataset."""
    cleaned_ds = await get_or_create_cleaned_dataset(db, project_id)
    cleaned_file_path = _cleaned_dir(project_id) / "cleaned.jsonl"

    existing_entries: list[dict] = []
    if cleaned_file_path.exists():
        with open(cleaned_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("source_document_id") != document_id:
                    existing_entries.append(entry)

    merged_entries = existing_entries + new_entries
    with open(cleaned_file_path, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(merged_entries, start=1):
            entry["id"] = idx
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    cleaned_ds.file_path = str(cleaned_file_path)
    cleaned_ds.record_count = len(merged_entries)
    return cleaned_file_path


def _clean_value(
    value: object,
    *,
    redact: bool,
    redact_toxic: bool,
    pii_findings: list[dict],
    toxicity_findings: list[dict],
) -> object:
    if isinstance(value, str):
        text = value.strip()
        pii = detect_pii(text)
        toxic = detect_toxicity(text)
        pii_findings.extend(pii)
        toxicity_findings.extend(toxic)
        if redact and pii:
            text = redact_pii(text)
        if redact_toxic and toxic:
            text = redact_toxicity(text)
        return text
    if isinstance(value, list):
        return [
            _clean_value(v, redact=redact, redact_toxic=redact_toxic,
                         pii_findings=pii_findings, toxicity_findings=toxicity_findings)
            for v in value
        ]
    if isinstance(value, dict):
        return {
            k: _clean_value(v, redact=redact, redact_toxic=redact_toxic,
                            pii_findings=pii_findings, toxicity_findings=toxicity_findings)
            for k, v in value.items()
        }
    return value


_ROW_BOOKKEEPING_KEYS = frozenset({"source_document_id", "source_doc", "chunk_id", "row_index"})


async def _clean_structured_document(
    db: AsyncSession,
    project_id: int,
    doc: RawDocument,
    rows: list[dict],
    *,
    redact: bool,
    redact_toxic: bool,
) -> dict:
    """Row-preserving cleaning: per-field PII/toxicity redaction, drop empty
    and exact-duplicate rows, keep every column. Fields starting with ``_``
    (adapter bookkeeping like ``_task_profile``) pass through untouched."""
    pii_findings: list[dict] = []
    toxicity_findings: list[dict] = []
    seen: set[str] = set()
    entries: list[dict] = []
    empty_dropped = duplicates_dropped = 0
    text_sample: list[str] = []

    for index, row in enumerate(rows):
        cleaned_row: dict = {}
        for key, value in row.items():
            if str(key).startswith("_"):
                cleaned_row[key] = value
                continue
            cleaned_row[key] = _clean_value(
                value,
                redact=redact,
                redact_toxic=redact_toxic,
                pii_findings=pii_findings,
                toxicity_findings=toxicity_findings,
            )
        content = {k: v for k, v in cleaned_row.items() if not str(k).startswith("_")}
        if not any(isinstance(v, str) and v for v in content.values()) and not any(
            isinstance(v, (list, dict, int, float)) for v in content.values()
        ):
            empty_dropped += 1
            continue
        fingerprint = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
        if fingerprint in seen:
            duplicates_dropped += 1
            continue
        seen.add(fingerprint)
        if "id" in cleaned_row:
            cleaned_row["source_row_id"] = cleaned_row.pop("id")
        for key in _ROW_BOOKKEEPING_KEYS & set(cleaned_row):
            cleaned_row[f"source_{key}"] = cleaned_row.pop(key)
        if len(text_sample) < 200:
            text_sample.append(_render_record_text(content))
        entries.append({
            **cleaned_row,
            "source_document_id": doc.id,
            "source_doc": doc.filename,
            "chunk_id": len(entries),
            "row_index": index,
        })

    if not entries:
        raise ValueError(f"{doc.filename} has no non-empty rows after cleaning.")

    cleaned_file_path = await _replace_document_entries(db, project_id, doc.id, entries)
    sample_text = "\n\n".join(t for t in text_sample if t)
    quality = compute_quality_score(sample_text) if sample_text else 0.0

    doc.quality_score = quality
    doc.chunk_count = len(entries)
    doc.metadata_ = {
        **(doc.metadata_ or {}),
        "structured": True,
        "cleaned_dataset_path": str(cleaned_file_path),
        "rows_in": len(rows),
        "rows_kept": len(entries),
        "empty_rows_dropped": empty_dropped,
        "duplicate_rows_dropped": duplicates_dropped,
        "pii_count": len(pii_findings),
        "pii_types": sorted({f["type"] for f in pii_findings}),
        "toxicity_count": len(toxicity_findings),
        "toxicity_types": sorted({f["type"] for f in toxicity_findings}),
        "chunk_count": len(entries),
    }
    await db.flush()
    await db.refresh(doc)
    return {
        "document_id": doc.id,
        "structured": True,
        "quality_score": quality,
        "pii_findings": pii_findings,
        "toxicity_findings": toxicity_findings,
        "chunk_count": len(entries),
        "rows_in": len(rows),
        "rows_kept": len(entries),
        "duplicate_rows_dropped": duplicates_dropped,
        "empty_rows_dropped": empty_dropped,
        "original_chars": len(sample_text),
        "cleaned_chars": len(sample_text),
        "text_hash": compute_text_hash(sample_text),
    }


# ── Main Cleaning Pipeline ─────────────────────────────────────────────

async def clean_document(
    db: AsyncSession,
    project_id: int,
    document_id: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    redact: bool = True,
    redact_toxic: bool = False,
) -> dict:
    """Run full cleaning pipeline on a document."""
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            RawDocument.id == document_id,
            Dataset.project_id == project_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise ValueError(f"Document {document_id} not found")

    # Row files (CSV/TSV/JSON(L)/XLSX/Parquet, incl. HF/Kaggle imports) keep
    # their columns: one cleaned row per record. Flattening them into text
    # and chunking at 1000 chars destroyed labelled data before it ever
    # reached training.
    source_path = Path(doc.file_path)
    if is_tabular(source_path) and source_path.exists():
        try:
            structured_rows = load_rows(source_path)
        except ValueError:
            structured_rows = []
        if structured_rows and all(isinstance(r, dict) for r in structured_rows):
            return await _clean_structured_document(
                db,
                project_id,
                doc,
                structured_rows,
                redact=redact,
                redact_toxic=redact_toxic,
            )

    # Read extracted text
    extracted_path = _materialize_extracted_text(doc)
    if not extracted_path.exists():
        raise ValueError("Document has no extractable text. Run ingestion processing first.")

    raw_text = extracted_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise ValueError("Document extracted text is empty. Re-process or re-import the document.")

    # Step 1: Remove boilerplate
    cleaned = remove_boilerplate(raw_text)

    # Step 2: PII detection
    pii_findings = detect_pii(cleaned)
    toxicity_findings = detect_toxicity(cleaned)

    # Step 3: Redact PII if requested
    if redact and pii_findings:
        cleaned = redact_pii(cleaned)
    if redact_toxic and toxicity_findings:
        cleaned = redact_toxicity(cleaned)

    # Step 4: Quality scoring
    quality = compute_quality_score(cleaned)

    # Step 5: Dedup hash
    text_hash = compute_text_hash(cleaned)

    # Step 6: Chunking
    chunks = chunk_text(cleaned, chunk_size, chunk_overlap)

    # Save cleaned text
    cleaned_path = Path(doc.file_path).with_suffix(".cleaned.txt")
    cleaned_path.write_text(cleaned, encoding="utf-8")

    # Save chunks
    chunks_path = Path(doc.file_path).with_suffix(".chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({"chunk_id": i, "text": chunk, "source_doc": doc.filename}) + "\n")

    new_entries = [
        {
            "source_document_id": doc.id,
            "source_doc": doc.filename,
            "chunk_id": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]
    cleaned_file_path = await _replace_document_entries(db, project_id, doc.id, new_entries)

    # Update document record
    doc.quality_score = quality
    doc.chunk_count = len(chunks)
    doc.metadata_ = {
        **(doc.metadata_ or {}),
        "extracted_text_path": str(extracted_path),
        "cleaned_path": str(cleaned_path),
        "chunks_path": str(chunks_path),
        "cleaned_dataset_path": str(cleaned_file_path),
        "text_hash": text_hash,
        "pii_count": len(pii_findings),
        "pii_types": list(set(f["type"] for f in pii_findings)),
        "toxicity_count": len(toxicity_findings),
        "toxicity_types": list(set(f["type"] for f in toxicity_findings)),
        "original_chars": len(raw_text),
        "cleaned_chars": len(cleaned),
        "chunk_count": len(chunks),
    }
    await db.flush()
    await db.refresh(doc)

    return {
        "document_id": doc.id,
        "quality_score": quality,
        "pii_findings": pii_findings,
        "toxicity_findings": toxicity_findings,
        "chunk_count": len(chunks),
        "original_chars": len(raw_text),
        "cleaned_chars": len(cleaned),
        "text_hash": text_hash,
    }


# ── Background-task plumbing ──────────────────────────────────────────
#
# Cleaning a single huge document (e.g. a 100K-row HF import that
# extracted to one multi-megabyte text dump) can run for several
# minutes per regex pass + chunk write. When the request stays open
# for the whole job, the Vite proxy's 10-minute timeout severs the
# connection and the user sees "Network error" — even though the
# worker is still happily cleaning. The fix is to detach the work
# from the request lifetime: the API returns a task_id immediately
# and the frontend polls a status endpoint.
#
# We use FastAPI's existing event loop (``asyncio.create_task``) with
# an in-memory job registry, mirroring ``cloud_burst_service``'s
# pattern. No Celery dependency added — cleaning is in-process and
# its lifecycle ends with the API process, which matches every other
# cleaning code path today.


@dataclass
class CleaningTask:
    """In-memory record of a cleaning job. The instance lives in the
    process-global ``_CLEANING_TASKS`` dict, keyed by task_id; the API
    layer reads it via :func:`get_clean_task_status`."""

    task_id: str
    project_id: int
    document_ids: list[int]
    chunk_size: int
    chunk_overlap: int
    redact_pii: bool
    redact_toxicity: bool

    status: str = "pending"  # pending | running | completed | failed
    completed: int = 0
    total: int = 0
    results: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    error: str | None = None
    current_document_id: int | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "current_document_id": self.current_document_id,
            "results": list(self.results),
            "errors": list(self.errors),
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
        }


_CLEANING_TASKS: dict[str, CleaningTask] = {}
_CLEANING_TASKS_LOCK = threading.Lock()

# Cap the in-memory registry so a process running for weeks doesn't
# accumulate completed task records forever. When we hit the cap we
# drop the oldest finished tasks; in-flight tasks are always kept.
_MAX_TRACKED_TASKS: int = 64


def _trim_finished_tasks() -> None:
    """Evict the oldest finished tasks when the registry grows past
    the cap. Holds the lock; safe to call while holding it
    (recursive lock semantics from threading.Lock — actually no, this
    lock is acquired by the caller already)."""

    if len(_CLEANING_TASKS) <= _MAX_TRACKED_TASKS:
        return
    finished = sorted(
        (
            task
            for task in _CLEANING_TASKS.values()
            if task.finished_at is not None
        ),
        key=lambda t: t.finished_at,  # type: ignore[arg-type]
    )
    overflow = len(_CLEANING_TASKS) - _MAX_TRACKED_TASKS
    for task in finished[:overflow]:
        _CLEANING_TASKS.pop(task.task_id, None)


async def _run_cleaning_task(task: CleaningTask) -> None:
    """Run a cleaning batch under its own DB session.

    Catches per-document failures so a bad row in the middle of a
    batch doesn't sink the whole job (matches the legacy
    ``clean-batch`` semantics). Updates the in-memory task record on
    each step so the polling endpoint sees live progress.
    """

    task.status = "running"
    task.total = len(task.document_ids)
    task.updated_at = datetime.now(timezone.utc)

    try:
        async with async_session_factory() as db:
            for doc_id in task.document_ids:
                task.current_document_id = doc_id
                task.updated_at = datetime.now(timezone.utc)
                try:
                    result = await clean_document(
                        db,
                        task.project_id,
                        doc_id,
                        task.chunk_size,
                        task.chunk_overlap,
                        task.redact_pii,
                        task.redact_toxicity,
                    )
                    await db.commit()
                    task.results.append(result)
                except Exception as exc:  # noqa: BLE001
                    # Roll back this document's changes before moving on.
                    await db.rollback()
                    task.errors.append({"document_id": doc_id, "error": str(exc)})
                finally:
                    task.completed += 1
                    task.updated_at = datetime.now(timezone.utc)
        task.status = "completed"
        task.current_document_id = None
    except Exception as exc:  # noqa: BLE001
        # Fatal failure (e.g. DB session couldn't open) — record + exit.
        task.status = "failed"
        task.error = str(exc)
    finally:
        task.finished_at = datetime.now(timezone.utc)
        task.updated_at = task.finished_at


def start_clean_batch_task(
    *,
    project_id: int,
    document_ids: list[int],
    chunk_size: int,
    chunk_overlap: int,
    redact_pii: bool,
    redact_toxicity: bool,
) -> CleaningTask:
    """Register a new cleaning task + start it on the event loop.

    Returns the task record so the caller can hand its ``task_id``
    back to the frontend immediately. The actual cleaning runs on
    ``asyncio.create_task``; the API request returns within
    milliseconds regardless of batch size.
    """

    task = CleaningTask(
        task_id=f"clean-{uuid4().hex[:12]}",
        project_id=project_id,
        document_ids=list(document_ids),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        redact_pii=redact_pii,
        redact_toxicity=redact_toxicity,
    )
    with _CLEANING_TASKS_LOCK:
        _CLEANING_TASKS[task.task_id] = task
        _trim_finished_tasks()
    asyncio.create_task(_run_cleaning_task(task))
    return task


def get_clean_task(task_id: str) -> CleaningTask | None:
    """Read-only lookup. Returns None when the id is unknown."""

    with _CLEANING_TASKS_LOCK:
        return _CLEANING_TASKS.get(task_id)


def get_clean_task_status(task_id: str) -> dict[str, Any] | None:
    task = get_clean_task(task_id)
    return task.to_dict() if task else None
