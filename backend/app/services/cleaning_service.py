"""Data Cleaning service — deduplication, PII detection, quality scoring, chunking."""

import csv
import hashlib
import json
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
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

    ext = file_path.suffix.lower()
    rows: list[object] = []

    if ext == ".jsonl":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                token = line.strip()
                if not token:
                    continue
                try:
                    rows.append(json.loads(token))
                except json.JSONDecodeError:
                    rows.append(token)
        return rows

    if ext == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return list(raw)
        return [raw]

    if ext == ".csv":
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    return [file_path.read_text(encoding="utf-8", errors="replace")]


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


# ── Main Cleaning Pipeline ─────────────────────────────────────────────

async def clean_document(
    db: AsyncSession,
    project_id: int,
    document_id: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    redact: bool = True,
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

    # Step 3: Redact PII if requested
    if redact and pii_findings:
        cleaned = redact_pii(cleaned)

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

    # Upsert project-level cleaned dataset so dataset prep can consume cleaned data.
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
                if entry.get("source_document_id") != doc.id:
                    existing_entries.append(entry)

    new_entries = [
        {
            "source_document_id": doc.id,
            "source_doc": doc.filename,
            "chunk_id": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]
    merged_entries = existing_entries + new_entries

    with open(cleaned_file_path, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(merged_entries, start=1):
            entry["id"] = idx
            f.write(json.dumps(entry) + "\n")

    cleaned_ds.file_path = str(cleaned_file_path)
    cleaned_ds.record_count = len(merged_entries)

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
        "chunk_count": len(chunks),
        "original_chars": len(raw_text),
        "cleaned_chars": len(cleaned),
        "text_hash": text_hash,
    }
