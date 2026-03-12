"""Lightweight RAG sandbox retrieval helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType, RawDocument


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]{2,}", str(text or "").lower())


def _score_overlap(query_tokens: set[str], chunk: str) -> float:
    if not query_tokens:
        return 0.0
    tokens = set(_tokenize(chunk))
    if not tokens:
        return 0.0
    inter = len(tokens.intersection(query_tokens))
    union = len(tokens.union(query_tokens))
    if union <= 0:
        return 0.0
    density = inter / max(1, len(query_tokens))
    jaccard = inter / union
    return (0.65 * density) + (0.35 * jaccard)


def _chunk_text(text: str, chunk_chars: int = 800, overlap: int = 120) -> list[str]:
    token = str(text or "").strip()
    if not token:
        return []
    if len(token) <= chunk_chars:
        return [token]
    rows: list[str] = []
    start = 0
    while start < len(token):
        end = min(len(token), start + chunk_chars)
        fragment = token[start:end].strip()
        if fragment:
            rows.append(fragment)
        if end >= len(token):
            break
        start = max(start + 1, end - overlap)
    return rows


def _read_doc_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


async def retrieve_project_rag_snippets(
    db: AsyncSession,
    *,
    project_id: int,
    query: str,
    top_k: int = 4,
    max_docs: int = 24,
) -> list[dict[str, Any]]:
    """Retrieve top lexical-overlap snippets from project raw docs."""
    safe_top_k = max(1, min(int(top_k), 10))
    safe_max_docs = max(1, min(int(max_docs), 100))
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []

    docs_query = (
        select(RawDocument, Dataset)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
        )
        .order_by(RawDocument.id.desc())
        .limit(safe_max_docs)
    )
    rows = (await db.execute(docs_query)).all()

    scored: list[tuple[float, dict[str, Any]]] = []
    snippet_id = 0
    for doc, _dataset in rows:
        doc_path = Path(str(doc.file_path or ""))
        if not doc_path.exists() or not doc_path.is_file():
            continue
        text = _read_doc_text(doc_path)
        if not text.strip():
            continue
        chunks = _chunk_text(text, chunk_chars=900, overlap=160)
        for chunk in chunks[:120]:
            score = _score_overlap(q_tokens, chunk)
            if score <= 0:
                continue
            snippet_id += 1
            scored.append(
                (
                    score,
                    {
                        "snippet_id": f"s{snippet_id}",
                        "document_id": int(doc.id),
                        "source_doc": str(doc.filename or doc_path.name),
                        "score": round(score, 4),
                        "text": chunk[:1200],
                    },
                )
            )

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [payload for _, payload in scored[:safe_top_k]]
    if not selected:
        return []

    max_score = max(float(item.get("score") or 0.0) for item in selected)
    for item in selected:
        score = float(item.get("score") or 0.0)
        item["normalized_score"] = round(score / max(0.0001, max_score), 4)
        item["rank"] = selected.index(item) + 1
    return selected


def build_rag_context_block(snippets: list[dict[str, Any]]) -> str:
    """Render snippet list into a grounding block."""
    if not snippets:
        return ""
    blocks: list[str] = []
    for item in snippets:
        snippet_id = str(item.get("snippet_id") or "s?")
        source_doc = str(item.get("source_doc") or "unknown")
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        blocks.append(f"[{snippet_id}] ({source_doc}) {text}")
    return "\n\n".join(blocks).strip()
