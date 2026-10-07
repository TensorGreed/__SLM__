"""Cross-encoder reranking for auto-RAG retrieval.

BM25 ranks passages by term overlap, so a question about "retaining a
record of disclosure" can pull the neighbouring section that shares the
words. A cross-encoder reads (question, passage) pairs together and scores
relevance; reranking BM25's top candidates with it and keeping the top-k
fixes most wrong-section retrievals at the cost of one small model
forward pass per candidate.

Plain ``transformers`` (``AutoModelForSequenceClassification``) — no
sentence-transformers dependency. The model is cached per process; fp32
on CPU, fp16 on CUDA. Best-effort: any failure returns the BM25 order so
retrieval never breaks because a reranker is missing.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

_LOG = logging.getLogger(__name__)

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# BM25 candidates handed to the reranker: enough to recover a passage BM25
# ranked below the cut, bounded so a reranking call stays cheap.
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 10
MAX_CANDIDATES = 40
RERANK_MAX_LENGTH = 512


def candidate_pool(k: int) -> int:
    """How many BM25 hits to fetch before reranking down to ``k``."""
    return max(MIN_CANDIDATES, min(MAX_CANDIDATES, int(k) * CANDIDATE_MULTIPLIER))


def normalize_reranker(value: Any) -> str | None:
    """``None`` / ``""`` / ``"none"`` / ``False`` → no reranker; ``True`` /
    ``"default"`` → the default model; else the model id as given."""
    if value is None or value is False:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "off", "false", "0"}:
        return None
    if text.lower() in {"default", "true", "1", "on"}:
        return DEFAULT_RERANKER
    return text


@lru_cache(maxsize=2)
def _load(model_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype).to(device).eval()
    return tokenizer, model, device


def score_pairs(query: str, texts: list[str], *, model_name: str = DEFAULT_RERANKER) -> list[float]:
    """Relevance score of ``query`` against each text (higher = more relevant)."""
    import torch

    if not texts:
        return []
    tokenizer, model, device = _load(model_name)
    scores: list[float] = []
    batch = 16
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            encoded = tokenizer(
                [query] * len(chunk), chunk, padding=True, truncation=True,
                max_length=RERANK_MAX_LENGTH, return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                logits = logits[:, -1]
            scores.extend(float(v) for v in logits.reshape(-1).float().cpu().tolist())
    return scores


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    model_name: str | None = DEFAULT_RERANKER,
    text_of: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Reorder BM25 ``hits`` by cross-encoder relevance and keep ``top_k``.
    Each kept hit gets ``rerank_score`` and ``bm25_rank`` (its position
    before reranking). Without a model name (or on any failure) returns the
    first ``top_k`` hits unchanged."""
    if not hits:
        return []
    model_name = normalize_reranker(model_name)
    if model_name is None:
        return hits[:top_k]

    def _text(hit: dict[str, Any]) -> str:
        if text_of is not None:
            return text_of(hit)
        payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else {}
        for key in ("text", "answer", "question"):
            if isinstance(payload.get(key), str) and payload[key].strip():
                return str(payload[key])
        return ""

    try:
        scores = score_pairs(query, [_text(h) for h in hits], model_name=model_name)
    except Exception as exc:  # noqa: BLE001 — keep BM25 order
        _LOG.warning("reranker %s unavailable, keeping BM25 order: %s", model_name, exc)
        return hits[:top_k]
    ranked = sorted(
        (dict(hit, rerank_score=score, bm25_rank=index + 1) for index, (hit, score) in enumerate(zip(hits, scores))),
        key=lambda h: h["rerank_score"],
        reverse=True,
    )
    return ranked[:top_k]
