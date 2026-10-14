"""Redact text with a trained token-classification run (Epic P5).

``POST /api/projects/{id}/redact`` tags a text with the run's span model
and returns the spans (with scores) plus the text masked (``[TYPE]``) or
tagged (``<TYPE>…</TYPE>``), and an offsets map from each output span back
to the input. One model is cached in-process (like ``local_chat_service``);
a lock serialises inference.

``apply_redaction`` is pure; ``load_tagger`` is the only model-loading
seam (tests patch it).
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment, ExperimentStatus

MODES = ("mask", "tag", "spans")
MAX_TEXT_CHARS = 200_000

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"key": None, "bundle": None}


class RedactionError(ValueError):
    """A request the endpoint answers with 4xx (no run, not a span run…)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def apply_redaction(text: str, spans: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """``{text, offsets}`` for ``mode`` mask / tag (``spans`` mode returns
    the input text unchanged). Overlapping spans keep the earlier one.
    ``offsets[i]`` = ``{type, start, end, out_start, out_end}``: the span in
    the input and its replacement in the output."""
    if mode not in MODES:
        raise RedactionError(f"mode must be one of {', '.join(MODES)}")
    ordered = sorted(
        (s for s in spans if 0 <= int(s["start"]) < int(s["end"]) <= len(text)),
        key=lambda s: (int(s["start"]), -int(s["end"])),
    )
    if mode == "spans":
        return {"text": text, "offsets": []}
    parts: list[str] = []
    offsets: list[dict[str, Any]] = []
    cursor = 0
    out_len = 0
    for span in ordered:
        start, end = int(span["start"]), int(span["end"])
        if start < cursor:
            continue  # overlaps a span already replaced
        parts.append(text[cursor:start])
        out_len += start - cursor
        ent_type = str(span["type"])
        if mode == "mask":
            replacement = f"[{ent_type.upper()}]"
        else:
            replacement = f"<{ent_type}>{text[start:end]}</{ent_type}>"
        parts.append(replacement)
        offsets.append({
            "type": ent_type, "start": start, "end": end,
            "out_start": out_len, "out_end": out_len + len(replacement),
        })
        out_len += len(replacement)
        cursor = end
    parts.append(text[cursor:])
    return {"text": "".join(parts), "offsets": offsets}


def _scripts_on_path() -> None:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


class TaggerBundle:
    def __init__(self, model: Any, tokenizer: Any, id2label: dict[int, str], max_length: int, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_length = max_length
        self.device = device


def load_tagger(artifacts: dict[str, Any]) -> TaggerBundle:
    """Load (or reuse) the token-classification model at
    ``artifacts["model_path"]``. One model resident at a time."""
    from app.services.local_chat_service import _cache_key

    key = _cache_key(artifacts["model_path"])
    if _CACHE["key"] == key and _CACHE["bundle"] is not None:
        return _CACHE["bundle"]
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    _CACHE["key"], _CACHE["bundle"] = None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(artifacts["model_path"])
    model = AutoModelForTokenClassification.from_pretrained(
        artifacts["model_path"], dtype=torch.float32,
    ).to(device).eval()
    bundle = TaggerBundle(model, tokenizer, artifacts["id2label"], tagger_window(tokenizer, artifacts), device)
    _CACHE["key"], _CACHE["bundle"] = key, bundle
    return bundle


def tagger_window(tokenizer: Any, artifacts: dict[str, Any]) -> int:
    """The training window (the saved tokenizer's ``model_max_length``),
    capped by the model's position embeddings; 512 when the tokenizer
    claims "unlimited"."""
    window = getattr(tokenizer, "model_max_length", None) or 512
    window = int(window) if window < 1_000_000 else 512
    return max(16, min(int(artifacts.get("max_length") or 512), window))


async def resolve_span_run(db: AsyncSession, project_id: int, experiment_id: int | None) -> Experiment:
    """The given run, else the project's newest COMPLETED token-classification run."""
    from app.services.post_training_eval_service import is_token_classification

    if experiment_id is not None:
        exp = await db.get(Experiment, experiment_id)
        if exp is None or exp.project_id != project_id:
            raise RedactionError(f"Run #{experiment_id} not found in project {project_id}", 404)
        return exp
    rows = (
        await db.execute(
            select(Experiment)
            .where(Experiment.project_id == project_id, Experiment.status == ExperimentStatus.COMPLETED)
            .order_by(desc(Experiment.id))
        )
    ).scalars().all()
    for exp in rows:
        if is_token_classification(exp):
            return exp
    raise RedactionError("This project has no completed span-tagging (token classification) run yet.", 404)


def _thresholds_for(model_path: str) -> dict[str, Any]:
    from app.services.span_model_package_service import read_thresholds

    return read_thresholds(Path(model_path)) or {}


def tag_text(
    bundle: TaggerBundle,
    text: str,
    *,
    min_score: float | None,
    recall_bias: float | None,
    per_type: dict[str, float] | None,
) -> list[dict[str, Any]]:
    _scripts_on_path()
    from train_token_classification import predict_spans

    with _LOCK:
        spans = predict_spans(
            [text], bundle.model, bundle.tokenizer, bundle.id2label,
            max_length=bundle.max_length, device=bundle.device,
            with_scores=True,
            thresholds=per_type or None,
            min_score=min_score,
            recall_bias=float(recall_bias or 0.0),
        )
    return spans[0] if spans else []


async def redact(
    db: AsyncSession,
    *,
    project_id: int,
    text: str,
    experiment_id: int | None = None,
    mode: str = "mask",
    min_score: float | None = None,
    recall_bias: float | None = None,
) -> dict[str, Any]:
    """Tag ``text`` with the run's span model and redact it. Request
    ``min_score`` / ``recall_bias`` override the run's thresholds.json
    (per-type thresholds there still apply above a request ``min_score``)."""
    from app.services.evaluation_service import _resolve_token_classifier_artifacts

    if mode not in MODES:
        raise RedactionError(f"mode must be one of {', '.join(MODES)}")
    if len(text) > MAX_TEXT_CHARS:
        raise RedactionError(f"text is longer than {MAX_TEXT_CHARS} characters", 413)
    exp = await resolve_span_run(db, project_id, experiment_id)
    artifacts = _resolve_token_classifier_artifacts(exp.output_dir) if exp.output_dir else None
    if artifacts is None:
        raise RedactionError(f"Run #{exp.id} has no token-classification model on disk.", 409)
    saved = _thresholds_for(artifacts["model_path"])
    per_type = {
        str(k): float(v) for k, v in (saved.get("per_type") or {}).items()
        if isinstance(v, (int, float)) and v > 0
    }
    effective_min = min_score if min_score is not None else (saved.get("default_min_score") or None)
    if min_score is not None:
        # A request floor raises every type to at least that value.
        per_type = {k: max(v, float(min_score)) for k, v in per_type.items()}
    effective_bias = recall_bias if recall_bias is not None else float(saved.get("recall_bias") or 0.0)
    bundle = await asyncio.to_thread(load_tagger, artifacts)
    spans = await asyncio.to_thread(
        tag_text, bundle, text,
        min_score=effective_min, recall_bias=effective_bias, per_type=per_type,
    )
    redacted = apply_redaction(text, spans, mode)
    return {
        "project_id": project_id,
        "experiment_id": exp.id,
        "experiment_name": exp.name,
        "mode": mode,
        "spans": spans,
        "text": redacted["text"],
        "offsets": redacted["offsets"],
        "settings": {
            "min_score": effective_min,
            "recall_bias": effective_bias,
            "per_type": per_type,
            "thresholds_file": bool(saved),
            "max_length": bundle.max_length,
        },
    }
