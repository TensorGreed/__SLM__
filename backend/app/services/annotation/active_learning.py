"""Active-learning row ranker for label-job queues (Epic F).

When a labeler has a queue of unlabeled rows, picking them FIFO wastes
budget on rows the model would have predicted correctly anyway. The
*active* strategy ranks unlabeled rows by model uncertainty so each
row the human labels carries more information per minute spent.

Phase 1 covers the **classification** task shape:

  uncertainty(row) = entropy of softmax over the project's most-recent
                     completed classification experiment's label-head
                     logits for that row's text.

High entropy → the model is genuinely unsure (probabilities spread
across classes) → labeling here moves the boundary the most. Rows the
model is already confident on rank last and stay in the FIFO tail.

This module is intentionally split into two layers:

* :func:`rank_rows_by_uncertainty` — pure: takes rows + a callable
  that returns a score per row, returns row ids sorted high-to-low.
  Trivially testable without a GPU or any HF/torch import.

* :func:`score_classification_rows` — wraps loading the experiment's
  trained model and computing softmax entropy over its label space.
  Lazily imports torch/transformers/peft so test environments without
  a CUDA build can still import this module.

The caller (``assign_next`` in ``annotation_service``) decides
whether to invoke the active path or fall back to FIFO based on the
job's label_type, the request's strategy, and whether a scoreable
experiment exists. The ranker itself never raises on "no usable
model" — it returns ``None`` and the caller falls back cleanly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment
from app.models.label_job import LabelJob, LabelRow


# Field names we try (in order) when extracting the human-readable
# text from a label row's ``raw_payload``. Matches the seed-from-
# dataset shape and the classification-label adapter's input fields
# so the ranker scores the same text the model would see at eval.
_TEXT_FIELD_CANDIDATES: tuple[str, ...] = (
    "text",
    "content",
    "input",
    "question",
    "prompt",
    "instruction",
    "body",
    "message",
)


def extract_row_text(raw_payload: dict[str, Any] | None) -> str | None:
    """Return the first non-empty text value from a label row's
    ``raw_payload``, or ``None`` when no usable text is present.
    Rows without text get skipped by the ranker (they rank last)
    rather than triggering a forward pass on an empty string.
    """
    if not isinstance(raw_payload, dict):
        return None
    for field in _TEXT_FIELD_CANDIDATES:
        value = raw_payload.get(field)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return None


def softmax_entropy(logits: Sequence[float]) -> float:
    """Shannon entropy of softmax(logits) in nats. Stable against
    large logits (subtracts the max before exp). Returns 0.0 for an
    empty sequence — treated as "no uncertainty signal."
    """
    if not logits:
        return 0.0
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    if z <= 0.0:
        return 0.0
    probs = [e / z for e in exps]
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def rank_rows_by_uncertainty(
    rows: Iterable[LabelRow],
    *,
    score_fn: Callable[[list[LabelRow]], list[float | None]],
) -> list[int]:
    """Return row ids sorted by descending uncertainty.

    ``score_fn`` receives the list of rows in input order and must
    return a same-length list of scores (or ``None`` for rows it
    can't score — those rank after any row with a real score). The
    indirection keeps this function pure: tests pass a stub
    callable, production passes :func:`score_classification_rows`.
    """
    materialized = list(rows)
    if not materialized:
        return []
    scores = score_fn(materialized)
    if len(scores) != len(materialized):
        raise ValueError(
            "score_fn returned wrong number of scores "
            f"(got {len(scores)}, expected {len(materialized)})"
        )
    indexed: list[tuple[float, int, int]] = []
    # Sentinel rank so ``None``-scored rows always trail any
    # real score, and within each band we preserve insertion order
    # via the row's ``id`` (stable secondary key).
    for idx, (row, raw) in enumerate(zip(materialized, scores)):
        has_score = raw is not None
        score = float(raw) if has_score else float("-inf")
        indexed.append(
            (score, -idx, int(row.id))
        )
    # Sort: highest score first; ties broken by earlier insertion
    # (`-idx` keeps lower indices ahead). Row id is in the tuple
    # only so we can pluck it out below.
    indexed.sort(reverse=True)
    return [row_id for _score, _neg_idx, row_id in indexed]


async def latest_scoreable_classification_experiment(
    db: AsyncSession, *, project_id: int
) -> Experiment | None:
    """Return the project's most-recent completed classification
    experiment, or ``None`` when there isn't one yet.

    "Scoreable" today means ``status == 'completed'`` and the
    config carries ``task_type == 'classification'`` (or the
    equivalent ``classification`` task_profile). A future
    refinement could check the saved adapter for a classifier
    head module, but for Phase 1 the experiment status is a
    sufficient gate — broken runs land as ``failed``.
    """
    result = await db.execute(
        select(Experiment)
        .where(
            Experiment.project_id == project_id,
            Experiment.status == "completed",
        )
        .order_by(desc(Experiment.id))
    )
    for experiment in result.scalars():
        cfg = experiment.config or {}
        task_type = str(cfg.get("task_type") or "").strip().lower()
        if task_type == "classification":
            return experiment
    return None


def score_classification_rows(
    rows: list[LabelRow],
    *,
    model_path: str,
    label_space: list[str],
) -> list[float | None]:
    """Run the experiment's classifier head over each row's text and
    return softmax entropy per row. Rows whose ``raw_payload``
    doesn't yield text get ``None``.

    This function lazily imports torch/transformers/peft so the
    pure-Python tests above can run without a CUDA wheel. The
    caller is responsible for catching ``Exception`` and falling
    back to FIFO — we don't want a model-load failure to deadlock
    the labeling queue.
    """
    texts: list[str | None] = [extract_row_text(row.raw_payload) for row in rows]
    if not any(t is not None for t in texts):
        return [None] * len(rows)

    import torch  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label_space),
        trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
    except Exception:
        # The saved checkpoint isn't a PEFT adapter (full-fine-tune);
        # use the base loader's output directly.
        model = base_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    out: list[float | None] = []
    with torch.inference_mode():
        for text in texts:
            if text is None:
                out.append(None)
                continue
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)
            logits = model(**tokens).logits[0].float().tolist()
            out.append(softmax_entropy(logits))
    return out


__all__ = [
    "extract_row_text",
    "softmax_entropy",
    "rank_rows_by_uncertainty",
    "latest_scoreable_classification_experiment",
    "score_classification_rows",
]
