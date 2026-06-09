"""Quality-Lift phase 4 slice 1 — Label-noise scoring (Confident-Learning-lite).

Walks the project's LABELED label_rows, runs each one through the
latest trained classifier head, and flags rows that satisfy the
dual condition:

  * ``predicted_label != given_label`` (model wants a different label)
  * ``predicted_prob >= confidence_threshold`` (model is *sure* about
    its prediction — default 0.85)
  * ``given_label_prob <= given_label_floor`` (given label is genuinely
    low-probability under the model — default 0.15). Without this the
    "model barely tipped the scale" cases sneak in as false positives.

Slice 1 honest caveat: **the model has seen these rows during
training**, so a confident disagreement means either (a) the label IS
noise (we want this) or (b) the model overfit and is wrong on a
clean row. The dual condition trims (b) substantially but slice 3's
review surface MUST show the prediction + the actual text so the
user makes the final call — we never auto-apply changes. Real K-fold
CL (phase 4b) catches the overfit-false-positive class because the
model never saw the row during scoring.

Wraps the same Epic F Phase 1 inference path phase 3 reuses — model
load once, batched per-row forward pass, returns probabilities so we
can read both predicted_prob AND given_label_prob in one pass.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment, ExperimentStatus
from app.models.label_job import LabelJob, LabelRow


DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_GIVEN_LABEL_FLOOR = 0.15
DEFAULT_TOP_K = 100
DEFAULT_SAMPLE_CAP = 2000
TEXT_PREVIEW_MAX_CHARS = 140
SUPPORTED_TASK_TYPES = frozenset({"classification"})


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def _softmax(logits: list[float]) -> list[float]:
    """Numerically-stable softmax. Subtracts the max before exp so
    over-saturating logits don't overflow. Returns a uniform
    distribution for an empty / all-equal input rather than NaN."""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def _truncate_preview(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= TEXT_PREVIEW_MAX_CHARS:
        return text
    return text[: TEXT_PREVIEW_MAX_CHARS - 1].rstrip() + "…"


async def _latest_completed_classification_experiment(
    db: AsyncSession, project_id: int,
) -> Experiment | None:
    """Latest COMPLETED experiment whose config.task_type is classification
    AND whose output_dir resolves to a path that exists on disk. Without
    a usable checkpoint there's nothing to score against."""
    rows = await db.execute(
        select(Experiment)
        .where(
            Experiment.project_id == project_id,
            Experiment.status == ExperimentStatus.COMPLETED,
        )
        .order_by(Experiment.completed_at.desc(), Experiment.id.desc())
    )
    for exp in rows.scalars().all():
        cfg = exp.config if isinstance(exp.config, dict) else {}
        task_type = str(cfg.get("task_type") or "").strip().lower()
        if task_type not in SUPPORTED_TASK_TYPES:
            continue
        raw_dir = (exp.output_dir or "").strip()
        if raw_dir and Path(raw_dir).exists():
            return exp
    return None


async def _labeled_rows_for_project(
    db: AsyncSession, project_id: int,
) -> tuple[list[LabelRow], list[str] | None]:
    """Pull labeled rows for the project AND the label space from the
    first classification label_job. Returns ([], None) when no rows
    match — caller turns that into a ``empty_labeled_pool`` skip."""
    job_rows = await db.execute(
        select(LabelJob).where(
            LabelJob.project_id == project_id,
            LabelJob.label_type == "classification",
        )
    )
    label_space: list[str] | None = None
    job_ids: list[int] = []
    for job in job_rows.scalars().all():
        job_ids.append(int(job.id))
        if label_space is None:
            schema = job.label_schema if isinstance(job.label_schema, dict) else {}
            raw = schema.get("allowed_labels")
            if isinstance(raw, list) and raw:
                label_space = [str(x) for x in raw if isinstance(x, (str, int, float))]
    if not job_ids:
        return [], None
    row_query = await db.execute(
        select(LabelRow).where(
            LabelRow.job_id.in_(job_ids),
            LabelRow.labeled_at.is_not(None),
        )
    )
    return list(row_query.scalars().all()), label_space


def _given_label_for_row(row: LabelRow) -> str | None:
    payload = row.label_payload if isinstance(row.label_payload, dict) else {}
    raw = payload.get("label")
    if raw is None:
        return None
    return str(raw)


def _build_skipped(
    *,
    base_experiment_id: int | None,
    skipped_reason: str,
    label_count_total: int = 0,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    given_label_floor: float = DEFAULT_GIVEN_LABEL_FLOOR,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The empty-snapshot shape used for every skipped path. Slice 2's
    Coach nudge + Data Studio card gate on ``top_k.length > 0`` and
    render the ``skipped_reason`` text inline, so every failure mode
    must produce a well-formed dict — never raise."""
    snap: dict[str, Any] = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "base_experiment_id": base_experiment_id,
        "label_count_total": label_count_total,
        "label_count_scored": 0,
        "suspected_count": 0,
        "confidence_threshold": confidence_threshold,
        "given_label_floor": given_label_floor,
        "top_k": [],
        "skipped_reason": skipped_reason,
    }
    if extra:
        snap.update(extra)
    return snap


# ────────────────────────────────────────────────────────────────────────
# Inference path (model load once, per-row probabilities)
# ────────────────────────────────────────────────────────────────────────


def _score_rows_with_classifier_head(
    rows: list[LabelRow],
    *,
    model_path: str,
    label_space: list[str],
) -> list[list[float] | None]:
    """Run the trained classifier head over each row's text and return
    softmax probability vectors aligned with ``label_space``. Rows
    with no extractable text get ``None`` (drop out of the top-K
    silently rather than crashing).

    Lazy imports of torch / transformers / peft so the pure-Python
    test paths don't need a CUDA wheel — the scoring service patches
    this function in unit tests rather than running real inference.
    """
    from app.services.annotation.active_learning import extract_row_text

    texts: list[str | None] = [extract_row_text(r.raw_payload) for r in rows]
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

    base = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label_space),
        trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(base, model_path)
    except Exception:
        model = base

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    out: list[list[float] | None] = []
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
            out.append(_softmax(logits))
    return out


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────


async def scan_labeled_rows_for_mislabels(
    db: AsyncSession,
    *,
    project_id: int,
    base_experiment_id: int | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    given_label_floor: float = DEFAULT_GIVEN_LABEL_FLOOR,
    top_k: int = DEFAULT_TOP_K,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """Score the project's labeled rows against the latest trained
    classifier and surface dual-condition suspected mislabels.

    Returns a result_payload dict for ``LabelNoiseScan.result_payload``.
    Always returns a dict — every failure mode (no checkpoint, no
    labeled rows, model load error, no label space, non-classification
    project) sets ``skipped_reason`` rather than raising. The runner
    wraps this in another try-block so a buggy scoring call can't
    leave the scan stuck in RUNNING.
    """
    # Resolve the checkpoint experiment. If caller pinned one (re-scan
    # of an older run), honor it; otherwise pick the latest viable.
    exp: Experiment | None = None
    if base_experiment_id is not None:
        result = await db.execute(
            select(Experiment).where(Experiment.id == base_experiment_id)
        )
        candidate = result.scalar_one_or_none()
        if candidate is not None:
            cfg = candidate.config if isinstance(candidate.config, dict) else {}
            task_type = str(cfg.get("task_type") or "").strip().lower()
            output_dir = (candidate.output_dir or "").strip()
            if task_type in SUPPORTED_TASK_TYPES and output_dir and Path(output_dir).exists():
                exp = candidate
    if exp is None:
        exp = await _latest_completed_classification_experiment(db, project_id)
    if exp is None:
        return _build_skipped(
            base_experiment_id=base_experiment_id,
            skipped_reason="no_classifier_checkpoint",
            confidence_threshold=confidence_threshold,
            given_label_floor=given_label_floor,
        )

    resolved_exp_id = int(exp.id)
    checkpoint_path = (exp.output_dir or "").strip()

    labeled_rows, label_space = await _labeled_rows_for_project(db, project_id)
    label_count_total = len(labeled_rows)
    if label_count_total == 0:
        return _build_skipped(
            base_experiment_id=resolved_exp_id,
            skipped_reason="empty_labeled_pool",
            confidence_threshold=confidence_threshold,
            given_label_floor=given_label_floor,
        )
    if not label_space:
        return _build_skipped(
            base_experiment_id=resolved_exp_id,
            skipped_reason="no_label_space_configured",
            label_count_total=label_count_total,
            confidence_threshold=confidence_threshold,
            given_label_floor=given_label_floor,
        )

    # Sample down before model load — scoring is per-row sequential
    # so the cost is linear. 2000 catches enough suspects in practice
    # to surface a useful queue without burning hours of GB10 time on
    # 50k-row datasets. Deterministic seed (defaults to experiment_id)
    # so a re-run produces the same sample.
    if label_count_total > sample_cap:
        seed = rng_seed if rng_seed is not None else resolved_exp_id
        rng = random.Random(seed)
        sampled = rng.sample(labeled_rows, sample_cap)
    else:
        sampled = list(labeled_rows)

    try:
        probs_per_row = _score_rows_with_classifier_head(
            sampled,
            model_path=checkpoint_path,
            label_space=label_space,
        )
    except Exception as exc:  # noqa: BLE001 — runner wraps this too
        return _build_skipped(
            base_experiment_id=resolved_exp_id,
            skipped_reason="scoring_failed",
            label_count_total=label_count_total,
            confidence_threshold=confidence_threshold,
            given_label_floor=given_label_floor,
            extra={
                "error": str(exc)[:512],
                "checkpoint_path": checkpoint_path,
            },
        )

    # Apply the dual condition row by row. Rows missing a `label` in
    # label_payload (corrupt / partial annotations) are skipped — we
    # can't disagree with a label that isn't there.
    label_to_idx = {lab: i for i, lab in enumerate(label_space)}
    suspected: list[dict[str, Any]] = []
    label_count_scored = 0
    for row, probs in zip(sampled, probs_per_row):
        if probs is None:
            continue
        label_count_scored += 1
        given_label = _given_label_for_row(row)
        if given_label is None:
            continue
        given_idx = label_to_idx.get(given_label)
        if given_idx is None:
            # Label not in current label_space — schema drift. Surface
            # via the row's data but don't treat as a mislabel signal
            # (the user changed their label inventory; reconciling
            # those rows is a different workflow).
            continue
        predicted_idx = max(range(len(probs)), key=lambda i: probs[i])
        predicted_prob = float(probs[predicted_idx])
        given_label_prob = float(probs[given_idx])
        if (
            predicted_idx != given_idx
            and predicted_prob >= confidence_threshold
            and given_label_prob <= given_label_floor
        ):
            from app.services.annotation.active_learning import extract_row_text

            text = extract_row_text(row.raw_payload)
            suspected.append({
                "label_row_id": int(row.id),
                "label_job_id": int(row.job_id),
                "given_label": given_label,
                "predicted_label": str(label_space[predicted_idx]),
                "predicted_prob": round(predicted_prob, 6),
                "given_label_prob": round(given_label_prob, 6),
                "mislabel_score": round(predicted_prob - given_label_prob, 6),
                "text_preview": _truncate_preview(text),
            })

    suspected.sort(key=lambda e: e["mislabel_score"], reverse=True)
    top_entries = suspected[: max(1, int(top_k))]

    return {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "base_experiment_id": resolved_exp_id,
        "label_count_total": label_count_total,
        "label_count_scored": label_count_scored,
        "suspected_count": len(suspected),
        "confidence_threshold": confidence_threshold,
        "given_label_floor": given_label_floor,
        "top_k": top_entries,
        "skipped_reason": None,
        "checkpoint_path": checkpoint_path,
        "label_space": label_space,
    }
