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


def top_two_margin(scores: Sequence[float]) -> float:
    """Uncertainty signal for span / token-tagging tasks (Phase 2).

    Returns ``top1 - top2`` of the candidate scores: small margin =
    high uncertainty, so the *negated* value is what we pass to the
    ranker (which sorts descending by uncertainty). One-candidate
    sequences return ``inf`` — fully certain by construction — and
    empty sequences return ``inf`` for the same reason. The caller
    negates: ``-top_two_margin(scores)``, so empty becomes ``-inf``
    and trails real scores in the ranker.
    """
    if len(scores) < 2:
        return float("inf")
    sorted_scores = sorted(scores, reverse=True)
    return float(sorted_scores[0] - sorted_scores[1])


def vote_disagreement(votes: Sequence[Any]) -> float:
    """Ensemble-disagreement uncertainty signal for preference-pair
    jobs (Phase 2). Higher = more disagreement = more uncertain.

    Each entry in ``votes`` is one model's pick (e.g., the preferred
    completion identifier ``"A"`` or ``"B"``; could also be class
    ids for any discrete prediction). The score is ``1 - majority
    fraction`` — 0.0 when every vote agrees, 0.5 when a binary
    vote splits 50/50, capped at ``(N-1)/N`` for full disagreement
    in larger ensembles. Empty input returns 0.0 (no signal).
    """
    if not votes:
        return 0.0
    counts: dict[Any, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    majority = max(counts.values())
    return 1.0 - (majority / len(votes))


def cohens_kappa(
    rater_a: Sequence[Any],
    rater_b: Sequence[Any],
) -> float | None:
    """Cohen's κ for two raters labeling the same N items.

    ``rater_a[i]`` and ``rater_b[i]`` must be the two raters' labels
    on the same item. Returns ``None`` when the inputs have
    different lengths, or are empty, or when both raters used a
    single label (expected agreement is degenerate; κ is undefined).

    Interpretation: κ=1.0 perfect agreement, κ=0.0 chance level,
    κ<0.0 worse than chance (raters systematically disagree).

    A pure-Python implementation so the math is testable without
    pulling in scikit-learn.
    """
    if not rater_a or len(rater_a) != len(rater_b):
        return None
    n = len(rater_a)
    # Observed agreement
    agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    p_o = agree / n

    # Expected agreement under independence
    labels = set(rater_a) | set(rater_b)
    if len(labels) <= 1:
        # Both raters used a single label — κ is undefined; report
        # ``None`` rather than silently returning 1.0 or 0.0.
        return None
    counts_a: dict[Any, int] = {}
    counts_b: dict[Any, int] = {}
    for a, b in zip(rater_a, rater_b):
        counts_a[a] = counts_a.get(a, 0) + 1
        counts_b[b] = counts_b.get(b, 0) + 1
    p_e = 0.0
    for label in labels:
        p_e += (counts_a.get(label, 0) / n) * (counts_b.get(label, 0) / n)
    if abs(1.0 - p_e) < 1e-12:
        # Perfect chance agreement — denominator collapses to 0,
        # κ is undefined. Treat as ``None`` so the caller can
        # report "agreement degenerate" rather than "κ = inf".
        return None
    return (p_o - p_e) / (1.0 - p_e)


def _spans_to_set(spans: Sequence[Any]) -> set[tuple[Any, Any, Any]]:
    """Normalise a list of span annotations to a set of
    ``(start, end, type)`` tuples. Annotations come in two common
    shapes — dict with ``start``/``end``/``type`` keys, or list/
    tuple of three values — and we accept both. Anything that
    doesn't fit is silently dropped (the F1 caller treats
    unparseable spans as "missing")."""
    out: set[tuple[Any, Any, Any]] = set()
    for span in spans:
        if isinstance(span, dict):
            start = span.get("start")
            end = span.get("end")
            kind = span.get("type") or span.get("label") or span.get("kind")
            if start is None or end is None:
                continue
            out.add((start, end, kind))
        elif isinstance(span, (list, tuple)) and len(span) >= 3:
            out.add((span[0], span[1], span[2]))
    return out


def span_f1(
    spans_a: Sequence[Any],
    spans_b: Sequence[Any],
) -> float | None:
    """Span-F1 treating one rater's spans as ground truth and the
    other's as predictions. Symmetric (F1 is precision and recall
    averaged), so the assignment doesn't matter. Returns ``None``
    when both sides are empty (undefined; the row has no spans
    either way, so there's no signal).
    """
    set_a = _spans_to_set(spans_a)
    set_b = _spans_to_set(spans_b)
    if not set_a and not set_b:
        return None
    tp = len(set_a & set_b)
    fp = len(set_b - set_a)
    fn = len(set_a - set_b)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def preference_agreement(
    rater_a: Sequence[Any],
    rater_b: Sequence[Any],
) -> float | None:
    """Simple agreement rate for preference-pair labels: fraction of
    items where both raters picked the same completion. Returns
    ``None`` for empty / length-mismatched inputs."""
    if not rater_a or len(rater_a) != len(rater_b):
        return None
    matches = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    return matches / len(rater_a)


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


async def _latest_completed_experiment_with_task_type(
    db: AsyncSession,
    *,
    project_id: int,
    task_types: frozenset[str],
    training_modes: frozenset[str] | None = None,
) -> Experiment | None:
    """Return the most-recent completed experiment whose config matches
    one of ``task_types`` (and optionally ``training_modes``). Returns
    ``None`` when nothing qualifies — caller is expected to fall back
    to FIFO rather than blocking the labeling queue.
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
        if task_type not in task_types:
            continue
        if training_modes is not None:
            mode = str(cfg.get("training_mode") or "").strip().lower()
            if mode not in training_modes:
                continue
        return experiment
    return None


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
    return await _latest_completed_experiment_with_task_type(
        db,
        project_id=project_id,
        task_types=frozenset({"classification"}),
    )


# Task types we'll accept as "scoreable" for span / NER-style
# uncertainty ranking. The platform currently trains span models
# under ``token_classification`` (multi-label per-token tag) or
# the generic ``sft``/``instruction`` path with span outputs. Phase
# 2 ships the dispatch layer; the actual scoreable inference still
# needs span-aware decode, which the scorer below handles lazily.
_SPAN_TASK_TYPES: frozenset[str] = frozenset(
    {"token_classification", "ner", "span", "structured", "sft", "instruction"}
)

# Preference-pair experiments are alignment runs. ``training_mode ==
# "dpo"`` is the canonical case; we also accept ``orpo`` / ``kto`` if
# the platform grows them later (string comparison is forward-
# compatible).
_PREFERENCE_TRAINING_MODES: frozenset[str] = frozenset(
    {"dpo", "orpo", "kto", "ipo"}
)


async def latest_scoreable_span_experiment(
    db: AsyncSession, *, project_id: int
) -> Experiment | None:
    """Most-recent completed experiment that *could* produce span-
    style scores. Returns ``None`` when none qualifies; the active
    strategy then falls back to FIFO."""
    return await _latest_completed_experiment_with_task_type(
        db,
        project_id=project_id,
        task_types=_SPAN_TASK_TYPES,
    )


async def latest_scoreable_preference_experiment(
    db: AsyncSession, *, project_id: int
) -> Experiment | None:
    """Most-recent completed alignment experiment that *could*
    produce preference-pair scores. Returns ``None`` when none
    qualifies; the active strategy then falls back to FIFO."""
    return await _latest_completed_experiment_with_task_type(
        db,
        project_id=project_id,
        # task_type may be left default ("sft") on alignment runs;
        # we gate on training_mode instead.
        task_types=frozenset(
            {"sft", "instruction", "alignment", "dpo", "preference"}
        ),
        training_modes=_PREFERENCE_TRAINING_MODES,
    )


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


def score_span_rows(
    rows: list[LabelRow],
    *,
    model_path: str,
    span_types: list[str],
) -> list[float | None]:
    """Score span/NER rows by **negative top-2 margin** of the
    token-classification head's per-token logits, averaged over
    each row's tokens. Smaller native top-2 margin → less certain
    tagging → larger negated score → ranks first in the
    descending-by-score ranker.

    Phase 2 ships this as a *graceful stub*: a real span-tagging
    model exposes per-token logits we can decode. If the saved
    checkpoint isn't actually a span tagger, the lazy load below
    raises and the caller catches it, returning ``[None] * len(rows)``.
    That keeps the labeler unblocked while we add proper span
    inference in a follow-up.
    """
    texts: list[str | None] = [extract_row_text(row.raw_payload) for row in rows]
    if not any(t is not None for t in texts):
        return [None] * len(rows)

    import torch  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=max(len(span_types), 2),
        trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
    except Exception:
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
            # logits shape: (1, seq_len, num_labels). Average the
            # per-token top-2 margin so rows whose every token is
            # confident score low (high margin → low uncertainty).
            logits = model(**tokens).logits[0].float().tolist()
            margins = [top_two_margin(row) for row in logits]
            if not margins:
                out.append(None)
                continue
            mean_margin = sum(margins) / len(margins)
            # Negate so smaller native margin → larger uncertainty.
            out.append(-mean_margin)
    return out


def score_preference_pair_rows(
    rows: list[LabelRow],
    *,
    model_path: str,
) -> list[float | None]:
    """Score preference-pair rows by **bootstrap disagreement** of
    the alignment model on ``(prompt, completion_a, completion_b)``
    pairs.

    Phase 2 *stub*: a true ensemble would require multiple
    independently-trained alignment models, which the platform
    doesn't materialize yet. The graceful fallback below catches
    any inference failure and returns ``None`` per row so the
    ranker treats them as FIFO. The wiring is in place; the actual
    multi-model ensemble is a follow-up once the platform exposes
    per-experiment alignment variants.
    """
    needs_text = []
    for row in rows:
        payload = row.raw_payload or {}
        prompt = payload.get("prompt") or ""
        comp_a = payload.get("completion_a") or ""
        comp_b = payload.get("completion_b") or ""
        if not (isinstance(prompt, str) and prompt.strip()) or not (
            isinstance(comp_a, str) and comp_a.strip()
        ) or not (isinstance(comp_b, str) and comp_b.strip()):
            needs_text.append(None)
        else:
            needs_text.append((prompt, comp_a, comp_b))
    if not any(t is not None for t in needs_text):
        return [None] * len(rows)

    # Without a real ensemble, return ``None`` per scoreable row so
    # the caller falls back to FIFO. Keeping the lazy import below
    # behind a deliberate ``raise`` documents the gap clearly and
    # tests don't need to mock heavyweight transformers loaders.
    raise NotImplementedError(
        "preference-pair scoring requires an alignment-model ensemble; "
        "wired to fall back to FIFO until Phase 3 lands the ensemble."
    )


__all__ = [
    "extract_row_text",
    "softmax_entropy",
    "top_two_margin",
    "vote_disagreement",
    "cohens_kappa",
    "span_f1",
    "preference_agreement",
    "rank_rows_by_uncertainty",
    "latest_scoreable_classification_experiment",
    "latest_scoreable_span_experiment",
    "latest_scoreable_preference_experiment",
    "score_classification_rows",
    "score_span_rows",
    "score_preference_pair_rows",
]
