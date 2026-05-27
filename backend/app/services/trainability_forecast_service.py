"""Trainability forecast service — USER-SUCCESS Epic 1.

Predicts whether a project's training run is likely to clear the default
Auto-Gates *before* the user clicks Train. The forecast is computed from
a recipe-aware set of signals against the project's prepared data +
recipe + base model.

Recipe-agnostic signals (always run):

1. Row count vs the recipe's minimum (`row_count_below_minimum`)
2. Gold-set diversity via token-set Jaccard (`goldset_diversity_low`)
3. Overall gate-pass probability via `estimate_gate_pass_prob()`

Per-recipe signals (dispatched by ``task_profile``):

* classification → ``class_imbalance``, ``per_class_minimum_unmet``,
  ``label_vocab_fragmented``, ``single_class_dominance``
* structured_extraction → ``format_inconsistency``,
  ``entity_type_coverage_thin``, ``span_offset_invalid``,
  ``negative_examples_missing``
* summarization → ``summary_doc_ratio_outliers``
* instruction_sft / qa-sft / generic-sft → (recipe-agnostic only)

The diversity check is the expensive bit, so the full result is cached on
``Project.training_forecast_cache`` keyed by
(dataset_version_signature, recipe_id, base_model_name). Cache invalidates
when any of those change.

The heuristic is deliberately interpretable — hand-tuned coefficients
calibrated against the 8 templates' shipped gold sets, not a learned
model. A learned-calibration v2 can plug into the same surface later.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.project import Project
from app.models.training_forecast_snapshot import TrainingForecastSnapshot
from app.services.dataset_service import _load_records_from_file
from app.services.recipe_service import get_recipe


# ─────────────────────────────────────────────────────────────────────
# Module-level constants — hand-tuned, interpretable knobs.
# ─────────────────────────────────────────────────────────────────────

# Per-task-profile "difficulty" on a 0-1 scale. 0 = easiest (constrained
# output space, fewer ways to be wrong); 1 = hardest. Classification is
# easiest because the label set is fixed; structured extraction is the
# hardest because both *which* spans and *exact* offsets have to match.
TASK_DIFFICULTY: dict[str, float] = {
    "classification": 0.30,
    "instruction_sft": 0.45,
    "summarization": 0.55,
    "structured_extraction": 0.70,
}

# Known parameter counts (millions) for the default base models the
# platform ships. Used in the capacity term of the heuristic. Fallback
# to 135M when the name isn't recognized — that's the default base
# model, and the conservative bet.
KNOWN_BASE_MODEL_PARAMS_M: dict[str, int] = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": 135,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 360,
    "Qwen/Qwen2.5-0.5B-Instruct": 500,
    "Qwen/Qwen2.5-1.5B-Instruct": 1500,
    "Qwen/Qwen2.5-3B-Instruct": 3000,
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": 1500,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1100,
    "microsoft/phi-2": 2700,
}
DEFAULT_BASE_MODEL_PARAMS_M = 135

# Diversity scoring: for projects with more than this many gold rows we
# sample pairs rather than do the full O(N^2) Jaccard sweep. 50 pairs
# is enough signal to detect "all rows look the same" pathologies.
MAX_DIVERSITY_PAIRS = 50

# Diversity-low threshold: mean pairwise Jaccard above this is a warn
# signal. Empirically calibrated against the 8 templates' gold sets
# (which all land in the 0.05-0.20 range).
DIVERSITY_WARN_THRESHOLD = 0.40

# Class-imbalance thresholds (Shannon entropy).
CLASS_ENTROPY_WARN = 1.0
CLASS_ENTROPY_BLOCK = 0.5

# Format-consistency check: max fraction of gold rows that can fail the
# schema check before we promote the signal from warn to block.
FORMAT_CONSISTENCY_BLOCK_FRAC = 0.20

# Per-class minimum (classification). Below this count for any class
# the model can't learn that class — F1 for the class collapses to
# guessing. 5 is the empirical floor where one or two correct
# generalisations start to land.
PER_CLASS_MINIMUM = 5
PER_CLASS_BLOCK = 2  # < this → block (basically no learning signal).

# Single-class dominance: any class above this fraction overwhelms the
# others and the model defaults to that class. 0.80 catches the 80/20
# binary skew and the 90/5/5 three-way skew.
SINGLE_CLASS_DOMINANCE_FRAC = 0.80

# Label-vocab fragmentation: case-insensitive near-duplicate detection.
# Two labels collide if they normalise to the same lowercase + stripped
# punctuation token ("positive" / "Positive" / "POSITIVE" / "positive ").
# Any collision is a warn — same drift the add-form already guards
# against, but the forecast is otherwise silent on it.

# Entity-type coverage: span-extraction gold sets with fewer than this
# many distinct types across the gold rows almost never generalise
# beyond the types present. 3 is the empirical floor (covers single-
# entity tasks like email-detection without false-positiving).
ENTITY_TYPE_COVERAGE_MIN = 3

# Span offset validity: fraction of span rows where ``text[start:end]``
# does not match the span's recorded ``text`` field. > 0 is always at
# least a warn (a single bad offset row poisons eval); > 10% → block.
SPAN_OFFSET_INVALID_BLOCK_FRAC = 0.10

# Summary-to-document length ratio. The summarization task assumes
# summary << doc; rows where summary is more than this fraction of
# the doc are usually mislabeled (the "summary" is just a paraphrase
# of similar length, or the wrong column was loaded).
SUMMARY_DOC_RATIO_WARN = 0.70

# Snapshot retention window. The trainability-forecast history table
# is pruned on every insert (cheap, no separate cron) so the sparkline
# always reflects the last 60 days. Anything older is opaque dead
# weight — the user's iteration cycle is measured in days, not months.
SNAPSHOT_RETENTION_DAYS = 60


# ─────────────────────────────────────────────────────────────────────
# Typed payloads.
# ─────────────────────────────────────────────────────────────────────


class ForecastSignal(TypedDict):
    id: str
    severity: Literal["ok", "warn", "block"]
    headline: str
    detail: str
    suggested_action: dict | None


class ForecastResult(TypedDict):
    overall: Literal["likely_pass", "borderline", "likely_fail"]
    confidence_pct: int
    signals: list[ForecastSignal]
    computed_at: str
    cache_key: str
    cache_hit: bool


# ─────────────────────────────────────────────────────────────────────
# Heuristic — the inverse-of-an-ML-model interpretable formula.
# ─────────────────────────────────────────────────────────────────────


def estimate_gate_pass_prob(
    row_count: int,
    recipe_difficulty: float,
    base_model_params_m: int,
    class_entropy: float | None,
    diversity_score: float,
) -> float:
    """Returns a 0.05-0.95 probability that the project will clear the
    default Auto-Gates on a first training run.

    Formula:
        data_floor = min(1.0, row_count / 200)
        capacity   = min(1.0, log(params_m / 50) / log(8))
        quality    = (1 - difficulty) * diversity * class_term
        raw        = 0.4 * data_floor + 0.25 * capacity + 0.35 * quality

    Clamped to [0.05, 0.95] so the surface never claims certainty.
    """
    data_floor = min(1.0, max(0.0, row_count) / 200.0)

    if base_model_params_m <= 50:
        capacity = 0.0
    else:
        capacity = min(1.0, math.log(base_model_params_m / 50.0) / math.log(8.0))

    quality = (1.0 - recipe_difficulty) * max(0.0, min(1.0, diversity_score))
    if class_entropy is not None:
        quality *= min(1.0, max(0.0, class_entropy) / 1.5)

    raw = 0.40 * data_floor + 0.25 * capacity + 0.35 * quality
    return max(0.05, min(0.95, raw))


# ─────────────────────────────────────────────────────────────────────
# Helpers — tokenization + Jaccard for the cheap diversity heuristic.
# We deliberately do *not* pull in sentence-transformers; the
# token-Jaccard signal is enough to detect "all rows look the same"
# pathologies, which is the failure mode the diversity check targets.
# ─────────────────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase whitespace+punct tokenizer. Returns a frozenset for
    cheap set ops downstream."""
    return frozenset(t.lower() for t in _TOKEN_RE.findall(text or ""))


def _row_to_text(row: dict[str, Any]) -> str:
    """Coerces a row dict to a single text blob for tokenization.
    Walks common gold-row shapes (input/expected, question/answer,
    text/label) and joins string values."""
    parts: list[str] = []
    for key in ("input", "expected", "question", "answer", "text", "prompt", "response", "output"):
        value = row.get(key)
        if isinstance(value, dict):
            for sub_value in value.values():
                if isinstance(sub_value, str):
                    parts.append(sub_value)
        elif isinstance(value, str):
            parts.append(value)
    if not parts:
        for value in row.values():
            if isinstance(value, str):
                parts.append(value)
    return " ".join(parts)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _mean_pairwise_jaccard(token_sets: list[frozenset[str]], *, seed: int = 0) -> float:
    """Mean pairwise Jaccard similarity. Bounded compute via sampling
    when there are more than enough pairs to need it."""
    n = len(token_sets)
    if n < 2:
        return 0.0
    full_pairs = n * (n - 1) // 2
    rng = random.Random(seed)
    if full_pairs <= MAX_DIVERSITY_PAIRS:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = set()
        while len(pairs) < MAX_DIVERSITY_PAIRS:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
        pairs = list(pairs)
    total = sum(_jaccard(token_sets[i], token_sets[j]) for i, j in pairs)
    return total / len(pairs)


# ─────────────────────────────────────────────────────────────────────
# Recipe-aware row normalisation.
#
# Project templates seed gold JSONL through a materializer that
# flattens every recipe shape into legacy ``{question, answer}`` keys
# (see ``demo_project_service._canonical_prepared_row`` and the gold-
# bundle conversion in the same file). For structured tasks ``answer``
# is JSON-encoded (e.g. ``{"entities": [...]}``, ``{"summary": "..."}``).
# Newer rows (LLM-gen + manual add via the per-recipe form) already
# carry recipe-shaped keys.
#
# The normalisers below mirror ``normalizeEntryForRecipe`` in
# ``frontend/src/components/data/GoldSetPanel.tsx`` — they extract a
# canonical view of each row regardless of whether it was written in
# legacy or recipe-shape form. Without this, the new per-recipe
# signals silently miss every template-instantiated project (which is
# the most common case for new users).
# ─────────────────────────────────────────────────────────────────────


def _try_parse_json_dict(value: Any) -> dict[str, Any] | None:
    """Return value parsed as a JSON dict, or None when not parseable."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_classification_labels(gold_rows: list[dict[str, Any]]) -> list[str]:
    """Walks gold rows to extract classification labels. Supports:

    - Top-level ``label`` field (synthetic / unit-test rows).
    - Nested ``expected.label`` (raw template JSONL before normalization).
    - Raw template ``expected`` string (when expected is the label itself).
    - Normalized ``answer`` field (legacy materialised gold-row shape).
    """
    labels: list[str] = []
    for row in gold_rows:
        label = _extract_classification_label(row)
        if label is not None:
            labels.append(label)
    return labels


def _extract_classification_label(row: dict[str, Any]) -> str | None:
    """Single-row classification label extractor. Returns None when no
    label is recoverable from the row."""
    if isinstance(row.get("label"), str):
        return row["label"]
    expected = row.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("label"), str):
        return expected["label"]
    if isinstance(expected, str):
        return expected
    # Legacy materialised row: classification label flattened into
    # ``answer`` as a plain string. Length-guard to avoid mistaking
    # free-text answers (qa-sft) for labels.
    answer = row.get("answer")
    if isinstance(answer, str) and 0 < len(answer) <= 64 and "\n" not in answer:
        return answer
    return None


def _extract_span_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the span-extraction payload from any of three shapes:

    - Raw template JSONL: ``row['expected'] = {'spans': [...]}`` or
      ``{'entities': [...]}``.
    - Normalized loader output: ``row['answer']`` is a JSON string of
      a dict containing ``spans`` or ``entities`` (the dataset_service
      coerces template rows to a QA-pair shape).
    - Pre-extracted: ``row['spans']`` or ``row['entities']`` at the
      top level.

    Returns the dict with the spans list, or None when no recognizable
    payload exists.
    """
    expected = row.get("expected")
    if isinstance(expected, dict) and (
        expected.get("spans") is not None or expected.get("entities") is not None
    ):
        return expected
    parsed_answer = _try_parse_json_dict(row.get("answer"))
    if parsed_answer is not None and (
        parsed_answer.get("spans") is not None or parsed_answer.get("entities") is not None
    ):
        return parsed_answer
    if row.get("spans") is not None or row.get("entities") is not None:
        return row
    return None


def _extract_span_source_text(row: dict[str, Any]) -> str | None:
    """Find the source text a span's start/end offsets index into.

    Walks the three shapes the platform emits:
    - Raw template JSONL: ``row['input'] = {'text': "..."}`` (or
      ``ticket``, ``advisory``, etc. — but span-extraction recipes use
      ``text``).
    - Top-level: ``row['text']``.
    - Legacy materialised: ``row['question']`` (the materializer
      flattens ``input.text`` into ``question``).
    """
    inp = row.get("input")
    if isinstance(inp, dict):
        text = inp.get("text")
        if isinstance(text, str):
            return text
    if isinstance(row.get("text"), str):
        return row["text"]
    if isinstance(row.get("question"), str):
        return row["question"]
    return None


def _extract_span_list(payload: dict[str, Any] | None) -> list[Any]:
    """Return the spans list from a span payload, or [] when missing.
    Accepts ``spans`` or ``entities`` interchangeably."""
    if payload is None:
        return []
    spans = payload.get("spans")
    if spans is None:
        spans = payload.get("entities")
    if isinstance(spans, list):
        return spans
    return []


def _extract_summary_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    """Pull (document, summary) from any of the shapes summarization
    rows arrive in. Returns None when either side can't be recovered.

    Shapes:
    - Raw template JSONL: ``input.advisory`` / ``input.document`` /
      ``input.text`` + ``expected.summary``.
    - Top-level: ``row['document']`` + ``row['summary']``.
    - Legacy materialised: ``row['question']`` (doc) + ``row['answer']``
      (summary text OR JSON-encoded ``{"summary": "..."}``).
    """
    # Document side ------------------------------------------------------
    doc: str | None = None
    inp = row.get("input")
    if isinstance(inp, dict):
        for key in ("document", "advisory", "article", "transcript", "text", "body"):
            val = inp.get(key)
            if isinstance(val, str):
                doc = val
                break
        if doc is None:
            # Last-ditch: first string value in the input dict.
            for val in inp.values():
                if isinstance(val, str):
                    doc = val
                    break
    if doc is None and isinstance(row.get("document"), str):
        doc = row["document"]
    if doc is None and isinstance(row.get("question"), str):
        doc = row["question"]

    # Summary side -------------------------------------------------------
    summary: str | None = None
    expected = row.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("summary"), str):
        summary = expected["summary"]
    elif isinstance(row.get("summary"), str):
        summary = row["summary"]
    else:
        # Legacy materialised: summary in ``answer`` (string or JSON dict).
        answer = row.get("answer")
        if isinstance(answer, str):
            parsed = _try_parse_json_dict(answer)
            if parsed is not None and isinstance(parsed.get("summary"), str):
                summary = parsed["summary"]
            else:
                summary = answer

    if not doc or not summary:
        return None
    return doc, summary


# ─────────────────────────────────────────────────────────────────────
# Cache key.
# ─────────────────────────────────────────────────────────────────────


def _compute_cache_key(
    *,
    dataset_signature: str,
    recipe_id: str,
    base_model_name: str,
) -> str:
    """Stable hash of the cache-key inputs. Truncated SHA-256 — collisions
    don't matter for cache correctness, only for hit rate."""
    raw = f"{dataset_signature}|{recipe_id}|{base_model_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def _load_gold_rows(db: AsyncSession, project_id: int) -> list[dict[str, Any]]:
    """Reads gold_dev + gold_test rows directly from the JSONL files on
    disk. Bypasses combine_datasets() because we want the raw rows
    without any adapter transformations."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_([DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]),
        )
    )
    datasets = list(result.scalars())
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if not dataset.file_path:
            continue
        path = Path(dataset.file_path)
        if not path.exists():
            continue
        rows.extend(_load_records_from_file(path))
    return rows


async def _estimate_train_row_count(db: AsyncSession, project_id: int) -> int:
    """Best-estimate of the *labeled-corpus size* the project has
    available for training.

    Semantically this is "how much data could the trainer see if you
    pulled every labeled row in the project," not "how many rows are
    currently in the TRAIN split." Gold rows count because Theme 8's
    active-learning promote flow moves them into training; raw +
    cleaned + synthetic count because they're the source rows that
    dataset_prep splits into train/val/test.

    We dedupe TRAIN against (cleaned + synthetic) since TRAIN is a
    subset of those — counting both would double-count. The
    practical formula:

        labeled_corpus = max(TRAIN, CLEANED + SYNTHETIC, RAW)
                        + GOLD_DEV + GOLD_TEST

    The first term picks the largest representation of the source
    data (TRAIN if prep ran, CLEANED/SYNTHETIC if cleaning ran, RAW
    if neither). The gold terms always add — gold is a separate
    labeled corpus from the source rows.
    """
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id)
    )
    by_type: dict[DatasetType, int] = {}
    for ds in result.scalars():
        by_type[ds.dataset_type] = by_type.get(ds.dataset_type, 0) + (ds.record_count or 0)

    source_corpus = max(
        by_type.get(DatasetType.TRAIN, 0),
        by_type.get(DatasetType.CLEANED, 0) + by_type.get(DatasetType.SYNTHETIC, 0),
        by_type.get(DatasetType.RAW, 0),
    )
    gold_corpus = by_type.get(DatasetType.GOLD_DEV, 0) + by_type.get(DatasetType.GOLD_TEST, 0)
    return source_corpus + gold_corpus


async def _build_dataset_signature(db: AsyncSession, project_id: int) -> str:
    """Stable signature over the project's gold + raw + cleaned datasets.
    Changes whenever any contributing dataset is updated (record_count or
    updated_at), which is the right invalidation trigger."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_([
                DatasetType.RAW,
                DatasetType.CLEANED,
                DatasetType.GOLD_DEV,
                DatasetType.GOLD_TEST,
                DatasetType.SYNTHETIC,
            ]),
        )
    )
    parts = []
    for dataset in sorted(result.scalars(), key=lambda d: d.id):
        ts = dataset.updated_at.isoformat() if dataset.updated_at else ""
        parts.append(f"{dataset.id}:{dataset.dataset_type.value}:{dataset.record_count}:{ts}")
    return "|".join(parts) or "empty"


# ─────────────────────────────────────────────────────────────────────
# Recipe-agnostic signals.
# ─────────────────────────────────────────────────────────────────────


def _signal_row_count(
    train_row_count: int,
    minimum_rows: int,
) -> ForecastSignal:
    """Block when below recipe minimum, warn when below 1.5×, ok above."""
    if minimum_rows <= 0:
        return {
            "id": "row_count_below_minimum",
            "severity": "ok",
            "headline": f"{train_row_count} training rows",
            "detail": "Recipe doesn't declare a minimum row count.",
            "suggested_action": None,
        }
    if train_row_count < minimum_rows:
        deficit = minimum_rows - train_row_count
        return {
            "id": "row_count_below_minimum",
            "severity": "block",
            "headline": (
                f"Only {train_row_count} training rows — recipe recommends "
                f"at least {minimum_rows}."
            ),
            "detail": (
                f"You're {deficit} rows short of the recipe's recommended "
                f"minimum. Training will likely fail its eval gates."
            ),
            "suggested_action": {
                "kind": "synth_augment",
                "params": {"target_rows": minimum_rows},
            },
        }
    if train_row_count < int(minimum_rows * 1.5):
        return {
            "id": "row_count_below_minimum",
            "severity": "warn",
            "headline": (
                f"{train_row_count} training rows — recipe recommends "
                f"{minimum_rows}+ for reliable results."
            ),
            "detail": (
                f"You're above the minimum but below the 1.5× comfort "
                f"zone. Borderline configurations sometimes pass and "
                f"sometimes don't."
            ),
            "suggested_action": {
                "kind": "synth_augment",
                "params": {"target_rows": int(minimum_rows * 1.5)},
            },
        }
    return {
        "id": "row_count_below_minimum",
        "severity": "ok",
        "headline": f"{train_row_count} training rows — above recipe minimum.",
        "detail": f"Recipe minimum is {minimum_rows}; you're comfortably above.",
        "suggested_action": None,
    }


def _signal_goldset_diversity(gold_rows: list[dict[str, Any]]) -> tuple[ForecastSignal, float]:
    """Returns (signal, diversity_score) where diversity_score is the
    0-1 inverse of mean pairwise Jaccard (1.0 = maximally diverse)."""
    token_sets = [_tokenize(_row_to_text(row)) for row in gold_rows]
    token_sets = [t for t in token_sets if t]

    if len(token_sets) < 2:
        return (
            {
                "id": "goldset_diversity_low",
                "severity": "ok",
                "headline": "Not enough rows to score diversity.",
                "detail": "Need at least 2 gold rows.",
                "suggested_action": None,
            },
            1.0,
        )

    mean_jaccard = _mean_pairwise_jaccard(token_sets)
    diversity_score = max(0.0, 1.0 - mean_jaccard)

    if mean_jaccard > DIVERSITY_WARN_THRESHOLD:
        return (
            {
                "id": "goldset_diversity_low",
                "severity": "warn",
                "headline": (
                    f"Gold-set rows look similar to each other "
                    f"(mean overlap {mean_jaccard:.2f})."
                ),
                "detail": (
                    f"High inter-row similarity means the model can't "
                    f"learn what varies across examples. Consider adding "
                    f"more diverse examples or running synth diversification."
                ),
                "suggested_action": {
                    "kind": "synth_diversify",
                    "params": {"target_rows": 50},
                },
            },
            diversity_score,
        )

    return (
        {
            "id": "goldset_diversity_low",
            "severity": "ok",
            "headline": f"Gold-set diversity looks healthy (overlap {mean_jaccard:.2f}).",
            "detail": "Rows vary enough that the model has signal to learn from.",
            "suggested_action": None,
        },
        diversity_score,
    )


def _signal_gate_pass_probability(
    *,
    train_row_count: int,
    task_profile: str,
    base_model_params_m: int,
    class_entropy: float | None,
    diversity_score: float,
) -> tuple[ForecastSignal, int]:
    """Returns (signal, confidence_pct 0-100)."""
    difficulty = TASK_DIFFICULTY.get(task_profile, 0.50)
    prob = estimate_gate_pass_prob(
        row_count=train_row_count,
        recipe_difficulty=difficulty,
        base_model_params_m=base_model_params_m,
        class_entropy=class_entropy,
        diversity_score=diversity_score,
    )
    pct = int(round(prob * 100))

    if prob >= 0.65:
        severity: Literal["ok", "warn", "block"] = "ok"
        headline = f"Predicted gate-pass probability: ~{pct}%"
        detail = "Most signals look healthy. Expect a usable first training run."
        action: dict | None = None
    elif prob >= 0.40:
        severity = "warn"
        headline = f"Predicted gate-pass probability: ~{pct}%"
        detail = (
            "Borderline configuration. The model may pass or fail; "
            "addressing the warnings above improves your odds."
        )
        action = None
    else:
        severity = "warn"
        headline = f"Predicted gate-pass probability: ~{pct}%"
        detail = (
            "Multiple signals suggest the first training run will not "
            "clear gates. Address the issues above before training, "
            "or expect to iterate."
        )
        action = {
            "kind": "synth_augment",
            "params": {"target_rows": max(200, train_row_count * 2)},
        }

    return (
        {
            "id": "gate_pass_probability",
            "severity": severity,
            "headline": headline,
            "detail": detail,
            "suggested_action": action,
        },
        pct,
    )


# ─────────────────────────────────────────────────────────────────────
# Classification signals.
# ─────────────────────────────────────────────────────────────────────


def _label_counts(gold_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in _extract_classification_labels(gold_rows):
        counts[label] = counts.get(label, 0) + 1
    return counts


def _shannon_entropy(counts: Iterable[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def _signal_class_imbalance(
    gold_rows: list[dict[str, Any]],
    task_profile: str,
) -> ForecastSignal | None:
    """Classification-only. Returns None if the task isn't classification
    or there's no usable label column."""
    if task_profile != "classification":
        return None

    counts = _label_counts(gold_rows)
    if not counts:
        return None

    total = sum(counts.values())
    entropy = _shannon_entropy(counts.values())

    # Under-represented = any class below 15% of the total when the
    # ideal balance is 1/n_classes. 15% catches the 90/10 binary skew
    # and any 5-way split where a class is below 1/3 of its share.
    under_classes = [k for k, c in counts.items() if c / total < 0.15]

    if entropy < CLASS_ENTROPY_BLOCK:
        severity: Literal["ok", "warn", "block"] = "block"
        headline = f"Severe class imbalance (entropy {entropy:.2f})."
    elif entropy < CLASS_ENTROPY_WARN:
        severity = "warn"
        headline = f"Class distribution is skewed (entropy {entropy:.2f})."
    else:
        return {
            "id": "class_imbalance",
            "severity": "ok",
            "headline": f"Class distribution looks healthy (entropy {entropy:.2f}).",
            "detail": f"{len(counts)} classes, distribution evenly spread.",
            "suggested_action": None,
        }

    return {
        "id": "class_imbalance",
        "severity": severity,
        "headline": headline,
        "detail": (
            f"{len(counts)} classes, "
            f"{', '.join(f'{k}={c}' for k, c in sorted(counts.items()))}. "
            f"Models trained on imbalanced data over-predict the majority "
            f"class."
        ),
        "suggested_action": {
            "kind": "synth_balance",
            "params": {"underrepresented_classes": under_classes},
        },
    }


def _signal_per_class_minimum(counts: dict[str, int]) -> ForecastSignal | None:
    """Flag classes with fewer than ``PER_CLASS_MINIMUM`` examples. The
    classification recipe sets a corpus-wide minimum, but the model
    needs a floor *per class* — a 200-row corpus with one class
    contributing 2 examples will still misclassify that class.

    Returns None when every class clears the warn floor."""
    if not counts:
        return None

    sparse = sorted(
        ((k, c) for k, c in counts.items() if c < PER_CLASS_MINIMUM),
        key=lambda kc: kc[1],
    )
    if not sparse:
        return None

    has_block = any(c < PER_CLASS_BLOCK for _, c in sparse)
    severity: Literal["warn", "block"] = "block" if has_block else "warn"

    rendered = ", ".join(f"{k}={c}" for k, c in sparse)
    return {
        "id": "per_class_minimum_unmet",
        "severity": severity,
        "headline": (
            f"{len(sparse)} class(es) below the {PER_CLASS_MINIMUM}-example "
            f"per-class floor."
        ),
        "detail": (
            f"Sparse classes: {rendered}. The model can't learn a class "
            f"with fewer than ~{PER_CLASS_MINIMUM} examples — F1 for those "
            f"classes will collapse to guessing."
        ),
        "suggested_action": {
            "kind": "synth_balance",
            "params": {
                "underrepresented_classes": [k for k, _ in sparse],
                "target_rows_per_class": PER_CLASS_MINIMUM,
            },
        },
    }


def _signal_label_vocab_fragmented(counts: dict[str, int]) -> ForecastSignal | None:
    """Detect case-insensitive label drift ("positive" vs "Positive").

    Same drift class the GoldEntryAddForm warns about, but the
    forecast was otherwise silent on it — and these fragments tank
    eval because each is treated as a distinct class. Always a warn
    (not a block) because the merge is a quick fix and the model can
    still train through it, just badly."""
    if len(counts) < 2:
        return None

    groups: dict[str, list[tuple[str, int]]] = {}
    for label, count in counts.items():
        key = re.sub(r"\s+", " ", label.strip().lower())
        groups.setdefault(key, []).append((label, count))

    fragments = [variants for variants in groups.values() if len(variants) > 1]
    if not fragments:
        return None

    rendered_groups = []
    for variants in fragments[:3]:
        rendered_groups.append(" / ".join(f"{label!r} ({c})" for label, c in variants))
    suffix = "" if len(fragments) <= 3 else f" (+{len(fragments) - 3} more)"

    return {
        "id": "label_vocab_fragmented",
        "severity": "warn",
        "headline": (
            f"{len(fragments)} label group(s) look like case/whitespace "
            f"duplicates."
        ),
        "detail": (
            f"Fragmented: {'; '.join(rendered_groups)}{suffix}. The trainer "
            f"treats each as a distinct class; merge them in the gold set "
            f"so the model learns the canonical label."
        ),
        "suggested_action": {
            "kind": "fix_gold_rows",
            "params": {
                "fragment_groups": [
                    [label for label, _ in variants] for variants in fragments[:25]
                ],
            },
        },
    }


def _signal_single_class_dominance(counts: dict[str, int]) -> ForecastSignal | None:
    """Flag when a single class exceeds ``SINGLE_CLASS_DOMINANCE_FRAC``
    of the corpus. Distinct from ``class_imbalance`` (which is entropy-
    based) — a 90/5/5 three-way split has higher entropy than a 80/20
    binary, but the dominant class still dominates. Always a warn."""
    if not counts:
        return None
    total = sum(counts.values())
    if total <= 0:
        return None
    top_label, top_count = max(counts.items(), key=lambda kc: kc[1])
    fraction = top_count / total
    if fraction < SINGLE_CLASS_DOMINANCE_FRAC:
        return None

    other_classes = [k for k in counts if k != top_label]
    return {
        "id": "single_class_dominance",
        "severity": "warn",
        "headline": (
            f"'{top_label}' is {fraction:.0%} of the gold set — the model "
            f"will default to it."
        ),
        "detail": (
            f"When one class dominates, the trainer's loss is minimised "
            f"by always predicting that class. Other classes won't get "
            f"learned even if their per-class counts look fine."
        ),
        "suggested_action": {
            "kind": "synth_balance",
            "params": {"underrepresented_classes": other_classes},
        },
    }


def _build_classification_signals(
    gold_rows: list[dict[str, Any]],
) -> tuple[list[ForecastSignal], float | None]:
    """Per-recipe builder for classification. Returns (signals,
    class_entropy) so the orchestrator can feed entropy into the
    gate-pass probability heuristic.

    Skips entirely when there are no recoverable labels — a brand-new
    classification project before its first row has been labeled
    shouldn't get a wall of label-shape signals."""
    signals: list[ForecastSignal] = []
    counts = _label_counts(gold_rows)
    if not counts:
        return signals, None

    imbalance = _signal_class_imbalance(gold_rows, "classification")
    if imbalance is not None:
        signals.append(imbalance)

    per_class = _signal_per_class_minimum(counts)
    if per_class is not None:
        signals.append(per_class)

    fragmented = _signal_label_vocab_fragmented(counts)
    if fragmented is not None:
        signals.append(fragmented)

    dominance = _signal_single_class_dominance(counts)
    if dominance is not None:
        signals.append(dominance)

    entropy = _shannon_entropy(counts.values())
    return signals, entropy


# ─────────────────────────────────────────────────────────────────────
# Span-extraction signals.
# ─────────────────────────────────────────────────────────────────────


def _signal_format_consistency(
    gold_rows: list[dict[str, Any]],
    task_profile: str,
) -> ForecastSignal | None:
    """Structured-task only. Returns None for non-structured recipes."""
    if task_profile != "structured_extraction":
        return None

    invalid_row_ids: list[int] = []
    for idx, row in enumerate(gold_rows):
        payload = _extract_span_payload(row)
        if payload is None:
            invalid_row_ids.append(idx)
            continue
        spans = _extract_span_list(payload)
        if not isinstance(spans, list):
            invalid_row_ids.append(idx)
            continue
        any_invalid = any(
            not isinstance(s, dict)
            or not isinstance(s.get("start"), int)
            or not isinstance(s.get("end"), int)
            or s.get("start", -1) > s.get("end", -1)
            for s in spans
        )
        if any_invalid:
            invalid_row_ids.append(idx)

    if not invalid_row_ids:
        return {
            "id": "format_inconsistency",
            "severity": "ok",
            "headline": "All gold-row span structures parse cleanly.",
            "detail": f"Checked {len(gold_rows)} rows for valid span schemas.",
            "suggested_action": None,
        }

    bad_frac = len(invalid_row_ids) / max(1, len(gold_rows))
    severity: Literal["warn", "block"] = "block" if bad_frac >= FORMAT_CONSISTENCY_BLOCK_FRAC else "warn"
    return {
        "id": "format_inconsistency",
        "severity": severity,
        "headline": f"{len(invalid_row_ids)} gold rows have invalid span structures.",
        "detail": (
            f"{bad_frac:.0%} of rows fail schema validation. The model "
            f"can't learn a structure that isn't consistent in the gold set."
        ),
        "suggested_action": {
            "kind": "fix_gold_rows",
            "params": {"invalid_row_ids": invalid_row_ids[:25]},
        },
    }


def _signal_entity_type_coverage(gold_rows: list[dict[str, Any]]) -> ForecastSignal | None:
    """Warn when the gold set covers fewer than ``ENTITY_TYPE_COVERAGE_MIN``
    distinct entity types. A NER-style task trained on a single type
    learns "find this one thing"; it won't generalise to types not in
    the gold set. Skipped when no spans are extractable (the
    ``format_inconsistency`` signal already covers that case)."""
    types: set[str] = set()
    rows_with_spans = 0
    for row in gold_rows:
        payload = _extract_span_payload(row)
        if payload is None:
            continue
        spans = _extract_span_list(payload)
        if not spans:
            continue
        rows_with_spans += 1
        for span in spans:
            if not isinstance(span, dict):
                continue
            type_value = span.get("type")
            if isinstance(type_value, str) and type_value.strip():
                types.add(type_value.strip())

    if rows_with_spans == 0:
        # No spans at all — handled by format_inconsistency /
        # negative_examples_missing depending on the shape.
        return None

    if len(types) >= ENTITY_TYPE_COVERAGE_MIN:
        return {
            "id": "entity_type_coverage_thin",
            "severity": "ok",
            "headline": (
                f"Gold set covers {len(types)} entity types — above the "
                f"{ENTITY_TYPE_COVERAGE_MIN}-type minimum."
            ),
            "detail": f"Types: {', '.join(sorted(types))}.",
            "suggested_action": None,
        }

    severity: Literal["warn", "block"] = "block" if len(types) <= 1 else "warn"
    return {
        "id": "entity_type_coverage_thin",
        "severity": severity,
        "headline": (
            f"Only {len(types)} entity type(s) in the gold set — recipe "
            f"benefits from at least {ENTITY_TYPE_COVERAGE_MIN}."
        ),
        "detail": (
            f"Types seen: {', '.join(sorted(types)) or '(none)'}. The "
            f"model can't generalise to types it never saw labeled."
        ),
        "suggested_action": {
            "kind": "synth_augment",
            "params": {
                "target_rows": max(50, len(gold_rows) * 2),
                "diversify": "entity_types",
            },
        },
    }


def _signal_span_offset_invalid(gold_rows: list[dict[str, Any]]) -> ForecastSignal | None:
    """Detect rows where ``text[start:end] != span.text``. A silently
    misaligned offset poisons exact-match scoring without any
    schema-validation error to point at."""
    invalid_row_ids: list[int] = []
    inspected = 0
    for idx, row in enumerate(gold_rows):
        payload = _extract_span_payload(row)
        if payload is None:
            continue
        spans = _extract_span_list(payload)
        if not spans:
            continue
        source_text = _extract_span_source_text(row)
        if source_text is None:
            # No text to validate against — skip rather than false-positive.
            continue
        inspected += 1
        for span in spans:
            if not isinstance(span, dict):
                continue
            start = span.get("start")
            end = span.get("end")
            recorded = span.get("text")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if not isinstance(recorded, str):
                continue
            if not (0 <= start <= end <= len(source_text)):
                invalid_row_ids.append(idx)
                break
            if source_text[start:end] != recorded:
                invalid_row_ids.append(idx)
                break

    if inspected == 0:
        return None

    if not invalid_row_ids:
        return {
            "id": "span_offset_invalid",
            "severity": "ok",
            "headline": "All span offsets line up with their recorded text.",
            "detail": f"Verified {inspected} rows with spans.",
            "suggested_action": None,
        }

    bad_frac = len(invalid_row_ids) / inspected
    severity: Literal["warn", "block"] = (
        "block" if bad_frac >= SPAN_OFFSET_INVALID_BLOCK_FRAC else "warn"
    )
    return {
        "id": "span_offset_invalid",
        "severity": severity,
        "headline": (
            f"{len(invalid_row_ids)} of {inspected} rows have spans whose "
            f"offsets don't match their text."
        ),
        "detail": (
            f"{bad_frac:.0%} of rows are silently broken: text[start:end] "
            f"!= span.text. Exact-match scoring will fail on these even "
            f"if the model predicts perfectly."
        ),
        "suggested_action": {
            "kind": "fix_gold_rows",
            "params": {"invalid_row_ids": invalid_row_ids[:25]},
        },
    }


def _signal_negative_examples_missing(gold_rows: list[dict[str, Any]]) -> ForecastSignal | None:
    """Warn when the gold set has no rows with an empty entities list.
    Without negatives the model learns "always extract something" —
    a common over-extraction failure mode for span tasks."""
    has_spans_rows = 0
    negatives = 0
    for row in gold_rows:
        payload = _extract_span_payload(row)
        if payload is None:
            continue
        has_spans_rows += 1
        spans = _extract_span_list(payload)
        if len(spans) == 0:
            negatives += 1

    if has_spans_rows < 5:
        # Too few rows to meaningfully assess.
        return None

    if negatives > 0:
        return {
            "id": "negative_examples_missing",
            "severity": "ok",
            "headline": (
                f"{negatives} negative example(s) present — the model can "
                f"learn 'sometimes there's nothing to extract'."
            ),
            "detail": (
                f"Recommended: 10-20% of the gold set as rows with empty "
                f"entities so the model doesn't over-extract."
            ),
            "suggested_action": None,
        }

    return {
        "id": "negative_examples_missing",
        "severity": "warn",
        "headline": "No negative examples in the gold set.",
        "detail": (
            "Every row has at least one span. Without rows that have an "
            "empty entities list, the model learns 'always extract "
            "something' and over-fires on inputs that genuinely have no "
            "entities. Add 10-20% negatives."
        ),
        "suggested_action": {
            "kind": "synth_augment",
            "params": {
                "target_rows": max(int(has_spans_rows * 0.15), 5),
                "diversify": "negative_examples",
            },
        },
    }


def _build_span_extraction_signals(
    gold_rows: list[dict[str, Any]],
) -> list[ForecastSignal]:
    """Per-recipe builder for span-extraction. Skips entirely when the
    gold set is empty — the row-count signal already covers that."""
    signals: list[ForecastSignal] = []
    if not gold_rows:
        return signals

    format_signal = _signal_format_consistency(gold_rows, "structured_extraction")
    if format_signal is not None:
        signals.append(format_signal)

    coverage = _signal_entity_type_coverage(gold_rows)
    if coverage is not None:
        signals.append(coverage)

    offset = _signal_span_offset_invalid(gold_rows)
    if offset is not None:
        signals.append(offset)

    negatives = _signal_negative_examples_missing(gold_rows)
    if negatives is not None:
        signals.append(negatives)

    return signals


# ─────────────────────────────────────────────────────────────────────
# Summarization signals.
# ─────────────────────────────────────────────────────────────────────


def _signal_summary_doc_ratio(gold_rows: list[dict[str, Any]]) -> ForecastSignal | None:
    """Flag rows where the summary is more than ``SUMMARY_DOC_RATIO_WARN``
    of the document length — usually a mislabeled paraphrase or the
    wrong column being loaded into the summary slot. The summarization
    task assumes summary << doc."""
    pairs: list[tuple[int, float, int, int]] = []  # (idx, ratio, doc_len, sum_len)
    inspected = 0
    for idx, row in enumerate(gold_rows):
        ds = _extract_summary_pair(row)
        if ds is None:
            continue
        doc, summary = ds
        doc_len = len(doc)
        sum_len = len(summary)
        if doc_len == 0:
            continue
        inspected += 1
        ratio = sum_len / doc_len
        if ratio > SUMMARY_DOC_RATIO_WARN:
            pairs.append((idx, ratio, doc_len, sum_len))

    if inspected == 0:
        return None

    if not pairs:
        return {
            "id": "summary_doc_ratio_outliers",
            "severity": "ok",
            "headline": "Every summary is meaningfully shorter than its document.",
            "detail": f"Checked {inspected} rows; all summary/doc ratios ≤ {SUMMARY_DOC_RATIO_WARN:.0%}.",
            "suggested_action": None,
        }

    bad_frac = len(pairs) / inspected
    severity: Literal["warn", "block"] = "block" if bad_frac >= 0.30 else "warn"
    worst = sorted(pairs, key=lambda p: -p[1])[:3]
    examples = ", ".join(
        f"row {i}: {sum_len}/{doc_len} chars ({ratio:.0%})"
        for i, ratio, doc_len, sum_len in worst
    )
    return {
        "id": "summary_doc_ratio_outliers",
        "severity": severity,
        "headline": (
            f"{len(pairs)} of {inspected} rows have summaries longer than "
            f"{SUMMARY_DOC_RATIO_WARN:.0%} of their document."
        ),
        "detail": (
            f"Suspicious rows are usually a mislabeled paraphrase or the "
            f"wrong column loaded into the summary slot — both poison "
            f"training. Worst offenders: {examples}."
        ),
        "suggested_action": {
            "kind": "fix_gold_rows",
            "params": {"invalid_row_ids": [i for i, _, _, _ in pairs[:25]]},
        },
    }


def _build_summarization_signals(
    gold_rows: list[dict[str, Any]],
) -> list[ForecastSignal]:
    """Per-recipe builder for summarization."""
    signals: list[ForecastSignal] = []
    if not gold_rows:
        return signals
    ratio = _signal_summary_doc_ratio(gold_rows)
    if ratio is not None:
        signals.append(ratio)
    return signals


# ─────────────────────────────────────────────────────────────────────
# Public entry point.
# ─────────────────────────────────────────────────────────────────────


def _overall_verdict(
    *,
    signals: list[ForecastSignal],
    confidence_pct: int,
) -> Literal["likely_pass", "borderline", "likely_fail"]:
    """Verdict logic:
    - Any 'block' signal → likely_fail.
    - confidence_pct >= 65 and no 'warn' signals → likely_pass.
    - confidence_pct >= 65 with warns → borderline.
    - confidence_pct 40-64 → borderline.
    - confidence_pct < 40 → likely_fail.
    """
    has_block = any(s["severity"] == "block" for s in signals)
    has_warn = any(s["severity"] == "warn" for s in signals)

    if has_block:
        return "likely_fail"
    if confidence_pct < 40:
        return "likely_fail"
    if confidence_pct >= 65 and not has_warn:
        return "likely_pass"
    return "borderline"


def _build_per_recipe_signals(
    *,
    task_profile: str,
    gold_rows: list[dict[str, Any]],
) -> tuple[list[ForecastSignal], float | None]:
    """Dispatch per-recipe signal building from the recipe's task
    profile. Returns (signals, class_entropy_for_heuristic) — only
    classification carries a class-entropy back into the gate-pass
    probability heuristic; other recipes return None."""
    if task_profile == "classification":
        return _build_classification_signals(gold_rows)
    if task_profile == "structured_extraction":
        return _build_span_extraction_signals(gold_rows), None
    if task_profile == "summarization":
        return _build_summarization_signals(gold_rows), None
    # instruction_sft / catch-all → no recipe-specific signals.
    return [], None


async def forecast_training(
    db: AsyncSession,
    project_id: int,
    *,
    use_cache: bool = True,
) -> ForecastResult:
    """Compute the trainability forecast for a project.

    Cache lookup keyed on (dataset_signature, recipe_id, base_model_name);
    cache miss triggers a full recompute including the gold-set token
    Jaccard sweep.

    Raises ValueError if the project doesn't exist or has no recipe
    selected — callers should translate to 404/400 at the API layer.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id") or ""
    if not recipe_id:
        raise ValueError("Project has no selected recipe")
    recipe = get_recipe(recipe_id)
    if recipe is None:
        raise ValueError(f"Recipe '{recipe_id}' not found in the catalog")

    base_model_name = (
        project.base_model_name
        or recipe.suggested_base_model
        or "HuggingFaceTB/SmolLM2-135M-Instruct"
    )

    dataset_signature = await _build_dataset_signature(db, project_id)
    cache_key = _compute_cache_key(
        dataset_signature=dataset_signature,
        recipe_id=recipe_id,
        base_model_name=base_model_name,
    )

    if use_cache and isinstance(project.training_forecast_cache, dict):
        cached = project.training_forecast_cache
        if cached.get("cache_key") == cache_key:
            return {
                **cached,
                "cache_hit": True,
            }

    gold_rows = await _load_gold_rows(db, project_id)
    train_row_count = await _estimate_train_row_count(db, project_id)
    minimum_rows = recipe.gold_template.min_rows_recommended

    signals: list[ForecastSignal] = []
    signals.append(_signal_row_count(train_row_count, minimum_rows))

    diversity_signal, diversity_score = _signal_goldset_diversity(gold_rows)
    signals.append(diversity_signal)

    recipe_signals, class_entropy = _build_per_recipe_signals(
        task_profile=recipe.task_profile,
        gold_rows=gold_rows,
    )
    signals.extend(recipe_signals)

    base_params = KNOWN_BASE_MODEL_PARAMS_M.get(base_model_name, DEFAULT_BASE_MODEL_PARAMS_M)
    prob_signal, confidence_pct = _signal_gate_pass_probability(
        train_row_count=train_row_count,
        task_profile=recipe.task_profile,
        base_model_params_m=base_params,
        class_entropy=class_entropy,
        diversity_score=diversity_score,
    )
    signals.append(prob_signal)

    overall = _overall_verdict(signals=signals, confidence_pct=confidence_pct)

    computed_at_dt = datetime.now(timezone.utc)
    result: ForecastResult = {
        "overall": overall,
        "confidence_pct": confidence_pct,
        "signals": signals,
        "computed_at": computed_at_dt.isoformat(),
        "cache_key": cache_key,
        "cache_hit": False,
    }

    project.training_forecast_cache = json.loads(json.dumps(result))
    await _persist_snapshot(
        db,
        project_id=project_id,
        cache_key=cache_key,
        computed_at=computed_at_dt,
        overall=overall,
        confidence_pct=confidence_pct,
        signals=signals,
    )
    await db.flush()

    return result


async def _persist_snapshot(
    db: AsyncSession,
    *,
    project_id: int,
    cache_key: str,
    computed_at: datetime,
    overall: str,
    confidence_pct: int,
    signals: list[ForecastSignal],
) -> None:
    """Persist one snapshot row + prune anything older than the
    retention window in the same call. Cheap enough that we don't
    need a separate retention job; the work scales with how often the
    user actually iterates."""
    db.add(
        TrainingForecastSnapshot(
            project_id=project_id,
            cache_key=cache_key,
            computed_at=computed_at,
            overall=overall,
            confidence_pct=confidence_pct,
            # Round-trip through json so we store plain dicts (not
            # TypedDicts) — matches the cache-payload convention above.
            signals=json.loads(json.dumps(signals)),
        )
    )
    cutoff = computed_at - timedelta(days=SNAPSHOT_RETENTION_DAYS)
    await db.execute(
        delete(TrainingForecastSnapshot).where(
            TrainingForecastSnapshot.project_id == project_id,
            TrainingForecastSnapshot.computed_at < cutoff,
        )
    )


async def list_forecast_history(
    db: AsyncSession,
    project_id: int,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Returns up to ``limit`` recent snapshots for the project,
    newest-first. The shape mirrors ``ForecastResult`` (minus the
    cache_hit field, which is always False for a persisted snapshot)
    so the panel can render historical entries with the same code
    path it uses for the live result.

    Limit is clamped to [1, 100]; the panel reads ~10 by default."""
    clamped = max(1, min(100, limit))
    result = await db.execute(
        select(TrainingForecastSnapshot)
        .where(TrainingForecastSnapshot.project_id == project_id)
        .order_by(TrainingForecastSnapshot.computed_at.desc())
        .limit(clamped)
    )
    return [
        {
            "id": snap.id,
            "cache_key": snap.cache_key,
            "computed_at": snap.computed_at.isoformat(),
            "overall": snap.overall,
            "confidence_pct": snap.confidence_pct,
            "signals": snap.signals or [],
        }
        for snap in result.scalars()
    ]
