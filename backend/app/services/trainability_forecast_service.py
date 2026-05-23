"""Trainability forecast service — USER-SUCCESS Epic 1.

Predicts whether a project's training run is likely to clear the default
Auto-Gates *before* the user clicks Train. The forecast is computed from
five signals against the project's prepared data + recipe + base model:

1. Row count vs the recipe's minimum (`row_count_below_minimum`)
2. Class-balance entropy for classification recipes (`class_imbalance`)
3. Gold-set diversity via token-set Jaccard (`goldset_diversity_low`)
4. Format consistency for structured tasks (`format_inconsistency`)
5. Overall gate-pass probability via `estimate_gate_pass_prob()`

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.project import Project
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
    # Cap row_count contribution at 200 rows — empirically the point
    # where doubling rows stops moving F1 much for small models.
    data_floor = min(1.0, max(0.0, row_count) / 200.0)

    # Capacity grows logarithmically: 50M = 0, 400M = ~1.0.
    if base_model_params_m <= 50:
        capacity = 0.0
    else:
        capacity = min(1.0, math.log(base_model_params_m / 50.0) / math.log(8.0))

    # Quality compounds difficulty + diversity + class balance.
    quality = (1.0 - recipe_difficulty) * max(0.0, min(1.0, diversity_score))
    if class_entropy is not None:
        # Entropy of 1.5 (≈4.5 equal classes) caps the class term at 1.0.
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
    # Fallback: include all top-level string values if nothing matched.
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
# The five signal checks.
# ─────────────────────────────────────────────────────────────────────


def _signal_row_count(
    train_row_count: int,
    minimum_rows: int,
) -> ForecastSignal:
    """Block when below recipe minimum, warn when below 1.5×, ok above."""
    if minimum_rows <= 0:
        # Defensive — should never happen with a valid recipe.
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


def _extract_classification_labels(gold_rows: list[dict[str, Any]]) -> list[str]:
    """Walks gold rows to extract classification labels. Supports:

    - Top-level ``label`` field (synthetic / unit-test rows).
    - Nested ``expected.label`` (raw template JSONL before normalization).
    - Raw template ``expected`` string (when expected is the label itself).
    - Normalized ``answer`` field (set by the dataset_service loader,
      which coerces template JSONL to a QA-pair shape regardless of
      original structure — for classification recipes the label
      ends up here).
    """
    labels: list[str] = []
    for row in gold_rows:
        if isinstance(row.get("label"), str):
            labels.append(row["label"])
            continue
        expected = row.get("expected")
        if isinstance(expected, dict) and isinstance(expected.get("label"), str):
            labels.append(expected["label"])
            continue
        if isinstance(expected, str):
            labels.append(expected)
            continue
        # Normalized loader path: classification gold rows arrive with
        # the label in the "answer" field. This is short + categorical
        # (vs free-text QA answers), so we add a length guard to avoid
        # mistaking actual prose for a label.
        answer = row.get("answer")
        if isinstance(answer, str) and 0 < len(answer) <= 64 and "\n" not in answer:
            labels.append(answer)
    return labels


def _signal_class_imbalance(
    gold_rows: list[dict[str, Any]],
    task_profile: str,
) -> ForecastSignal | None:
    """Classification-only. Returns None if the task isn't classification
    or there's no usable label column."""
    if task_profile != "classification":
        return None

    labels = _extract_classification_labels(gold_rows)
    if not labels:
        return None

    # Shannon entropy in nats over the class distribution.
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

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
    if isinstance(expected, dict) and (expected.get("spans") is not None or expected.get("entities") is not None):
        return expected
    answer = row.get("answer")
    if isinstance(answer, str):
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and (parsed.get("spans") is not None or parsed.get("entities") is not None):
            return parsed
    if row.get("spans") is not None or row.get("entities") is not None:
        return row  # spans live at the top level
    return None


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
            # No span payload found — this *is* a schema issue.
            invalid_row_ids.append(idx)
            continue
        spans = payload.get("spans") or payload.get("entities") or []
        if not isinstance(spans, list):
            invalid_row_ids.append(idx)
            continue
        # Empty spans is valid (negative examples), but malformed
        # entries fail validation.
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
        severity = "warn"  # never block on probability alone — that's advisory
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

    # Cache hit path.
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

    class_entropy: float | None = None
    class_signal = _signal_class_imbalance(gold_rows, recipe.task_profile)
    if class_signal is not None:
        signals.append(class_signal)
        labels = _extract_classification_labels(gold_rows)
        if labels:
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            total = sum(counts.values())
            class_entropy = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)

    diversity_signal, diversity_score = _signal_goldset_diversity(gold_rows)
    signals.append(diversity_signal)

    format_signal = _signal_format_consistency(gold_rows, recipe.task_profile)
    if format_signal is not None:
        signals.append(format_signal)

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

    result: ForecastResult = {
        "overall": overall,
        "confidence_pct": confidence_pct,
        "signals": signals,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "cache_hit": False,
    }

    # Persist the cache. We serialize through json round-trip so the
    # JSON column stores plain dicts (not TypedDicts).
    project.training_forecast_cache = json.loads(json.dumps(result))
    await db.flush()

    return result
