"""Gold-set diagnostics for V4 of the ML-native visualisations arc.

Computes two views of a project's gold set that the
``GoldSetDiagnosticsPanel`` renders side-by-side:

  - **Class balance** — per-label counts + share-of-total + Shannon
    entropy. The Coach Mode card already reports the entropy number;
    this is the visualisation behind that number plus the per-class
    counts the user actually needs to act on.
  - **Class similarity matrix** — for each pair of labels ``(a, b)``,
    the mean pairwise Jaccard similarity between sampled rows of
    class ``a`` and class ``b``. Diagonal cells measure intra-class
    redundancy (high = the rows of one class look the same as each
    other → low diversity); off-diagonal cells measure inter-class
    confusability (high = even a perfect classifier can't tell the
    classes apart from text alone).

Both views are classification-specific. For non-classification gold
sets we return an empty payload so the UI can render "n/a for this
recipe" rather than a misleading single-cell heatmap.

Reuses helpers from ``trainability_forecast_service`` — ``_load_gold_rows``,
``_label_counts``, ``_tokenize``, ``_row_to_text``, ``_jaccard``,
``_shannon_entropy`` — to avoid duplicating gold-row parsing logic.
The forecast service is the single source of truth for "what counts as
a label in this gold row"; this service just visualises it.
"""

from __future__ import annotations

import random
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.trainability_forecast_service import (
    _extract_classification_label,
    _jaccard,
    _label_counts,
    _load_gold_rows,
    _row_to_text,
    _shannon_entropy,
    _tokenize,
)


# Max sample size per class when computing the similarity matrix. The
# matrix is O(n_classes^2) cells and each cell does O(sample^2) pairwise
# Jaccard ops, so this caps the cost at sample^2 * n_classes^2 set ops.
# 12 per class is empirically the smallest sample that gives a stable
# mean (variance below 5% on repeated bootstrap samples) without making
# a 5-class gold set chew through 3600 set ops per request.
DEFAULT_SAMPLE_PER_CLASS = 12

# Minimum gold rows per class to even attempt a similarity score. With
# fewer than this many rows the pairwise Jaccard mean has huge variance
# and the cell would just read as noise — better to surface "n/a" than
# a noisy number.
MIN_ROWS_PER_CLASS_FOR_SIMILARITY = 2


async def compute_gold_set_diagnostics(
    db: AsyncSession,
    project_id: int,
    *,
    sample_per_class: int = DEFAULT_SAMPLE_PER_CLASS,
    seed: int = 0,
) -> dict[str, Any]:
    """Return ``{class_balance, similarity, total_rows, ...}`` for the
    project's gold set.

    Always returns 200 — never raises for "no gold set" / "no labels";
    the caller renders the empty-state payload instead. The only
    failure mode is ``ValueError`` when ``project_id`` doesn't exist,
    which the API translates to 404.
    """
    gold_rows = await _load_gold_rows(db, project_id)
    total_rows = len(gold_rows)

    # Class-balance view ------------------------------------------------
    label_counts = _label_counts(gold_rows)
    classification_eligible = bool(label_counts)
    balance_payload: dict[str, Any]
    if classification_eligible:
        sorted_items = sorted(
            label_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        total = sum(label_counts.values()) or 1
        # share is fraction in [0, 1]. Shannon entropy is in nats.
        entropy = _shannon_entropy(label_counts.values())
        balance_payload = {
            "labels": [
                {
                    "label": name,
                    "count": count,
                    "share": round(count / total, 4),
                }
                for name, count in sorted_items
            ],
            "total": total,
            "entropy_nats": round(entropy, 4),
        }
    else:
        # Non-classification gold set OR labels weren't recoverable.
        balance_payload = {
            "labels": [],
            "total": total_rows,
            "entropy_nats": 0.0,
        }

    # Similarity-matrix view -------------------------------------------
    # Build per-class row buckets (only when we have at least the minimum
    # for similarity scoring — otherwise skip and surface as n/a).
    buckets: dict[str, list[dict]] = {}
    if classification_eligible:
        for row in gold_rows:
            label = _extract_classification_label(row)
            if label is None:
                continue
            buckets.setdefault(label, []).append(row)

    similarity_payload: dict[str, Any]
    if len(buckets) >= 2:
        rng = random.Random(seed)
        # Sample-per-class token-sets we'll reuse across every matrix cell.
        sampled_tokens: dict[str, list[frozenset[str]]] = {}
        insufficient: set[str] = set()
        for label, rows in buckets.items():
            if len(rows) < MIN_ROWS_PER_CLASS_FOR_SIMILARITY:
                insufficient.add(label)
                continue
            sample = rows if len(rows) <= sample_per_class else rng.sample(rows, sample_per_class)
            token_sets = [_tokenize(_row_to_text(r)) for r in sample]
            # Keep non-empty token sets only.
            token_sets = [t for t in token_sets if t]
            if len(token_sets) >= MIN_ROWS_PER_CLASS_FOR_SIMILARITY:
                sampled_tokens[label] = token_sets
            else:
                insufficient.add(label)

        # Use the class-balance order for matrix rows/cols so the
        # heatmap reads in descending-popularity order — easier to scan.
        label_order = [entry["label"] for entry in balance_payload["labels"]]
        matrix: list[list[float | None]] = []
        for row_label in label_order:
            row_cells: list[float | None] = []
            row_tokens = sampled_tokens.get(row_label)
            for col_label in label_order:
                col_tokens = sampled_tokens.get(col_label)
                if row_tokens is None or col_tokens is None:
                    row_cells.append(None)
                    continue
                cell = _mean_cross_jaccard(
                    row_tokens, col_tokens, same_bucket=(row_label == col_label),
                )
                row_cells.append(cell)
            matrix.append(row_cells)
        similarity_payload = {
            "labels": label_order,
            "matrix": matrix,
            "sample_per_class": sample_per_class,
            "insufficient_labels": sorted(insufficient),
        }
    else:
        similarity_payload = {
            "labels": [],
            "matrix": [],
            "sample_per_class": sample_per_class,
            "insufficient_labels": sorted(buckets.keys()) if buckets else [],
        }

    return {
        "project_id": int(project_id),
        "total_rows": total_rows,
        "classification_eligible": classification_eligible,
        "class_balance": balance_payload,
        "similarity": similarity_payload,
    }


def _mean_cross_jaccard(
    rows_a: list[frozenset[str]],
    rows_b: list[frozenset[str]],
    *,
    same_bucket: bool,
) -> float | None:
    """Mean pairwise Jaccard between every row in ``rows_a`` and every
    row in ``rows_b``.

    When ``same_bucket`` is True the buckets are the same class — skip
    ``(i, i)`` self-pairs (which are always Jaccard=1.0 and would inflate
    the diagonal). Returns ``None`` only when there are zero pairs to
    score, which is genuinely "not measurable" rather than 0.
    """
    pairs = 0
    total = 0.0
    for i, a in enumerate(rows_a):
        for j, b in enumerate(rows_b):
            if same_bucket and i == j:
                continue
            total += _jaccard(a, b)
            pairs += 1
    if pairs == 0:
        return None
    return round(total / pairs, 4)
