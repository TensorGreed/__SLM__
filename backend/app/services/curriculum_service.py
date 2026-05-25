"""Curriculum-style training row ranking (USER-SUCCESS Epic 6 Phase 6a).

Ranks training rows by **difficulty** so the trainer can show easy
examples first and add the hard ones later — a classic curriculum-
learning recipe that routinely lifts F1 by 10-20% on thin datasets
without changing the data or the model.

This module is the **ranking surface only.** Training-pipeline
integration (the two-epoch schedule) lands in Phase 6b; the UI
toggle + default-on heuristic land in Phase 6d. Phase 6a ships:

  * ``rank_rows(rows, *, scoring_mode=...)`` — pure function over a
    list of rows + a scoring mode token.
  * ``recommended_scoring_mode_for_recipe(recipe_id)`` — recipe →
    scoring mode mapping. Classification ships in Phase 6a; the
    other recipes plug their own scoring modes in later phases.

The API is intentionally **recipe-agnostic**: the ``scoring_mode``
parameter carries any per-recipe logic so future scoring modes
(``length_complexity`` for span-extraction, ``paraphrase_distance``
for QA-SFT) plug in alongside without renaming the entry point. Per
the ``keep-brewslm-general`` design rule, no public function is
named after one recipe.

Scoring modes:

  * ``"prototype_entropy"`` (Phase 6a): per-row entropy across cosine
    similarity to other rows of the **same grouping key**. Works for
    any task where rows carry a discrete group label (classification
    labels today; span types, conversation tones, etc. later). Low
    entropy = high similarity to classmates = prototypical row =
    easy. High entropy = outlier = hard / noisy / potentially
    mislabeled.

Embeddings: uses ``sentence-transformers/all-MiniLM-L6-v2`` (22 MB,
CPU-fast) via the same pattern as
``dataset_intelligence_service._sentence_transformer_embeddings``.
Unlike that service, curriculum **hard-fails** when the lib isn't
installed — a hashing-fallback embedding would produce essentially
random rankings, which is worse than not running curriculum at all.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict


# Public type — scoring modes are a closed enum at any given phase.
# Add a new value here when a new scoring mode ships.
ScoringMode = Literal["prototype_entropy"]


# Recipe → scoring mode map. Recipes that don't have a curriculum
# implementation yet return None — callers (training pipeline, Coach
# Mode) treat None as "curriculum not available for this recipe."
_RECIPE_TO_SCORING_MODE: dict[str, ScoringMode] = {
    "classification": "prototype_entropy",
}


def recommended_scoring_mode_for_recipe(recipe_id: str) -> ScoringMode | None:
    """Maps recipe_id → the scoring mode best-suited for it, or None
    when no curriculum scoring mode ships for that recipe yet."""
    return _RECIPE_TO_SCORING_MODE.get(recipe_id)


class CurriculumEntry(TypedDict):
    """One row's place in the curriculum.

    - ``row_id`` mirrors the row's own id field (or its 0-indexed
      position in the input list when no id was present).
    - ``difficulty`` is in [0, 1]; lower = easier.
    - ``rank`` is the 0-indexed position when the list is sorted
      ascending by difficulty (rank 0 = easiest, rank N-1 = hardest).
    """

    row_id: int | str
    difficulty: float
    rank: int


class CurriculumUnavailable(RuntimeError):
    """Raised when curriculum ranking can't be computed (sentence-
    transformers missing, unsupported scoring mode, empty rows…)."""


def rank_rows(
    rows: list[dict[str, Any]],
    *,
    scoring_mode: ScoringMode,
    group_key: str = "label",
    text_keys: tuple[str, ...] = ("text", "input", "question"),
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
    cache_dir: Path | None = None,
) -> list[CurriculumEntry]:
    """Return a curriculum order for ``rows``.

    Arguments:
      rows:          training rows. Each row must carry text (under one
                     of ``text_keys``) and, for ``prototype_entropy``,
                     a grouping value under ``group_key``.
      scoring_mode:  which difficulty signal to use. See the module
                     docstring for the supported modes.
      group_key:     which row field carries the discrete grouping
                     key (default ``"label"`` for classification; will
                     be ``"span_type"`` etc. for other recipes).
      text_keys:     ordered fallbacks for extracting the row's text.
                     First non-empty hit wins.
      embedder:      optional dependency-injection for tests. When
                     None, the sentence-transformers embedder is
                     used (and a CurriculumUnavailable error is
                     raised if that lib isn't installed).
      cache_dir:     optional on-disk cache for embeddings; keyed by
                     ``hash(text)`` so unchanged rows aren't re-
                     embedded across calls. ``None`` skips caching.

    Returns a list of ``CurriculumEntry`` ordered ascending by
    difficulty. The list is always the same length as ``rows``
    (degenerate cases assign difficulty = 0.5).

    Raises:
      CurriculumUnavailable: when sentence-transformers isn't
        installed (and no explicit embedder was provided), or the
        scoring mode is unknown, or ``rows`` is empty.
    """
    if not rows:
        raise CurriculumUnavailable(
            "Cannot rank an empty row list — pass at least one row."
        )
    if scoring_mode != "prototype_entropy":
        raise CurriculumUnavailable(
            f"Unknown scoring_mode: {scoring_mode!r}. "
            f"Known modes: prototype_entropy."
        )

    texts = [_extract_text(row, text_keys) for row in rows]
    groups = [_extract_group(row, group_key) for row in rows]
    row_ids: list[int | str] = [
        row.get("id") if isinstance(row.get("id"), (int, str)) else idx
        for idx, row in enumerate(rows)
    ]

    vectors = _embed_texts(texts, embedder=embedder, cache_dir=cache_dir)
    raw_difficulties = _prototype_entropy_difficulty(vectors, groups)

    indexed = list(enumerate(raw_difficulties))
    indexed.sort(key=lambda pair: pair[1])
    rank_by_idx: dict[int, int] = {idx: rank for rank, (idx, _) in enumerate(indexed)}

    return [
        {
            "row_id": row_ids[idx],
            "difficulty": float(raw_difficulties[idx]),
            "rank": rank_by_idx[idx],
        }
        for idx in range(len(rows))
    ]


# ─────────────────────────────────────────────────────────────────────
# prototype_entropy scoring
# ─────────────────────────────────────────────────────────────────────


def _prototype_entropy_difficulty(
    vectors: list[list[float]],
    groups: list[str],
) -> list[float]:
    """For each row, difficulty = 1 - normalized mean cosine similarity
    to other rows in the **same group**.

    Intuition:
      - High mean similarity to classmates → prototypical → easy.
      - Low mean similarity to classmates → outlier (noisy /
        mislabeled / boundary case) → hard.

    Edge cases:
      - Singleton class (row is alone in its group) → no classmates
        to compare against → max difficulty (1.0). These rows are
        usually outliers by definition.
      - Empty / missing group label → treat as a singleton.
      - Zero-norm vector → max difficulty (1.0); we'd have to divide
        by zero to compute similarity otherwise.
    """
    n = len(vectors)
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        key = group or f"__ungrouped_{idx}"  # ungrouped rows are singletons
        by_group.setdefault(key, []).append(idx)

    norms = [_norm(v) for v in vectors]
    difficulties = [1.0] * n  # default to max difficulty (outlier-safe)

    for group_indices in by_group.values():
        if len(group_indices) < 2:
            # Singleton class — no classmates → outlier by definition.
            for idx in group_indices:
                difficulties[idx] = 1.0
            continue
        for idx in group_indices:
            if norms[idx] == 0:
                difficulties[idx] = 1.0
                continue
            sims: list[float] = []
            for other in group_indices:
                if other == idx or norms[other] == 0:
                    continue
                sims.append(_cosine(vectors[idx], vectors[other], norms[idx], norms[other]))
            if not sims:
                difficulties[idx] = 1.0
                continue
            mean_sim = sum(sims) / len(sims)
            # Cosine over normalized embeddings lives in [-1, 1] but
            # for sentence embeddings is almost always in [0, 1]. Clip
            # to [0, 1] then invert so high similarity = low difficulty.
            mean_sim = max(0.0, min(1.0, mean_sim))
            difficulties[idx] = 1.0 - mean_sim
    return difficulties


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cosine(a: list[float], b: list[float], norm_a: float, norm_b: float) -> float:
    if norm_a == 0 or norm_b == 0:
        return 0.0
    size = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(size))
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────
# Text + group extraction
# ─────────────────────────────────────────────────────────────────────


def _extract_text(row: dict[str, Any], text_keys: tuple[str, ...]) -> str:
    for key in text_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            # Nested ``input: {question: "..."}`` shape used by some templates.
            for sub in value.values():
                if isinstance(sub, str) and sub.strip():
                    return sub
    return ""


def _extract_group(row: dict[str, Any], group_key: str) -> str:
    value = row.get(group_key)
    if isinstance(value, str):
        return value
    # Templates sometimes nest the label under ``expected.label``.
    expected = row.get("expected")
    if isinstance(expected, dict):
        nested = expected.get(group_key)
        if isinstance(nested, str):
            return nested
    return ""


# ─────────────────────────────────────────────────────────────────────
# Embedding plumbing
# ─────────────────────────────────────────────────────────────────────


_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _embed_texts(
    texts: list[str],
    *,
    embedder: Callable[[list[str]], list[list[float]]] | None,
    cache_dir: Path | None,
) -> list[list[float]]:
    """Compute embeddings, hitting the on-disk cache when set.

    Cache key = sha256(text). We cache **per row** (not per call) so
    adding a new row to a project doesn't re-embed the existing ones.
    The cache is JSON for inspectability; a numpy ``.npz`` would be
    smaller but harder to debug, and at ~384 floats per row the JSON
    overhead is < 5KB / row.
    """
    if embedder is None:
        embedder = _sentence_transformer_embedder

    if cache_dir is None:
        return embedder(texts)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "embeddings.json"
    cached: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = {}

    keys = [_text_cache_key(t) for t in texts]
    missing_indices = [i for i, k in enumerate(keys) if k not in cached]
    if missing_indices:
        fresh = embedder([texts[i] for i in missing_indices])
        for src_idx, vec in zip(missing_indices, fresh):
            cached[keys[src_idx]] = list(vec)
        try:
            cache_path.write_text(json.dumps(cached), encoding="utf-8")
        except OSError:
            # Cache write failure is non-fatal — we still return the vectors.
            pass

    return [list(cached[k]) for k in keys]


def _text_cache_key(text: str) -> str:
    # Include the model name in the key so a future embedder swap
    # invalidates the cache automatically.
    h = hashlib.sha256()
    h.update(_DEFAULT_EMBEDDING_MODEL.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Phase 6b — on-disk curriculum shards for the training pipeline.
# ─────────────────────────────────────────────────────────────────────


class CurriculumShardManifest(TypedDict):
    """Manifest describing the on-disk curriculum shards.

    The training pipeline reads ``shard_path`` as its new ``train_file``
    and threads the rest of these fields into ``runtime_config`` so
    the UI / eval surfaces can show "this experiment ran with
    curriculum learning, easy-half = N rows."
    """

    scoring_mode: ScoringMode
    shard_path: str           # path to the ordered easy_half + full_set JSONL
    meta_path: str            # path to the JSON-serialized ranking + counts
    total_rows: int           # rows in the full training set
    easy_count: int           # rows in the bottom-50% (easy) half
    full_count: int           # rows in the full set (= total_rows; alias for symmetry)


def build_curriculum_shards(
    rows: list[dict[str, Any]],
    *,
    scoring_mode: ScoringMode,
    output_dir: Path,
    group_key: str = "label",
    text_keys: tuple[str, ...] = ("text", "input", "question"),
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
    cache_dir: Path | None = None,
) -> CurriculumShardManifest:
    """Rank ``rows`` by difficulty and write the curriculum shard to
    disk: a single JSONL file whose contents are ``[bottom_50%] +
    [full_set]`` in that order.

    Why one ordered file + one trainer run (not two trainer runs):
      - HF Trainer doesn't expose a per-epoch dataset-swap hook, so
        a true "epoch 1 on easy half, epoch 2..N on full set" schedule
        would require chaining two training runs with intermediate
        checkpoint plumbing — substantial orchestration with no
        empirical mandate yet. Phase 6c's A/B harness will tell us
        whether the simpler "easy rows seen first" effect lifts F1
        on classification; if it does AND loss-curve diagnostics
        suggest easy-row overfitting, Phase 6e adds the strict
        two-stage schedule as a follow-up.
      - The trainer must run with shuffle disabled so the easy-first
        ordering is preserved. The caller (``training_service``)
        passes ``curriculum_disable_shuffle=True`` through the
        training config; ``train.py`` reads it and swaps the
        Trainer's default ``RandomSampler`` for a ``SequentialSampler``.

    Returns a manifest with paths + counts so the training_service
    can persist a curriculum block under ``runtime_config`` (for
    observability + the UI's "this run used curriculum" badge).

    Raises ``CurriculumUnavailable`` on the same conditions as
    ``rank_rows`` (empty input, unknown scoring mode, missing
    sentence-transformers).
    """
    if not rows:
        raise CurriculumUnavailable(
            "Cannot build curriculum shards from an empty row list."
        )
    ranked = rank_rows(
        rows,
        scoring_mode=scoring_mode,
        group_key=group_key,
        text_keys=text_keys,
        embedder=embedder,
        cache_dir=cache_dir,
    )

    # Match each ranked entry back to its source row. Falls back to
    # 0-indexed position when the row lacks an ``id`` (mirrors the
    # rank_rows behavior).
    rows_by_key: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        rid = row.get("id") if isinstance(row.get("id"), (int, str)) else idx
        rows_by_key[f"id:{rid}"] = row

    # Sort ranked entries ascending by difficulty → easiest first.
    ranked_sorted = sorted(ranked, key=lambda entry: entry["rank"])
    total = len(ranked_sorted)
    # Bottom-50% — round up so a 3-row project still gets 2 easy
    # rows in the warmup shard. Single-row projects fall through
    # with easy_count = 1 (= full set; curriculum is a no-op but the
    # shard is still well-formed).
    easy_count = max(1, (total + 1) // 2)
    easy_entries = ranked_sorted[:easy_count]
    full_entries = ranked_sorted  # full set; preserves easy-first ordering for non-shuffled trainer.

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / "train.curriculum.jsonl"
    meta_path = output_dir / "train.curriculum_meta.json"

    with shard_path.open("w", encoding="utf-8") as f:
        # Pass 1: easy half (each row once).
        for entry in easy_entries:
            row = rows_by_key.get(f"id:{entry['row_id']}")
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Pass 2: the full set, easy → hard. After pass 1 the easy
        # rows have been seen once; in pass 2 they show up again
        # (so the bottom half gets 2x exposure), then the hard rows
        # are seen for the first time near the end. This is the
        # ordered concatenation Phase 6c will A/B against uniform
        # training.
        for entry in full_entries:
            row = rows_by_key.get(f"id:{entry['row_id']}")
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write the ranking metadata for observability + debugging. JSON
    # so a human can `cat` it. Small enough (~80 bytes / row) that
    # we keep the full ranking, not a truncated preview.
    meta_payload = {
        "scoring_mode": scoring_mode,
        "total_rows": total,
        "easy_count": easy_count,
        "ranked": ranked_sorted,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return {
        "scoring_mode": scoring_mode,
        "shard_path": str(shard_path),
        "meta_path": str(meta_path),
        "total_rows": total,
        "easy_count": easy_count,
        "full_count": total,
    }


def _sentence_transformer_embedder(texts: list[str]) -> list[list[float]]:
    """Default embedder. Hard-fails when sentence-transformers isn't
    installed (unlike dataset_intelligence_service, which falls back
    to hashing for its diversity heuristic — curriculum rankings
    built on hashing would be essentially random)."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
    except ImportError as e:
        raise CurriculumUnavailable(
            "Curriculum ranking needs `sentence-transformers` (≈ 22MB "
            "for all-MiniLM-L6-v2). Install it with "
            "`pip install sentence-transformers` and retry. The "
            "hashing-fallback used elsewhere in the platform would "
            "produce essentially random rankings here, so we don't "
            "silently degrade."
        ) from e
    model = SentenceTransformer(_DEFAULT_EMBEDDING_MODEL)
    matrix = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors: list[list[float]] = []
    for row in matrix:
        if hasattr(row, "tolist"):
            vectors.append([float(v) for v in row.tolist()])
        else:
            vectors.append([float(v) for v in row])
    return vectors
