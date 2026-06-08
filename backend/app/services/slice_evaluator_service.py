"""Quality-Lift phase 2, slice 2 — Slice predicate evaluator + handler wrapper.

The predicate side of slice 1's slice_definitions service. Given a
validated slice_definitions payload + a list of prediction dicts,
bucket the predictions into named subsets, then run any TaskHandler's
``score`` method on each subset to produce per-slice metrics.

Design (locked with the user in phase 2 design walkthrough):
  - No string-DSL eval. Predicates dispatch via a closed op set; the
    grammar matches ``slice_definitions_service.SLICE_OPERATORS``.
  - Platform-computed fields (``input_length``, ``prediction_length``,
    ``reference_length``, ``latency_ms``, ``_dataset_index``) are
    injected on every row before predicate matching so the user can
    slice on shape-derived signals without dataset-level changes.
    ``input_token_count`` is best-effort word-count when no tokenizer
    is wired (slice 3 will plumb the real tokenizer through ctx).
  - Predicate resolution falls back through the row's own dict, then
    its ``extras`` / metadata children. Single-level dot-paths only —
    ``metadata.source``, not ``metadata.tags[0]``.
  - ``score_with_slices`` reuses the handler's own ``score`` method on
    each subset; no per-handler changes required. The 9-or-so handlers
    (Classification, QA, Extraction, RAG, Safety, …) all get
    per-slice metrics for free because they already loop over the
    prediction list.
  - Empty subset → emit ``{"support": 0}`` rather than skipping the
    slice entirely. Silent skipping would mask "your slice matches 0
    rows" which is almost always a slice-definition bug worth
    surfacing in the UI.
  - Per-slice metrics land under ``metrics["per_slice"][<slice_id>]``,
    mirroring the per-class structure from Gap-#6 so the multi-seed
    aggregator from phase 1 walks it transparently (``compute_variance_stats``
    already recurses through nested dicts — phase 2 + phase 1 compose
    at the aggregator without aggregator changes).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable


# ────────────────────────────────────────────────────────────────────────
# Platform-computed field injection
# ────────────────────────────────────────────────────────────────────────


def _input_length_chars(row: dict[str, Any]) -> int:
    """Char length of the row's input. Handlers normalize to ``prompt``
    in the prediction shape, but some legacy paths still write ``input``
    or ``question``; we check all three to keep slicing robust against
    handler-version drift."""
    raw = (
        row.get("prompt")
        or row.get("input")
        or row.get("question")
        or ""
    )
    return len(str(raw))


def _output_length_chars(row: dict[str, Any], key: str) -> int:
    value = row.get(key) or ""
    return len(str(value))


def _input_token_count(row: dict[str, Any], tokenizer: Any | None) -> int:
    """Best-effort token count.

    With a tokenizer wired (slice 3 plumbs ctx.tokenizer), use its
    ``encode`` call. Without one, fall back to whitespace word count —
    not great, but predictable and ordering-correct for length-based
    slicing (longer inputs → more "tokens"). Documented in the platform
    field table so the user knows what they're getting.
    """
    raw = (
        row.get("prompt")
        or row.get("input")
        or row.get("question")
        or ""
    )
    text = str(raw)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(text.split()) if text else 0


def inject_platform_fields(
    row: dict[str, Any],
    *,
    index: int,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Return a new dict with ``input_length``, ``input_token_count``,
    ``prediction_length``, ``reference_length``, ``latency_ms``,
    ``_dataset_index`` added under the canonical keys the slice
    predicates can resolve. The original row is not mutated — the
    eval-time prediction list lives across multiple downstream
    summaries / details payloads, and silently mutating it would
    surprise downstream code.
    """
    enriched = dict(row)
    enriched.setdefault("input_length", _input_length_chars(row))
    enriched.setdefault("input_token_count", _input_token_count(row, tokenizer))
    enriched.setdefault("prediction_length", _output_length_chars(row, "prediction"))
    enriched.setdefault("reference_length", _output_length_chars(row, "reference"))
    # ``latency_ms`` is already on the row when inference ran through the
    # generation path. Don't fabricate a value when it's missing —
    # ``exists`` predicates would lie if we did.
    if "latency_ms" not in enriched and "latency_ms" in row:
        enriched["latency_ms"] = row["latency_ms"]
    enriched.setdefault("_dataset_index", index)
    return enriched


# ────────────────────────────────────────────────────────────────────────
# Field-path resolution + clause matching
# ────────────────────────────────────────────────────────────────────────


def _resolve_path(row: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Resolve a dot-path against the row dict. Returns ``(found, value)``.

    ``found=False`` is the only honest signal for ``exists: false``
    predicates and for "missing metric" semantics — a None value is
    NOT the same as a missing field (user might genuinely have
    ``language=None`` rows they want to slice on).
    """
    parts = path.split(".")
    current: Any = row
    for segment in parts:
        if isinstance(current, dict) and segment in current:
            current = current[segment]
            continue
        return False, None
    return True, current


_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda lhs, rhs: lhs == rhs,
    "neq": lambda lhs, rhs: lhs != rhs,
    # Numeric ops require lhs to be a number; protect against the
    # comparisons-between-different-types blow-up by failing closed
    # (predicate doesn't match) when lhs isn't numeric. The validator
    # already enforces rhs is numeric, so only the row-side check
    # matters here.
    "gt": lambda lhs, rhs: isinstance(lhs, (int, float)) and not isinstance(lhs, bool) and lhs > rhs,
    "gte": lambda lhs, rhs: isinstance(lhs, (int, float)) and not isinstance(lhs, bool) and lhs >= rhs,
    "lt": lambda lhs, rhs: isinstance(lhs, (int, float)) and not isinstance(lhs, bool) and lhs < rhs,
    "lte": lambda lhs, rhs: isinstance(lhs, (int, float)) and not isinstance(lhs, bool) and lhs <= rhs,
    "in": lambda lhs, rhs: lhs in rhs,
    "not_in": lambda lhs, rhs: lhs not in rhs,
    # contains is case-insensitive substring on string-coerced lhs;
    # users almost always mean "contains" with case-folded semantics.
    "contains": lambda lhs, rhs: isinstance(lhs, str) and rhs.lower() in lhs.lower(),
}


def _match_clause(row: dict[str, Any], clause: dict[str, Any]) -> bool:
    field = clause["field"]
    op = clause["op"]
    value = clause["value"]

    found, lhs = _resolve_path(row, field)
    # ``exists`` is the one op that hinges on field presence rather
    # than field value, so it gets handled separately.
    if op == "exists":
        # ``exists: true`` means the field is present AND non-null.
        present_and_non_null = found and lhs is not None
        return present_and_non_null if value else not present_and_non_null

    # Every other op fails closed when the field is missing — the
    # caller has no meaningful comparison to perform on absence.
    if not found:
        return False

    if op == "regex":
        # The validator compiled this at write time, so re.error here
        # would be a real bug; fail closed.
        if not isinstance(lhs, str):
            return False
        try:
            return re.search(value, lhs) is not None
        except re.error:
            return False

    matcher = _OPS.get(op)
    if matcher is None:
        # Unknown ops should never land here — the validator rejected
        # them at write time. Fail closed.
        return False
    try:
        return bool(matcher(lhs, value))
    except TypeError:
        return False


def _matches_all(row: dict[str, Any], clauses: list[dict[str, Any]]) -> bool:
    return all(_match_clause(row, c) for c in clauses)


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────


def apply_slices(
    predictions: list[dict[str, Any]],
    slice_definitions: list[dict[str, Any]] | None,
    *,
    tokenizer: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Bucket predictions by slice definition.

    Returns ``{slice_id: [matching predictions]}`` — empty list when
    no row matches, NOT omitted. Slice 2's contract: the handler's
    per-slice metric write emits ``{"support": 0}`` for those, so the
    UI can render "this slice matched 0 rows" rather than silently
    dropping it.

    ``slice_definitions`` is expected to be the ``slices`` list from
    ``Project.slice_definitions`` (i.e. already validated by
    slice_definitions_service.validate_slice_definitions). Passing
    ``None`` or empty list returns ``{}`` — no-op fast path so
    single-handler eval code doesn't pay any cost when slicing is off.
    """
    if not slice_definitions:
        return {}
    enriched = [
        inject_platform_fields(row, index=i, tokenizer=tokenizer)
        for i, row in enumerate(predictions)
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for slice_def in slice_definitions:
        slice_id = slice_def["slice_id"]
        clauses = slice_def["where"]
        out[slice_id] = [row for row in enriched if _matches_all(row, clauses)]
    return out


def score_with_slices(
    handler: Any,
    predictions: list[dict[str, Any]],
    ctx: Any,
    *,
    slice_definitions: list[dict[str, Any]] | None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Wrap any TaskHandler.score call to also produce per-slice metrics.

    Always returns the same overall-metric dict the handler would
    produce on its own; when ``slice_definitions`` is non-empty,
    adds a ``per_slice: {slice_id: {<handler metrics>}}`` key.

    Critical: empty subsets emit ``{"support": 0}`` rather than
    triggering the handler's "no predictions" edge case (which
    different handlers handle differently — some return zeros, some
    return None). The ``support`` semantics match how
    ClassificationHandler reports per-class support; the slice 3 gate
    evaluator uses this to honor a ``min_slice_support`` threshold
    when computing worst-slice gates.

    No handler signature change: the handler's existing
    ``score(subset, ctx)`` runs on each bucket as-is. If a handler
    happens to look at ``len(predictions)`` for support, it gets the
    subset length — exactly what we want.
    """
    overall = dict(handler.score(predictions, ctx))
    if not slice_definitions:
        return overall

    buckets = apply_slices(predictions, slice_definitions, tokenizer=tokenizer)
    per_slice: dict[str, dict[str, Any]] = {}
    for slice_id, subset in buckets.items():
        if not subset:
            per_slice[slice_id] = {"support": 0}
            continue
        slice_metrics = dict(handler.score(subset, ctx))
        # Carry support explicitly even when the handler emitted its
        # own ``total`` / ``support`` counts — slice 3's worst-slice
        # gate reads ``support`` uniformly across handlers, so a
        # canonical key matters more than handler-specific synonyms.
        slice_metrics.setdefault("support", len(subset))
        per_slice[slice_id] = slice_metrics
    overall["per_slice"] = per_slice
    return overall


# Re-export for callers that want to enumerate the field menu for the
# slice editor without pulling in the whole module.
PLATFORM_FIELDS: tuple[tuple[str, str], ...] = (
    ("input_length", "Char count of the prompt/input."),
    ("input_token_count", "Token count via the eval tokenizer (word count fallback)."),
    ("prediction_length", "Char count of the model's prediction."),
    ("reference_length", "Char count of the gold reference."),
    ("latency_ms", "Per-row inference latency (when available)."),
    ("_dataset_index", "Row position in the eval set; useful for sampling slices."),
)
