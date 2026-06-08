"""Quality-Lift phase 2, slice 1 — Project slice-definitions validator.

Users define named subsets of eval rows via JSON predicates. Phase 2
slice 2 wires these into the eval handler base so every metric is
also reported per-slice; phase 2 slice 3 makes them gateable
(``slice_name: "long_input"`` or aggregate ``operator: "worst_slice_gte"``).

This service is the schema gatekeeper. **No string expression DSL**:
predicates are JSON-only and the operator set is closed, so there is
no eval-injection surface and the editor UI can render a complete
op picker.

Design constraints locked with the user (2026-06-08):
  - Storage: ``Project.slice_definitions`` JSON column (not pack-scoped).
    Slices describe user-data shape — one slice ("long inputs") often
    applies across multiple packs.
  - Predicate grammar: closed JSON ops; AND across clauses within a
    slice; OR is achieved by defining multiple slices.
  - Auto-derived row fields are deliberately a small fixed set —
    ``input_length``, ``input_token_count``, ``prediction_length``,
    ``reference_length``, ``latency_ms``, ``_dataset_index``. Richer
    slicing requires the dataset row to carry the field; the slice
    layer is not a derivation DSL.
  - Per-project cap: 20 slices. Each slice adds linear cost to eval.
  - Slice support floor (gate-time): default 5 rows. Tiny slices have
    too much noise to gate on — set when defining worst-slice gates in
    slice 3.
"""

from __future__ import annotations

import re
from typing import Any


# Closed set. Adding ops here is a deliberate decision — the editor +
# slice 2 evaluator + slice 3 gate suggestions all read this tuple.
SLICE_OPERATORS: tuple[str, ...] = (
    "eq", "neq",
    "gt", "gte", "lt", "lte",
    "in", "not_in",
    "contains",
    "regex",
    "exists",
)

# Ops that compare numerically — value must be a number.
_NUMERIC_OPS = frozenset({"gt", "gte", "lt", "lte"})
# Ops that take a list value.
_LIST_OPS = frozenset({"in", "not_in"})
# Ops that take a string value (substring / regex).
_STRING_OPS = frozenset({"contains", "regex"})
# Ops that take a boolean — exists checks field presence + non-null.
_BOOL_OPS = frozenset({"exists"})

# slice_id grammar: lowercase ASCII, must start with a letter, ≤64 chars.
# Stricter than per-class label normalization (which accepts arbitrary
# strings) because slice_ids appear in metric keys like
# ``per_slice.<slice_id>.f1`` and gate metric_ids must be writable in
# the YAML pack contract without escaping.
_SLICE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

MAX_SLICES_PER_PROJECT = 20
MAX_CLAUSES_PER_SLICE = 8  # AND-only; if you need 9, restructure.
MAX_FIELD_PATH_DEPTH = 4   # dot-path resolver depth on the row dict.


class SliceValidationError(ValueError):
    """Raised when slice_definitions payload is malformed. The first
    error wins — the API surfaces this directly so the editor can show
    a precise inline error rather than a fuzzy "invalid input."
    """


def _normalize_clauses(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise SliceValidationError("`where` must be a non-empty list of clauses")
    if len(raw) > MAX_CLAUSES_PER_SLICE:
        raise SliceValidationError(
            f"`where` has {len(raw)} clauses (cap is {MAX_CLAUSES_PER_SLICE}). "
            "Define a second slice if you need OR semantics."
        )
    return [c for c in raw if isinstance(c, dict)] if any(
        isinstance(c, dict) for c in raw
    ) else []


def _validate_field_path(path: Any, *, slice_id: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise SliceValidationError(
            f"slice `{slice_id}`: clause is missing a string `field`"
        )
    cleaned = path.strip()
    parts = cleaned.split(".")
    if len(parts) > MAX_FIELD_PATH_DEPTH:
        raise SliceValidationError(
            f"slice `{slice_id}`: field path `{cleaned}` exceeds depth "
            f"{MAX_FIELD_PATH_DEPTH} (use a flatter dataset shape)"
        )
    for segment in parts:
        if not segment or not re.match(r"^[A-Za-z_][A-Za-z0-9_\-]*$", segment):
            raise SliceValidationError(
                f"slice `{slice_id}`: field path segment `{segment}` is not a valid identifier"
            )
    return cleaned


def _validate_clause_value(op: str, value: Any, *, slice_id: str) -> Any:
    """Per-op value type validation. The clause as a whole stays a dict,
    but we return the (possibly normalized) value the eval-time
    matcher will receive.
    """
    if op in _BOOL_OPS:
        # `exists`: value is a bool (presence-required vs. presence-forbidden).
        # Default to True if omitted so {"field": "x", "op": "exists"} reads
        # as "field x must exist."
        if value is None:
            return True
        if not isinstance(value, bool):
            raise SliceValidationError(
                f"slice `{slice_id}`: `exists` op takes a boolean value"
            )
        return value
    if op in _NUMERIC_OPS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SliceValidationError(
                f"slice `{slice_id}`: op `{op}` requires a numeric value, got {type(value).__name__}"
            )
        return float(value)
    if op in _LIST_OPS:
        if not isinstance(value, list) or not value:
            raise SliceValidationError(
                f"slice `{slice_id}`: op `{op}` requires a non-empty list of values"
            )
        # Each item must be a JSON scalar — keeps the matcher simple.
        for item in value:
            if not isinstance(item, (str, int, float, bool)) and item is not None:
                raise SliceValidationError(
                    f"slice `{slice_id}`: op `{op}` list values must be scalars"
                )
        return list(value)
    if op in _STRING_OPS:
        if not isinstance(value, str) or not value:
            raise SliceValidationError(
                f"slice `{slice_id}`: op `{op}` requires a non-empty string value"
            )
        if op == "regex":
            try:
                re.compile(value)
            except re.error as exc:
                raise SliceValidationError(
                    f"slice `{slice_id}`: regex `{value}` is invalid: {exc}"
                ) from exc
        return value
    # eq / neq: any JSON scalar (str, number, bool, None).
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise SliceValidationError(
            f"slice `{slice_id}`: op `{op}` value must be a JSON scalar "
            f"(string/number/bool/null), got {type(value).__name__}"
        )
    return value


def _validate_clause(raw: dict, *, slice_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SliceValidationError(
            f"slice `{slice_id}`: each clause must be an object"
        )
    field = _validate_field_path(raw.get("field"), slice_id=slice_id)
    op = str(raw.get("op") or "").strip().lower()
    if op not in SLICE_OPERATORS:
        raise SliceValidationError(
            f"slice `{slice_id}`: unknown op `{op}` "
            f"(allowed: {', '.join(SLICE_OPERATORS)})"
        )
    value = _validate_clause_value(op, raw.get("value"), slice_id=slice_id)
    return {"field": field, "op": op, "value": value}


def _validate_slice(raw: dict, *, seen_ids: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SliceValidationError("each slice must be an object")
    slice_id_raw = raw.get("slice_id")
    if not isinstance(slice_id_raw, str) or not slice_id_raw.strip():
        raise SliceValidationError("slice is missing `slice_id`")
    slice_id = slice_id_raw.strip()
    if not _SLICE_ID_RE.match(slice_id):
        raise SliceValidationError(
            f"slice_id `{slice_id}` must match {_SLICE_ID_RE.pattern} "
            "(lowercase ASCII, starts with a letter, ≤64 chars, no dots/spaces)"
        )
    if slice_id in seen_ids:
        raise SliceValidationError(f"duplicate slice_id `{slice_id}`")
    seen_ids.add(slice_id)

    display_raw = raw.get("display_name")
    display_name = (display_raw or "").strip() if isinstance(display_raw, str) else ""
    if display_name and len(display_name) > 128:
        raise SliceValidationError(
            f"slice `{slice_id}`: display_name exceeds 128 chars"
        )

    clauses_raw = raw.get("where")
    if not isinstance(clauses_raw, list) or not clauses_raw:
        raise SliceValidationError(
            f"slice `{slice_id}`: `where` must be a non-empty list"
        )
    if len(clauses_raw) > MAX_CLAUSES_PER_SLICE:
        raise SliceValidationError(
            f"slice `{slice_id}`: where has {len(clauses_raw)} clauses "
            f"(cap is {MAX_CLAUSES_PER_SLICE}); split into multiple slices"
        )
    clauses = [_validate_clause(c, slice_id=slice_id) for c in clauses_raw]

    return {
        "slice_id": slice_id,
        "display_name": display_name or slice_id,
        "where": clauses,
    }


def validate_slice_definitions(payload: Any) -> dict[str, Any]:
    """Validate + normalize a slice_definitions payload.

    Returns the cleaned dict ready to persist on ``Project.slice_definitions``.
    Idempotent: passing a previously-validated payload returns an equal
    dict (good for the PUT-then-GET round-trip).

    Empty / None / ``{"slices": []}`` is a valid "no slices configured"
    state — the project drops back to overall-metric-only behavior.
    """
    if payload is None:
        return {"slices": []}
    if not isinstance(payload, dict):
        raise SliceValidationError("slice_definitions must be an object")
    slices_raw = payload.get("slices")
    if slices_raw is None:
        return {"slices": []}
    if not isinstance(slices_raw, list):
        raise SliceValidationError("`slices` must be a list")
    if len(slices_raw) > MAX_SLICES_PER_PROJECT:
        raise SliceValidationError(
            f"too many slices: {len(slices_raw)} (cap is {MAX_SLICES_PER_PROJECT}). "
            "Each slice adds linear cost to every eval."
        )

    seen_ids: set[str] = set()
    cleaned = [_validate_slice(item, seen_ids=seen_ids) for item in slices_raw]
    return {"slices": cleaned}
