"""Gate-options catalog for the eval-pack scaffold editor (Gap #5 slice 1).

The scaffold editor in the UI lets a user add/remove/edit gates on a
project's eval pack. Today every tutorial documents the JSON-edit
workaround because there's no FE surface for the underlying choices
(which metrics are valid? which operators does the eval engine
actually support? which metric_ids are recommended for *this* recipe?).
This module answers those questions in one place so:

  * The frontend dropdowns are populated from a single source of truth.
  * The server-side validator (``validate_draft_pack_gates``) rejects
    malformed gates with specific error codes instead of silently
    defaulting to ``gte`` or trusting unknown metric_ids.

The catalog is intentionally derived from the existing
``_BASE_METRIC_SCHEMA`` and ``_RECIPE_SCAFFOLD`` rather than maintained
as a parallel list — that way a metric added to the eval engine flows
into the editor automatically.
"""

from __future__ import annotations

from typing import Any

from app.services.evaluation_pack_service import _BASE_METRIC_SCHEMA


# Operators the eval engine actually implements. The engine silently
# defaults anything else to ``gte`` (see
# ``evaluation_pack_service._evaluate_gate``); we surface the whitelist
# here so the validator can return a real error instead.
GATE_OPERATORS: list[dict[str, str]] = [
    {"value": "gte", "label": "≥ (at least)"},
    {"value": "lte", "label": "≤ (at most)"},
]
VALID_GATE_OPERATORS: set[str] = {op["value"] for op in GATE_OPERATORS}


# Metrics where lower-is-better, so the natural default operator is
# ``lte`` instead of ``gte``. Sourced from each metric's documented
# semantics in ``_BASE_METRIC_SCHEMA``.
LTE_DEFAULT_METRIC_IDS: set[str] = {"hallucination_rate"}


# Gap-#5 slice 3 — reason_code → ordered list of candidate (metric_id,
# operator, default_threshold) tuples. The adopt-gate-from-cluster
# action walks this list and picks the first candidate whose
# metric_id is in the recipe's recommended set (or the catalog at
# all); the threshold is a sensible starter the user can tighten in
# the editor. Reason-code vocabulary mirrors
# ``evaluation_remediation_service`` (safety_failure / hallucination /
# coverage_gap / formatting_mismatch).
REASON_CODE_GATE_SUGGESTIONS: dict[str, list[tuple[str, str, float]]] = {
    "safety_failure": [
        # The canonical lever for safety regressions — bump the gate so
        # any future drop below the chosen bar blocks ship.
        ("safety_pass_rate", "gte", 0.95),
    ],
    "hallucination": [
        # rag-protocol recipes track hallucination_rate directly (lower
        # is better); other recipes fall back to groundedness or judge.
        ("hallucination_rate", "lte", 0.05),
        ("groundedness", "gte", 0.85),
        ("citation_rate", "gte", 0.80),
        ("llm_judge_pass_rate", "gte", 0.75),
    ],
    "coverage_gap": [
        # Coverage gaps usually surface as F1/recall drops. Start with
        # the recipe's headline metric — the editor's "recommended"
        # list narrows this down per recipe.
        ("f1", "gte", 0.60),
        ("macro_f1", "gte", 0.65),
        ("span_set_recall", "gte", 0.60),
        ("llm_judge_pass_rate", "gte", 0.70),
    ],
    "formatting_mismatch": [
        # Format drift — the rag-protocol recipe has a dedicated metric
        # (slice-2 implementation); other recipes use exact_match as a
        # proxy for stable formatting.
        ("format_consistency", "gte", 0.90),
        ("exact_match", "gte", 0.55),
    ],
}


def suggest_gate_for_reason_code(
    reason_code: str,
    *,
    recipe_id: str | None = None,
) -> dict[str, Any] | None:
    """Pick a starter gate for a failure cluster's reason_code.

    Walks ``REASON_CODE_GATE_SUGGESTIONS[reason_code]`` and returns the
    first candidate whose metric_id is in the recipe's recommended
    metric set; falls back to the first candidate whose metric_id is
    in ``known_metric_ids()`` at all. Returns ``None`` if reason_code
    is unknown and no fallback candidate matches — the caller surfaces
    that as "no obvious gate for this cluster — add one manually".

    The returned dict matches the scaffolder's gate shape (gate_id,
    metric_id, operator, threshold, required) so callers can append
    it to a task_spec's gate list verbatim. ``required=False`` so the
    new gate doesn't break a passing eval the moment it's added —
    the user can promote it after watching it for a run or two.
    """
    candidates = REASON_CODE_GATE_SUGGESTIONS.get(
        (reason_code or "").strip().lower(), [],
    )
    if not candidates:
        return None

    recommended = set(
        _RECOMMENDED_METRIC_IDS_PER_RECIPE.get(
            (recipe_id or "").strip().lower(), [],
        )
    )
    known = known_metric_ids()

    pick: tuple[str, str, float] | None = None
    # Prefer a candidate the recipe recommends.
    for entry in candidates:
        if entry[0] in recommended:
            pick = entry
            break
    if pick is None:
        for entry in candidates:
            if entry[0] in known:
                pick = entry
                break
    if pick is None:
        return None

    metric_id, operator, threshold = pick
    prefix = "max" if operator == "lte" else "min"
    return {
        "gate_id": f"{prefix}_{metric_id}_from_cluster",
        "metric_id": metric_id,
        "operator": operator,
        "threshold": float(threshold),
        "required": False,
    }


# Recipe → list of recommended metric_ids. Mirrors the
# ``required_metric_ids`` in ``_RECIPE_SCAFFOLD`` so the editor's
# "recommended" badges match what the scaffolder pre-populates.
# Kept here (not imported from the scaffold service) so the catalog
# stays a flat data file the validator can import without pulling in
# the scaffold service's gold-set query path.
_RECOMMENDED_METRIC_IDS_PER_RECIPE: dict[str, list[str]] = {
    "classification": ["macro_f1", "accuracy", "safety_pass_rate"],
    "span-extraction": [
        "span_set_f1", "span_set_precision", "span_set_recall", "safety_pass_rate",
    ],
    "summarization": ["rouge_l", "groundedness", "safety_pass_rate"],
    "qa-sft": ["exact_match", "f1", "llm_judge_pass_rate", "safety_pass_rate"],
    "generic-sft": ["exact_match", "f1", "llm_judge_pass_rate", "safety_pass_rate"],
    "code-review": ["llm_judge_pass_rate", "f1", "safety_pass_rate"],
    "rag-protocol": [
        "citation_rate", "hallucination_rate",
        "appropriate_refusal_rate", "format_consistency",
        "safety_pass_rate",
    ],
}


def _metric_label(metric_id: str) -> str:
    """Render a human label from a snake_case metric_id."""
    return metric_id.replace("_", " ").replace(".", " · ").title()


def build_gate_options(recipe_id: str | None = None) -> dict[str, Any]:
    """Return the catalog payload the frontend gate editor reads.

    Shape:
      {
        "operators": [{"value": "gte", "label": "≥ (at least)"}, ...],
        "metrics": [
          {
            "metric_id": str,
            "label": str,
            "description": str,
            "expected_range": [float, float],
            "default_operator": "gte" | "lte",
            "recommended": bool,
          },
          ...
        ],
        "recipe_id": str | None,
      }

    All metric_ids in ``_BASE_METRIC_SCHEMA`` are returned. Entries
    flagged ``recommended=True`` are the ones the scaffolder would
    have pre-filled for this recipe — the UI uses that to badge or
    sort them to the top.
    """
    recommended = set(
        _RECOMMENDED_METRIC_IDS_PER_RECIPE.get(
            (recipe_id or "").strip().lower(), [],
        )
    )
    metrics: list[dict[str, Any]] = []
    for metric_id, spec in sorted(_BASE_METRIC_SCHEMA.items()):
        metrics.append(
            {
                "metric_id": metric_id,
                "label": _metric_label(metric_id),
                "description": str(spec.get("description") or ""),
                "expected_range": list(spec.get("expected_range") or [0.0, 1.0]),
                "default_operator": "lte" if metric_id in LTE_DEFAULT_METRIC_IDS else "gte",
                "recommended": metric_id in recommended,
            }
        )
    return {
        "operators": list(GATE_OPERATORS),
        "metrics": metrics,
        "recipe_id": (recipe_id or "").strip().lower() or None,
    }


import re as _re


# Gap #6 slice 1 — per-class metric ID patterns the validator should
# accept even though the class names are project-specific and can't
# be enumerated up-front. The classification handler emits per-class
# precision/recall/f1/support flattened by ``_flatten_per_class_metrics``
# into both ``precision_<label>`` short keys and ``per_class.<label>.<m>``
# dot-paths; the editor + the validator both have to know about these
# shapes so users picking ``precision_benign`` from the per-class
# dropdown don't trip ``unknown_metric_id``.
_PER_CLASS_METRIC_PREFIXES: tuple[str, ...] = ("precision_", "recall_", "f1_", "support_")
_PER_CLASS_DOT_PATTERN = _re.compile(r"^per_class\.[a-z0-9_]+\.(precision|recall|f1|support)$")


def is_per_class_metric_id(metric_id: str) -> bool:
    """True when ``metric_id`` matches a per-class shape — either the
    short form (``precision_<label>``) or the dot-path form
    (``per_class.<label>.precision``). Both are emitted by the
    flattener so the validator accepts both.
    """
    if not metric_id:
        return False
    token = metric_id.strip().lower()
    if any(token.startswith(p) and len(token) > len(p) for p in _PER_CLASS_METRIC_PREFIXES):
        return True
    if _PER_CLASS_DOT_PATTERN.match(token):
        return True
    return False


def known_metric_ids() -> set[str]:
    """Set of metric_ids the gate validator accepts.

    Unions the keys of ``_BASE_METRIC_SCHEMA`` with the metric_ids
    referenced by the recipe scaffolder. Several scaffolder recipes
    (classification, span-extraction, summarization) emit metrics
    like ``rouge_l`` / ``span_set_f1`` / ``min_per_class_f1`` that
    the eval engine accepts via its [0, 1]-range fallback but that
    aren't in the base schema. Without this union the validator
    would reject the scaffolder's own output as ``unknown_metric_id``,
    which would be a hilarious bug.

    Lazy import of ``_RECIPE_SCAFFOLD`` keeps this module decoupled
    from ``eval_pack_scaffold_service`` (which imports back into
    here from ``save_scaffolded_pack``) — avoids a module-import
    cycle while still pulling the canonical metric list.
    """
    from app.services.eval_pack_scaffold_service import _RECIPE_SCAFFOLD

    out = set(_BASE_METRIC_SCHEMA.keys())
    for spec in _RECIPE_SCAFFOLD.values():
        for metric_id in spec.get("required_metric_ids", []):
            if isinstance(metric_id, str) and metric_id.strip():
                out.add(metric_id.strip().lower())
        for entry in spec.get("gates", []):
            # Scaffolder gate tuples: (gate_id, metric_id, threshold, required).
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                metric_id = entry[1]
                if isinstance(metric_id, str) and metric_id.strip():
                    out.add(metric_id.strip().lower())
    return out


async def build_per_class_metric_options(
    db: Any,
    *,
    project_id: int,
) -> dict[str, Any]:
    """Gap #6 slice 1 — discover the per-class metric IDs the project's
    classification handler is currently emitting, so the editor's gate
    dropdown can group them under a "Per-class" section.

    Reads the latest classification-style eval result for the project
    and pulls ``metrics["per_class"]`` keys (class labels). For each
    label, returns the three short-form metric IDs the flattener emits
    (``precision_<label>``, ``recall_<label>``, ``f1_<label>``) so the
    editor doesn't have to know about the dot-path form unless the
    user types it manually.

    Returns:
      {
        "classes": [str, ...],            # discovered class labels
        "metrics": [
          {"metric_id": "precision_benign", "label": "Precision · benign",
           "default_operator": "gte", "class_name": "benign",
           "metric_kind": "precision"},
          ...
        ],
        "source_eval_result_id": int | None,
      }

    When the project has no eval results yet, returns an empty
    ``classes`` + ``metrics`` list with ``source_eval_result_id=None``;
    the FE renders that as "Run an eval first to discover classes".
    """
    from sqlalchemy import select
    from app.models.experiment import EvalResult, Experiment

    # Latest eval result for the project, in any classification-shaped
    # eval_type. We don't filter by eval_type because the per_class
    # dict is what we key on — any handler that emits it is fair game.
    rows = await db.execute(
        select(EvalResult)
        .join(Experiment, Experiment.id == EvalResult.experiment_id)
        .where(Experiment.project_id == project_id)
        .order_by(EvalResult.created_at.desc(), EvalResult.id.desc())
        .limit(50)
    )

    class_labels: list[str] = []
    source_eval_result_id: int | None = None
    for row in rows.scalars().all():
        metrics_dict = row.metrics if isinstance(row.metrics, dict) else {}
        per_class = metrics_dict.get("per_class")
        if not isinstance(per_class, dict) or not per_class:
            continue
        # Stable order: alphabetical class names so the dropdown
        # doesn't reshuffle between renders.
        class_labels = sorted(
            str(label).strip()
            for label in per_class.keys()
            if str(label).strip()
        )
        source_eval_result_id = int(row.id)
        break

    metrics_out: list[dict[str, Any]] = []
    for label in class_labels:
        normalized_label = label.lower().replace(" ", "_").replace("-", "_")
        for kind, default_op in (
            ("precision", "gte"),
            ("recall", "gte"),
            ("f1", "gte"),
        ):
            metric_id = f"{kind}_{normalized_label}"
            metrics_out.append({
                "metric_id": metric_id,
                "label": f"{kind.title()} · {label}",
                "description": (
                    f"Per-class {kind} for class '{label}' "
                    f"(flattened from per_class.{label}.{kind} on the "
                    f"latest classification eval result)."
                ),
                "default_operator": default_op,
                "expected_range": [0.0, 1.0],
                "class_name": label,
                "metric_kind": kind,
                "recommended": False,
            })

    return {
        "classes": class_labels,
        "metrics": metrics_out,
        "source_eval_result_id": source_eval_result_id,
    }


def validate_draft_pack_gates(draft_pack: dict[str, Any]) -> None:
    """Walk every task_spec's gate list and raise on the first
    malformed gate. Codes are stable so the API layer can map them
    to specific 400 messages and the FE can highlight the bad row.

    Codes raised (all via ``ValueError``):
      * ``invalid_gate_shape`` — a gate entry isn't a dict.
      * ``missing_gate_id`` — gate has no gate_id or it's empty.
      * ``duplicate_gate_id:<id>`` — two gates share a gate_id within
        the same task_spec.
      * ``missing_metric_id:<gate_id>`` — gate has no metric_id.
      * ``unknown_metric_id:<id>`` — metric_id isn't recognised by
        ``known_metric_ids()`` (union of base schema + scaffolder
        metrics). Specific so the FE can offer a suggestion picker.
      * ``invalid_gate_operator:<op>`` — operator isn't in
        ``VALID_GATE_OPERATORS`` (the engine only implements
        gte/lte).
      * ``missing_threshold:<gate_id>`` — threshold is None / unparseable.
      * ``threshold_out_of_range:<gate_id>`` — threshold falls outside
        the metric's expected_range.

    This function only validates the *shape* of gates; the larger
    ``save_scaffolded_pack`` flow handles missing task_specs and
    other pack-level shape issues.
    """
    task_specs = draft_pack.get("task_specs") if isinstance(draft_pack, dict) else None
    if not isinstance(task_specs, list):
        # Pack-level shape is the caller's responsibility — return
        # silently here so we don't double-raise on the same issue.
        return

    known = known_metric_ids()

    for task_spec in task_specs:
        if not isinstance(task_spec, dict):
            continue
        gates = task_spec.get("gates")
        if not isinstance(gates, list):
            continue

        seen_ids: set[str] = set()
        for gate in gates:
            if not isinstance(gate, dict):
                raise ValueError("invalid_gate_shape")

            gate_id = str(gate.get("gate_id") or "").strip()
            if not gate_id:
                raise ValueError("missing_gate_id")
            if gate_id in seen_ids:
                raise ValueError(f"duplicate_gate_id:{gate_id}")
            seen_ids.add(gate_id)

            metric_id = str(gate.get("metric_id") or "").strip().lower()
            if not metric_id:
                raise ValueError(f"missing_metric_id:{gate_id}")
            # Per-class metric IDs (precision_benign, per_class.attack.f1,
            # …) are accepted without needing a base-schema entry — the
            # class names are project-specific and surface via the
            # /per-class-metric-options endpoint, not the static catalog.
            if metric_id not in known and not is_per_class_metric_id(metric_id):
                raise ValueError(f"unknown_metric_id:{metric_id}")

            operator = str(gate.get("operator") or "gte").strip().lower()
            if operator not in VALID_GATE_OPERATORS:
                raise ValueError(f"invalid_gate_operator:{operator}")

            raw_threshold = gate.get("threshold")
            try:
                threshold = float(raw_threshold) if raw_threshold is not None else None
            except (TypeError, ValueError):
                threshold = None
            if threshold is None:
                raise ValueError(f"missing_threshold:{gate_id}")

            schema_entry = _BASE_METRIC_SCHEMA.get(metric_id) or {}
            expected_range = schema_entry.get("expected_range")
            if (
                isinstance(expected_range, list)
                and len(expected_range) == 2
                and not (expected_range[0] <= threshold <= expected_range[1])
            ):
                raise ValueError(f"threshold_out_of_range:{gate_id}")
