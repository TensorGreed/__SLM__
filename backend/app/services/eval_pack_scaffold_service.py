"""Recipe-aware evaluation-pack scaffolder (E5).

When a project has no `evaluation_preferred_pack_id` set, the eval-pack
picker can ask this service for a draft pack tailored to the project's
`selected_recipe.recipe_id` and gold-set shape. The user reviews the
draft inline + clicks "Use scaffold" to persist it on
`project.runtime_config["scaffolded_evaluation_pack"]`.

Pure-function: ``scaffold_pack(recipe_id, gold_set_summary)`` →
draft-pack dict that already round-trips through
``evaluation_pack_service._build_pack_contract`` so the eval engine
sees a fully-formed task-spec.

Per-recipe metric defaults:
  - classification     → macro_f1 + accuracy with class-balance gate
  - span-extraction    → span_set_f1 + per-row precision/recall
  - summarization      → rouge_l + groundedness
  - qa-sft, generic-sft, code-review, instruction_sft → exact_match + llm_judge
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.project import Project
from app.services.recipe_service import get_recipe


# Pack id reserved for scaffolded packs. Distinct from the builtin
# ids so callers can detect "this is a project-scoped scaffold" via
# the prefix without touching pack contents.
SCAFFOLDED_PACK_ID = "evalpack.project.scaffolded"


# Recipe → (task_profile, required_metric_ids, gates list).
# Each gate spec: (gate_id, metric_id, threshold, required).
_RECIPE_SCAFFOLD: dict[str, dict[str, Any]] = {
    "classification": {
        "task_profile": "classification",
        "display_name": "Classification",
        "required_metric_ids": ["macro_f1", "accuracy"],
        "gates": [
            # macro_f1 is the headline metric for classification — it's
            # the one that catches class-imbalance bugs that accuracy
            # alone misses (a 90/10 binary can hit 0.9 accuracy with
            # macro_f1=0.47).
            ("min_macro_f1", "macro_f1", 0.65, True),
            ("min_accuracy", "accuracy", 0.70, True),
            # Class-balance health check — gates per-class minimum
            # F1 so a single starved class can't ride on the rest.
            ("min_per_class_f1", "min_per_class_f1", 0.50, True),
            ("min_safety_pass_rate", "safety_pass_rate", 0.93, False),
        ],
    },
    "span-extraction": {
        "task_profile": "structured_extraction",
        "display_name": "Structured span extraction",
        "required_metric_ids": ["span_set_f1", "span_set_precision", "span_set_recall"],
        "gates": [
            ("min_span_set_f1", "span_set_f1", 0.65, True),
            ("min_span_set_precision", "span_set_precision", 0.70, True),
            ("min_span_set_recall", "span_set_recall", 0.60, True),
            ("min_safety_pass_rate", "safety_pass_rate", 0.93, False),
        ],
    },
    "summarization": {
        "task_profile": "summarization",
        "display_name": "Summarization",
        "required_metric_ids": ["rouge_l", "groundedness"],
        "gates": [
            ("min_rouge_l", "rouge_l", 0.30, True),
            # Groundedness catches summaries that fabricate content
            # — the failure mode summarization gold sets are built
            # to test.
            ("min_groundedness", "groundedness", 0.82, True),
            ("min_safety_pass_rate", "safety_pass_rate", 0.93, False),
        ],
    },
    "qa-sft": {
        "task_profile": "instruction_sft",
        "display_name": "Question & Answer",
        "required_metric_ids": ["exact_match", "f1", "llm_judge_pass_rate"],
        "gates": [
            ("min_exact_match", "exact_match", 0.45, True),
            ("min_f1", "f1", 0.60, True),
            # LLM-judge handles semantic equivalence the exact-match
            # gate misses — "Paris is the capital." vs "The capital
            # is Paris." both pass under a judge but only one passes
            # exact_match.
            ("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.75, True),
            ("min_safety_pass_rate", "safety_pass_rate", 0.93, False),
        ],
    },
    "generic-sft": {
        "task_profile": "instruction_sft",
        "display_name": "Generic instruction",
        "required_metric_ids": ["exact_match", "f1"],
        "gates": [
            ("min_exact_match", "exact_match", 0.35, True),
            ("min_f1", "f1", 0.55, True),
            ("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.65, False),
            ("min_safety_pass_rate", "safety_pass_rate", 0.93, False),
        ],
    },
    "code-review": {
        "task_profile": "instruction_sft",
        "display_name": "Code review",
        # Code-review eval leans heavily on LLM-judge because the
        # "correct" review is rarely an exact-match — the rubric
        # checks whether the review hits the same critique theme.
        "required_metric_ids": ["llm_judge_pass_rate", "f1"],
        "gates": [
            ("min_llm_judge_pass_rate", "llm_judge_pass_rate", 0.70, True),
            ("min_f1", "f1", 0.40, False),
            ("min_safety_pass_rate", "safety_pass_rate", 0.93, False),
        ],
    },
    # Arc R-2 — rag-protocol projects point directly at the built-in
    # ``evalpack.rag_protocol.discipline`` pack (Recipe.eval_pack_id),
    # so no scaffold entry is needed here. The discipline pack uses
    # the ``lte`` operator on hallucination_rate which the scaffold
    # tuple shape doesn't currently support; keeping the scaffold
    # path focused on simple gte-only gates avoids expanding that
    # tuple for one recipe.
}


def _fallback_spec() -> dict[str, Any]:
    """For unknown recipes — generic SFT-style scaffold so the user
    still gets a draft they can edit. Mirrors the generic-sft spec
    so the gate thresholds match the platform's existing default."""
    return _RECIPE_SCAFFOLD["generic-sft"]


def _gate_dict(gate_id: str, metric_id: str, threshold: float, required: bool) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "metric_id": metric_id,
        "operator": "gte",
        "threshold": threshold,
        "required": required,
    }


def scaffold_pack(
    recipe_id: str,
    *,
    project_id: int | None = None,
    gold_set_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a draft eval-pack tailored to ``recipe_id``. The draft
    is JSON-serialisable + already shaped to pass through the eval
    pack contract builder.

    ``gold_set_summary`` is the dict returned by ``_summarise_gold_set``
    — used to populate the description with row-count context. v1 only
    reads ``row_count``; future tunings (relaxed thresholds when row
    counts are thin, etc.) can plug in here.
    """
    spec = _RECIPE_SCAFFOLD.get(recipe_id.strip().lower(), _fallback_spec())
    task_spec = {
        "task_profile": spec["task_profile"],
        "display_name": spec["display_name"],
        "description": f"Scaffolded from the '{recipe_id}' recipe.",
        "required_metric_ids": list(spec["required_metric_ids"]),
        "gates": [_gate_dict(*entry) for entry in spec["gates"]],
    }
    row_count = (gold_set_summary or {}).get("row_count")
    suffix = f" (gold set: {row_count} rows)" if row_count else ""
    return {
        "pack_id": SCAFFOLDED_PACK_ID,
        "display_name": f"Scaffolded · {spec['display_name']}",
        "description": (
            f"Auto-generated starter pack tuned for the '{recipe_id}' recipe"
            f"{suffix}. Edit the gates inline before saving — nothing is "
            f"persisted until you click 'Use scaffold'."
        ),
        "version": "1.0.0",
        "owner": "project_scaffold",
        "tags": ["scaffolded", recipe_id],
        "default_task_profile": spec["task_profile"],
        "task_specs": [task_spec],
        # Backward-compat top-level gates mirror the default task spec
        # — the existing pack-summary endpoint reads this field.
        "gates": list(task_spec["gates"]),
    }


async def _summarise_gold_set(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Lightweight gold-set summary the scaffolder uses for context.
    Returns ``{row_count, dataset_types_seen}`` — best-effort, errors
    return zeros so the scaffold call never fails on a fresh project."""
    try:
        result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type.in_([
                    DatasetType.GOLD_DEV, DatasetType.GOLD_TEST,
                ]),
            )
        )
        rows = list(result.scalars())
        total = sum(int(r.record_count or 0) for r in rows)
        return {
            "row_count": total,
            "dataset_types_seen": sorted({r.dataset_type.value for r in rows}),
        }
    except Exception:
        return {"row_count": 0, "dataset_types_seen": []}


async def scaffold_pack_for_project(
    db: AsyncSession, *, project_id: int
) -> dict[str, Any]:
    """Public entry point used by the API. Reads the project's
    recipe + gold-set summary and returns the draft pack.

    Raises ``ValueError("project_not_found")`` when the project is
    missing. Raises ``ValueError("recipe_required")`` when the
    project has no recipe selected — a scaffold without a recipe
    would be guessing.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError("project_not_found")
    selected = project.selected_recipe or {}
    recipe_id = str(selected.get("recipe_id") or "")
    if not recipe_id:
        raise ValueError("recipe_required")
    recipe = get_recipe(recipe_id)
    if recipe is None:
        # The recipe id was set but the catalog doesn't know it — fall
        # back to generic but flag in the description so the user
        # sees the mismatch.
        recipe_id = "generic-sft"

    summary = await _summarise_gold_set(db, project_id)
    return {
        "project_id": project_id,
        "recipe_id": recipe_id,
        "gold_set_summary": summary,
        "draft_pack": scaffold_pack(
            recipe_id,
            project_id=project_id,
            gold_set_summary=summary,
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Persistence: save the user's edited draft onto Project.runtime_config
# and flip the preference. Reuses runtime_config so we don't need a new
# table for what's effectively a per-project JSON blob.
# ─────────────────────────────────────────────────────────────────────


RUNTIME_CONFIG_KEY = "scaffolded_evaluation_pack"


def is_scaffolded_pack_id(pack_id: str | None) -> bool:
    return (pack_id or "").strip().lower() == SCAFFOLDED_PACK_ID


async def save_scaffolded_pack(
    db: AsyncSession,
    *,
    project_id: int,
    draft_pack: dict[str, Any],
) -> dict[str, Any]:
    """Persist the user's edited draft to
    ``project.runtime_config["scaffolded_evaluation_pack"]`` and flip
    ``evaluation_preferred_pack_id`` to ``evalpack.project.scaffolded``.

    Validates that the draft has a recognisable pack_id and at least
    one task_spec — beyond that we trust the caller (the UI surfaces
    the draft that came from this same service, just with edits).
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError("project_not_found")

    if not isinstance(draft_pack, dict):
        raise ValueError("invalid_draft_pack")
    task_specs = draft_pack.get("task_specs")
    if not isinstance(task_specs, list) or not task_specs:
        raise ValueError("draft_pack_missing_task_specs")

    # Gap-#5 slice 1: validate gates against the operator whitelist +
    # metric catalog so the FE editor can surface a precise error
    # instead of the eval engine silently coercing bad input.
    from app.services.evaluation_gate_catalog import validate_draft_pack_gates
    validate_draft_pack_gates(draft_pack)

    # Normalise: force the scaffolded pack id even if the client
    # mutated it — the resolver keys on this id, and renaming would
    # silently disconnect the saved pack from the active-pack path.
    normalised = dict(draft_pack)
    normalised["pack_id"] = SCAFFOLDED_PACK_ID

    rc = dict(project.runtime_config or {})
    rc[RUNTIME_CONFIG_KEY] = normalised
    project.runtime_config = rc
    project.evaluation_preferred_pack_id = SCAFFOLDED_PACK_ID
    await db.flush()
    return {
        "project_id": project_id,
        "preferred_pack_id": SCAFFOLDED_PACK_ID,
        "scaffolded_pack": normalised,
    }


async def adopt_gate_from_cluster(
    db: AsyncSession,
    *,
    project_id: int,
    cluster_id: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gap-#5 slice 3 — one-click "Adopt this gate" action that turns a
    failure cluster into a starter gate on the project's scaffolded
    eval pack. Loads the active scaffold (or scaffolds a fresh one
    when none is persisted), derives a gate from the cluster's
    ``reason_code`` via the catalog's mapping, appends it to the
    first task_spec's gates, and persists via ``save_scaffolded_pack``.

    The optional ``overrides`` dict lets the FE override any of
    ``{metric_id, operator, threshold, required, gate_id}`` — useful
    if the user wants to tighten the starter threshold before saving
    rather than after.

    Raises:
      * ``project_not_found`` — project missing.
      * ``cluster_not_found`` — cluster row missing or belongs to
        another project (defensive — we don't trust the client to
        scope its own request).
      * ``no_gate_suggestion_for_reason_code:<code>`` — the cluster's
        reason_code has no mapping in the catalog AND no overrides
        were supplied. The user can still add the gate by hand in
        the editor.
      * Any validator error from ``validate_draft_pack_gates`` (e.g.
        ``duplicate_gate_id:<id>``) when the resulting pack is malformed
        — surfaces back through the slice-1 contract.
    """
    from app.models.failure_cluster import FailureCluster
    from app.services.evaluation_gate_catalog import suggest_gate_for_reason_code

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError("project_not_found")

    cluster = await db.get(FailureCluster, int(cluster_id))
    if cluster is None or int(cluster.project_id) != int(project_id):
        raise ValueError("cluster_not_found")

    # Recipe drives the metric pick (the catalog prefers recommended
    # metrics for the project's recipe).
    selected = project.selected_recipe or {}
    recipe_id = str(selected.get("recipe_id") or "").strip().lower() or None
    overrides = dict(overrides or {})

    suggestion = suggest_gate_for_reason_code(
        str(cluster.reason_code or ""),
        recipe_id=recipe_id,
    )
    # Apply any client overrides on top of the suggestion. When no
    # suggestion exists, the overrides MUST cover the full gate shape
    # — otherwise we have no metric_id to attach to.
    if suggestion is None and "metric_id" not in overrides:
        raise ValueError(f"no_gate_suggestion_for_reason_code:{cluster.reason_code}")
    base = suggestion or {}
    new_gate = {
        "gate_id": str(overrides.get("gate_id") or base.get("gate_id") or "").strip(),
        "metric_id": str(overrides.get("metric_id") or base.get("metric_id") or "").strip(),
        "operator": str(overrides.get("operator") or base.get("operator") or "gte").strip(),
        "threshold": float(overrides.get("threshold")) if "threshold" in overrides else float(base.get("threshold") or 0.5),
        "required": bool(overrides.get("required", base.get("required", False))),
    }
    if not new_gate["gate_id"]:
        prefix = "max" if new_gate["operator"] == "lte" else "min"
        new_gate["gate_id"] = f"{prefix}_{new_gate['metric_id']}_from_cluster"

    # Pull the current scaffolded pack if one's persisted; otherwise
    # build a fresh one off the project's recipe. Either way we end up
    # with a draft we can append to.
    existing = get_scaffolded_pack(project)
    if existing is not None:
        draft = dict(existing)
        draft["task_specs"] = [dict(ts) for ts in (draft.get("task_specs") or [])]
        for ts in draft["task_specs"]:
            ts["gates"] = list(ts.get("gates") or [])
        if "gates" in draft:
            draft["gates"] = list(draft["gates"])
    else:
        # No persisted scaffold yet — build one from the recipe.
        summary = await _summarise_gold_set(db, project_id)
        if not recipe_id:
            # No recipe → fall back to generic-sft so the user still
            # gets a valid pack instead of an error.
            recipe_id = "generic-sft"
        draft = scaffold_pack(
            recipe_id,
            project_id=project_id,
            gold_set_summary=summary,
        )

    if not draft.get("task_specs"):
        raise ValueError("draft_pack_missing_task_specs")

    # If a gate with the same gate_id is already in the first
    # task_spec, suffix it so we don't collide. Mirrors the FE's
    # collision-handling in the editor.
    first_spec = draft["task_specs"][0]
    existing_ids = {str(g.get("gate_id") or "") for g in (first_spec.get("gates") or [])}
    if new_gate["gate_id"] in existing_ids:
        for i in range(2, 1000):
            candidate = f"{new_gate['gate_id']}_{i}"
            if candidate not in existing_ids:
                new_gate["gate_id"] = candidate
                break

    first_spec.setdefault("gates", []).append(new_gate)
    # Keep top-level gates synced with the default task spec (the
    # eval-pack contract reads this as a fallback for the legacy
    # gate-list shape).
    draft["gates"] = list(first_spec["gates"])

    result = await save_scaffolded_pack(
        db,
        project_id=project_id,
        draft_pack=draft,
    )
    result["new_gate"] = new_gate
    result["cluster_reason_code"] = str(cluster.reason_code or "")
    return result


def get_scaffolded_pack(project: Project) -> dict[str, Any] | None:
    """Read the saved scaffold from the project, or None when absent.
    Used by ``resolve_project_evaluation_pack`` to surface the
    scaffolded pack as the active one when the preference points
    at ``evalpack.project.scaffolded``."""
    rc = project.runtime_config or {}
    pack = rc.get(RUNTIME_CONFIG_KEY)
    if isinstance(pack, dict) and pack:
        return pack
    return None
