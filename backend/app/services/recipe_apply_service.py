"""Apply a Theme 2 recipe selection to a Project record.

When the user picks a recipe in the DatasetImportWizard, we snapshot
the recipe metadata into the project's `selected_recipe` JSON column
and propagate the recipe's suggested base model into the project's
`base_model_name` field so downstream training defaults pick it up
automatically.

The snapshot is intentional — recipe definitions can evolve
(`builtin-v2`, plugins) but we want the project to remember the
shape it was set up against. Comparing the live recipe to the
snapshot is how a future "your recipe has changed" surface would
work; out of scope for this commit.

What we do NOT auto-apply (and why):
- `target_profile_id` — has a meaningful global default
  (`vllm_server`) and switching it silently could reroute deploys.
  Surfaced as a future explicit prompt, not a silent default.
- `training_preferred_plan_profile` — most recipes pick "balanced"
  anyway, so applying it is a no-op; not worth the surface area.
- `evaluation_preferred_pack_id` — eval-pack selection has its
  own UX in the Eval tab and shouldn't be quietly reset.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.services import recipe_service


def build_recipe_snapshot(recipe: recipe_service.Recipe) -> dict[str, Any]:
    """The shape stored in `Project.selected_recipe`. Compact by
    design — full recipe definitions are available via the catalog
    API; the snapshot just carries enough context for downstream
    code paths and for a "what shape did this project pick?" UI."""
    return {
        "recipe_id": recipe.id,
        "name": recipe.name,
        "icon": recipe.icon,
        "task_profile": recipe.task_profile,
        "adapter_id": recipe.adapter_id,
        "scoring_mode": recipe.scoring_mode,
        "suggested_base_model": recipe.suggested_base_model,
        "target_profile": recipe.target_profile,
        "training_plan_profile": recipe.training_plan_profile,
        "eval_pack_id": recipe.eval_pack_id,
        "default_input_column": recipe.default_input_column,
        "default_output_column": recipe.default_output_column,
        "catalog_version": recipe.catalog_version,
        "catalog_source": recipe.catalog_source,
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }


class RecipeNotFoundError(ValueError):
    """Raised when the supplied recipe_id doesn't match the catalog."""


async def apply_recipe_to_project(
    db: AsyncSession,
    project_id: int,
    recipe_id: str,
) -> Project:
    """Snapshot the recipe onto the project and adopt its
    suggested base model. Returns the updated Project (flushed,
    refreshed) so the caller can serialize to ProjectResponse.

    Idempotent: re-applying the same recipe re-snapshots with a
    fresh `applied_at` but leaves all other fields effectively
    unchanged. The `base_model_name` is updated unconditionally
    because picking a recipe is an explicit user action — newer
    pick wins over older state, even if the user had typed a
    custom value in between.
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    recipe = recipe_service.get_recipe(recipe_id)
    if recipe is None:
        raise RecipeNotFoundError(f"Recipe '{recipe_id}' not found in catalog")

    project.selected_recipe = build_recipe_snapshot(recipe)
    project.base_model_name = recipe.suggested_base_model
    _sync_dataset_adapter_preset(project, recipe)

    await db.flush()
    await db.refresh(project)
    return project


def _sync_dataset_adapter_preset(project: Project, recipe) -> None:  # noqa: ANN001
    """Dataset prep, training and eval read ``dataset_adapter_preset``, not
    ``selected_recipe``. Without this a recipe picked in the wizard never
    reached training. Keeps an existing adapter config / field mapping when
    the adapter doesn't change."""
    from app.services.task_shape_service import canonical_task_profile

    adapter_id = str(getattr(recipe, "adapter_id", "") or "").strip()
    if not adapter_id:
        return
    existing = project.dataset_adapter_preset if isinstance(project.dataset_adapter_preset, dict) else {}
    recipe_profile = canonical_task_profile(getattr(recipe, "task_profile", None))
    if (
        existing.get("adapter_id")
        and recipe_profile
        and canonical_task_profile(existing.get("task_profile")) == recipe_profile
    ):
        # Already set up for this task shape (possibly with a more specific
        # adapter, e.g. a demo's structured-extraction) — leave it alone.
        return
    same_adapter = existing.get("adapter_id") == adapter_id
    project.dataset_adapter_preset = {
        # Merge: callers (e.g. demo seeding) keep their own keys.
        **existing,
        "adapter_id": adapter_id,
        "adapter_config": dict(existing.get("adapter_config") or {}) if same_adapter else {},
        "field_mapping": dict(existing.get("field_mapping") or {}) if same_adapter else {},
        "task_profile": canonical_task_profile(getattr(recipe, "task_profile", None))
        or getattr(recipe, "task_profile", None),
        # Provenance: surfaces as "recipe" (not a hand-set project override)
        # in Data Studio's mapping preview.
        "origin": "recipe",
    }


async def clear_recipe_from_project(db: AsyncSession, project_id: int) -> Project:
    """Clear the snapshot. Does NOT roll back `base_model_name` —
    once a model name is set the user is presumed to want it kept;
    they can edit it in the project-settings surface."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ValueError(f"Project {project_id} not found")
    project.selected_recipe = None
    await db.flush()
    await db.refresh(project)
    return project
