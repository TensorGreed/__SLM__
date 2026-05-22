"""Project templates — cloneable starting kits for new projects.

A project template carries the same shape as a demo bundle (manifest +
CSV + gold.jsonl) but with two key differences:

  1. **Cloneable.** Templates are designed to instantiate any number
     of projects. The demo-bundle path (`seed_demo_project`) is
     idempotent by name; the template path takes a user-chosen name
     each call so they can spin up 3 different "Ticket Router"
     projects with different teams or different training data
     additions.

  2. **Richer metadata.** Templates carry `minimum_dataset_size`,
     `recommended_base_models` (a list), and a `recipe_id` that the
     instantiation flow uses to apply the recipe snapshot to the
     new project. Demo bundles infer all of this from the
     task_profile alone.

Templates live at `backend/data/project_templates/<slug>/`. The
on-disk layout mirrors `demo_samples/` so we reuse the existing
`_materialize_demo_bundle_into_project` helper for the actual
data write — the only template-specific work is loading the
manifest from the templates dir + applying the recipe + setting
the user-chosen name.

Stable error codes (raised as ValueError; mapped to HTTP by the API):

  - ``template_slug_unknown:<slug>`` (404)
  - ``template_manifest_invalid:<slug>[:<reason>]`` (400)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import PipelineStage, Project, ProjectStatus
from app.services.demo_project_service import (
    _materialize_demo_bundle_into_project,
)
from app.services.recipe_apply_service import (
    RecipeNotFoundError,
    apply_recipe_to_project,
)


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_TEMPLATES_DIR = BACKEND_DIR / "data" / "project_templates"


def _resolve_template_dir(slug: str) -> Path:
    """Resolve a template slug to its directory, with the same
    path-traversal guard `demo_project_service._resolve_demo_dir`
    uses."""
    safe = "".join(ch for ch in slug if ch.isalnum() or ch in "-_").strip("-_")
    if not safe or safe != slug:
        raise ValueError(f"template_slug_unknown:{slug}")
    candidate = (PROJECT_TEMPLATES_DIR / safe).resolve()
    if (
        not candidate.is_dir()
        or PROJECT_TEMPLATES_DIR.resolve() not in candidate.parents
    ):
        raise ValueError(f"template_slug_unknown:{slug}")
    return candidate


def _load_template_manifest(slug: str) -> dict[str, Any]:
    template_dir = _resolve_template_dir(slug)
    manifest_path = template_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"template_manifest_invalid:{slug}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"template_manifest_invalid:{slug}:{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"template_manifest_invalid:{slug}:not_a_dict")
    payload.setdefault("_dir", str(template_dir))
    return payload


def _template_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Public-safe summary of a template manifest — used by the
    catalog endpoint and the gallery UI."""
    return {
        "slug": str(manifest.get("slug") or "").strip(),
        "name": str(manifest.get("name") or manifest.get("slug") or ""),
        "headline": str(manifest.get("headline") or ""),
        "description": str(manifest.get("description") or ""),
        "icon": str(manifest.get("icon") or "🧰"),
        "recipe_id": str(manifest.get("recipe_id") or "").strip() or None,
        "task_profile": str(manifest.get("task_profile") or "").strip(),
        "target_profile": str(manifest.get("target_profile") or "vllm_server"),
        "training_preferred_plan_profile": str(
            manifest.get("training_preferred_plan_profile") or "balanced"
        ),
        "evaluation_preferred_pack_id": manifest.get("evaluation_preferred_pack_id"),
        "minimum_dataset_size": int(manifest.get("minimum_dataset_size") or 0),
        "recommended_base_models": [
            str(item)
            for item in (manifest.get("recommended_base_models") or [])
            if str(item).strip()
        ],
        "labels": [
            str(item)
            for item in (manifest.get("labels") or [])
            if str(item).strip()
        ],
        "suggested_brief": str(manifest.get("suggested_brief") or ""),
        "template_version": str(manifest.get("template_version") or "v1"),
        "dataset_input_field": str(manifest.get("dataset_input_field") or "input"),
        "dataset_output_field": str(manifest.get("dataset_output_field") or "output"),
    }


def list_project_templates() -> list[dict[str, Any]]:
    """List every project template available on disk. Order is stable
    (alphabetical by slug) so the UI gallery doesn't reshuffle on
    refresh."""
    if not PROJECT_TEMPLATES_DIR.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for child in sorted(PROJECT_TEMPLATES_DIR.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not (child / "manifest.json").exists():
            continue
        try:
            manifest = _load_template_manifest(child.name)
        except ValueError:
            # Skip malformed manifests but don't break the catalog
            # — the UI shouldn't disappear because one template's
            # JSON has a stray comma.
            continue
        summaries.append(_template_summary(manifest))
    return summaries


def get_project_template(slug: str) -> dict[str, Any] | None:
    """Fetch a single template's full summary. Returns None when the
    slug is unknown so callers (API) can render a 404 cleanly."""
    try:
        manifest = _load_template_manifest(slug)
    except ValueError:
        return None
    return _template_summary(manifest)


async def _unique_project_name(
    db: AsyncSession, requested: str,
) -> str:
    """Pick a project name that doesn't collide with an existing one.
    If the requested name is taken, append ' (2)', ' (3)', … until a
    free slot opens. Project.name has a UNIQUE constraint so this is
    necessary to support multiple clones of the same template."""
    base = requested.strip() or "Untitled project"
    candidate = base
    counter = 2
    while True:
        result = await db.execute(
            select(func.count(Project.id)).where(Project.name == candidate)
        )
        count = int(result.scalar() or 0)
        if count == 0:
            return candidate
        candidate = f"{base} ({counter})"
        counter += 1


async def instantiate_project_template(
    db: AsyncSession,
    slug: str,
    *,
    project_name: str | None = None,
    actor_user_id: int | None = None,
) -> tuple[Project, dict[str, Any]]:
    """Create a brand-new project from a template. Materializes the
    template's data into the project (raw + gold + prepared splits)
    and applies the template's recipe so `selected_recipe` lands
    populated.

    Unlike `seed_demo_project`, this is NOT idempotent by name —
    every call creates a new Project row, so the same template can
    back any number of projects. Name collisions are resolved by
    appending ' (2)', ' (3)', etc.

    Raises:
        ValueError("template_slug_unknown:{slug}") for unknown slugs.
        ValueError("template_manifest_invalid:{slug}") for malformed manifests.
    """
    manifest = _load_template_manifest(slug)
    template_dir = Path(manifest["_dir"])

    summary_meta = _template_summary(manifest)
    description = str(manifest.get("description") or "")
    suggested_brief = str(manifest.get("suggested_brief") or "")
    target_profile = str(manifest.get("target_profile") or "vllm_server")
    plan_profile = str(
        manifest.get("training_preferred_plan_profile") or "balanced"
    )
    eval_pack = manifest.get("evaluation_preferred_pack_id") or None
    task_profile = summary_meta["task_profile"] or "instruction_sft"
    input_field = summary_meta["dataset_input_field"]
    output_field = summary_meta["dataset_output_field"]
    recommended_base_models = summary_meta["recommended_base_models"]

    base_template_name = str(manifest.get("name") or slug)
    requested = (project_name or base_template_name).strip() or base_template_name
    final_name = await _unique_project_name(db, requested)

    project = Project(
        name=final_name,
        description=description,
        status=ProjectStatus.ACTIVE,
        pipeline_stage=PipelineStage.TRAINING,
        beginner_mode=True,
        target_profile_id=target_profile,
        training_preferred_plan_profile=plan_profile,
        evaluation_preferred_pack_id=eval_pack,
        # Pre-fill the base model from the template's first
        # recommendation so the Quickstart Train tile is unlocked
        # without a separate recipe-apply round trip. The recipe
        # apply call below also sets base_model_name, but that
        # uses the recipe's `suggested_base_model` which may
        # differ from the template's curated recommendation.
        base_model_name=recommended_base_models[0] if recommended_base_models else "",
        dataset_adapter_preset={
            "template_slug": slug,
            "template_version": summary_meta["template_version"],
            "suggested_brief": suggested_brief,
            "task_profile": task_profile,
            "field_mapping": {"input": input_field, "output": output_field},
        },
    )
    db.add(project)
    await db.flush()  # populate project.id

    materialize_summary = await _materialize_demo_bundle_into_project(
        db,
        project,
        slug,
        manifest,
        template_dir,
        project_name=final_name,
        actor_user_id=actor_user_id,
    )

    recipe_id = summary_meta["recipe_id"]
    if recipe_id:
        try:
            await apply_recipe_to_project(db, project.id, recipe_id)
        except RecipeNotFoundError:
            # The template references a recipe that doesn't exist
            # in the catalog. Non-fatal: the project still has its
            # template-curated base_model_name + task_profile.
            pass
        # Recipe-apply overrides base_model_name with the recipe's
        # suggested model; re-assert the template's pick so the
        # template's first recommendation wins. The recipe is
        # still snapshotted onto `selected_recipe` so the rest of
        # the platform sees the right task_profile/scoring_mode.
        if recommended_base_models:
            project.base_model_name = recommended_base_models[0]
            await db.flush()

    summary = {
        **materialize_summary,
        "template_slug": slug,
        "project_id": project.id,
        "project_name": project.name,
        "requested_name": requested,
        "name_adjusted": project.name != requested,
        "minimum_dataset_size": summary_meta["minimum_dataset_size"],
        "recommended_base_models": recommended_base_models,
    }
    return project, summary


__all__ = [
    "PROJECT_TEMPLATES_DIR",
    "list_project_templates",
    "get_project_template",
    "instantiate_project_template",
]
