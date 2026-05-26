"""Archetype API routes (USER-SUCCESS Epic 8 Phase 8a).

Read-only surface exposing per-recipe structural archetypes for
the UI + Coach Mode to consume in later phases. Phase 8a ships
the endpoint and the underlying computation; Phase 8b adds the
per-project comparison + UI panel; Phase 8c adds the Coach nudge.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.archetype_service import (
    compare_project_to_archetype,
    compute_recipe_archetype,
)


router = APIRouter(tags=["Archetypes"])


@router.get("/archetypes/{recipe_id}")
async def get_recipe_archetype(
    recipe_id: str,
    refresh: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Return the per-recipe archetype payload (distribution stats
    per structural feature, plus cohort provenance).

    The cohort merges passing user projects (latest eval pass_rate
    >= 0.6) with seed contributions from the shipped templates
    when the user-project pool is thin.

    Status codes:
      * 200 — archetype computed; ``n_passing_projects >= 1``.
      * 400 — unknown ``recipe_id``.
      * 404 — empty cohort (no user projects AND no template seeds
        for this recipe — only possible for recipes with no
        shipped template, today ``code-review``).
    """
    if refresh:
        from app.services.archetype_service import clear_archetype_cache

        clear_archetype_cache()
    try:
        return await compute_recipe_archetype(db, recipe_id)
    except ValueError as exc:
        detail = str(exc)
        if detail.startswith("unknown_recipe_id"):
            raise HTTPException(400, detail) from exc
        if detail.startswith("empty_cohort"):
            raise HTTPException(404, detail) from exc
        raise HTTPException(400, detail) from exc


@router.get("/projects/{project_id}/archetype-comparison")
async def get_project_archetype_comparison(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """USER-SUCCESS Epic 8 Phase 8b — per-project comparison
    against the recipe's archetype. Returns one row per applicable
    feature with status (below / above / ok / missing) and an
    optional ``suggested_action`` payload that matches the Coach
    Mode contract so the frontend reuses the existing handlers
    (``run_playbook`` fires through the Jobs framework;
    ``navigate`` uses ``window.location.assign``).

    Status codes:
      * 200 — comparison computed.
      * 400 — project has no selected recipe (can't compare to an
        archetype without knowing the recipe) OR the recipe's
        archetype cohort is empty (no user projects, no template
        seed — only ``code-review`` today).
      * 404 — project not found.
    """
    try:
        return await compare_project_to_archetype(db, project_id)
    except ValueError as exc:
        detail = str(exc)
        if detail == "project_not_found":
            raise HTTPException(404, detail) from exc
        if detail.startswith("empty_cohort"):
            raise HTTPException(400, detail) from exc
        raise HTTPException(400, detail) from exc
