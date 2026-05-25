"""Curriculum-ranking API routes (USER-SUCCESS Epic 6 Phase 6a).

Phase 6a ships a single read-only preview endpoint:

  ``GET /api/projects/{project_id}/curriculum/preview``

Returns the curriculum ordering the training pipeline *would* use
when curriculum learning is enabled (Phase 6b adds that toggle).
Phase 6a is preview-only so the user can eyeball "are the rankings
reasonable for my data?" before betting training quality on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.project import Project
from app.services.curriculum_service import (
    CurriculumUnavailable,
    rank_rows,
    recommended_scoring_mode_for_recipe,
)


router = APIRouter(
    prefix="/projects/{project_id}/curriculum", tags=["Curriculum"]
)


@router.get("/preview")
async def preview_curriculum(
    project_id: int,
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description=(
            "Number of ranked rows to return in the preview. The full "
            "ranking is computed; this just trims the response payload."
        ),
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Compute the curriculum order for the project's gold + synthetic
    accepted rows and return it as a sample for the UI / A/B harness.

    Status codes:
      200 — ranking computed; payload includes scoring_mode +
            ordered ``ranked`` list (capped by ``limit``).
      400 — project missing a recipe, or recipe has no curriculum
            scoring mode shipped yet.
      404 — project not found.
      503 — sentence-transformers isn't installed (the embedder
            dependency curriculum needs).
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Project has no selected recipe — curriculum ranking "
                "needs the recipe to pick the right scoring mode."
            ),
        )

    scoring_mode = recommended_scoring_mode_for_recipe(recipe_id)
    if scoring_mode is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No curriculum scoring mode ships for recipe "
                f"{recipe_id!r} yet. Phase 6a covers classification "
                f"only; other recipes plug in in later phases."
            ),
        )

    rows = await _load_curriculum_input_rows(db, project_id)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "Project has no training rows yet — import a gold set "
                "or generate synthetic rows first."
            ),
        )

    cache_dir = settings.DATA_DIR / "projects" / str(project_id) / "curriculum"
    try:
        ranked = rank_rows(
            rows,
            scoring_mode=scoring_mode,
            cache_dir=cache_dir,
        )
    except CurriculumUnavailable as e:
        # 503 — feature is structurally unavailable on this install.
        # The error message names sentence-transformers + the install
        # command so the user can fix it without grepping docs.
        raise HTTPException(status_code=503, detail=str(e)) from e

    # Return ranking sorted ascending by difficulty (easy first), then
    # trim to ``limit`` for the preview payload. Total counts surface
    # the full size so the UI can show "showing 50 of N."
    ranked_sorted = sorted(ranked, key=lambda entry: entry["rank"])
    preview = ranked_sorted[:limit]
    # Attach a sample text snippet per previewed row so the user can
    # tell what made it easy or hard at a glance — without leaking the
    # entire training corpus over the wire.
    by_row_id = {_row_id_key(row): row for row in rows}
    payload_rows: list[dict[str, Any]] = []
    for entry in preview:
        source_row = by_row_id.get(_entry_row_id_key(entry))
        snippet = _row_snippet(source_row) if source_row else ""
        payload_rows.append({
            **entry,
            "snippet": snippet,
        })

    return {
        "project_id": project_id,
        "recipe_id": recipe_id,
        "scoring_mode": scoring_mode,
        "total_rows": len(ranked),
        "returned": len(payload_rows),
        "ranked": payload_rows,
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


async def _load_curriculum_input_rows(
    db: AsyncSession, project_id: int
) -> list[dict[str, Any]]:
    """Load the rows the curriculum should rank.

    Reuses the dataset loader's existing gold + accepted-synth merge
    semantics — by default ``_load_records_from_file`` excludes
    pending synth rows, which is what we want (curriculum should only
    rank what's actually going into training)."""
    from sqlalchemy import select

    from app.models.dataset import Dataset, DatasetType
    from app.services.dataset_service import _load_records_from_file

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [
                    DatasetType.GOLD_DEV,
                    DatasetType.GOLD_TEST,
                    DatasetType.SYNTHETIC,
                ]
            ),
        )
    )
    rows: list[dict[str, Any]] = []
    for dataset in result.scalars():
        if not dataset.file_path:
            continue
        path = Path(dataset.file_path)
        if not path.exists():
            continue
        rows.extend(_load_records_from_file(path))
    return rows


def _row_id_key(row: dict[str, Any]) -> str:
    """Stable key for matching ranked entries back to their source row."""
    value = row.get("id")
    if isinstance(value, (int, str)):
        return f"id:{value}"
    return f"pos:{id(row)}"


def _entry_row_id_key(entry: dict[str, Any]) -> str:
    return f"id:{entry['row_id']}"


def _row_snippet(row: dict[str, Any] | None, *, limit: int = 160) -> str:
    if row is None:
        return ""
    for key in ("text", "input", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:limit]
        if isinstance(value, dict):
            for sub in value.values():
                if isinstance(sub, str) and sub.strip():
                    return sub.strip()[:limit]
    return ""
