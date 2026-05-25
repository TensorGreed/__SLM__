"""Auto-RAG API routes (USER-SUCCESS Epic 9 Phase 9a).

Phase 9a ships a single read-only preview endpoint:

  ``GET /api/projects/{project_id}/auto-rag/preview``

Builds (or rebuilds, on cache miss) the BM25 index over the project's
training rows and returns the top-K retrievals for a given query. The
caller passes the query as a URL param so a human can eyeball "for
*this* question, would the index find a reasonable Q&A pair?" before
Phase 9b wires retrieval into actual inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.project import Project
from app.services.auto_rag_service import (
    AutoRagUnavailable,
    build_bm25_index,
    recommended_text_keys_for_recipe,
    retrieve,
)


router = APIRouter(
    prefix="/projects/{project_id}/auto-rag", tags=["AutoRAG"]
)


@router.get("/preview")
async def preview_auto_rag(
    project_id: int,
    query: str = Query(
        ...,
        min_length=1,
        description=(
            "The query to retrieve against. Phase 9b will pass the "
            "user's playground prompt here; for Phase 9a's preview, "
            "the caller passes whatever they want to spot-check."
        ),
    ),
    k: int = Query(
        3,
        ge=1,
        le=20,
        description="Number of retrievals to return. Default 3 (matches Phase 9b's planned prepend size).",
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Build the BM25 index (if needed) and return top-K retrievals.

    Status codes:
      200 — index loaded + query scored; ``retrieved`` is up to ``k`` hits.
      400 — project missing a recipe; recipe has no RAG corpus shape
            (Phase 9a covers qa-sft only); no training data on disk.
      404 — project not found.
      503 — BM25 index built but unreadable (corrupt file on disk —
            rare; would indicate a partial write).
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
                "Project has no selected recipe — auto-RAG needs the "
                "recipe to pick the right text fields for the corpus."
            ),
        )

    if recommended_text_keys_for_recipe(recipe_id) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Recipe {recipe_id!r} has no auto-RAG corpus shape "
                f"yet. Phase 9a covers qa-sft only; other recipes "
                f"plug in in later phases."
            ),
        )

    rows = await _load_rag_corpus_rows(db, project_id)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "Project has no training rows yet — import a gold set "
                "or generate synthetic rows first."
            ),
        )

    index_dir = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
    index_path = index_dir / "bm25_index.json"

    # Rebuild the index when the file is missing OR when the row
    # count has changed since the last build (cheap signal that rows
    # were added/removed — proper change detection is Phase 9b's
    # job, the preview just wants the latest possible answer).
    needs_rebuild = (
        not index_path.exists()
        or _index_row_count(index_path) != len(rows)
    )
    if needs_rebuild:
        try:
            manifest = build_bm25_index(
                rows,
                recipe_id=recipe_id,
                output_dir=index_dir,
            )
        except AutoRagUnavailable as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        manifest = {
            "index_path": str(index_path),
            "recipe_id": recipe_id,
            "text_keys": list(recommended_text_keys_for_recipe(recipe_id) or ()),
            "doc_count": len(rows),
            # avg_doc_length only surfaced for new builds; on cache
            # hit we'd need to re-read the file. Cheap to omit here.
            "avg_doc_length": None,
        }

    try:
        retrieved = retrieve(query, index_dir=index_dir, k=k)
    except AutoRagUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return {
        "project_id": project_id,
        "recipe_id": recipe_id,
        "query": query,
        "k": k,
        "index": manifest,
        "retrieved": retrieved,
    }


async def _load_rag_corpus_rows(
    db: AsyncSession, project_id: int
) -> list[dict[str, Any]]:
    """Load the rows the BM25 index should be built over.

    Reuses ``dataset_service._load_records_from_file`` so pending
    synth rows stay out of the corpus (consistent with what the
    training pipeline trains on). For Phase 9a the corpus is the
    union of gold + accepted-synth, same as the training prep step's
    output."""
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


def _index_row_count(index_path: Path) -> int:
    """Cheap doc_count read for the cache-validity check. Returns
    -1 on any read error so the caller treats it as "needs rebuild"
    rather than crashing on a malformed index file."""
    import json

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return -1
    return int(payload.get("doc_count") or 0)
