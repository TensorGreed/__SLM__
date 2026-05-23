"""Coach Mode service (USER-SUCCESS Epic 4).

Generates context-aware suggestions for each workflow stage. The UI
mounts a ``CoachStrip`` per panel that calls
``GET /api/projects/{id}/coach/{stage}`` and renders the returned
suggestions with click-to-execute action buttons.

Design constraints:
- Suggestions must carry a self-contained ``action`` payload so the
  frontend can route the click without per-suggestion glue code.
- Every suggestion includes a ``severity`` (info | warning | critical)
  so the UI can color the stripe consistently.
- Generators must NOT mutate project state — they're read-only. Any
  side effect happens later when the user clicks the action button.

Phase 1 ships the ``"data"`` stage end-to-end. Subsequent phases will
add ``"cleaning"`` / ``"gold_set"`` / ``"training"`` / ``"eval"`` by
adding generators to ``_STAGE_HANDLERS``.
"""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.project import Project


CoachStage = Literal["data", "cleaning", "gold_set", "training", "eval"]
Severity = Literal["info", "warning", "critical"]

# Thresholds describing what a "thin" / "comfortable" gold-set looks
# like for the typical narrow task. These are deliberately conservative
# — most useful first models on BrewSLM need at least 100 rows to learn
# any non-trivial pattern; 300+ is comfortable for classification with
# a handful of labels.
GOLD_ROW_THIN_MAX: int = 99
GOLD_ROW_COMFORTABLE_MIN: int = 300
# Default top-up the coach proposes when gold is thin: bring the user
# to ``GOLD_ROW_COMFORTABLE_MIN``, capped by the playbook's own
# ``target_count`` ceiling (500). Floor at 20 so the suggestion is
# never a no-op.
SUGGESTED_TOPUP_FLOOR: int = 20
SUGGESTED_TOPUP_CEILING: int = 500


async def _read_gold_row_count(db: AsyncSession, project_id: int) -> int:
    """Read-only sum of ``record_count`` across the project's gold
    datasets (dev + test). Avoids the ``get_or_create_gold_dataset``
    side effect so the coach call doesn't materialize empty Dataset
    rows on first read."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_([DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]),
        )
    )
    rows = result.scalars().all()
    return sum(int(ds.record_count or 0) for ds in rows)


def _topup_count(current: int) -> int:
    """How many synthetic rows to suggest generating to lift the gold
    set toward the comfortable threshold. Clamped to the playbook
    endpoint's accepted range."""
    delta = GOLD_ROW_COMFORTABLE_MIN - max(0, current)
    return max(SUGGESTED_TOPUP_FLOOR, min(SUGGESTED_TOPUP_CEILING, delta))


def _recipe_id_for(project: Project) -> str | None:
    recipe = project.selected_recipe or {}
    rid = recipe.get("recipe_id")
    if isinstance(rid, str) and rid.strip():
        return rid
    return None


async def _data_stage_suggestions(
    db: AsyncSession, project: Project
) -> list[dict[str, Any]]:
    """Suggestions surfaced on the Data tab.

    Phase 1 covers exactly one signal: gold-set row count vs. the
    "comfortable for a narrow task" threshold. Phase 2 will add
    class-imbalance + format-consistency suggestions.
    """
    suggestions: list[dict[str, Any]] = []
    row_count = await _read_gold_row_count(db, project.id)

    if row_count < GOLD_ROW_COMFORTABLE_MIN:
        topup = _topup_count(row_count)
        severity: Severity = "critical" if row_count <= GOLD_ROW_THIN_MAX else "warning"
        recipe_id = _recipe_id_for(project)

        # Build the click-to-execute action. When the project has a
        # selected recipe, point the action at the recipe's
        # ``positives_paraphrase`` playbook (it's the universal mode
        # — registered for every recipe). When there's no recipe yet,
        # we can't trigger run-playbook (the endpoint requires a
        # selected recipe), so fall back to a navigation hint and
        # mark the suggestion as ``navigate`` so the UI can route
        # the user to the recipe picker first.
        if recipe_id:
            action: dict[str, Any] = {
                "kind": "run_playbook",
                "label": f"Generate {topup} synthetic positives",
                "params": {
                    "mode": "positives_paraphrase",
                    "target_count": topup,
                    "target_class": None,
                },
            }
        else:
            action = {
                "kind": "navigate",
                "label": "Pick a recipe first",
                "params": {"target": "recipe-picker"},
            }

        # Headline framing depends on which bucket we're in: "thin" is
        # the urgent one ("most narrow tasks need 100+ rows"); the
        # warning case is the less-urgent "you could comfortably
        # train but more rows would help" framing.
        if row_count <= GOLD_ROW_THIN_MAX:
            title = f"Your gold set has {row_count} rows"
            body = (
                "Most useful first models need at least 100 rows of labeled "
                f"examples. Generating ~{topup} synthetic positives via the "
                "recipe's paraphrase playbook bridges the gap fast."
            )
        else:
            title = f"Your gold set has {row_count} rows — could be stronger"
            body = (
                f"You're past the 100-row floor, but {GOLD_ROW_COMFORTABLE_MIN}+ "
                "rows is the comfortable zone for narrow tasks. Generating "
                f"~{topup} more synthetic positives improves headroom on "
                "your eval splits."
            )

        suggestions.append({
            "id": "data:gold-row-count",
            "title": title,
            "body": body,
            "severity": severity,
            "action": action,
            "context": {
                "gold_row_count": row_count,
                "comfortable_threshold": GOLD_ROW_COMFORTABLE_MIN,
                "thin_threshold": GOLD_ROW_THIN_MAX,
            },
        })

    return suggestions


_STAGE_HANDLERS = {
    "data": _data_stage_suggestions,
}


async def suggest_for_stage(
    db: AsyncSession, project_id: int, stage: CoachStage
) -> dict[str, Any]:
    """Top-level entry point. Returns a serializable payload the
    ``CoachStrip`` UI renders directly.

    The frontend treats an empty ``suggestions`` array as "nothing to
    coach right now" — the strip should remain mounted (so users see
    it light up when state changes) but render an unobtrusive "Looks
    healthy" pill rather than a blank space.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    handler = _STAGE_HANDLERS.get(stage)
    if handler is None:
        # Unknown stage isn't an error — it just means no coach
        # generators have been wired for that surface yet. Return
        # an empty list so the UI degrades gracefully when a new
        # CoachStrip is mounted before its backend handler ships.
        return {
            "project_id": project_id,
            "stage": stage,
            "suggestions": [],
            "handler_available": False,
        }

    suggestions = await handler(db, project)
    return {
        "project_id": project_id,
        "stage": stage,
        "suggestions": suggestions,
        "handler_available": True,
    }
