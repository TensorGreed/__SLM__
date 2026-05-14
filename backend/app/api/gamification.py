"""Lab Journal API — per-project progression state + achievement catalog.

Two read-only endpoints feed the frontend:

- ``GET /api/projects/{project_id}/gamification`` — current XP /
  level / unlocked achievements / recent unlocks. The frontend
  polls this every 10s while the ProgressChip is visible.
- ``GET /api/projects/{project_id}/gamification/achievements`` — the
  full declarative catalog (with per-achievement ``unlocked`` +
  ``unlocked_at`` flags) for the Lab Journal drawer.

No write endpoints: gamification only mutates via the RunEvent tap
in ``run_event_service.emit_event``. Treating this as a side-effect
of the real workflow events keeps the user's progression honest —
they can't cheat by hitting an endpoint.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.gamification.achievements import ACHIEVEMENTS, level_title
from app.services.gamification_service import (
    default_state,
    get_progression,
    level_for_total_xp,
)


router = APIRouter(
    prefix="/projects/{project_id}/gamification", tags=["Gamification"]
)


@router.get("")
async def read_progression(
    project_id: int, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Current progression for the chip + drawer status section."""

    try:
        return await get_progression(db, project_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/achievements")
async def read_achievements(
    project_id: int, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Full catalog + per-achievement ``unlocked`` + ``unlocked_at``
    flags. The drawer renders this directly into its Unlocked /
    Locked / Discovery sections."""

    try:
        state = await get_progression(db, project_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    unlocked_set = set(state.get("achievements_unlocked") or [])
    milestones = state.get("milestones") or {}

    items = []
    for achievement in ACHIEVEMENTS:
        items.append(
            {
                **achievement.to_dict(),
                "unlocked": achievement.id in unlocked_set,
                "unlocked_at": milestones.get(achievement.id),
            }
        )

    return {
        "achievements": items,
        "summary": {
            "total": len(ACHIEVEMENTS),
            "unlocked": len(unlocked_set),
            "level": state["level"],
            "level_title": state.get("level_title", level_title(state["level"])),
            "xp_balance": state["xp_balance"],
        },
    }


__all__ = ["router", "default_state", "level_for_total_xp"]
