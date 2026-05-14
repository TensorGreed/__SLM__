"""Gamification subpackage — Lab Journal progression system.

Re-exports the achievement catalog so callers can use the short
`from app.services.gamification import ACHIEVEMENTS` path.
"""

from app.services.gamification.achievements import (
    ACHIEVEMENTS,
    ACHIEVEMENT_BY_ID,
    Achievement,
)

__all__ = ["ACHIEVEMENTS", "ACHIEVEMENT_BY_ID", "Achievement"]
