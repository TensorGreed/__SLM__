"""Declarative achievement catalog (Lab Journal).

Single source of truth for the achievement spine. Each entry is a
plain :class:`Achievement` dataclass; the gamification service reads
``id`` + ``xp`` when unlocking, the API returns the full record for
the Lab Journal drawer.

Tiers shape the drawer layout:

- ``onboarding`` — mirrors the ProjectGuidePage checklist. Sets the
  newbie's first-week trajectory. Highest reward density.
- ``mastery`` — real ML-skill milestones (F1 thresholds,
  preference-pair training, RAG, compression, multi-model breadth).
  The mid- to long-game.
- ``discovery`` — easter eggs. ``hidden=True`` so the drawer shows
  them as ``▢ ???`` until unlocked.

Adding a new achievement is one entry here + one branch in
``gamification_service.process_run_event`` for the trigger. No
migration needed (unlocked IDs are stored as a list on the project's
gamification JSON column, so the catalog is free to evolve).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AchievementTier = Literal["onboarding", "mastery", "discovery"]


@dataclass(frozen=True)
class Achievement:
    """One achievement definition.

    ``id`` is the stable string we store in
    ``project.gamification.achievements_unlocked``. The frontend
    treats it as opaque.
    """

    id: str
    title: str
    description: str
    xp: int
    tier: AchievementTier
    hidden: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "xp": self.xp,
            "tier": self.tier,
            "hidden": self.hidden,
        }


# Order matters: this is also the display order in the drawer.
ACHIEVEMENTS: tuple[Achievement, ...] = (
    # ── Onboarding ────────────────────────────────────────────────
    Achievement(
        id="domain_set",
        title="Domain locked in",
        description="Picked a domain pack or profile for the project.",
        xp=50,
        tier="onboarding",
    ),
    Achievement(
        id="first_ingest",
        title="Data flows",
        description="Imported your first dataset.",
        xp=100,
        tier="onboarding",
    ),
    Achievement(
        id="first_clean",
        title="Spotless",
        description="Ran a successful cleaning pass on at least one document.",
        xp=100,
        tier="onboarding",
    ),
    Achievement(
        id="first_train",
        title="First Forge",
        description="Trained your first model to completion.",
        xp=200,
        tier="onboarding",
    ),
    Achievement(
        id="first_eval",
        title="Benchmark Run",
        description="Ran your first evaluation against a trained model.",
        xp=150,
        tier="onboarding",
    ),
    Achievement(
        id="first_export",
        title="Artifact Sealed",
        description="Exported a trained model — ready to ship.",
        xp=250,
        tier="onboarding",
    ),
    Achievement(
        id="first_deploy",
        title="Shipped to Production",
        description="Promoted a deployment version to live.",
        xp=500,
        tier="onboarding",
    ),
    # ── Mastery ───────────────────────────────────────────────────
    Achievement(
        id="f1_above_80",
        title="Decent Hacker",
        description="Eval pass rate crossed 80%.",
        xp=200,
        tier="mastery",
    ),
    Achievement(
        id="f1_above_90",
        title="Tuner",
        description="Eval pass rate crossed 90%.",
        xp=400,
        tier="mastery",
    ),
    Achievement(
        id="f1_above_95",
        title="Surgeon",
        description="Eval pass rate crossed 95%. Most teams plateau before this.",
        xp=800,
        tier="mastery",
    ),
    Achievement(
        id="multi_model",
        title="Three Recipes",
        description="Trained on at least three different base models.",
        xp=300,
        tier="mastery",
    ),
    Achievement(
        id="multi_dataset",
        title="Polyglot",
        description="Imported data from three different source connectors.",
        xp=300,
        tier="mastery",
    ),
    Achievement(
        id="compression_used",
        title="Quantized",
        description="Shipped a compressed export. Fits on weaker hardware.",
        xp=200,
        tier="mastery",
    ),
    Achievement(
        id="dpo_done",
        title="Aligned",
        description="Trained on a preference-pair dataset (DPO / ORPO).",
        xp=400,
        tier="mastery",
    ),
    Achievement(
        id="rag_done",
        title="Grounded",
        description="Trained a RAG pipeline with explicit context.",
        xp=400,
        tier="mastery",
    ),
    Achievement(
        id="saved_mapping_reused",
        title="Repeat Customer",
        description="Re-ran an import from a saved mapping. Reproducibility.",
        xp=150,
        tier="mastery",
    ),
    Achievement(
        id="ten_trainings",
        title="Ten Forges",
        description="Completed ten successful training runs in this project.",
        xp=500,
        tier="mastery",
    ),
    # ── Discovery (hidden until unlocked) ─────────────────────────
    Achievement(
        id="night_owl",
        title="Night Owl",
        description="Started a training run between midnight and 5am.",
        xp=100,
        tier="discovery",
        hidden=True,
    ),
    Achievement(
        id="recovered_from_oom",
        title="OOM Survivor",
        description="Recovered from a training OOM via autopilot.",
        xp=250,
        tier="discovery",
        hidden=True,
    ),
    Achievement(
        id="force_used",
        title="With Authority",
        description="Used --force on a low-confidence dataset import. You knew what you were doing.",
        xp=50,
        tier="discovery",
        hidden=True,
    ),
    Achievement(
        id="plugin_loaded",
        title="Modder",
        description="Registered a custom mapper plugin.",
        xp=300,
        tier="discovery",
        hidden=True,
    ),
    Achievement(
        id="llm_assist_used",
        title="Phone a Friend",
        description="Triggered LLM-assisted mapping suggestion.",
        xp=150,
        tier="discovery",
        hidden=True,
    ),
    Achievement(
        id="clean_speedrun",
        title="Speedrunner",
        description="Cleaned a document end-to-end in under 30 seconds.",
        xp=100,
        tier="discovery",
        hidden=True,
    ),
)


ACHIEVEMENT_BY_ID: dict[str, Achievement] = {a.id: a for a in ACHIEVEMENTS}


# Level title ladder. Index 0 unused; index 1 = L1 title.
# Levels beyond the explicit list keep the last entry ("Distinguished").
LEVEL_TITLES: tuple[str, ...] = (
    "",  # 0 — never displayed
    "Intern",          # 1
    "Lab Tech",        # 2
    "ML Engineer",     # 3
    "ML Engineer",     # 4
    "Senior",          # 5
    "Senior",          # 6
    "Senior",          # 7
    "Staff",           # 8
    "Staff",           # 9
    "Principal",       # 10
    "Principal",       # 11
    "Principal",       # 12
    "Principal",       # 13
    "Principal",       # 14
    "Distinguished",   # 15+
)


def level_title(level: int) -> str:
    """Return the human-readable rank for ``level``. Levels beyond
    the ladder collapse to the last title."""

    if level < 1:
        return ""
    if level >= len(LEVEL_TITLES):
        return LEVEL_TITLES[-1]
    return LEVEL_TITLES[level]
