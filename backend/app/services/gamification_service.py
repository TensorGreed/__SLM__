"""Lab Journal — per-project gamification progression service.

Hangs off the existing ``run_events`` stream as its XP feedstock.
The service is intentionally a small, self-contained module:

- ``award_xp`` is the only path that mutates the JSON column.
- ``check_and_unlock`` is set-membership idempotent — firing the
  same trigger twice never double-pays.
- ``process_run_event`` is the dispatcher called from
  ``run_event_service.emit_event`` (best-effort, wrapped in
  try/except at the call site).
- All writes happen on the caller's session via ``await db.flush()``
  — never a standalone commit. The caller owns the transaction.

The whole module is designed to fail soft: a bug in here must
never break the data write path. Callers wrap us in try/except;
internally we use ``with contextlib.suppress(Exception)`` around
anything optional so we degrade silently.

Persistence shape (lives in ``Project.gamification`` JSON column)::

    {
        "xp_balance": 1240,
        "level": 3,
        "achievements_unlocked": ["domain_set", "first_ingest", "first_train"],
        "milestones": {
            "first_train": "2026-05-14T12:34:56+00:00",
            ...
        },
        "recent_unlocks": [
            {
                "achievement_id": "first_train",
                "unlocked_at": "2026-05-14T12:34:56+00:00",
                "xp_awarded": 200,
                "level_after": 3,    # null if no level change
            },
            ...
        ],
        "counters": {
            "base_models_trained": ["llama3-8b", "qwen2-7b"],
            "import_sources_used": ["jsonl", "kaggle"],
            "successful_training_runs": 4,
        }
    }
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.models.project import Project
from app.services.gamification.achievements import (
    ACHIEVEMENT_BY_ID,
    ACHIEVEMENTS,
    level_title,
)


# Cap on how many recent unlocks the JSON column carries inline. The
# drawer's "show all" path re-fetches and renders the longer
# ``achievements_unlocked`` list; ``recent_unlocks`` only drives the
# toast-diff loop on the frontend.
_RECENT_UNLOCKS_CAP: int = 50

# Toast-spam buffer: per-process dict mapping ``(project_id,
# reason_code)`` → last-grant timestamp. A repeat continuous-event
# award within this window drips XP silently (no recent_unlocks
# entry, no toast). Achievement unlocks aren't affected — they're
# one-time by definition.
_TOAST_SUPPRESS_WINDOW_SECONDS: int = 30
_RECENT_GRANTS: dict[tuple[int, str], datetime] = {}
_RECENT_GRANTS_LOCK = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── State shape ────────────────────────────────────────────────────────


def default_state() -> dict[str, Any]:
    """Canonical empty progression for a fresh project."""

    return {
        "xp_balance": 0,
        "level": 1,
        "achievements_unlocked": [],
        "milestones": {},
        "recent_unlocks": [],
        "counters": {
            "base_models_trained": [],
            "import_sources_used": [],
            "successful_training_runs": 0,
        },
    }


def _coerce_state(raw: Any) -> dict[str, Any]:
    """Defensive normalize: a real project might have ``None``, a
    partial dict, or a dict with extra keys from a future version.
    Always return a dict matching :func:`default_state`'s key set."""

    base = default_state()
    if not isinstance(raw, dict):
        return base
    for key in base:
        if key in raw:
            base[key] = raw[key]
    # The counters sub-dict needs the same shallow merge.
    if isinstance(raw.get("counters"), dict):
        counters = default_state()["counters"]
        counters.update(raw["counters"])
        base["counters"] = counters
    return base


# ── XP curve ───────────────────────────────────────────────────────────


def xp_to_next(level: int) -> int:
    """XP required to advance from ``level`` to ``level + 1``.

    ``floor(100 * level ** 1.5)`` — fast onboarding, steep mastery.
    L1→2 = 100, L2→3 = 282, L5→6 = 1118, L10→11 = 3162.
    """

    if level < 1:
        return 0
    return int(math.floor(100 * (level ** 1.5)))


def level_for_total_xp(total_xp: int) -> tuple[int, int, int]:
    """Walk the XP curve, returning ``(level, xp_into_level,
    xp_to_next_level)``. Used after every ``award_xp`` so the
    frontend can render the progress bar without re-deriving."""

    if total_xp <= 0:
        return 1, 0, xp_to_next(1)
    level = 1
    remaining = total_xp
    # Hard cap iteration at 200 so a corrupted XP balance can't loop
    # forever; the curve at L200 is well above any realistic balance.
    for _ in range(200):
        cost = xp_to_next(level)
        if remaining < cost:
            return level, remaining, cost
        remaining -= cost
        level += 1
    return level, remaining, xp_to_next(level)


# ── State reads + writes ───────────────────────────────────────────────


async def _load_project(
    db: AsyncSession, project_id: int
) -> Project | None:
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    return result.scalar_one_or_none()


async def get_progression(
    db: AsyncSession, project_id: int
) -> dict[str, Any]:
    """Return the project's progression state (creating the empty
    shape lazily for projects that predate the column).

    This is a *read-only* view — the persisted column is not written
    here. Callers that need to ensure persistence go through
    :func:`award_xp` / :func:`check_and_unlock`.
    """

    project = await _load_project(db, project_id)
    if project is None:
        raise ValueError(f"project_{project_id}_not_found")
    state = _coerce_state(project.gamification)
    # Compute progress-bar fields fresh on every read; cheap.
    level, into_level, to_next = level_for_total_xp(int(state["xp_balance"] or 0))
    state["level"] = level
    state["xp_into_level"] = into_level
    state["xp_to_next_level"] = to_next
    state["level_title"] = level_title(level)
    return state


async def _persist_state(
    db: AsyncSession, project: Project, state: dict[str, Any]
) -> None:
    """Strip the derived progress-bar fields before persisting so the
    column stays minimal + the next ``get_progression`` recomputes
    from the canonical xp_balance.

    Use :func:`flag_modified` because SQLAlchemy's default JSON
    column comparator doesn't reliably detect repeated full-dict
    re-assignments within the same session — the second + third
    ``project.gamification = new_dict`` calls in a single
    ``process_run_event`` invocation can silently no-op without
    this flag.
    """

    persisted = {
        k: v
        for k, v in state.items()
        if k not in {"xp_into_level", "xp_to_next_level", "level_title"}
    }
    project.gamification = persisted
    flag_modified(project, "gamification")
    await db.flush()


def _should_suppress_toast(project_id: int, reason_code: str | None) -> bool:
    """Recent-grants dedup: a repeat continuous-event XP award for
    the same project + reason within
    :data:`_TOAST_SUPPRESS_WINDOW_SECONDS` drips XP silently."""

    if not reason_code:
        return False
    key = (project_id, reason_code)
    now = _utcnow()
    with _RECENT_GRANTS_LOCK:
        last = _RECENT_GRANTS.get(key)
        _RECENT_GRANTS[key] = now
    if last is None:
        return False
    return (now - last).total_seconds() < _TOAST_SUPPRESS_WINDOW_SECONDS


# ── Award + unlock ─────────────────────────────────────────────────────


async def award_xp(
    db: AsyncSession,
    project_id: int,
    amount: int,
    reason: str,
    *,
    suppress_toast: bool = False,
) -> dict[str, Any]:
    """Add ``amount`` XP to the project, recompute level, and append
    a level-up entry to ``recent_unlocks`` when a boundary is crossed.

    ``reason`` is a short stable string ("dataset_import_run",
    "training_complete", etc.) used for the toast-spam dedup buffer.
    Returns the updated progression dict (suitable for an API
    response).
    """

    if amount <= 0:
        return await get_progression(db, project_id)

    project = await _load_project(db, project_id)
    if project is None:
        raise ValueError(f"project_{project_id}_not_found")

    state = _coerce_state(project.gamification)
    prev_level = int(state.get("level") or 1)
    new_balance = int(state["xp_balance"]) + amount
    state["xp_balance"] = new_balance
    new_level, _, _ = level_for_total_xp(new_balance)
    state["level"] = new_level

    if new_level > prev_level and not suppress_toast:
        # Level-up earns its own recent-unlocks slot — the frontend
        # treats this like a special achievement-style toast.
        state.setdefault("recent_unlocks", []).insert(
            0,
            {
                "kind": "level_up",
                "level_after": new_level,
                "title": level_title(new_level),
                "xp_awarded": 0,
                "unlocked_at": _utcnow().isoformat(),
            },
        )
        state["recent_unlocks"] = state["recent_unlocks"][:_RECENT_UNLOCKS_CAP]

    await _persist_state(db, project, state)
    return await get_progression(db, project_id)


async def check_and_unlock(
    db: AsyncSession,
    project_id: int,
    achievement_id: str,
) -> dict[str, Any] | None:
    """Grant ``achievement_id`` if it isn't already in
    ``achievements_unlocked``. Idempotent — the second call is a
    no-op.

    Returns the achievement record (with timestamp) when newly
    unlocked, ``None`` when already had it.
    """

    achievement = ACHIEVEMENT_BY_ID.get(achievement_id)
    if achievement is None:
        return None

    project = await _load_project(db, project_id)
    if project is None:
        return None

    state = _coerce_state(project.gamification)
    if achievement_id in state["achievements_unlocked"]:
        return None

    now_iso = _utcnow().isoformat()
    state["achievements_unlocked"].append(achievement_id)
    state["milestones"][achievement_id] = now_iso
    new_balance = int(state["xp_balance"]) + achievement.xp
    state["xp_balance"] = new_balance
    new_level, _, _ = level_for_total_xp(new_balance)
    level_changed = new_level > int(state.get("level") or 1)
    state["level"] = new_level

    unlock_entry = {
        "kind": "achievement",
        "achievement_id": achievement_id,
        "title": achievement.title,
        "description": achievement.description,
        "tier": achievement.tier,
        "xp_awarded": achievement.xp,
        "unlocked_at": now_iso,
        "level_after": new_level if level_changed else None,
    }
    state.setdefault("recent_unlocks", []).insert(0, unlock_entry)
    state["recent_unlocks"] = state["recent_unlocks"][:_RECENT_UNLOCKS_CAP]

    await _persist_state(db, project, state)
    return unlock_entry


# ── Counter bumpers ────────────────────────────────────────────────────


async def _bump_counter(
    db: AsyncSession, project_id: int, mutator
) -> dict[str, Any]:
    """Apply ``mutator(state["counters"])`` and persist. The mutator
    is a small inline function the caller passes so we don't have to
    name every increment in the service surface."""

    project = await _load_project(db, project_id)
    if project is None:
        return default_state()
    state = _coerce_state(project.gamification)
    mutator(state["counters"])
    await _persist_state(db, project, state)
    return state


# ── Process RunEvent → dispatcher ──────────────────────────────────────


async def process_run_event(db: AsyncSession, event: Any) -> None:
    """Translate a freshly-flushed :class:`RunEvent` into XP +
    achievement updates.

    Called from ``run_event_service.emit_event`` after ``db.flush()``.
    The whole function is wrapped in try/except at the call site, so
    any per-handler failure is observed but doesn't leak.

    Dispatch is keyed by ``stage`` + ``reason_code``; payload shape
    is interpreted per handler.
    """

    project_id = int(getattr(event, "project_id", 0) or 0)
    if project_id <= 0:
        return
    stage = str(getattr(event, "stage", "") or "")
    severity = str(getattr(event, "severity", "") or "")
    reason = str(getattr(event, "reason_code", "") or "")
    payload = dict(getattr(event, "payload", None) or {})

    # ── Dataset import (Phase A–H) ────────────────────────────────
    if reason == "dataset_import_run":
        suppress = _should_suppress_toast(project_id, "dataset_import_run")
        await award_xp(
            db,
            project_id,
            amount=20,
            reason="dataset_import_run",
            suppress_toast=suppress,
        )
        await check_and_unlock(db, project_id, "first_ingest")

        # multi_dataset: count distinct source connectors used.
        src = payload.get("source_id")
        if isinstance(src, str) and src:
            def _bump_source(counters, _src=src):
                used = counters.setdefault("import_sources_used", [])
                if _src not in used:
                    used.append(_src)
            await _bump_counter(db, project_id, _bump_source)
            project = await _load_project(db, project_id)
            if project is not None:
                state = _coerce_state(project.gamification)
                if len(set(state["counters"]["import_sources_used"])) >= 3:
                    await check_and_unlock(db, project_id, "multi_dataset")

        # saved_mapping_reused: payload carries config_id when the
        # run came from a saved config.
        if payload.get("config_id") is not None:
            await check_and_unlock(db, project_id, "saved_mapping_reused")
        return

    # ── Cleaning ──────────────────────────────────────────────────
    if stage == "cleaning" and severity == "info":
        await award_xp(
            db, project_id, amount=20, reason="cleaning_complete",
            suppress_toast=_should_suppress_toast(project_id, "cleaning_complete"),
        )
        await check_and_unlock(db, project_id, "first_clean")
        return

    # ── Training run completed ────────────────────────────────────
    if stage == "training" and severity == "info" and (
        "experiment_id" in payload or str(getattr(event, "run_id", "")).startswith("exp-")
    ):
        await award_xp(
            db, project_id, amount=50, reason="training_complete",
            suppress_toast=_should_suppress_toast(project_id, "training_complete"),
        )
        await check_and_unlock(db, project_id, "first_train")

        # Counter: successful training runs. ten_trainings @ 10.
        def _bump_trainings(counters):
            counters["successful_training_runs"] = int(
                counters.get("successful_training_runs", 0) or 0
            ) + 1
        await _bump_counter(db, project_id, _bump_trainings)
        project = await _load_project(db, project_id)
        if project is not None:
            state = _coerce_state(project.gamification)
            if int(state["counters"]["successful_training_runs"]) >= 10:
                await check_and_unlock(db, project_id, "ten_trainings")

        # Counter: distinct base models trained on. multi_model @ 3.
        base_model = payload.get("base_model") or payload.get("model_name")
        if isinstance(base_model, str) and base_model:
            def _bump_models(counters, _bm=base_model):
                used = counters.setdefault("base_models_trained", [])
                if _bm not in used:
                    used.append(_bm)
            await _bump_counter(db, project_id, _bump_models)
            project = await _load_project(db, project_id)
            if project is not None:
                state = _coerce_state(project.gamification)
                if len(set(state["counters"]["base_models_trained"])) >= 3:
                    await check_and_unlock(db, project_id, "multi_model")

        # Discovery: night_owl if started 00:00–05:00 in UTC. (We
        # don't know the user's local timezone; UTC is a defensible
        # approximation for "the wee hours.")
        ts = getattr(event, "ts", None)
        if isinstance(ts, datetime) and 0 <= ts.hour < 5:
            await check_and_unlock(db, project_id, "night_owl")
        return

    # ── Training failure: recovered-from-OOM is checked when the
    # autopilot resolves it, not here.

    # ── Eval ──────────────────────────────────────────────────────
    if stage == "eval" and severity == "info":
        pass_rate = payload.get("pass_rate")
        try:
            pr = float(pass_rate) if pass_rate is not None else None
        except (TypeError, ValueError):
            pr = None

        await check_and_unlock(db, project_id, "first_eval")

        if pr is not None:
            tier_xp = 0
            if pr >= 0.9:
                tier_xp = 120
            elif pr >= 0.8:
                tier_xp = 60
            elif pr >= 0.6:
                tier_xp = 30
            if tier_xp:
                await award_xp(
                    db, project_id, amount=tier_xp,
                    reason=f"eval_pass_rate_{int(pr * 100)}",
                    suppress_toast=_should_suppress_toast(
                        project_id, f"eval_pass_rate_{int(pr * 100)}"
                    ),
                )
            # F1 ladder: passing 0.95 grants all three tiers; each
            # check is idempotent so this is safe to call greedily.
            if pr >= 0.80:
                await check_and_unlock(db, project_id, "f1_above_80")
            if pr >= 0.90:
                await check_and_unlock(db, project_id, "f1_above_90")
            if pr >= 0.95:
                await check_and_unlock(db, project_id, "f1_above_95")
        return

    # ── Export ────────────────────────────────────────────────────
    if stage == "export" and severity == "info":
        await award_xp(
            db, project_id, amount=40, reason="export_complete",
            suppress_toast=_should_suppress_toast(project_id, "export_complete"),
        )
        await check_and_unlock(db, project_id, "first_export")
        # Compression: format field is "gguf" or includes "q4_k_m"
        # etc. Treat anything with a quantization hint as compressed.
        fmt = str(payload.get("format") or "").lower()
        if any(token in fmt for token in ("gguf", "q4", "q5", "q8", "awq", "gptq")):
            await check_and_unlock(db, project_id, "compression_used")
        return

    # ── Deployment ────────────────────────────────────────────────
    if stage == "deployment" and severity == "info" and payload.get("action") == "promote":
        await award_xp(
            db, project_id, amount=60, reason="deployment_promote",
            suppress_toast=_should_suppress_toast(project_id, "deployment_promote"),
        )
        await check_and_unlock(db, project_id, "first_deploy")
        return
