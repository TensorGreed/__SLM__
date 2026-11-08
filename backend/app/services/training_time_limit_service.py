"""Training time limit for real (external-runtime) runs.

- ``resolve_training_timeout`` — a run's ``external_timeout_seconds`` overrides the server's
  ``EXTERNAL_COMMAND_TIMEOUT_SECONDS`` (default 6 hours) for that run only. Unset keeps the
  server limit, so existing runs behave as before.
- ``time_limit_preflight_warnings`` — warns before start when the limit is longer than the
  task queue's ``CELERY_VISIBILITY_TIMEOUT_SECONDS``: past that, the queue can hand the same
  training task to a worker again while the first is still running.
- ``TimeLimitProjector`` — during a run, once real speed is known, says (once) when the run
  is on pace to need more time than its limit allows.
"""

from __future__ import annotations

from typing import Any

from app.config import settings

TIMEOUT_KEY = "external_timeout_seconds"
SERVER_SETTING = "EXTERNAL_COMMAND_TIMEOUT_SECONDS"
VISIBILITY_SETTING = "CELERY_VISIBILITY_TIMEOUT_SECONDS"
MIN_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 7 * 24 * 3600


def format_hours(seconds: float) -> str:
    """5400 → "1.5 h", 75 → "75 s", 600 → "10 min"."""
    seconds = float(seconds)
    if seconds >= 3600:
        hours = seconds / 3600
        return f"{hours:.0f} h" if abs(hours - round(hours)) < 0.05 else f"{hours:.1f} h"
    if seconds >= 600 or (seconds >= 60 and seconds % 60 == 0):
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def _run_value(config: dict | None) -> Any:
    return (config or {}).get(TIMEOUT_KEY)


def validate_training_timeout(config: dict | None) -> list[str]:
    """Errors for a run's ``external_timeout_seconds`` (none when unset)."""
    value = _run_value(config)
    if value is None or value == "":
        return []
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        return [f"{TIMEOUT_KEY} must be a whole number of seconds (got {value!r})."]
    if not MIN_TIMEOUT_SECONDS <= int(value) <= MAX_TIMEOUT_SECONDS:
        return [
            f"{TIMEOUT_KEY} must be between {MIN_TIMEOUT_SECONDS} seconds and "
            f"{MAX_TIMEOUT_SECONDS // 86400} days (got {int(value)})."
        ]
    return []


def resolve_training_timeout(config: dict | None) -> tuple[int, str]:
    """(seconds, source): the run's own limit when valid (``"run"``), else the server's (``"server"``)."""
    if not validate_training_timeout(config) and _run_value(config) not in (None, ""):
        return int(_run_value(config)), "run"
    return int(settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS), "server"


def timeout_setting_name(source: str | None) -> str:
    """What to change to get a longer limit, as named in messages."""
    return f"{TIMEOUT_KEY} for this run" if source == "run" else SERVER_SETTING


def time_limit_preflight_warnings(config: dict | None) -> list[str]:
    """Start-time warnings about the training time limit (external runs only)."""
    limit, source = resolve_training_timeout(config)
    visibility = int(settings.CELERY_VISIBILITY_TIMEOUT_SECONDS)
    if limit <= visibility:
        return []
    return [
        f"This run's training time limit ({format_hours(limit)}, {timeout_setting_name(source)}) is longer "
        f"than the task queue's visibility timeout ({format_hours(visibility)}, {VISIBILITY_SETTING}). "
        f"A run that lasts longer than {format_hours(visibility)} can be handed to the worker again; "
        f"BrewSLM ignores that second delivery while the first is still training, but raise "
        f"{VISIBILITY_SETTING} above the limit to avoid it."
    ]


class TimeLimitProjector:
    """Warns once when a run is on pace to need more time than its limit.

    Speed is measured from the first observed step, so model loading does not count against
    it; it needs ``min_observe_seconds`` of progress (10 minutes, or 5% of the limit when that
    is shorter) before projecting.
    """

    def __init__(self, limit_seconds: int, started_at: float, *, min_observe_seconds: float | None = None):
        self.limit = float(limit_seconds)
        self.started_at = float(started_at)
        self.min_observe = (
            float(min_observe_seconds) if min_observe_seconds is not None else min(600.0, self.limit * 0.05)
        )
        self._first: tuple[float, int] | None = None
        self.warned = False

    def observe(self, step: int | None, total: int | None, now: float, *, source: str | None = None) -> dict | None:
        if self.warned or not step or not total or step >= total:
            return None
        if self._first is None:
            self._first = (now, int(step))
            return None
        t0, s0 = self._first
        if now - t0 < self.min_observe or step <= s0:
            return None
        rate = (step - s0) / (now - t0)  # steps per second
        elapsed = now - self.started_at
        projected = elapsed + (total - step) / rate
        if projected <= self.limit:
            return None
        self.warned = True
        return {
            "step": int(step),
            "total_steps": int(total),
            "elapsed_seconds": round(elapsed),
            "projected_seconds": round(projected),
            "limit_seconds": int(self.limit),
            "seconds_per_step": round(1 / rate, 2),
            "message": (
                f"On pace to need about {format_hours(projected)}, past its {format_hours(self.limit)} limit: "
                f"step {int(step)} of {int(total)} after {format_hours(elapsed)} "
                f"({1 / rate:.1f} s per step). It will stop at the limit and keep its checkpoints; "
                f"continue from the last one in a run with a longer limit ({timeout_setting_name(source)})."
            ),
        }


async def record_time_limit_warning(experiment_id: int, warning: dict) -> None:
    """Save the warning on ``_runtime.time_limit_warning`` and the run timeline. Best effort."""
    try:
        from sqlalchemy import select

        from app.database import async_session_factory
        from app.models.experiment import Experiment
        from app.models.run_event import SEVERITY_WARNING, STAGE_TRAINING
        from app.services.run_event_service import emit_event

        async with async_session_factory() as db:
            exp = (await db.execute(select(Experiment).where(Experiment.id == int(experiment_id)))).scalar_one_or_none()
            if exp is None:
                return
            config = dict(exp.config or {})
            runtime = dict(config.get("_runtime") or {})
            runtime["time_limit_warning"] = warning
            config["_runtime"] = runtime
            exp.config = config
            await emit_event(
                db,
                project_id=int(exp.project_id),
                run_id=f"exp-{int(experiment_id)}",
                stage=STAGE_TRAINING,
                severity=SEVERITY_WARNING,
                reason_code="training_time_limit_projected",
                summary=f"Training on pace to hit its time limit: {exp.name} — {warning.get('message')}",
                payload={"experiment_id": int(experiment_id), **warning},
            )
            await db.commit()
    except Exception as exc:  # noqa: BLE001 - a warning never disturbs the run
        print(f"[time_limit] could not record warning for experiment {experiment_id}: {exc}", flush=True)
