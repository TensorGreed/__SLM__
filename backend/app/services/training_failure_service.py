"""Why a real (external-runtime) training run stopped, saved once as ``_runtime.failure``.

A run that ends without success used to keep only ``returncode`` (e.g. ``-15``), and the
error text quoted the last output line — often a metric. ``describe_training_failure``
turns what the worker saw into one record:

- ``code`` — ``timeout`` (the worker's training time limit stopped it), ``signal`` (killed
  by a signal it did not send, e.g. out of memory), ``exit`` (non-zero exit), or ``error``
  (the run could not be monitored).
- ``message`` — one sentence for people, e.g. "Stopped at step 60 of 250: reached the
  6-hour limit (EXTERNAL_COMMAND_TIMEOUT_SECONDS). Last checkpoint: step 50."
- ``last_step`` / ``total_steps`` — from the streamed metrics, progress bars and the
  newest ``checkpoint-<step>/trainer_state.json``.
- ``last_checkpoint_step`` / ``resumable_checkpoint_step`` — the newest checkpoint on
  disk, and the newest one a run can continue from (``POST .../resume-from/{step}``).
- ``error_line`` — the last output line that reads like an error; metric, event and
  progress lines are skipped.

The worker's job error, the ``training_start`` Job error and the timeline event all use
``message``, so the three say the same thing.
"""

from __future__ import annotations

import ast
import json
import re
import signal as signal_module
from pathlib import Path
from typing import Any

FAILURE_CODES = ("timeout", "signal", "exit", "error")
TIMEOUT_SETTING = "EXTERNAL_COMMAND_TIMEOUT_SECONDS"

_METRIC_PREFIX = "SLM_METRIC "
_EVENT_PREFIX = "SLM_EVENT "
_PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[")
_TQDM_RE = re.compile(r"^\s*\d+%\|.*\|\s*\d+/\d+")
_ERRORISH_RE = re.compile(r"(error|exception|traceback|killed|out of memory|oom)\b", re.IGNORECASE)
_ERROR_LINE_CAP = 300
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _lines(*streams: str | None) -> list[str]:
    out: list[str] = []
    for stream in streams:
        for raw in _ANSI_RE.sub("", str(stream or "")).replace("\r", "\n").splitlines():
            line = raw.strip()
            if line:
                out.append(line)
    return out


def _is_noise(line: str) -> bool:
    """Metric, event and progress lines say nothing about why a run stopped."""
    if _METRIC_PREFIX in line or _EVENT_PREFIX in line or _TQDM_RE.match(line):
        return True
    if line.startswith("[progress]"):
        return True
    if line.startswith("{") and line.endswith("}"):
        try:
            parsed: Any = json.loads(line)
        except ValueError:
            try:
                parsed = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, dict):
            return True
    return False


def last_error_line(stdout_text: str | None, stderr_text: str | None) -> str:
    """The last output line that reads like an error (stderr first), else the last real line."""
    stderr_lines = [line for line in _lines(stderr_text) if not _is_noise(line)]
    stdout_lines = [line for line in _lines(stdout_text) if not _is_noise(line)]
    chosen = ""
    for pool in (stderr_lines, stdout_lines):
        for line in reversed(pool):
            if _ERRORISH_RE.search(line):
                chosen = line
                break
        if chosen:
            break
    if not chosen:
        chosen = (stderr_lines or stdout_lines or [""])[-1]
    if len(chosen) > _ERROR_LINE_CAP:
        chosen = f"{chosen[:_ERROR_LINE_CAP].rstrip()}..."
    return chosen


def _stream_progress(stdout_text: str | None, stderr_text: str | None) -> tuple[int | None, int | None]:
    """Highest step seen in metric lines and progress bars, and the bar's total."""
    last_step: int | None = None
    total: int | None = None
    for line in _lines(stdout_text, stderr_text):
        idx = line.find(_METRIC_PREFIX)
        if idx >= 0:
            try:
                payload = json.loads(line[idx + len(_METRIC_PREFIX):])
            except ValueError:
                payload = None
            step = payload.get("step") if isinstance(payload, dict) else None
            if isinstance(step, (int, float)) and not isinstance(step, bool):
                last_step = max(last_step or 0, int(step))
            continue
        match = _PROGRESS_RE.search(line) if "%|" in line else None
        if match:
            done, of = int(match.group(1)), int(match.group(2))
            if of > 0:
                total = of
                last_step = max(last_step or 0, done)
    return last_step, total


def _checkpoint_dirs(output_dir: Path | None) -> list[tuple[int, Path]]:
    if output_dir is None:
        return []
    found: list[tuple[int, Path]] = []
    try:
        for path in Path(output_dir).glob("checkpoint-*"):
            suffix = path.name.split("-", 1)[-1]
            if path.is_dir() and suffix.isdigit():
                found.append((int(suffix), path))
    except OSError:
        return []
    return sorted(found)


def _checkpoint_progress(output_dir: Path | None) -> dict[str, int | None]:
    from app.services.training_service import checkpoint_dir_details

    result: dict[str, int | None] = {
        "last_checkpoint_step": None,
        "resumable_checkpoint_step": None,
        "checkpoint_global_step": None,
        "max_steps": None,
    }
    for step, path in reversed(_checkpoint_dirs(output_dir)):
        details = checkpoint_dir_details(path)
        if result["last_checkpoint_step"] is None:
            result["last_checkpoint_step"] = step
            result["checkpoint_global_step"] = details.get("global_step")
            try:
                state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
                max_steps = state.get("max_steps") if isinstance(state, dict) else None
                if isinstance(max_steps, (int, float)) and max_steps > 0:
                    result["max_steps"] = int(max_steps)
            except (OSError, ValueError):
                pass
        if details.get("resumable"):
            result["resumable_checkpoint_step"] = step
            break
    return result


def format_duration(seconds: int | float | None) -> str:
    """6-hour / 90-minute / 75-second — the limit as people say it."""
    try:
        value = int(seconds or 0)
    except (TypeError, ValueError):
        return "time"
    if value <= 0:
        return "time"
    for unit, size in (("hour", 3600), ("minute", 60)):
        if value >= size and value % size == 0:
            return f"{value // size}-{unit}"
    if value >= 3600:
        return f"{value / 3600:.1f}-hour"
    return f"{value}-second"


def _signal_name(returncode: int) -> str:
    try:
        return signal_module.Signals(-returncode).name
    except (ValueError, TypeError):
        return f"signal {-returncode}"


def describe_training_failure(
    *,
    returncode: int | None,
    stdout_text: str | None = None,
    stderr_text: str | None = None,
    output_dir: Path | str | None = None,
    stop_reason: str | None = None,
    timeout_seconds: int | None = None,
    timeout_source: str | None = None,
    error: str | None = None,
    total_steps_hint: int | None = None,
) -> dict[str, Any]:
    """Build the ``_runtime.failure`` record for a run that did not complete."""
    out_dir = Path(output_dir) if output_dir else None
    stream_step, stream_total = _stream_progress(stdout_text, stderr_text)
    ckpt = _checkpoint_progress(out_dir)
    steps = [s for s in (stream_step, ckpt["checkpoint_global_step"], ckpt["last_checkpoint_step"]) if s]
    last_step = max(steps) if steps else None
    total_steps = ckpt["max_steps"] or stream_total or (int(total_steps_hint) if total_steps_hint else None)
    error_line = last_error_line(stdout_text, stderr_text)

    if stop_reason == "timeout":
        code = "timeout"
        from app.services.training_time_limit_service import timeout_setting_name

        setting = timeout_setting_name(timeout_source)
        reason = f"reached the {format_duration(timeout_seconds)} limit ({setting})"
    elif error:
        code = "error"
        reason = f"the run could not be monitored: {str(error)[:_ERROR_LINE_CAP]}"
    elif isinstance(returncode, int) and returncode < 0:
        code = "signal"
        name = _signal_name(returncode)
        reason = f"the process was stopped by {name}"
        if name == "SIGKILL":
            reason += " (often the system running out of memory)"
        if error_line:
            reason += f"; last error: {error_line}"
    else:
        code = "exit"
        reason = f"the process exited with code {returncode}"
        if error_line:
            reason += f": {error_line}"

    if last_step and total_steps:
        head = f"Stopped at step {last_step} of {total_steps}"
    elif last_step:
        head = f"Stopped at step {last_step}"
    else:
        head = "Stopped before the first logged step"
    resumable = ckpt["resumable_checkpoint_step"]
    if resumable:
        tail = f" Last checkpoint: step {resumable}."
    elif ckpt["last_checkpoint_step"]:
        tail = f" Checkpoint step {ckpt['last_checkpoint_step']} is incomplete and can't be resumed."
    else:
        tail = " No checkpoint was saved."
    message = f"{head}: {reason}.{tail}"

    record: dict[str, Any] = {
        "code": code,
        "message": message,
        "returncode": returncode,
        "last_step": last_step,
        "total_steps": total_steps,
        "last_checkpoint_step": ckpt["last_checkpoint_step"],
        "resumable_checkpoint_step": resumable,
        "error_line": error_line or None,
    }
    if code == "timeout":
        record["timeout_seconds"] = int(timeout_seconds) if timeout_seconds else None
        record["setting"] = setting
        record["timeout_source"] = timeout_source or "server"
    if code == "signal" and isinstance(returncode, int):
        record["signal"] = _signal_name(returncode)
    return record


def failure_from_config(config: dict | None) -> dict | None:
    """The saved ``_runtime.failure`` record, if any."""
    runtime = (config or {}).get("_runtime")
    failure = runtime.get("failure") if isinstance(runtime, dict) else None
    return failure if isinstance(failure, dict) and failure.get("message") else None


async def emit_failure_event(*, project_id: int, experiment_id: int, name: str, failure: dict) -> None:
    """``training_<code>`` error event on the run timeline. Best effort."""
    try:
        from app.database import async_session_factory
        from app.models.run_event import SEVERITY_ERROR, STAGE_TRAINING
        from app.services.run_event_service import emit_event

        async with async_session_factory() as event_db:
            await emit_event(
                event_db,
                project_id=int(project_id),
                run_id=f"exp-{int(experiment_id)}",
                stage=STAGE_TRAINING,
                severity=SEVERITY_ERROR,
                reason_code=f"training_{failure.get('code') or 'exit'}",
                summary=f"Training failed: {name} — {failure.get('message')}",
                payload={"experiment_id": int(experiment_id), "backend": "external", "failure": failure},
            )
            await event_db.commit()
    except Exception as event_exc:  # noqa: BLE001 - observability never blocks the status
        print(
            f"[run_event] training_failed_emit_failed experiment_id={experiment_id}: {event_exc}",
            flush=True,
        )


async def load_failure_message(experiment_id: int) -> str:
    """The saved failure message for a run, or "" (never raises)."""
    try:
        from sqlalchemy import select

        from app.database import async_session_factory
        from app.models.experiment import Experiment

        async with async_session_factory() as db:
            config = (
                await db.execute(select(Experiment.config).where(Experiment.id == int(experiment_id)))
            ).scalar_one_or_none()
        failure = failure_from_config(config)
        return str(failure.get("message")) if failure else ""
    except Exception:  # noqa: BLE001
        return ""
