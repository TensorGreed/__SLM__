"""Track + stop the OS process behind an external training run.

The external runtime (``builtin.external_celery``) runs ``scripts/train.py``
as a subprocess of the Celery worker. Cancelling the experiment from the API
process has to reach that subprocess, which lives in another process tree
(the worker's). Celery's ``revoke(terminate=True)`` cannot do it: the worker
runs ``--pool=threads`` (terminate is a no-op there) and, even under prefork,
it would signal the pool child, not the training process.

So the worker records the process in a small file next to the run's outputs
and starts it in its own session (process group):

* ``training_process.json`` — ``{pid, pgid, start_time, command, ...}``,
  written by the worker right after spawning, removed after it exits.
* ``cancel_requested.json`` — written by ``cancel_training`` before it
  signals anything. The worker checks it before spawning (a cancel that
  landed while the task was queued) and every couple of seconds while the
  process runs (a second, worker-side path to stop it), and the completion
  monitor reads it so an exit caused by the cancel is recorded CANCELLED,
  never COMPLETED / FAILED.

``terminate_training_process`` verifies the recorded PID is still the same
process before signalling (``/proc/<pid>/stat`` start time + command line —
a PID can be reused after the run exits), then sends SIGTERM to the process
group and SIGKILL after a grace period.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROCESS_FILE_NAME = "training_process.json"
CANCEL_MARKER_NAME = "cancel_requested.json"

_SHELL_NAMES = {"sh", "bash", "dash", "zsh"}
_PROC = Path("/proc")


# ── /proc helpers (Linux; every caller tolerates None) ─────────────────


def _read_stat_fields(pid: int) -> list[str] | None:
    """Fields of /proc/<pid>/stat after the ``(comm)`` field, or None."""
    try:
        raw = (_PROC / str(int(pid)) / "stat").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    # rest[0] is field 3 (state); field N lives at rest[N - 3].
    return raw[close + 1:].split()


def process_start_time(pid: int) -> str | None:
    """Kernel start time (clock ticks since boot) — stable across exec."""
    fields = _read_stat_fields(pid)
    if not fields or len(fields) < 20:
        return None
    return fields[19]  # field 22: starttime


def _process_state(pid: int) -> str | None:
    fields = _read_stat_fields(pid)
    return fields[0] if fields else None


def _process_pgid(pid: int) -> int | None:
    fields = _read_stat_fields(pid)
    if not fields or len(fields) < 3:
        return None
    try:
        return int(fields[2])  # field 5: pgrp
    except ValueError:
        return None


def process_cmdline(pid: int) -> list[str] | None:
    try:
        raw = (_PROC / str(int(pid)) / "cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    if not raw:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.rstrip(b"\0").split(b"\0")]


def _proc_available() -> bool:
    return (_PROC / "self" / "stat").exists()


def pid_alive(pid: int) -> bool:
    """True when ``pid`` exists and is not a zombie."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except (OSError, ValueError):
        return False
    return _process_state(pid) != "Z"


def cmdline_matches_command(argv: list[str] | None, command: str) -> bool:
    """Is ``argv`` the recorded training command (or the shell running it)?"""
    if not argv or not command:
        return False
    if len(argv) >= 3 and os.path.basename(argv[0]) in _SHELL_NAMES and argv[1] == "-c":
        return argv[2] == command
    try:
        expected = shlex.split(command)
    except ValueError:
        return False
    return argv == expected


def _group_members(pgid: int) -> list[int]:
    """Live (non-zombie) pids in process group ``pgid``. Needs /proc."""
    members: list[int] = []
    try:
        entries = list(_PROC.iterdir())
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        fields = _read_stat_fields(pid)
        if not fields or len(fields) < 3:
            continue
        if fields[0] == "Z":
            continue
        try:
            if int(fields[2]) == int(pgid):
                members.append(pid)
        except ValueError:
            continue
    return members


def _group_alive(pgid: int) -> bool:
    if _proc_available():
        return bool(_group_members(pgid))
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ── record files ───────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_file(output_dir: Path | str) -> Path:
    return Path(output_dir) / PROCESS_FILE_NAME


def cancel_marker_file(output_dir: Path | str) -> Path:
    return Path(output_dir) / CANCEL_MARKER_NAME


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def record_training_process(
    output_dir: Path | str,
    *,
    pid: int,
    command: str,
    experiment_id: int | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Write the process record. The process must lead its own session."""
    pgid: int | None
    try:
        pgid = os.getpgid(int(pid))
    except OSError:
        pgid = None
    record = {
        "pid": int(pid),
        "pgid": pgid,
        "start_time": process_start_time(int(pid)),
        "command": command,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "recorded_at": _now(),
    }
    _write_json(process_file(output_dir), record)
    return record


def read_training_process(output_dir: Path | str) -> dict[str, Any] | None:
    record = _read_json(process_file(output_dir))
    if not record or not isinstance(record.get("pid"), int):
        return None
    return record


def clear_training_process(output_dir: Path | str, *, pid: int | None = None) -> None:
    """Remove the record (only if it still names ``pid`` when given)."""
    path = process_file(output_dir)
    if pid is not None:
        record = read_training_process(output_dir)
        if record and int(record.get("pid", -1)) != int(pid):
            return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def write_cancel_marker(output_dir: Path | str, *, reason: str = "user_cancel") -> None:
    _write_json(cancel_marker_file(output_dir), {"requested_at": _now(), "reason": reason})


def cancel_requested(output_dir: Path | str | None) -> bool:
    if not output_dir:
        return False
    return cancel_marker_file(output_dir).exists()


def clear_cancel_marker(output_dir: Path | str) -> None:
    try:
        cancel_marker_file(output_dir).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ── identity + termination ─────────────────────────────────────────────


def identify_process(record: dict[str, Any]) -> str:
    """``match`` / ``gone`` / ``mismatch`` for the recorded leader pid."""
    pid = int(record.get("pid") or 0)
    if pid <= 0 or not pid_alive(pid):
        return "gone"
    recorded_start = record.get("start_time")
    current_start = process_start_time(pid)
    if recorded_start and current_start and str(recorded_start) != str(current_start):
        return "mismatch"
    argv = process_cmdline(pid)
    command = str(record.get("command") or "")
    if argv is not None and command and not cmdline_matches_command(argv, command):
        # Same start time but a different program can only mean the
        # record was written before exec finished; the start time is the
        # stronger signal. Without a start time, the cmdline decides.
        if not (recorded_start and current_start):
            return "mismatch"
    return "match"


def _orphaned_group_is_ours(record: dict[str, Any], pgid: int) -> bool:
    """Leader gone but the group lives on: kill only if a member is the
    recorded command and started no earlier than the leader did."""
    command = str(record.get("command") or "")
    recorded_start = record.get("start_time")
    for member in _group_members(pgid):
        if not cmdline_matches_command(process_cmdline(member), command):
            continue
        member_start = process_start_time(member)
        if recorded_start and member_start:
            try:
                if int(member_start) < int(recorded_start):
                    continue
            except ValueError:
                continue
        return True
    return False


def _signal_target(pid: int, pgid: int | None, sig: int) -> None:
    own_group = os.getpgrp()
    if pgid and pgid > 1 and pgid != own_group:
        os.killpg(pgid, sig)
    else:
        # Never signal our own process group; fall back to the pid alone.
        os.kill(pid, sig)


def _target_alive(pid: int, pgid: int | None) -> bool:
    if pgid and pgid > 1 and pgid != os.getpgrp():
        return _group_alive(pgid)
    return pid_alive(pid)


async def _wait_dead(pid: int, pgid: int | None, timeout: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, float(timeout))
    while True:
        if not _target_alive(pid, pgid):
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.1)


async def terminate_process_record(
    record: dict[str, Any] | None,
    *,
    grace_seconds: float = 10.0,
    kill_wait_seconds: float = 5.0,
) -> dict[str, Any]:
    """SIGTERM the recorded process group, SIGKILL after ``grace_seconds``.

    ``status``: ``no_process_record`` | ``process_not_found`` (already
    exited) | ``process_mismatch`` (pid reused — nothing signalled) |
    ``process_terminated`` (exited on SIGTERM) | ``process_killed``
    (needed SIGKILL) | ``process_kill_failed`` (still alive / not allowed).
    """
    if not record:
        return {"status": "no_process_record"}
    pid = int(record.get("pid") or 0)
    pgid_raw = record.get("pgid")
    pgid = int(pgid_raw) if isinstance(pgid_raw, int) else None
    out: dict[str, Any] = {"pid": pid, "pgid": pgid}

    identity = identify_process(record)
    if identity == "mismatch":
        return {**out, "status": "process_mismatch"}
    if identity == "gone":
        if not (pgid and _proc_available() and _orphaned_group_is_ours(record, pgid)):
            return {**out, "status": "process_not_found"}
    else:
        live_pgid = _process_pgid(pid)
        if live_pgid is None:
            try:
                live_pgid = os.getpgid(pid)
            except OSError:
                live_pgid = None
        if pgid is None or (live_pgid is not None and live_pgid != pgid):
            # The process left its recorded group; signal it alone.
            pgid = None
            out["pgid"] = None

    try:
        _signal_target(pid, pgid, signal.SIGTERM)
    except ProcessLookupError:
        return {**out, "status": "process_not_found"}
    except PermissionError as exc:
        return {**out, "status": "process_kill_failed", "error": str(exc)}
    out["signal"] = "SIGTERM"
    if await _wait_dead(pid, pgid, grace_seconds):
        return {**out, "status": "process_terminated"}

    try:
        _signal_target(pid, pgid, signal.SIGKILL)
    except ProcessLookupError:
        return {**out, "status": "process_terminated"}
    except PermissionError as exc:
        return {**out, "status": "process_kill_failed", "error": str(exc)}
    out["signal"] = "SIGKILL"
    if await _wait_dead(pid, pgid, kill_wait_seconds):
        return {**out, "status": "process_killed"}
    return {**out, "status": "process_kill_failed", "error": "still alive after SIGKILL"}


async def terminate_training_process(
    output_dir: Path | str,
    *,
    grace_seconds: float = 10.0,
    kill_wait_seconds: float = 5.0,
) -> dict[str, Any]:
    return await terminate_process_record(
        read_training_process(output_dir),
        grace_seconds=grace_seconds,
        kill_wait_seconds=kill_wait_seconds,
    )
