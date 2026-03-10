"""Ephemeral local serve run manager (start/status/stop + live log tail)."""

from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


MAX_LOG_LINES = 2000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


@dataclass
class ServeRunJob:
    run_id: str
    project_id: int
    source: str
    export_id: int | None
    model_id: int | None
    template_id: str
    template_name: str
    cwd: str
    argv: list[str]
    env_overrides: dict[str, str]
    healthcheck_curl: str | None = None
    smoke_curl: str | None = None
    command_display: str | None = None
    status: str = "pending"
    created_at: str = field(default_factory=_utcnow_iso)
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    pid: int | None = None
    cancel_requested: bool = False
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    process: asyncio.subprocess.Process | None = None
    monitor_task: asyncio.Task | None = None


_SERVE_RUNS: dict[str, ServeRunJob] = {}
_RUN_LOCK = asyncio.Lock()


def _job_by_id(project_id: int, run_id: str) -> ServeRunJob:
    run = _SERVE_RUNS.get(run_id)
    if run is None or int(run.project_id) != int(project_id):
        raise ValueError(f"Serve run {run_id} not found in project {project_id}")
    return run


def _format_log(prefix: str, line: str) -> str:
    stamp = _utcnow().strftime("%H:%M:%S")
    text = line.rstrip("\n")
    if not text:
        return ""
    return f"[{stamp}] {prefix} {text}".rstrip()


def _append_log(run: ServeRunJob, line: str) -> None:
    text = _coerce_text(line)
    if not text:
        return
    run.logs.append(text)


def _serialize_run(run: ServeRunJob, *, logs_tail: int = 200) -> dict[str, Any]:
    logs_tail = max(0, min(int(logs_tail or 200), MAX_LOG_LINES))
    log_items = list(run.logs)[-logs_tail:] if logs_tail > 0 else []
    return {
        "run_id": run.run_id,
        "project_id": run.project_id,
        "source": run.source,
        "export_id": run.export_id,
        "model_id": run.model_id,
        "template_id": run.template_id,
        "template_name": run.template_name,
        "status": run.status,
        "pid": run.pid,
        "return_code": run.return_code,
        "error": run.error,
        "command": run.command_display or "",
        "cwd": run.cwd,
        "argv": list(run.argv),
        "healthcheck_curl": run.healthcheck_curl,
        "smoke_curl": run.smoke_curl,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "cancel_requested": run.cancel_requested,
        "can_stop": run.status in {"pending", "running", "stopping"},
        "logs_tail_count": len(log_items),
        "logs_tail": log_items,
    }


async def _consume_stream(
    run: ServeRunJob,
    stream: asyncio.StreamReader | None,
    *,
    prefix: str,
) -> None:
    if stream is None:
        return
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace")
        formatted = _format_log(prefix, text)
        if formatted:
            _append_log(run, formatted)


async def _monitor_run(run: ServeRunJob) -> None:
    process = run.process
    if process is None:
        return

    stdout_task = asyncio.create_task(_consume_stream(run, process.stdout, prefix="[OUT]"))
    stderr_task = asyncio.create_task(_consume_stream(run, process.stderr, prefix="[ERR]"))
    return_code = await process.wait()
    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    run.return_code = int(return_code)
    run.finished_at = _utcnow_iso()
    if run.cancel_requested:
        run.status = "cancelled"
        _append_log(run, _format_log("[SYS]", f"Process cancelled (code={return_code})."))
    elif return_code == 0:
        run.status = "completed"
        _append_log(run, _format_log("[SYS]", "Process exited successfully."))
    else:
        run.status = "failed"
        _append_log(run, _format_log("[SYS]", f"Process exited with code {return_code}."))


async def start_serve_run(
    *,
    project_id: int,
    source: str,
    export_id: int | None,
    model_id: int | None,
    template: dict[str, Any],
) -> dict[str, Any]:
    launch_spec = template.get("launch_spec") if isinstance(template.get("launch_spec"), dict) else {}
    argv = [str(item) for item in list(launch_spec.get("argv") or []) if _coerce_text(item)]
    if not argv:
        raise ValueError("Selected serve template does not include executable launch spec.")

    cwd = _coerce_text(launch_spec.get("cwd"))
    if not cwd:
        raise ValueError("Serve template is missing working directory.")
    cwd_path = Path(cwd).expanduser()
    if not cwd_path.exists() or not cwd_path.is_dir():
        raise ValueError(f"Serve run directory does not exist: {cwd_path}")

    env_overrides = {
        str(key): str(value)
        for key, value in dict(launch_spec.get("env") or {}).items()
        if _coerce_text(key)
    }

    run_id = uuid4().hex
    run = ServeRunJob(
        run_id=run_id,
        project_id=int(project_id),
        source=_coerce_text(source) or "export",
        export_id=int(export_id) if export_id is not None else None,
        model_id=int(model_id) if model_id is not None else None,
        template_id=_coerce_text(template.get("template_id")) or "unknown",
        template_name=_coerce_text(template.get("display_name")) or _coerce_text(template.get("template_id")),
        cwd=str(cwd_path),
        argv=argv,
        env_overrides=env_overrides,
        healthcheck_curl=_coerce_text((template.get("healthcheck") or {}).get("curl")) or None,
        smoke_curl=_coerce_text((template.get("smoke_test") or {}).get("curl")) or None,
        command_display=_coerce_text(template.get("command")),
    )
    _append_log(run, _format_log("[SYS]", f"Starting serve run for template {run.template_id}..."))
    run.status = "running"
    run.started_at = _utcnow_iso()

    env = dict(os.environ)
    env.update(env_overrides)
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd_path),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        run.status = "failed"
        run.error = str(e)
        run.finished_at = _utcnow_iso()
        _append_log(run, _format_log("[ERR]", f"Failed to start process: {e}"))
        async with _RUN_LOCK:
            _SERVE_RUNS[run_id] = run
        return _serialize_run(run)
    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        run.finished_at = _utcnow_iso()
        _append_log(run, _format_log("[ERR]", f"Unexpected start failure: {e}"))
        async with _RUN_LOCK:
            _SERVE_RUNS[run_id] = run
        return _serialize_run(run)

    run.process = process
    run.pid = int(process.pid) if process.pid is not None else None
    _append_log(run, _format_log("[SYS]", f"Process started with pid={run.pid}."))

    run.monitor_task = asyncio.create_task(_monitor_run(run))
    async with _RUN_LOCK:
        _SERVE_RUNS[run_id] = run
    return _serialize_run(run)


async def get_serve_run_status(
    *,
    project_id: int,
    run_id: str,
    logs_tail: int = 200,
) -> dict[str, Any]:
    run = _job_by_id(project_id, run_id)
    return _serialize_run(run, logs_tail=logs_tail)


async def stop_serve_run(
    *,
    project_id: int,
    run_id: str,
) -> dict[str, Any]:
    run = _job_by_id(project_id, run_id)
    if run.status not in {"pending", "running", "stopping"}:
        return _serialize_run(run)

    process = run.process
    if process is None:
        run.cancel_requested = True
        run.status = "cancelled"
        run.finished_at = _utcnow_iso()
        _append_log(run, _format_log("[SYS]", "Stop requested before process spawn; marked cancelled."))
        return _serialize_run(run)

    run.cancel_requested = True
    run.status = "stopping"
    _append_log(run, _format_log("[SYS]", "Stop requested; sending terminate signal..."))
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except TimeoutError:
            _append_log(run, _format_log("[SYS]", "Terminate timeout; sending kill signal..."))
            process.kill()
            await process.wait()

    if run.monitor_task is not None:
        await asyncio.wait([run.monitor_task], timeout=2.0)
    if run.status in {"running", "stopping", "pending"}:
        run.status = "cancelled"
        run.finished_at = _utcnow_iso()
    return _serialize_run(run)
