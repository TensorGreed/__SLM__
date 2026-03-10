"""Ephemeral local serve run manager (start/status/stop + live log tail)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from app.config import settings


MAX_LOG_LINES = 2000
MAX_HEALTH_HISTORY = 160
MAX_SMOKE_HISTORY = 40
HEALTH_PROBE_TIMEOUT_SECONDS = 2.5
STARTUP_HEALTH_POLL_SECONDS = 1.0
STEADY_HEALTH_POLL_SECONDS = 5.0
MAX_STARTUP_SMOKE_ATTEMPTS = 3


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


def _normalize_method(value: Any, *, default: str) -> str:
    token = _coerce_text(value).upper()
    return token or default


def _normalize_expected_status(value: Any, *, default: int = 200) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        return default
    return parsed


def _safe_headers(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        header_key = _coerce_text(key)
        header_value = _coerce_text(raw)
        if not header_key or not header_value:
            continue
        out[header_key] = header_value
    return out


def _safe_json_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    return None


def _serve_run_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "serve_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _telemetry_path(project_id: int, run_id: str) -> str:
    return str(_serve_run_dir(project_id) / f"{run_id}.telemetry.json")


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
    healthcheck_url: str | None = None
    healthcheck_method: str = "GET"
    healthcheck_expected_status: int = 200
    smoke_url: str | None = None
    smoke_method: str = "POST"
    smoke_headers: dict[str, str] = field(default_factory=dict)
    smoke_json_body: dict[str, Any] | None = None
    smoke_expected_status: int = 200
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
    probe_task: asyncio.Task | None = None
    telemetry_path: str | None = None
    started_monotonic: float | None = None
    first_healthy_at: str | None = None
    startup_latency_ms: int | None = None
    smoke_passed: bool | None = None
    health_checks: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=MAX_HEALTH_HISTORY)
    )
    smoke_checks: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=MAX_SMOKE_HISTORY)
    )


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


def _run_telemetry_payload(run: ServeRunJob) -> dict[str, Any]:
    return {
        "schema": "slm.serve_runtime.telemetry/v1",
        "run_id": run.run_id,
        "project_id": run.project_id,
        "source": run.source,
        "export_id": run.export_id,
        "model_id": run.model_id,
        "template_id": run.template_id,
        "template_name": run.template_name,
        "status": run.status,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "pid": run.pid,
        "return_code": run.return_code,
        "cancel_requested": run.cancel_requested,
        "error": run.error,
        "runtime": {
            "cwd": run.cwd,
            "argv": list(run.argv),
            "command": run.command_display or "",
        },
        "healthcheck": {
            "url": run.healthcheck_url,
            "method": run.healthcheck_method,
            "expected_status": run.healthcheck_expected_status,
            "first_healthy_at": run.first_healthy_at,
            "startup_latency_ms": run.startup_latency_ms,
            "checks": list(run.health_checks),
        },
        "smoke_test": {
            "url": run.smoke_url,
            "method": run.smoke_method,
            "expected_status": run.smoke_expected_status,
            "passed": run.smoke_passed,
            "checks": list(run.smoke_checks),
        },
        "updated_at": _utcnow_iso(),
    }


def _persist_telemetry(run: ServeRunJob) -> None:
    path_text = _coerce_text(run.telemetry_path)
    if not path_text:
        return
    try:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _run_telemetry_payload(run)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _append_log(run, _format_log("[ERR]", f"Failed to persist serve telemetry: {exc}"))


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
        "telemetry": {
            "path": run.telemetry_path,
            "first_healthy_at": run.first_healthy_at,
            "startup_latency_ms": run.startup_latency_ms,
            "smoke_passed": run.smoke_passed,
            "healthcheck_url": run.healthcheck_url,
            "health_checks": list(run.health_checks),
            "smoke_checks": list(run.smoke_checks),
        },
    }


def _http_probe_sync(
    *,
    url: str,
    method: str,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    expected_status: int = 200,
) -> dict[str, Any]:
    request_headers = dict(headers or {})
    data: bytes | None = None
    if json_body is not None:
        data = json.dumps(json_body, ensure_ascii=True).encode("utf-8")
        lower_headers = {str(key).lower() for key in request_headers.keys()}
        if "content-type" not in lower_headers:
            request_headers["Content-Type"] = "application/json"

    started = time.perf_counter()
    status_code: int | None = None
    body_preview = ""
    error = ""
    ok = False
    try:
        request = Request(url=url, method=method, data=data, headers=request_headers)
        with urlopen(request, timeout=HEALTH_PROBE_TIMEOUT_SECONDS) as response:  # noqa: S310
            status_code = int(response.status)
            raw = response.read(512)
            body_preview = raw.decode("utf-8", errors="replace").strip()
            ok = status_code == expected_status
    except HTTPError as exc:
        status_code = int(exc.code)
        try:
            body_preview = exc.read(512).decode("utf-8", errors="replace").strip()
        except Exception:
            body_preview = ""
        error = str(exc)
    except URLError as exc:
        error = str(exc.reason or exc)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000.0))
    return {
        "timestamp": _utcnow_iso(),
        "ok": bool(ok),
        "status_code": status_code,
        "expected_status": expected_status,
        "latency_ms": latency_ms,
        "error": error,
        "body_preview": body_preview[:240],
    }


async def _http_probe(
    *,
    url: str,
    method: str,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    expected_status: int = 200,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _http_probe_sync,
        url=url,
        method=method,
        headers=headers,
        json_body=json_body,
        expected_status=expected_status,
    )


def _record_health_probe(run: ServeRunJob, probe: dict[str, Any]) -> bool:
    run.health_checks.append(dict(probe))
    became_healthy = bool(probe.get("ok")) and run.first_healthy_at is None
    if became_healthy:
        run.first_healthy_at = str(probe.get("timestamp") or _utcnow_iso())
        if run.started_monotonic is not None:
            elapsed = max(0.0, time.monotonic() - run.started_monotonic)
            run.startup_latency_ms = int(elapsed * 1000.0)
        _append_log(
            run,
            _format_log(
                "[SYS]",
                (
                    "Health check passed"
                    + (
                        f" (startup_latency_ms={run.startup_latency_ms})"
                        if run.startup_latency_ms is not None
                        else ""
                    )
                    + "."
                ),
            ),
        )
    return bool(probe.get("ok"))


def _record_smoke_probe(run: ServeRunJob, probe: dict[str, Any]) -> bool:
    run.smoke_checks.append(dict(probe))
    passed = bool(probe.get("ok"))
    run.smoke_passed = passed if run.smoke_passed is None else bool(run.smoke_passed or passed)
    if passed:
        _append_log(run, _format_log("[SYS]", "Smoke test probe passed."))
    else:
        code = probe.get("status_code")
        err = _coerce_text(probe.get("error"))
        detail = f"status={code}" if code is not None else err or "unknown error"
        _append_log(run, _format_log("[SYS]", f"Smoke test probe failed ({detail})."))
    return passed


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


async def _probe_runtime_telemetry(run: ServeRunJob) -> None:
    startup_smoke_attempts = 0
    while True:
        process = run.process
        if process is None or process.returncode is not None:
            break

        has_health = bool(run.healthcheck_url)
        has_smoke = bool(run.smoke_url)
        if not has_health and not has_smoke:
            break

        try:
            if has_health:
                health_probe = await _http_probe(
                    url=str(run.healthcheck_url),
                    method=run.healthcheck_method,
                    expected_status=run.healthcheck_expected_status,
                )
                _record_health_probe(run, health_probe)
                _persist_telemetry(run)

            should_probe_smoke = (
                has_smoke
                and startup_smoke_attempts < MAX_STARTUP_SMOKE_ATTEMPTS
                and run.smoke_passed is not True
                and (run.first_healthy_at is not None or not has_health)
            )
            if should_probe_smoke:
                startup_smoke_attempts += 1
                smoke_probe = await _http_probe(
                    url=str(run.smoke_url),
                    method=run.smoke_method,
                    headers=run.smoke_headers,
                    json_body=run.smoke_json_body,
                    expected_status=run.smoke_expected_status,
                )
                _record_smoke_probe(run, smoke_probe)
                _persist_telemetry(run)
        except Exception as exc:  # noqa: BLE001
            _append_log(run, _format_log("[ERR]", f"Telemetry probe failed: {exc}"))
            _persist_telemetry(run)

        sleep_seconds = (
            STARTUP_HEALTH_POLL_SECONDS
            if run.first_healthy_at is None
            else STEADY_HEALTH_POLL_SECONDS
        )
        await asyncio.sleep(sleep_seconds)


async def _monitor_run(run: ServeRunJob) -> None:
    process = run.process
    if process is None:
        return

    stdout_task = asyncio.create_task(_consume_stream(run, process.stdout, prefix="[OUT]"))
    stderr_task = asyncio.create_task(_consume_stream(run, process.stderr, prefix="[ERR]"))
    run.probe_task = asyncio.create_task(_probe_runtime_telemetry(run))
    return_code = await process.wait()
    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    if run.probe_task is not None and not run.probe_task.done():
        run.probe_task.cancel()
        await asyncio.gather(run.probe_task, return_exceptions=True)

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
    _persist_telemetry(run)


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

    healthcheck = template.get("healthcheck") if isinstance(template.get("healthcheck"), dict) else {}
    smoke_test = template.get("smoke_test") if isinstance(template.get("smoke_test"), dict) else {}

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
        healthcheck_curl=_coerce_text(healthcheck.get("curl")) or None,
        smoke_curl=_coerce_text(smoke_test.get("curl")) or None,
        command_display=_coerce_text(template.get("command")),
        healthcheck_url=_coerce_text(healthcheck.get("url")) or None,
        healthcheck_method=_normalize_method(healthcheck.get("method"), default="GET"),
        healthcheck_expected_status=_normalize_expected_status(healthcheck.get("expected_status"), default=200),
        smoke_url=_coerce_text(smoke_test.get("url")) or None,
        smoke_method=_normalize_method(
            smoke_test.get("method"),
            default="POST",
        ),
        smoke_headers=_safe_headers(smoke_test.get("headers")),
        smoke_json_body=_safe_json_mapping(smoke_test.get("json_body")),
        smoke_expected_status=_normalize_expected_status(smoke_test.get("expected_status"), default=200),
        telemetry_path=_telemetry_path(int(project_id), run_id),
    )
    _append_log(run, _format_log("[SYS]", f"Starting serve run for template {run.template_id}..."))
    run.status = "running"
    run.started_at = _utcnow_iso()
    run.started_monotonic = time.monotonic()
    _persist_telemetry(run)

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
        _persist_telemetry(run)
        async with _RUN_LOCK:
            _SERVE_RUNS[run_id] = run
        return _serialize_run(run)
    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        run.finished_at = _utcnow_iso()
        _append_log(run, _format_log("[ERR]", f"Unexpected start failure: {e}"))
        _persist_telemetry(run)
        async with _RUN_LOCK:
            _SERVE_RUNS[run_id] = run
        return _serialize_run(run)

    run.process = process
    run.pid = int(process.pid) if process.pid is not None else None
    _append_log(run, _format_log("[SYS]", f"Process started with pid={run.pid}."))
    _persist_telemetry(run)

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
        _persist_telemetry(run)
        return _serialize_run(run)

    run.cancel_requested = True
    run.status = "stopping"
    _append_log(run, _format_log("[SYS]", "Stop requested; sending terminate signal..."))
    _persist_telemetry(run)
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
    _persist_telemetry(run)
    return _serialize_run(run)
