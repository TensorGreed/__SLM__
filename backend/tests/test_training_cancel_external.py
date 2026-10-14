"""Cancelling an external training run stops its process.

Regression for experiment 66 (2026-10-14): cancel answered
``cancel_requested_without_task_id`` and flipped the row to CANCELLED while
``scripts/train.py`` kept running on the GPU. Two defects:

1. ``start_training`` stored the same ``runtime_config`` dict in
   ``exp.config`` before and after ``start_runtime`` mutated it, so
   SQLAlchemy saw old == new and never persisted the Celery ``task_id``.
2. Even with a task id, ``revoke(terminate=True)`` cannot reach a
   subprocess of a ``--pool=threads`` worker.

The fix records the process (``training_process.json``, own process group)
and cancel signals the group. These tests use a harmless ``sleep`` process
in place of train.py; no GPU, Celery broker or Redis.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU from this file

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import training_process_service as tps

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/stat").exists() or not hasattr(os, "killpg"),
    reason="needs POSIX process groups + /proc",
)

SLEEPER = f'{sys.executable} -c "import time; time.sleep(60)"'
# ``; true`` keeps the shell alive as the group leader with python as its
# child — the shape asyncio.create_subprocess_shell produces in the worker.
SHELL_SLEEPER = f"{SLEEPER}; true"
TERM_IGNORING = (
    f"{sys.executable} -c "
    "\"import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "print('ready', flush=True); time.sleep(60)\""
)

_CLIENT_CM: TestClient | None = None
_CLIENT: TestClient | None = None


def setUpModule() -> None:  # noqa: N802
    global _CLIENT_CM, _CLIENT
    _CLIENT_CM = TestClient(app)
    _CLIENT = _CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    if _CLIENT_CM is not None:
        _CLIENT_CM.__exit__(None, None, None)


def _spawn(command: str, tmp: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        command,
        shell=True,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(tmp),
    )
    # Wait until the shell has exec'd/forked so the cmdline is final.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if len(tps._group_members(proc.pid)) >= 1 and tps.process_cmdline(proc.pid):
            break
        time.sleep(0.05)
    return proc


class _ProcessCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(settings.DATA_DIR) / f"cancel-proc-{uuid.uuid4().hex[:8]}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._procs: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        # Only ever signal processes this test started.
        for proc in self._procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc.wait(timeout=5)

    def spawn(self, command: str) -> subprocess.Popen:
        proc = _spawn(command, self.tmp)
        self._procs.append(proc)
        return proc


class TerminateProcessTests(_ProcessCase):
    def test_sigterm_stops_the_whole_group(self):
        proc = self.spawn(SHELL_SLEEPER)
        tps.record_training_process(self.tmp, pid=proc.pid, command=SHELL_SLEEPER, experiment_id=1)
        time.sleep(0.2)
        members = tps._group_members(proc.pid)
        self.assertGreaterEqual(len(members), 2, "shell + python child expected")

        result = asyncio.run(tps.terminate_training_process(self.tmp, grace_seconds=5))

        self.assertEqual(result["status"], "process_terminated", result)
        self.assertEqual(result["signal"], "SIGTERM")
        proc.wait(timeout=5)
        self.assertEqual(tps._group_members(proc.pid), [])

    def test_sigkill_after_grace_when_sigterm_is_ignored(self):
        proc = self.spawn(TERM_IGNORING)
        self.assertEqual(proc.stdout.readline().strip(), b"ready")
        tps.record_training_process(self.tmp, pid=proc.pid, command=TERM_IGNORING)

        result = asyncio.run(tps.terminate_training_process(self.tmp, grace_seconds=0.5))

        self.assertEqual(result["status"], "process_killed", result)
        self.assertEqual(result["signal"], "SIGKILL")
        self.assertIsNotNone(proc.wait(timeout=5))

    def test_already_exited_process_is_not_found(self):
        proc = self.spawn(f'{sys.executable} -c "pass"')
        proc.wait(timeout=10)
        record = {"pid": proc.pid, "pgid": proc.pid, "start_time": "1", "command": "x"}
        result = asyncio.run(tps.terminate_process_record(record, grace_seconds=1))
        self.assertEqual(result["status"], "process_not_found", result)

    def test_reused_pid_is_never_signalled(self):
        proc = self.spawn(SLEEPER)
        start = tps.process_start_time(proc.pid)
        self.assertIsNotNone(start)
        # Same pid, different start time → a different process now owns it.
        stale = {"pid": proc.pid, "pgid": proc.pid, "start_time": str(int(start) - 100),
                 "command": SLEEPER}
        result = asyncio.run(tps.terminate_process_record(stale, grace_seconds=1))
        self.assertEqual(result["status"], "process_mismatch", result)
        self.assertIsNone(proc.poll(), "a mismatched pid must not be signalled")

        # No start time recorded → the command line decides.
        other = {"pid": proc.pid, "pgid": proc.pid, "start_time": None,
                 "command": "python scripts/train.py --experiment 999"}
        result = asyncio.run(tps.terminate_process_record(other, grace_seconds=1))
        self.assertEqual(result["status"], "process_mismatch", result)
        self.assertIsNone(proc.poll())

    def test_no_record(self):
        result = asyncio.run(tps.terminate_training_process(self.tmp))
        self.assertEqual(result["status"], "no_process_record")


class _FakeRedis:
    async def publish(self, channel, message):
        return 0

    async def aclose(self):
        return None


class WorkerCancelTests(_ProcessCase):
    """The worker task itself honours the cancel marker (second path)."""

    def _run_worker(self, command: str, result_box: dict) -> threading.Thread:
        from app import worker

        async def fake_monitor(experiment_id, process, command, log_path, output_dir, **kw):
            return "cancelled" if tps.cancel_requested(output_dir) else (
                "completed" if process.returncode == 0 else "failed"
            )

        def target():
            with patch.object(worker.aioredis, "from_url", return_value=_FakeRedis()), patch(
                "app.services.training_service._monitor_external_training", fake_monitor
            ):
                try:
                    result_box["result"] = worker.run_training_job(
                        experiment_id=1,
                        command=command,
                        log_path=str(self.tmp / "external_training.log"),
                        output_dir=str(self.tmp),
                    )
                except Exception as exc:  # pragma: no cover - surfaced below
                    result_box["error"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def test_marker_before_spawn_never_starts_the_process(self):
        tps.write_cancel_marker(self.tmp)
        box: dict = {}
        thread = self._run_worker(SLEEPER, box)
        thread.join(timeout=20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(box.get("result", {}).get("status"), "cancelled", box)
        self.assertIsNone(tps.read_training_process(self.tmp))

    def test_marker_while_running_stops_the_process(self):
        box: dict = {}
        thread = self._run_worker(SHELL_SLEEPER, box)
        deadline = time.monotonic() + 15
        record = None
        while time.monotonic() < deadline and record is None:
            record = tps.read_training_process(self.tmp)
            time.sleep(0.1)
        self.assertIsNotNone(record, "worker must record the spawned process")
        self.assertEqual(record["pgid"], record["pid"], "own process group")
        pid = record["pid"]

        tps.write_cancel_marker(self.tmp)
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive(), "worker did not stop the process")
        self.assertEqual(box.get("result", {}).get("status"), "cancelled", box)
        self.assertEqual(tps._group_members(pid), [])
        self.assertIsNone(tps.read_training_process(self.tmp), "record cleared on exit")


class CancelEndpointTests(_ProcessCase):
    def _wait_jobs_idle(self, project_id: int, experiment_id: int, timeout_s: float = 30.0) -> None:
        # A run a failed assertion left RUNNING keeps its watcher polling;
        # cancel it (409 when already cancelled) so the watcher ends.
        with patch("app.services.job_service.cancel_task"):
            _CLIENT.post(f"/api/projects/{project_id}/training/experiments/{experiment_id}/cancel")
        deadline = time.monotonic() + timeout_s
        jobs: list = []
        while time.monotonic() < deadline:
            resp = _CLIENT.get("/api/jobs/active", params={"include_recently_completed": "false"})
            jobs = resp.json().get("jobs", []) if resp.status_code == 200 else []
            if not jobs:
                return
            time.sleep(0.5)
        self.fail(f"background jobs still running: {jobs}")

    def _start_external_run(self) -> tuple[int, int, Path]:
        resp = _CLIENT.post(
            "/api/projects",
            json={"name": f"cancel-ext-{uuid.uuid4().hex[:8]}", "description": "cancel"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        project_id = int(resp.json()["id"])
        resp = _CLIENT.post(
            f"/api/projects/{project_id}/training/experiments",
            json={"name": "cancel-ext", "config": {"base_model": "microsoft/phi-2", "num_epochs": 1}},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        experiment_id = int(resp.json()["id"])
        prepared = Path(settings.DATA_DIR) / "projects" / str(project_id) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (prepared / f"{split}.jsonl").write_text(
                '{"prompt":"hi","response":"hello"}\n{"prompt":"bye","response":"goodbye"}\n',
                encoding="utf-8",
            )
        with patch.object(settings, "TRAINING_BACKEND", "external"), patch.object(
            settings,
            "TRAINING_EXTERNAL_CMD",
            "python scripts/train.py --project {project_id} --experiment {experiment_id} --output {output_dir}",
        ), patch(
            "app.worker.celery_app.send_task",
            return_value=SimpleNamespace(id="task-cancel-test"),
        ) as send_task:
            resp = _CLIENT.post(
                f"/api/projects/{project_id}/training/experiments/{experiment_id}/start"
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(send_task.call_count, 1)
        output_dir = Path(send_task.call_args.kwargs["kwargs"]["output_dir"])
        return project_id, experiment_id, output_dir

    def _experiment(self, project_id: int, experiment_id: int) -> dict:
        resp = _CLIENT.get(f"/api/projects/{project_id}/training/experiments")
        self.assertEqual(resp.status_code, 200, resp.text)
        matches = [row for row in resp.json() if int(row["id"]) == experiment_id]
        self.assertEqual(len(matches), 1, resp.text)
        return matches[0]

    def test_task_id_is_persisted_and_cancel_kills_the_process(self):
        project_id, experiment_id, output_dir = self._start_external_run()
        try:
            exp = self._experiment(project_id, experiment_id)
            runtime = (exp.get("config") or {}).get("_runtime") or {}
            # Root cause #1: the runtime's updates were silently dropped.
            self.assertEqual(runtime.get("task_id"), "task-cancel-test", runtime)
            self.assertEqual(runtime.get("runtime_kind"), "external_celery")

            proc = self.spawn(SHELL_SLEEPER)
            tps.record_training_process(
                output_dir, pid=proc.pid, command=SHELL_SLEEPER,
                experiment_id=experiment_id, task_id="task-cancel-test",
            )
            with patch("app.services.job_service.cancel_task") as revoke:
                resp = _CLIENT.post(
                    f"/api/projects/{project_id}/training/experiments/{experiment_id}/cancel"
                )
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(body["status"], "cancelled")
            self.assertEqual(body["cancel_status"], "process_terminated", body)
            self.assertEqual(body["task_id"], "task-cancel-test")
            self.assertTrue(body["task_revoked"])
            self.assertEqual(body["pid"], proc.pid)
            revoke.assert_called_once_with("task-cancel-test", terminate=False)
            proc.wait(timeout=5)
            self.assertEqual(tps._group_members(proc.pid), [])
            self.assertTrue(tps.cancel_requested(output_dir))
        finally:
            self._wait_jobs_idle(project_id, experiment_id)

        exp = self._experiment(project_id, experiment_id)
        self.assertEqual(str(exp["status"]).lower(), "cancelled")
        self.assertEqual(
            exp["config"]["_runtime"]["cancel_status"], "process_terminated"
        )

        # The worker's completion monitor runs after the killed process
        # exits — even a 0 exit must not overwrite CANCELLED.
        from app.services.training_service import _monitor_external_training

        for returncode in (-15, 0):
            status = asyncio.run(
                _monitor_external_training(
                    experiment_id,
                    SimpleNamespace(returncode=returncode),
                    SHELL_SLEEPER,
                    output_dir / "external_training.log",
                    output_dir,
                    captured_stdout="",
                    captured_stderr="",
                )
            )
            self.assertEqual(status, "cancelled")
        exp = self._experiment(project_id, experiment_id)
        self.assertEqual(str(exp["status"]).lower(), "cancelled")

    def test_cancel_with_queued_task_and_no_process_reports_revoked(self):
        project_id, experiment_id, output_dir = self._start_external_run()
        try:
            with patch("app.services.job_service.cancel_task") as revoke:
                resp = _CLIENT.post(
                    f"/api/projects/{project_id}/training/experiments/{experiment_id}/cancel"
                )
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(body["cancel_status"], "revoked", body)
            revoke.assert_called_once()
            self.assertEqual(
                str(self._experiment(project_id, experiment_id)["status"]).lower(), "cancelled"
            )
            # The worker reads this before spawning and never starts train.py.
            self.assertTrue(tps.cancel_requested(output_dir))
        finally:
            self._wait_jobs_idle(project_id, experiment_id)

    def test_cancel_after_process_exited_reports_not_found(self):
        project_id, experiment_id, output_dir = self._start_external_run()
        try:
            proc = self.spawn(f'{sys.executable} -c "pass"')
            pid = proc.pid
            tps.record_training_process(output_dir, pid=pid, command="gone")
            proc.wait(timeout=10)
            with patch("app.services.job_service.cancel_task"):
                resp = _CLIENT.post(
                    f"/api/projects/{project_id}/training/experiments/{experiment_id}/cancel"
                )
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(body["cancel_status"], "process_not_found", body)
            self.assertEqual(body["status"], "cancelled")
        finally:
            self._wait_jobs_idle(project_id, experiment_id)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
