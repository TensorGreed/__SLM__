"""Training time limit: per-run override, queue-visibility warning, re-delivery guard, pace warning.

Covers:
- ``resolve_training_timeout`` / ``validate_training_timeout`` (``external_timeout_seconds``).
- Preflight: invalid limit is an error; a limit above ``CELERY_VISIBILITY_TIMEOUT_SECONDS``
  warns on the external runtime only.
- The external runtime sends the resolved limit to the worker task.
- ``TimeLimitProjector`` warns once, only when the pace overruns the limit.
- ``redelivery_skip_reason`` + the worker: a task id that already finished, or a live
  training process, never spawns a second process; a gone process does not block.
- The failure message names the per-run setting when the run's own limit stopped it.

No GPU, broker or Redis: a ``sleep`` process stands in for train.py.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU from this file

from app.config import settings
from app.services import training_process_service as tps
from app.services.training_failure_service import describe_training_failure
from app.services.training_time_limit_service import (
    TimeLimitProjector,
    format_hours,
    resolve_training_timeout,
    time_limit_preflight_warnings,
    validate_training_timeout,
)

posix_only = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")


class ResolveTests(unittest.TestCase):
    def test_unset_uses_server_limit(self):
        self.assertEqual(resolve_training_timeout({}), (int(settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS), "server"))
        self.assertEqual(resolve_training_timeout({"external_timeout_seconds": None})[1], "server")

    def test_run_value_overrides(self):
        self.assertEqual(resolve_training_timeout({"external_timeout_seconds": 86400}), (86400, "run"))
        self.assertEqual(resolve_training_timeout({"external_timeout_seconds": 7200.0}), (7200, "run"))

    def test_invalid_values_are_errors_and_fall_back(self):
        for bad in (5, 10**9, "6h", True, 90.5):
            cfg = {"external_timeout_seconds": bad}
            self.assertTrue(validate_training_timeout(cfg), bad)
            self.assertEqual(resolve_training_timeout(cfg)[1], "server", bad)
        self.assertEqual(validate_training_timeout({"external_timeout_seconds": 60}), [])

    def test_format_hours(self):
        self.assertEqual([format_hours(s) for s in (21600, 5400, 600, 75)], ["6 h", "1.5 h", "10 min", "75 s"])


class PreflightTests(unittest.TestCase):
    def test_warns_when_limit_exceeds_visibility(self):
        with patch.object(settings, "CELERY_VISIBILITY_TIMEOUT_SECONDS", 43200):
            self.assertEqual(time_limit_preflight_warnings({"external_timeout_seconds": 43200}), [])
            [warning] = time_limit_preflight_warnings({"external_timeout_seconds": 86400})
        self.assertIn("24 h, external_timeout_seconds for this run", warning)
        self.assertIn("12 h, CELERY_VISIBILITY_TIMEOUT_SECONDS", warning)

    def test_preflight_reports_error_and_external_only_warning(self):
        from app.services.training_preflight_service import run_training_preflight

        base = {"base_model": "HuggingFaceTB/SmolLM2-135M-Instruct", "training_mode": "sft"}
        with patch.object(settings, "CELERY_VISIBILITY_TIMEOUT_SECONDS", 3600):
            bad = run_training_preflight(project_id=999999, config={**base, "external_timeout_seconds": 5})
            ext = run_training_preflight(project_id=999999, config={
                **base, "external_timeout_seconds": 7200, "training_runtime_id": "builtin.external_celery"})
            sim = run_training_preflight(project_id=999999, config={
                **base, "external_timeout_seconds": 7200, "training_runtime_id": "builtin.simulate"})
        self.assertTrue(any("external_timeout_seconds must be between" in e for e in bad["errors"]))
        self.assertTrue(any("CELERY_VISIBILITY_TIMEOUT_SECONDS" in w for w in ext["warnings"]))
        self.assertFalse(any("CELERY_VISIBILITY_TIMEOUT_SECONDS" in w for w in sim["warnings"]))


class DispatchTests(unittest.TestCase):
    def test_external_runtime_sends_resolved_limit(self):
        import asyncio

        from app.services import training_runtime_service as trs

        tmp = Path(tempfile.mkdtemp(dir=settings.DATA_DIR))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        sent = {}

        class _Task:
            id = "task-1"

        def send_task(name, kwargs):
            sent.update(kwargs)
            return _Task()

        ctx = trs.TrainingRuntimeStartContext(
            project_id=1, experiment_id=2, base_model="m", config={"external_timeout_seconds": 86400},
            output_dir=tmp, config_path=tmp / "c.json", prepared_dir=tmp, train_file=tmp / "t", val_file=tmp / "v")
        with patch.object(settings, "TRAINING_EXTERNAL_CMD", "python3 train.py"), \
                patch("app.worker.celery_app.send_task", side_effect=send_task):
            result = asyncio.run(trs._start_builtin_external_celery(ctx))
        self.assertEqual((sent["timeout_seconds"], sent["timeout_source"]), (86400, "run"))
        self.assertEqual(result.runtime_updates["training_timeout_seconds"], 86400)


class ProjectorTests(unittest.TestCase):
    def test_warns_once_when_pace_overruns(self):
        p = TimeLimitProjector(21600, started_at=0.0)  # 6 h limit; min observe 10 min
        self.assertIsNone(p.observe(1, 250, 300.0))  # first sighting after 5 min of loading
        self.assertIsNone(p.observe(2, 250, 600.0))  # not observed long enough yet
        warning = p.observe(4, 250, 1200.0, source="server")  # 3 steps / 900 s = 300 s per step
        self.assertIsNotNone(warning)
        self.assertEqual(warning["seconds_per_step"], 300.0)
        self.assertGreater(warning["projected_seconds"], 21600)
        self.assertIn("past its 6 h limit: step 4 of 250", warning["message"])
        self.assertIn("(EXTERNAL_COMMAND_TIMEOUT_SECONDS)", warning["message"])
        self.assertIsNone(p.observe(5, 250, 1500.0))  # once only

    def test_no_warning_when_on_pace(self):
        p = TimeLimitProjector(3600, started_at=0.0)
        p.observe(10, 100, 60.0)
        self.assertIsNone(p.observe(50, 100, 400.0))  # 40 steps / 340 s → done in ~13 min

    def test_needs_step_and_total(self):
        p = TimeLimitProjector(60, started_at=0.0, min_observe_seconds=0)
        self.assertIsNone(p.observe(None, 100, 1.0))
        self.assertIsNone(p.observe(5, None, 2.0))


class FailureMessageSourceTests(unittest.TestCase):
    def test_run_limit_named_in_message(self):
        f = describe_training_failure(returncode=-15, stop_reason="timeout", timeout_seconds=86400, timeout_source="run")
        self.assertIn("reached the 24-hour limit (external_timeout_seconds for this run)", f["message"])
        self.assertEqual(f["timeout_source"], "run")


class _FakeRedis:
    async def publish(self, channel, message):
        return 0

    async def aclose(self):
        return None


@posix_only
class RedeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=settings.DATA_DIR))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, command: str, task_id: str) -> dict:
        from app import worker

        async def fake_monitor(experiment_id, process, command, log_path, output_dir, **kw):
            return "completed" if process.returncode == 0 else "failed"

        with patch.object(worker.aioredis, "from_url", return_value=_FakeRedis()), \
                patch("app.services.training_service._monitor_external_training", fake_monitor):
            return worker.run_training_job.apply(kwargs={
                "experiment_id": 1, "command": command, "log_path": str(self.tmp / "external_training.log"),
                "output_dir": str(self.tmp)}, task_id=task_id).get()

    def test_finished_task_id_is_not_run_again(self):
        marker = self.tmp / "ran.txt"
        cmd = f"{sys.executable} -c \"open('{marker}', 'a').write('x')\""
        self.assertEqual(self._run(cmd, "task-A")["status"], "success")
        self.assertEqual(marker.read_text(), "x")
        again = self._run(cmd, "task-A")  # the queue re-delivers the same task
        self.assertEqual(again, {"status": "skipped", "reason": "task_already_finished", "experiment_id": 1})
        self.assertEqual(marker.read_text(), "x", "train command must not run twice")
        self.assertEqual(self._run(cmd, "task-B")["status"], "success")  # a new start runs
        self.assertEqual(marker.read_text(), "xx")

    def test_live_training_process_blocks_a_second_one(self):
        cmd = f"{sys.executable} -c \"import time; time.sleep(30)\""
        proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
        self.addCleanup(lambda: (proc.kill(), proc.wait()))
        time.sleep(0.3)
        tps.record_training_process(self.tmp, pid=proc.pid, command=cmd, experiment_id=1, task_id="task-A")
        self.assertEqual(tps.redelivery_skip_reason(self.tmp, task_id="task-A"), "training_process_still_running")
        result = self._run(f"{sys.executable} -c pass", "task-A")
        self.assertEqual(result["reason"], "training_process_still_running")
        self.assertEqual(tps.read_training_process(self.tmp)["pid"], proc.pid, "the live record is kept")

    def test_gone_process_does_not_block_crash_recovery(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        tps.record_training_process(self.tmp, pid=proc.pid, command="x", experiment_id=1, task_id="task-A")
        self.assertIsNone(tps.redelivery_skip_reason(self.tmp, task_id="task-A"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
