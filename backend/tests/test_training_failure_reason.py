"""A real run that does not complete says why: ``_runtime.failure``, the job error, the timeline.

Covers:
- ``describe_training_failure`` codes (``timeout`` / ``signal`` / ``exit`` / ``error``), the
  step-of-total head, and the last-checkpoint tail.
- ``last_error_line`` skips metric, event and progress lines.
- ``_monitor_external_training`` saves the record on FAILED, emits a ``training_<code>``
  RunEvent, and saves nothing on a cancel.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode
from app.models.run_event import RunEvent
from app.services.training_failure_service import (
    describe_training_failure,
    failure_from_config,
    format_duration,
    last_error_line,
)
from app.services.training_service import _monitor_external_training
from sqlalchemy import select

_CLIENT_CM: TestClient | None = None
_CLIENT: TestClient | None = None


def setUpModule():
    global _CLIENT_CM, _CLIENT
    _CLIENT_CM = TestClient(app)
    _CLIENT = _CLIENT_CM.__enter__()


def tearDownModule():
    if _CLIENT_CM is not None:
        _CLIENT_CM.__exit__(None, None, None)


def write_checkpoint(root: Path, step: int, *, max_steps: int = 250, state: bool = True) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(b"w")
    if state:
        (path / "trainer_state.json").write_text(json.dumps({
            "global_step": step, "max_steps": max_steps, "epoch": 0.4,
            "log_history": [{"step": step, "loss": 0.2}],
        }))
    return path


METRICS = "\n".join(f'SLM_METRIC {{"step": {s}, "train_loss": 0.2}}' for s in (20, 40, 60))


class DescribeFailureTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(settings.DATA_DIR) / "failure-describe" / uuid.uuid4().hex[:8]
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_timeout_names_the_limit_step_and_last_checkpoint(self):
        write_checkpoint(self.root, 50)
        f = describe_training_failure(returncode=-15, stdout_text=METRICS, stderr_text="",
                                      output_dir=self.root, stop_reason="timeout", timeout_seconds=21600)
        self.assertEqual(f["code"], "timeout")
        self.assertEqual(f["message"], "Stopped at step 60 of 250: reached the 6-hour limit "
                                       "(EXTERNAL_COMMAND_TIMEOUT_SECONDS). Last checkpoint: step 50.")
        self.assertEqual((f["last_step"], f["total_steps"], f["resumable_checkpoint_step"]), (60, 250, 50))
        self.assertEqual(f["timeout_seconds"], 21600)

    def test_signal_without_timeout_names_the_signal(self):
        f = describe_training_failure(returncode=-9, stdout_text=METRICS, stderr_text="", output_dir=self.root)
        self.assertEqual(f["code"], "signal")
        self.assertEqual(f["signal"], "SIGKILL")
        self.assertIn("stopped by SIGKILL (often the system running out of memory)", f["message"])
        self.assertTrue(f["message"].endswith("No checkpoint was saved."))
        self.assertEqual(f["total_steps"], None)
        self.assertTrue(f["message"].startswith("Stopped at step 60:"))

    def test_exit_quotes_the_error_not_the_trailing_metric(self):
        stderr = "Traceback (most recent call last):\nRuntimeError: CUDA error: device-side assert\n" \
                 " 24%|██▍       | 60/250 [00:30<01:30,  2.0it/s]"
        f = describe_training_failure(returncode=1, stdout_text=METRICS + "\n{'loss': 0.2, 'epoch': 0.4}",
                                      stderr_text=stderr, output_dir=self.root)
        self.assertEqual(f["code"], "exit")
        self.assertEqual(f["error_line"], "RuntimeError: CUDA error: device-side assert")
        self.assertEqual(f["total_steps"], 250)  # from the progress bar
        self.assertIn("exited with code 1: RuntimeError: CUDA error", f["message"])

    def test_incomplete_newest_checkpoint_points_at_an_older_resumable_one(self):
        write_checkpoint(self.root, 50)
        write_checkpoint(self.root, 100, state=False)
        f = describe_training_failure(returncode=-15, output_dir=self.root, stop_reason="timeout", timeout_seconds=75)
        self.assertEqual(f["last_checkpoint_step"], 100)
        self.assertEqual(f["resumable_checkpoint_step"], 50)
        self.assertIn("75-second limit", f["message"])
        self.assertTrue(f["message"].endswith("Last checkpoint: step 50."))

    def test_only_incomplete_checkpoint_says_so(self):
        write_checkpoint(self.root, 100, state=False)
        f = describe_training_failure(returncode=1, output_dir=self.root)
        self.assertIsNone(f["resumable_checkpoint_step"])
        self.assertIn("Checkpoint step 100 is incomplete", f["message"])

    def test_total_comes_from_the_training_bar_not_eval_or_data_bars(self):
        # Seen on a real run: an evaluation bar ("3/12") printed after the training bar
        # made the message read "step 440 of 12".
        stderr = "\n".join([
            "Map:  50%|█████     | 136/272 [00:00<00:00, 900.1 examples/s]",
            "Filter: 100%|██████████| 360/360 [00:00<00:00, 45199.95 examples/s]",
            " 40%|████      | 440/1080 [00:50<01:10,  9.0it/s]",
            " 25%|██▌       | 3/12 [00:00<00:01,  9.9it/s]",
        ])
        write_checkpoint(self.root, 400, max_steps=1080)
        write_checkpoint(self.root, 440, state=False)  # stopped mid-save
        f = describe_training_failure(returncode=-15, stderr_text=stderr, output_dir=self.root,
                                      stop_reason="timeout", timeout_seconds=75)
        self.assertEqual((f["last_step"], f["total_steps"]), (440, 1080))
        self.assertTrue(f["message"].startswith("Stopped at step 440 of 1080:"))
        f = describe_training_failure(returncode=-15, stderr_text=stderr, output_dir=None)
        self.assertEqual(f["total_steps"], 1080)  # no checkpoint: the training bar's total

    def test_monitor_error_code(self):
        f = describe_training_failure(returncode=None, output_dir=self.root, error="disk full")
        self.assertEqual(f["code"], "error")
        self.assertIn("could not be monitored: disk full", f["message"])
        self.assertTrue(f["message"].startswith("Stopped before the first logged step"))

    def test_helpers(self):
        self.assertEqual(last_error_line(METRICS, 'SLM_EVENT {"event": "epoch_end"}'), "")
        self.assertEqual(last_error_line("plain last line\n" + METRICS, ""), "plain last line")
        # Progress bars leave cursor-movement codes behind: never an error line.
        self.assertEqual(last_error_line("", "real problem\n\x1b[A\n\x1b[A \x1b[2K"), "real problem")
        self.assertEqual([format_duration(s) for s in (21600, 5400, 75, 86400, 0)],
                         ["6-hour", "90-minute", "75-second", "24-hour", "time"])
        self.assertIsNone(failure_from_config({"_runtime": {"returncode": -15}}))
        self.assertEqual(failure_from_config({"_runtime": {"failure": {"message": "m"}}}), {"message": "m"})


class FailureCodesAreInTheTaxonomyTests(unittest.TestCase):
    def test_every_failure_code_is_a_known_training_reason_code(self):
        from app.models.reason_codes import KNOWN_REASON_CODES, REASON_CODES_BY_STAGE
        from app.models.run_event import STAGE_TRAINING

        cases = [
            describe_training_failure(returncode=-15, stop_reason="timeout", timeout_seconds=60),
            describe_training_failure(returncode=-9),
            describe_training_failure(returncode=1),
            describe_training_failure(returncode=None, error="boom"),
        ]
        self.assertEqual({c["code"] for c in cases}, {"timeout", "signal", "exit", "error"})
        for case in cases:
            reason = f"training_{case['code']}"
            self.assertIn(reason, KNOWN_REASON_CODES)
            self.assertIn(reason, REASON_CODES_BY_STAGE[STAGE_TRAINING])


class MonitorSavesFailureTests(unittest.TestCase):
    def _run(self) -> tuple[int, int, Path]:
        resp = _CLIENT.post("/api/projects", json={"name": f"fail-reason-{uuid.uuid4().hex[:8]}", "description": "t"})
        self.assertEqual(resp.status_code, 201, resp.text)
        project_id = int(resp.json()["id"])

        async def go():
            async with async_session_factory() as db:
                exp = Experiment(project_id=project_id, name="r", status=ExperimentStatus.RUNNING,
                                 base_model="b", training_mode=TrainingMode.SFT,
                                 config={"_runtime": {"backend": "external"}})
                db.add(exp)
                await db.commit()
                await db.refresh(exp)
                out = Path(settings.DATA_DIR) / "projects" / str(project_id) / "experiments" / str(exp.id)
                exp.output_dir = str(out)
                await db.commit()
                return int(exp.id), out
        exp_id, out = asyncio.run(go())
        out.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        return project_id, exp_id, out

    def _monitor(self, exp_id: int, out: Path, returncode: int = -15, **kw) -> str:
        return asyncio.run(_monitor_external_training(
            exp_id, SimpleNamespace(returncode=returncode), "train.py", out / "external_training.log", out,
            captured_stdout=METRICS, captured_stderr="", **kw))

    def _state(self, project_id: int, exp_id: int):
        async def go():
            async with async_session_factory() as db:
                exp = await db.get(Experiment, exp_id)
                events = (await db.execute(select(RunEvent).where(
                    RunEvent.project_id == project_id, RunEvent.run_id == f"exp-{exp_id}"))).scalars().all()
                return exp.status, dict(exp.config or {}), list(events)
        return asyncio.run(go())

    def test_timed_out_run_saves_failure_and_emits_event(self):
        project_id, exp_id, out = self._run()
        write_checkpoint(out, 50)
        self.assertEqual(self._monitor(exp_id, out, stop_reason="timeout", timeout_seconds=21600), "failed")
        status, config, events = self._state(project_id, exp_id)
        self.assertEqual(status, ExperimentStatus.FAILED)
        failure = config["_runtime"]["failure"]
        self.assertEqual(failure["code"], "timeout")
        self.assertEqual(failure["resumable_checkpoint_step"], 50)
        self.assertEqual(config["_runtime"]["returncode"], -15)
        failed = [e for e in events if e.reason_code == "training_timeout"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].severity, "error")
        self.assertIn(failure["message"], failed[0].summary)
        self.assertEqual(failed[0].payload["failure"]["code"], "timeout")

    def _assert_one_stored_event(self, project_id: int, exp_id: int, code: str) -> None:
        _, config, events = self._state(project_id, exp_id)
        self.assertEqual(config["_runtime"]["failure"]["code"], code)
        stored = [e for e in events if e.reason_code == f"training_{code}"]
        self.assertEqual(len(stored), 1, [e.reason_code for e in events])
        self.assertEqual(stored[0].severity, "error")
        self.assertEqual(stored[0].stage, "training")
        self.assertEqual(stored[0].payload["failure"]["code"], code)

    def test_signal_killed_run_emits_stored_event(self):
        # A run stopped by a signal without a timeout (e.g. an OOM kill) used to fail
        # ``invalid_reason_code`` inside emit_event and never reach the timeline.
        project_id, exp_id, out = self._run()
        self.assertEqual(self._monitor(exp_id, out, returncode=-9), "failed")
        self._assert_one_stored_event(project_id, exp_id, "signal")

    def test_nonzero_exit_run_emits_stored_event(self):
        project_id, exp_id, out = self._run()
        self.assertEqual(self._monitor(exp_id, out, returncode=1), "failed")
        self._assert_one_stored_event(project_id, exp_id, "exit")

    def test_monitor_error_failure_emits_stored_event(self):
        from app.services.training_failure_service import emit_failure_event

        project_id, exp_id, _ = self._run()
        failure = describe_training_failure(returncode=None, error="lost the process handle")
        asyncio.run(emit_failure_event(project_id=project_id, experiment_id=exp_id, name="r", failure=failure))
        _, _, events = self._state(project_id, exp_id)
        stored = [e for e in events if e.reason_code == "training_error"]
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].payload["failure"]["code"], "error")

    def test_cancelled_run_saves_no_failure(self):
        project_id, exp_id, out = self._run()

        async def cancel():
            async with async_session_factory() as db:
                exp = await db.get(Experiment, exp_id)
                exp.status = ExperimentStatus.CANCELLED
                await db.commit()
        asyncio.run(cancel())
        self.assertEqual(self._monitor(exp_id, out), "cancelled")
        _, config, events = self._state(project_id, exp_id)
        self.assertNotIn("failure", config["_runtime"])
        self.assertFalse([e for e in events if (e.reason_code or "").startswith("training_")])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
