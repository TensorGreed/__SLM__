"""Phase 7i fix pass: the trainer's epoch plan reaches the run, one rounding rule, no summary-row metric.

- ``record_epoch_plan`` saves ``total_epochs`` / ``total_steps`` on a seeded Experiment; a fractional plan
  (a step cap) is rounded UP by ``epoch_total``, the same rule the end-of-run report reader uses.
- The worker task relays a ``SLM_EVENT epoch_plan`` line over the run channel (``{"type": "epoch_plan"}``),
  logs it, and saves it on the run.
- The worker never publishes HF's end-of-run summary row (``train_runtime`` + run-average ``train_loss``) as a
  metric, while ordinary metric lines still go out.

No GPU, broker or Redis: a short ``python -c`` process stands in for train.py, Redis is a fake.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU from this file

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode
from app.services.training_epoch_plan_service import epoch_total, record_epoch_plan

posix_only = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")

client: TestClient | None = None


def setUpModule():  # noqa: N802 - unittest convention
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():  # noqa: N802 - unittest convention
    client.__exit__(None, None, None)


def _seed_experiment() -> int:
    pid = client.post("/api/projects", json={"name": f"epoch-plan-{time.time_ns()}", "description": "x"}).json()["id"]

    async def seed() -> int:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=pid,
                name="auto-epochs-run",
                status=ExperimentStatus.RUNNING,
                training_mode=TrainingMode.SFT,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                config={"num_epochs": 3, "auto_epochs": True},
            )
            db.add(exp)
            await db.commit()
            return int(exp.id)

    return asyncio.run(seed())


def _totals(experiment_id: int) -> tuple[int | None, int | None]:
    async def read():
        async with async_session_factory() as db:
            exp = await db.get(Experiment, experiment_id)
            return exp.total_epochs, exp.total_steps

    return asyncio.run(read())


class EpochTotalRuleTests(unittest.TestCase):
    def test_rounds_up_to_a_whole_epoch(self):
        self.assertEqual(epoch_total(8), 8)
        self.assertEqual(epoch_total(8.0), 8)
        self.assertEqual(epoch_total(2.5), 3)
        self.assertEqual(epoch_total(2.0001), 3)
        for bad in (0, -1, None, "3", True, float("nan"), float("inf")):
            self.assertIsNone(epoch_total(bad), bad)


class RecordEpochPlanTests(unittest.TestCase):
    def test_plan_is_saved_on_the_run(self):
        exp_id = _seed_experiment()
        self.assertEqual(_totals(exp_id), (None, None))
        asyncio.run(record_epoch_plan(exp_id, {"num_epochs": 8, "total_steps": 216}))
        self.assertEqual(_totals(exp_id), (8, 216))

    def test_fractional_plan_is_rounded_up(self):
        exp_id = _seed_experiment()
        asyncio.run(record_epoch_plan(exp_id, {"num_epochs": 2.5, "total_steps": 50}))
        self.assertEqual(_totals(exp_id), (3, 50))

    def test_missing_run_and_empty_plan_are_no_ops(self):
        asyncio.run(record_epoch_plan(10**9, {"num_epochs": 8}))  # never raises
        exp_id = _seed_experiment()
        asyncio.run(record_epoch_plan(exp_id, {}))
        self.assertEqual(_totals(exp_id), (None, None))


class _RecordingRedis:
    def __init__(self):
        self.messages: list[dict] = []

    async def publish(self, channel, message):
        self.messages.append(json.loads(message))
        return 0

    async def aclose(self):
        return None


@posix_only
class WorkerRelayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=settings.DATA_DIR))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_epoch_plan_is_relayed_and_saved_and_the_summary_row_is_not_a_metric(self):
        from app import worker

        exp_id = _seed_experiment()
        script = self.tmp / "fake_train.py"
        script.write_text(
            "import json\n"
            "print('SLM_EVENT ' + json.dumps({'event': 'epoch_plan', 'num_epochs': 8, 'total_steps': 216}), flush=True)\n"
            "print('SLM_METRIC ' + json.dumps({'loss': 3.83, 'epoch': 0.74, 'step': 20}), flush=True)\n"
            "print('SLM_METRIC ' + json.dumps({'train_runtime': 29.5, 'train_loss': 2.205, 'epoch': 8.0, 'step': 216}),"
            " flush=True)\n"
            "print(str({'train_runtime': 29.5, 'train_loss': 2.205, 'epoch': 8.0}), flush=True)\n",
            encoding="utf-8",
        )
        fake_redis = _RecordingRedis()

        async def fake_monitor(experiment_id, process, command, log_path, output_dir, **kw):
            return "completed" if process.returncode == 0 else "failed"

        with patch.object(worker.aioredis, "from_url", return_value=fake_redis), \
                patch("app.services.training_service._monitor_external_training", fake_monitor):
            result = worker.run_training_job.apply(kwargs={
                "experiment_id": exp_id, "command": f"{sys.executable} {script}",
                "log_path": str(self.tmp / "external_training.log"), "output_dir": str(self.tmp),
            }, task_id=f"task-{exp_id}").get()

        self.assertEqual(result["status"], "success")
        plans = [m for m in fake_redis.messages if m.get("type") == "epoch_plan"]
        self.assertEqual(plans, [{"type": "epoch_plan", "payload": {"num_epochs": 8, "total_steps": 216}}])
        logs = [m.get("text") for m in fake_redis.messages if m.get("type") == "log"]
        self.assertIn("[event] training plan: 8 epochs, 216 optimizer steps", logs)
        metrics = [m["metric"] for m in fake_redis.messages if m.get("type") == "metric"]
        self.assertEqual([m.get("step") for m in metrics], [20])
        self.assertFalse(any(m.get("train_loss") == 2.205 for m in metrics))
        self.assertEqual(_totals(exp_id), (8, 216))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
