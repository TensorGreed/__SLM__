"""Export refuses a run with nothing to package (simulated, or no saved weights): a clear 400."""

from __future__ import annotations

import asyncio
import time
import unittest

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    if client is not None:
        client.__exit__(None, None, None)


def _run(pid: int, config: dict, output_dir: str | None = None) -> int:
    async def _go():
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=pid, name="r", status=ExperimentStatus.COMPLETED,
                base_model="toy", output_dir=output_dir, config=config,
            )
            db.add(exp)
            await db.flush()
            await db.commit()
            return exp.id

    return asyncio.run(_go())


class WeightlessExportTests(unittest.TestCase):
    def _create(self, pid: int, exp_id: int):
        return client.post(
            f"/api/projects/{pid}/export/create",
            json={"experiment_id": exp_id, "export_format": "gguf", "quantization": "4-bit"},
        )

    def test_simulated_and_weightless_runs_get_a_400_with_error_code(self):
        pid = client.post("/api/projects", json={"name": f"wl-{time.time_ns()}", "description": "x"}).json()["id"]
        sim = _run(pid, {"_runtime": {"backend": "simulate"}})
        bare = _run(pid, {}, output_dir=None)
        for exp_id, reason in ((sim, "simulated_run"), (bare, "no_model_weights")):
            res = self._create(pid, exp_id)
            self.assertEqual(res.status_code, 400, res.text)
            body = res.json()
            self.assertEqual(body.get("error_code"), "RUN_HAS_NO_WEIGHTS")
            self.assertEqual((body.get("metadata") or {}).get("reason"), reason)

    def test_baseline_run_gets_a_400_at_create(self):
        # 7h follow-up (G2): Ship hides the baseline, but the API used to create its export and fail only at /run.
        pid = client.post("/api/projects", json={"name": f"wl-{time.time_ns()}", "description": "x"}).json()["id"]
        baseline = _run(pid, {"is_baseline": True}, output_dir=None)
        res = self._create(pid, baseline)
        self.assertEqual(res.status_code, 400, res.text)
        body = res.json()
        self.assertEqual(body.get("error_code"), "RUN_HAS_NO_WEIGHTS")
        self.assertEqual((body.get("metadata") or {}).get("reason"), "baseline_experiment")
        self.assertIn("baseline run", str(body.get("detail")))

    def test_unknown_run_is_still_a_404(self):
        pid = client.post("/api/projects", json={"name": f"wl-{time.time_ns()}", "description": "x"}).json()["id"]
        self.assertEqual(self._create(pid, 987654).status_code, 404)
