"""SLM-vs-frontier benchmark report (Track 1, Epic D).

Covers the honest report: quality is a pure ratio over stored EvalResults (with
soft-fallbacks when no frontier baseline / no SLM eval), cost + latency render
with explicit provenance from the project's benchmark sweep + the published
frontier reference (never fabricated), and the headline is composed only from
the numbers that exist.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "frontier_cmp.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "frontier_cmp_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode  # noqa: E402
from app.services.frontier_comparison_service import (  # noqa: E402
    FRONTIER_REFERENCE,
    _blended_usd_per_1m,
    _build_headline,
    _quality_pct,
)
from app.services.training_telemetry_service import record_model_benchmark_run  # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


class PureLogicTests(unittest.TestCase):
    def test_reference_table(self):
        self.assertIn("gpt-4o-mini", FRONTIER_REFERENCE)
        self.assertEqual(_blended_usd_per_1m(FRONTIER_REFERENCE["gpt-4o-mini"]), 0.375)

    def test_quality_pct(self):
        self.assertEqual(_quality_pct(0.46, 0.5), 0.92)
        self.assertIsNone(_quality_pct(0.4, 0.0))  # exceeds a zero baseline
        self.assertEqual(_quality_pct(0.0, 0.0), 1.0)

    def test_headline_variants(self):
        self.assertEqual(
            _build_headline("GPT-4o mini", 0.92, 12.5, 0.4),
            "Your model is 92% as good as GPT-4o mini at 12.5% of the cost and 0.4× the latency.",
        )
        self.assertIn("runs at", _build_headline("GPT-4o mini", None, 12.5, 0.4))
        self.assertIn("Not enough data", _build_headline("GPT-4o mini", None, None, None))


class FrontierComparisonApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _create_project(self) -> int:
        resp = self.client.post("/api/projects", json={"name": f"frontier-{uuid.uuid4().hex[:8]}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _create_experiment(self, project_id: int, *, config: dict | None = None) -> int:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=project_id, name=f"exp-{uuid.uuid4().hex[:6]}", base_model=BASE,
                training_mode=TrainingMode.SFT, status=ExperimentStatus.COMPLETED, config=config or {},
            )
            db.add(exp)
            await db.commit()
            await db.refresh(exp)
            return int(exp.id)

    async def _add_eval(self, experiment_id: int, metrics: dict) -> None:
        async with async_session_factory() as db:
            db.add(EvalResult(
                experiment_id=experiment_id, dataset_name="gold", eval_type="qa",
                metrics=metrics, pass_rate=metrics.get("pass_rate"),
            ))
            await db.commit()

    def _get(self, project_id: int, experiment_id: int):
        return self.client.get(
            f"/api/projects/{project_id}/evaluation/frontier-comparison/{experiment_id}"
        )

    def test_no_slm_eval_softfalls_back_but_renders_cost_latency(self):
        project_id = self._create_project()
        exp_id = asyncio.run(self._create_experiment(project_id))
        resp = self._get(project_id, exp_id)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["quality"]["status"], "no_slm_eval")
        self.assertEqual(body["frontier_model"]["id"], "gpt-4o-mini")
        # Frontier reference numbers are always present (published).
        self.assertEqual(body["cost"]["frontier_usd_per_1m_tokens"], 0.375)
        self.assertGreater(body["latency"]["frontier_latency_ms"], 0)
        # No benchmark sweep yet → SLM side unavailable, not fabricated.
        self.assertEqual(body["cost"]["provenance"], "unavailable")
        self.assertIn("Not enough data", body["headline"])

    def test_no_frontier_eval_when_no_baseline_set(self):
        project_id = self._create_project()
        exp_id = asyncio.run(self._create_experiment(project_id))
        asyncio.run(self._add_eval(exp_id, {"f1": 0.46}))
        body = self._get(project_id, exp_id).json()
        self.assertEqual(body["quality"]["status"], "no_frontier_eval")
        self.assertIsNone(body["quality"]["headline_quality_pct"])

    def test_full_quality_comparison_when_baseline_present(self):
        project_id = self._create_project()
        frontier_id = asyncio.run(self._create_experiment(project_id))
        asyncio.run(self._add_eval(frontier_id, {"f1": 0.50, "accuracy": 0.60}))
        slm_id = asyncio.run(
            self._create_experiment(project_id, config={"frontier_baseline_run_id": frontier_id})
        )
        asyncio.run(self._add_eval(slm_id, {"f1": 0.46, "accuracy": 0.54}))

        body = self._get(project_id, slm_id).json()
        self.assertEqual(body["quality"]["status"], "ok")
        self.assertEqual(body["quality"]["headline_quality_pct"], 0.92)  # f1 0.46/0.50
        by_metric = {r["metric_id"]: r for r in body["quality"]["metric_comparisons"]}
        self.assertEqual(by_metric["f1"]["quality_pct"], 0.92)
        self.assertEqual(by_metric["f1"]["direction"], "behind")
        self.assertIn("92% as good as GPT-4o mini", body["headline"])

    def test_cost_latency_from_benchmark_run(self):
        project_id = self._create_project()
        # Seed a benchmark sweep row for the SLM's base model.
        record_model_benchmark_run(project_id, payload={
            "run_id": "bench1", "status": "completed",
            "matrix": [{"model_id": BASE, "estimated_latency_ms": 70.0, "estimated_throughput_tps": 120.0}],
        })
        exp_id = asyncio.run(self._create_experiment(project_id))
        asyncio.run(self._add_eval(exp_id, {"f1": 0.46}))
        body = self._get(project_id, exp_id).json()
        self.assertEqual(body["latency"]["provenance"], "estimated")
        self.assertEqual(body["latency"]["slm_latency_ms"], 70.0)
        self.assertIsNotNone(body["latency"]["latency_ratio"])
        self.assertEqual(body["cost"]["provenance"], "estimated")
        self.assertIsNotNone(body["cost"]["slm_usd_per_1m_tokens"])
        self.assertIsNotNone(body["cost"]["cost_pct"])
        # Headline now carries cost + latency (quality still no_frontier_eval).
        self.assertIn("the cost", body["headline"])

    def test_frontier_model_override(self):
        project_id = self._create_project()
        exp_id = asyncio.run(self._create_experiment(project_id))
        resp = self.client.get(
            f"/api/projects/{project_id}/evaluation/frontier-comparison/{exp_id}",
            params={"frontier_model_id": "gpt-4o"},
        )
        self.assertEqual(resp.json()["frontier_model"]["id"], "gpt-4o")

    def test_404_for_unknown_experiment(self):
        project_id = self._create_project()
        resp = self._get(project_id, 999999)
        self.assertEqual(resp.status_code, 404, resp.text)


if __name__ == "__main__":
    unittest.main()
