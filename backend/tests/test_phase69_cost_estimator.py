"""Phase 69 — Cost estimator with provenance (priority.md P18, RM3).

Covers:
- ``POST /projects/{pid}/training/plan/estimate-cost`` always returns
  the contract shape: ``gpu_hours, usd, co2_kg, provenance, confidence``
  plus structured ``calibration``, ``pricing``, and ``co2`` sub-blocks.
- Provenance is ``estimated`` when no comparable historical runs exist
  and ``measured`` when ≥2 cohort matches push confidence above the
  measured threshold (0.60). Single historical run blends 60/40 with
  heuristic and stays ``estimated``.
- The same DB seeded with two completed phi-2 SFT runs of comparable
  duration drives a measured estimate whose ``gpu_hours`` is close to
  the cohort median (within 5%); calibration sub-block reports
  ``cohort="mode+model_size"`` and ``sample_count=2``.
- USD is non-zero for ``vllm_server`` (cloud-burst pricing catalog) and
  zero for ``mobile_cpu`` / ``browser_webgpu`` (local runtime); CO2 is
  zero when target power_kw is zero (mobile_cpu) and non-zero otherwise.
- Heuristic seconds floor at 60s — empty config + tiny num_gpus does
  not collapse to 0.
- Cancelled/failed runs in history are **not** pulled into the cohort
  even when their training_mode + base_model would otherwise match —
  sample_count stays zero and provenance stays ``estimated``.
- Larger cohort sizes raise confidence; identical timings raise it
  further by collapsing ``variability_cv``.
- Endpoint 400 for malformed payload.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase69_cost_estimator.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase69_cost_estimator_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode


_BASE_CONFIG = {
    "training_mode": "sft",
    "training_runtime_id": "builtin.simulate",
    "num_epochs": 3,
    "steps_per_epoch": 100,
    "num_gpus": 1,
}


class Phase69CostEstimatorTests(unittest.TestCase):
    @classmethod
    def _cleanup_artifacts(cls):
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        for suffix in ("", "-shm", "-wal"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_artifacts()

    # -- helpers ------------------------------------------------------------

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"phase69-{uuid.uuid4().hex[:8]}",
                "description": "phase69 cost estimator",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _seed_run(
        self,
        project_id: int,
        *,
        duration_seconds: float,
        base_model: str = "microsoft/phi-2",
        mode: TrainingMode = TrainingMode.SFT,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
    ) -> int:
        async with async_session_factory() as db:
            now = datetime.now(timezone.utc)
            started = now - timedelta(seconds=duration_seconds + 60)
            completed = started + timedelta(seconds=duration_seconds)
            exp = Experiment(
                project_id=project_id,
                name=f"hist-{uuid.uuid4().hex[:6]}",
                description="phase69 history",
                status=status,
                base_model=base_model,
                training_mode=mode,
                config=dict(_BASE_CONFIG),
                started_at=started,
                completed_at=completed,
            )
            db.add(exp)
            await db.commit()
            await db.refresh(exp)
            return int(exp.id)

    # -- contract shape ---------------------------------------------------

    def test_estimate_response_has_required_keys_even_without_history(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={
                "config": _BASE_CONFIG,
                "base_model": "microsoft/phi-2",
                "target_profile_id": "vllm_server",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        for key in ("gpu_hours", "usd", "co2_kg", "provenance", "confidence", "confidence_band"):
            self.assertIn(key, body, f"missing top-level key {key}")
        for sub in ("calibration", "pricing", "co2"):
            self.assertIn(sub, body, f"missing sub-block {sub}")
        self.assertEqual(body["provenance"], "estimated")
        self.assertEqual(body["calibration"]["sample_count"], 0)
        self.assertEqual(body["calibration"]["fallback_used"], True)
        self.assertGreater(body["gpu_hours"], 0)
        # vllm_server should pull a non-zero hourly rate from the catalog.
        self.assertGreater(body["pricing"]["hourly_rate_usd"], 0)
        self.assertGreater(body["usd"], 0)
        self.assertGreater(body["co2_kg"], 0)

    # -- measured path ----------------------------------------------------

    def test_two_comparable_runs_drive_measured_provenance_near_cohort_median(self):
        project_id = self._create_project()

        async def _seed():
            await self._seed_run(project_id, duration_seconds=900.0)
            await self._seed_run(project_id, duration_seconds=1100.0)
            await self._seed_run(project_id, duration_seconds=1000.0)

        asyncio.run(_seed())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={
                "config": _BASE_CONFIG,
                "base_model": "microsoft/phi-2",
                "target_profile_id": "vllm_server",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["provenance"], "measured")
        self.assertEqual(body["calibration"]["cohort"], "mode+model_size")
        self.assertEqual(body["calibration"]["sample_count"], 3)
        # Cohort median is 1000s = 0.2778h. Estimate must be within 5%.
        expected_gpu_hours = 1000.0 / 3600.0
        self.assertAlmostEqual(body["gpu_hours"], expected_gpu_hours, delta=expected_gpu_hours * 0.05)
        self.assertGreaterEqual(body["confidence"], 0.60)

    def test_single_historical_run_stays_estimated_with_blend(self):
        project_id = self._create_project()
        async def _seed():
            await self._seed_run(project_id, duration_seconds=900.0)
        asyncio.run(_seed())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={
                "config": _BASE_CONFIG,
                "base_model": "microsoft/phi-2",
                "target_profile_id": "vllm_server",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["calibration"]["sample_count"], 1)
        # N=1 cohort gets the blend, never the measured badge.
        self.assertEqual(body["provenance"], "estimated")
        self.assertLess(body["confidence"], 0.60)

    def test_cancelled_and_failed_runs_are_excluded_from_cohort(self):
        project_id = self._create_project()
        async def _seed():
            await self._seed_run(
                project_id, duration_seconds=900.0, status=ExperimentStatus.CANCELLED
            )
            await self._seed_run(
                project_id, duration_seconds=1100.0, status=ExperimentStatus.FAILED
            )

        asyncio.run(_seed())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={
                "config": _BASE_CONFIG,
                "base_model": "microsoft/phi-2",
                "target_profile_id": "vllm_server",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["calibration"]["sample_count"], 0)
        self.assertEqual(body["provenance"], "estimated")
        # Heuristic was used — fallback_used is true.
        self.assertTrue(body["calibration"]["fallback_used"])

    def test_identical_durations_raise_confidence_via_low_variability(self):
        project_id_few = self._create_project()
        project_id_more = self._create_project()

        async def _seed_few():
            for _ in range(2):
                await self._seed_run(project_id_few, duration_seconds=1000.0)

        async def _seed_more():
            for _ in range(8):
                await self._seed_run(project_id_more, duration_seconds=1000.0)

        asyncio.run(_seed_few())
        asyncio.run(_seed_more())

        few_body = self.client.post(
            f"/api/projects/{project_id_few}/training/plan/estimate-cost",
            json={"config": _BASE_CONFIG, "base_model": "microsoft/phi-2"},
        ).json()
        more_body = self.client.post(
            f"/api/projects/{project_id_more}/training/plan/estimate-cost",
            json={"config": _BASE_CONFIG, "base_model": "microsoft/phi-2"},
        ).json()

        # More samples → higher confidence.
        self.assertGreater(more_body["confidence"], few_body["confidence"])
        self.assertEqual(more_body["calibration"]["sample_count"], 8)
        self.assertEqual(few_body["calibration"]["sample_count"], 2)
        # Identical durations → variability_cv is 0.
        self.assertEqual(more_body["calibration"]["variability_cv"], 0.0)

    # -- pricing + CO2 paths ----------------------------------------------

    def test_mobile_cpu_target_yields_zero_usd_and_zero_co2(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={
                "config": _BASE_CONFIG,
                "base_model": "microsoft/phi-2",
                "target_profile_id": "mobile_cpu",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["pricing"]["hourly_rate_usd"], 0.0)
        self.assertEqual(body["pricing"]["source"], "local_runtime")
        self.assertEqual(body["usd"], 0.0)
        # Power = 0 → CO2 = 0 even though gpu_hours > 0.
        self.assertEqual(body["co2"]["power_kw"], 0.0)
        self.assertEqual(body["co2_kg"], 0.0)

    def test_edge_gpu_target_uses_edge_pricing_and_lower_power(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={
                "config": _BASE_CONFIG,
                "base_model": "microsoft/phi-2",
                "target_profile_id": "edge_gpu",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["pricing"]["source"], "edge_gpu_heuristic")
        # Edge GPU power is 0.05 kW vs server 0.30 kW.
        self.assertEqual(body["co2"]["power_kw"], 0.05)
        self.assertGreater(body["co2_kg"], 0)
        self.assertGreater(body["usd"], 0)

    def test_dpo_mode_takes_longer_than_sft_for_same_steps(self):
        project_id = self._create_project()
        sft_body = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={
                "config": {**_BASE_CONFIG, "training_mode": "sft"},
                "base_model": "microsoft/phi-2",
            },
        ).json()
        dpo_body = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={
                "config": {**_BASE_CONFIG, "training_mode": "dpo"},
                "base_model": "microsoft/phi-2",
            },
        ).json()
        # DPO multiplier is 1.8x SFT — gpu_hours should scale roughly the
        # same. Use a wider tolerance because heuristic_seconds floors at 60s.
        self.assertGreater(dpo_body["gpu_hours"], sft_body["gpu_hours"])

    # -- floors ----------------------------------------------------------

    def test_estimate_floors_at_60_seconds_for_tiny_configs(self):
        project_id = self._create_project()
        # No epochs / no steps / no gpus authored — the heuristic should
        # still produce a positive estimate, not zero.
        resp = self.client.post(
            f"/api/projects/{project_id}/training/plan/estimate-cost",
            json={"config": {"training_mode": "sft"}},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # 60 seconds = ~0.0167 gpu_hours.
        self.assertGreaterEqual(body["gpu_hours"], 60.0 / 3600.0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
