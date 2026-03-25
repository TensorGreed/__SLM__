"""Phase 41 tests: autopilot estimate uses telemetry calibration with sparse fallback."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase41_autopilot_estimator_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase41_autopilot_estimator_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app
from app.services.newbie_autopilot_service import estimate_newbie_autopilot_run


class Phase41AutopilotEstimatorCalibrationTests(unittest.TestCase):
    @classmethod
    def _cleanup_db_files(cls):
        for suffix in ("", "-shm", "-wal"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                path.unlink()

    @classmethod
    def setUpClass(cls):
        cls._cleanup_db_files()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_db_files()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "phase41 autopilot estimator"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_estimator_uses_measured_history_for_matching_task_profile_target(self):
        history = [
            {
                "duration_seconds": 300.0,
                "dataset_size_rows": 1000,
                "plan_profile": "balanced",
                "target_profile_id": "vllm_server",
                "task_profile": "qa",
            },
            {
                "duration_seconds": 360.0,
                "dataset_size_rows": 1000,
                "plan_profile": "balanced",
                "target_profile_id": "vllm_server",
                "task_profile": "qa",
            },
            {
                "duration_seconds": 330.0,
                "dataset_size_rows": 1000,
                "plan_profile": "balanced",
                "target_profile_id": "vllm_server",
                "task_profile": "qa",
            },
        ]
        estimate = estimate_newbie_autopilot_run(
            plan_profile="balanced",
            target_profile_id="vllm_server",
            dataset_size_rows=2000,
            task_profile="qa",
            run_history=history,
        )

        self.assertEqual(str(estimate.get("metric_source") or ""), "measured", estimate)
        self.assertEqual(str(estimate.get("unit") or ""), "USD", estimate)
        self.assertGreater(float(estimate.get("estimated_cost") or 0.0), 0.0, estimate)
        self.assertIn(str(estimate.get("confidence_band") or ""), {"medium", "high"}, estimate)
        calibration = dict(estimate.get("calibration") or {})
        self.assertEqual(str(calibration.get("cohort") or ""), "target+profile+task", calibration)
        self.assertGreaterEqual(int(calibration.get("sample_count") or 0), 2, calibration)

    def test_estimator_falls_back_with_low_confidence_when_history_is_sparse(self):
        history = [
            {
                "duration_seconds": 1200.0,
                "dataset_size_rows": 1000,
                "plan_profile": "safe",
                "target_profile_id": "mobile_cpu",
                "task_profile": "classification",
            }
        ]
        estimate = estimate_newbie_autopilot_run(
            plan_profile="max_quality",
            target_profile_id="vllm_server",
            dataset_size_rows=1500,
            task_profile="summarization",
            run_history=history,
        )

        self.assertEqual(str(estimate.get("metric_source") or ""), "estimated", estimate)
        self.assertEqual(str(estimate.get("confidence_band") or ""), "low", estimate)
        self.assertIn("telemetry", str(estimate.get("note") or "").lower(), estimate)
        calibration = dict(estimate.get("calibration") or {})
        self.assertEqual(bool(calibration.get("fallback_used")), True, calibration)

    @patch("app.api.training._load_autopilot_run_history", new_callable=AsyncMock)
    def test_autopilot_estimate_endpoint_surfaces_calibration_fields(self, mock_history: AsyncMock):
        mock_history.return_value = [
            {
                "duration_seconds": 300.0,
                "dataset_size_rows": 1000,
                "plan_profile": "balanced",
                "target_profile_id": "vllm_server",
                "task_profile": "qa",
            },
            {
                "duration_seconds": 330.0,
                "dataset_size_rows": 1000,
                "plan_profile": "balanced",
                "target_profile_id": "vllm_server",
                "task_profile": "qa",
            },
        ]
        project_id = self._create_project("phase41-autopilot-estimate-api")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/estimate",
            json={
                "plan_profile": "balanced",
                "target_profile_id": "vllm_server",
                "dataset_size_rows": 1000,
                "task_profile": "qa",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        estimate = dict(payload.get("estimate") or {})
        self.assertEqual(str(estimate.get("unit") or ""), "USD", estimate)
        self.assertTrue(str(estimate.get("confidence_band") or "").strip(), estimate)
        self.assertTrue(str(estimate.get("metric_source") or "").strip(), estimate)
        self.assertIsInstance(estimate.get("calibration"), dict)
        self.assertEqual(mock_history.await_count, 1)


if __name__ == "__main__":
    unittest.main()
