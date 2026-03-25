"""Phase 39 tests: target VRAM incompatibility hard-blocks training start."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase39_target_vram_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase39_target_vram_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase39TargetVramBlockerTests(unittest.TestCase):
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
            json={"name": name, "description": "phase39 target vram"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_experiment(self, project_id: int, *, base_model: str) -> int:
        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase39-exp",
                "description": "phase39 target vram",
                "config": {"base_model": base_model},
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _set_target_profile(self, project_id: int, target_profile_id: str) -> None:
        resp = self.client.put(
            f"/api/projects/{project_id}",
            json={"target_profile_id": target_profile_id},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    @patch("app.services.target_profile_service.introspect_hf_model")
    def test_start_training_hard_blocks_when_vram_is_clearly_over_target(self, mock_introspect):
        project_id = self._create_project("phase39-vram-block")
        self._set_target_profile(project_id, "edge_gpu")
        experiment_id = self._create_experiment(project_id, base_model="microsoft/phi-2")

        mock_introspect.return_value = {
            "model_id": "microsoft/phi-2",
            "resolved": True,
            "source": "hf_config",
            "params_estimate_b": 6.0,
            "memory_profile": {"estimated_min_vram_gb": 7.6, "estimated_ideal_vram_gb": 10.2},
            "architecture": "causal_lm",
            "context_length": 4096,
            "license": "apache-2.0",
        }

        start_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/{experiment_id}/start",
        )
        self.assertEqual(start_resp.status_code, 400, start_resp.text)
        payload = start_resp.json()
        self.assertEqual(str(payload.get("error_code") or ""), "TARGET_INCOMPATIBLE", payload)
        self.assertEqual(str(payload.get("stage") or ""), "training", payload)
        self.assertTrue(str(payload.get("actionable_fix") or "").strip(), payload)
        self.assertTrue(str(payload.get("docs_url") or "").strip(), payload)

        metadata = dict(payload.get("metadata") or {})
        reasons = [str(item) for item in list(metadata.get("reasons") or [])]
        self.assertTrue(any("VRAM" in item and "exceeds target baseline" in item for item in reasons), reasons)
        vram_check = dict(metadata.get("vram_check") or {})
        self.assertEqual(str(vram_check.get("status") or ""), "blocked", vram_check)
        unblock_actions = [row for row in list(metadata.get("unblock_actions") or []) if isinstance(row, dict)]
        self.assertGreaterEqual(len(unblock_actions), 2, metadata)

    @patch("app.api.training.start_training", new_callable=AsyncMock)
    @patch("app.services.target_profile_service.introspect_hf_model")
    def test_start_training_allows_unknown_vram_when_other_constraints_pass(
        self,
        mock_introspect,
        mock_start_training: AsyncMock,
    ):
        project_id = self._create_project("phase39-vram-unknown")
        self._set_target_profile(project_id, "edge_gpu")
        experiment_id = self._create_experiment(project_id, base_model="microsoft/phi-2")
        mock_start_training.return_value = {"status": "queued", "task_id": "phase39-task"}

        mock_introspect.return_value = {
            "model_id": "microsoft/phi-2",
            "resolved": False,
            "source": "none",
            "params_estimate_b": None,
            "memory_profile": {},
            "architecture": "unknown",
            "context_length": None,
            "license": None,
        }

        start_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/{experiment_id}/start",
        )
        self.assertEqual(start_resp.status_code, 200, start_resp.text)
        payload = dict(start_resp.json() or {})
        self.assertEqual(str(payload.get("status") or ""), "queued", payload)
        self.assertEqual(str(payload.get("task_id") or ""), "phase39-task", payload)
        self.assertEqual(mock_start_training.await_count, 1)


if __name__ == "__main__":
    unittest.main()
