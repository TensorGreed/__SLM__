"""Phase 21 tests: model selection wizard recommendation API."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase21_model_selection_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase21_model_selection_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase21ModelSelectionWizardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
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
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _create_project(self, name: str) -> int:
        resp = self.client.post("/api/projects", json={"name": name, "description": "phase21"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_model_recommendation_returns_ranked_results(self):
        project_id = self._create_project("phase21-model-wizard-1")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "laptop",
                "primary_language": "coding",
                "available_vram_gb": 12,
                "task_profile": "instruction_sft",
                "top_k": 3,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(int(payload.get("project_id", 0)), project_id)

        request = payload.get("request", {})
        self.assertEqual(request.get("target_device"), "laptop")
        self.assertEqual(request.get("primary_language"), "coding")
        self.assertEqual(int(request.get("top_k", 0)), 3)

        rows = [item for item in payload.get("recommendations", []) if isinstance(item, dict)]
        self.assertEqual(len(rows), 3)
        scores = [float(item.get("match_score", 0.0)) for item in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))

        first = rows[0]
        self.assertTrue(bool(first.get("model_id")))
        defaults = first.get("suggested_defaults", {})
        self.assertIn(defaults.get("task_type"), {"causal_lm", "seq2seq", "classification"})
        self.assertTrue(bool(defaults.get("chat_template")))

    def test_model_recommendation_classification_profile_sets_task_type(self):
        project_id = self._create_project("phase21-model-wizard-2")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "server",
                "primary_language": "multilingual",
                "available_vram_gb": 24,
                "task_profile": "classification",
                "top_k": 2,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = [item for item in resp.json().get("recommendations", []) if isinstance(item, dict)]
        self.assertEqual(len(rows), 2)
        for item in rows:
            defaults = item.get("suggested_defaults", {})
            self.assertEqual(defaults.get("task_type"), "classification")

    def test_model_recommendation_warns_when_vram_is_too_low(self):
        project_id = self._create_project("phase21-model-wizard-3")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "mobile",
                "primary_language": "english",
                "available_vram_gb": 2,
                "task_profile": "chat_sft",
                "top_k": 2,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        warnings = [str(item) for item in payload.get("warnings", [])]
        self.assertTrue(any("available_vram_gb" in item for item in warnings), warnings)
        rows = [item for item in payload.get("recommendations", []) if isinstance(item, dict)]
        self.assertEqual(len(rows), 2)

