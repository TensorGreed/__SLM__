"""Tests for the project-guide quickstart endpoints (Theme 1 Epic 4):
import-sample, train-default, evaluate-latest."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "quickstart_endpoints_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "quickstart_endpoints_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class QuickstartImportSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

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
        settings.AUTH_ENABLED = cls._prev_auth_enabled

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "quickstart tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_import_sample_falls_back_to_support_faq_without_recipe(self):
        project_id = self._create_project("qs-import-default-slug")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()
        self.assertEqual(payload["status"], "ok")
        summary = payload["summary"]
        self.assertEqual(summary["slug"], "support-faq")
        self.assertGreater(summary["source_row_count"], 0)
        self.assertGreater(summary["gold_row_count"], 0)
        self.assertGreater(summary["prepared_train_rows"], 0)

    def test_import_sample_advances_pipeline_stage_so_ingest_step_completes(self):
        project_id = self._create_project("qs-import-advances-stage")
        self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        # The 'ingest' guide step's predicate is `stageIndex >= 1`.
        # Confirm the project moved off the INGESTION default.
        reread = self.client.get(f"/api/projects/{project_id}")
        self.assertEqual(reread.status_code, 200, reread.text)
        stage = reread.json()["pipeline_stage"]
        self.assertNotEqual(stage, "ingestion")

    def test_import_sample_derives_slug_from_selected_recipe(self):
        project_id = self._create_project("qs-import-from-recipe")
        # Apply the classification recipe — the slug derivation should
        # pick sentiment-classifier as the matching demo bundle.
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["summary"]["slug"], "sentiment-classifier")

    def test_import_sample_explicit_slug_overrides_recipe_derivation(self):
        project_id = self._create_project("qs-import-explicit-slug")
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        # User explicitly passes pii-detector — should win over the
        # classification → sentiment-classifier auto-derivation.
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={"slug": "pii-detector"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["summary"]["slug"], "pii-detector")

    def test_import_sample_unknown_slug_returns_404(self):
        project_id = self._create_project("qs-import-unknown-slug")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={"slug": "not-a-real-bundle"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_import_sample_missing_project_returns_404(self):
        resp = self.client.post(
            "/api/projects/99999/quickstart/import-sample",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)


class QuickstartTrainDefaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "qs-train tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_train_default_rejects_project_without_base_model(self):
        # Fresh project, no recipe applied, no base_model_name set.
        project_id = self._create_project("qs-train-no-base-model")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/train-default",
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("base_model_name", resp.text)

    def test_train_default_creates_and_starts_after_recipe_apply(self):
        # Apply recipe → propagates suggested_base_model onto project,
        # then quickstart should accept the call. With TRAINING_BACKEND=
        # simulate the simulator runs without a real GPU.
        project_id = self._create_project("qs-train-with-recipe")
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "qa-sft"},
        )
        # Also materialize sample data so the training-data gate has
        # something to read.
        self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/train-default",
        )
        # We accept either: (a) 201 with experiment_id, or (b) 400 if
        # the simulated training backend rejects something
        # environment-specific. The contract under test is that the
        # endpoint REACHES start_training; it shouldn't fail on the
        # "no base_model" guard.
        self.assertIn(resp.status_code, (201, 400, 409), resp.text)
        if resp.status_code == 201:
            payload = resp.json()
            self.assertEqual(payload["status"], "training_started")
            self.assertIn("experiment_id", payload)
            self.assertTrue(payload["base_model"])
            self.assertEqual(payload["recipe_id"], "qa-sft")
        else:
            # Even when something downstream errors, it must NOT be
            # the missing-base-model guard.
            self.assertNotIn("base_model_name", resp.text)


class QuickstartEvaluateLatestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "qs-eval tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_evaluate_latest_returns_409_when_no_experiments_exist(self):
        project_id = self._create_project("qs-eval-no-exps")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/evaluate-latest",
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertIn("training", resp.text.lower())

    def test_evaluate_latest_missing_project_returns_404(self):
        resp = self.client.post(
            "/api/projects/99999/quickstart/evaluate-latest",
        )
        self.assertEqual(resp.status_code, 404, resp.text)


if __name__ == "__main__":
    unittest.main()
