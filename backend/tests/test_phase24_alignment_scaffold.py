"""Phase 24 tests: alignment contract and judge scoring scaffolds."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase24_alignment_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase24_alignment_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase24AlignmentScaffoldTests(unittest.TestCase):
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
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "phase24"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_alignment_recipe_resolve_returns_training_mode(self):
        project_id = self._create_project("phase24-alignment-1")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/recipes/resolve",
            json={
                "recipe_id": "recipe.alignment.dpo.fast",
                "base_config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        resolved = resp.json().get("resolved_config", {})
        self.assertEqual(resolved.get("training_mode"), "dpo")
        self.assertEqual(resolved.get("task_type"), "causal_lm")

    def test_contract_validation_reports_missing_fields(self):
        project_id = self._create_project("phase24-alignment-2")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/preference-contract/validate",
            json={
                "rows": [
                    {"prompt": "q1", "chosen": "a1", "rejected": "bad"},
                    {"prompt": "q2", "chosen": "a2"},
                ],
                "min_coverage": 0.9,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertFalse(bool(payload.get("ok")))
        self.assertEqual(int(payload.get("total_rows", 0)), 2)
        self.assertEqual(int(payload.get("valid_rows", 0)), 1)
        self.assertEqual(int(payload.get("invalid_rows", 0)), 1)
        self.assertGreater(len(payload.get("errors", [])), 0)

    def test_alignment_judge_scores_rows(self):
        project_id = self._create_project("phase24-alignment-3")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/judge/score",
            json={
                "quality_threshold": 3.0,
                "rows": [
                    {
                        "prompt": "Summarize TLS best practices.",
                        "chosen": "Use TLS 1.2+ with strong ciphers and rotate certs.",
                        "rejected": "I like turtles.",
                    },
                    {
                        "prompt": "How to rotate keys?",
                        "chosen": "Rotate keys every 90 days.",
                        "rejected": "Rotate keys every 90 days.",
                    },
                ],
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(int(payload.get("scored_count", 0)), 1)
        self.assertGreaterEqual(int(payload.get("keep_count", 0)), 0)
        self.assertIn("contract", payload)


if __name__ == "__main__":
    unittest.main()
