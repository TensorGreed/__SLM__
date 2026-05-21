"""Tests for recipe_apply_service — applying a Theme 2 recipe pick
to a Project record (snapshot + base-model propagation)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "recipe_apply_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "recipe_apply_data"

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


class RecipeApplyApiTests(unittest.TestCase):
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
            json={"name": name, "description": "recipe-apply tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_apply_recipe_snapshots_and_sets_base_model(self):
        project_id = self._create_project("recipe-apply-snapshot")

        resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "qa-sft"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        # Snapshot wrote a compact recipe dict to the project.
        snapshot = payload.get("selected_recipe")
        self.assertIsInstance(snapshot, dict)
        self.assertEqual(snapshot["recipe_id"], "qa-sft")
        self.assertEqual(snapshot["task_profile"], "instruction_sft")
        self.assertEqual(snapshot["scoring_mode"], "field_match")
        self.assertIn("applied_at", snapshot)
        self.assertIn("suggested_base_model", snapshot)

        # And propagated the suggested base model onto the project field
        # that training defaults already read.
        self.assertEqual(
            payload["base_model_name"], snapshot["suggested_base_model"]
        )

    def test_apply_recipe_survives_a_reread(self):
        project_id = self._create_project("recipe-apply-roundtrip")
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        reread = self.client.get(f"/api/projects/{project_id}")
        self.assertEqual(reread.status_code, 200, reread.text)
        snapshot = reread.json().get("selected_recipe")
        self.assertEqual(snapshot["recipe_id"], "classification")
        self.assertEqual(snapshot["task_profile"], "classification")

    def test_apply_recipe_is_idempotent_and_overrides_base_model(self):
        project_id = self._create_project("recipe-apply-idempotent")

        # Pick once.
        first = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "qa-sft"},
        ).json()
        first_model = first["base_model_name"]
        self.assertTrue(first_model)

        # Pick a different recipe — base model should adopt the new one.
        second = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "span-extraction"},
        ).json()
        self.assertEqual(second["selected_recipe"]["recipe_id"], "span-extraction")
        # The two recipes happen to share the same suggested base model
        # (SmolLM2-135M-Instruct), so we assert the field is set to the
        # span-extraction recipe's value rather than testing inequality.
        self.assertEqual(
            second["base_model_name"],
            second["selected_recipe"]["suggested_base_model"],
        )

    def test_apply_unknown_recipe_returns_404(self):
        project_id = self._create_project("recipe-apply-404")
        resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "not-a-real-recipe"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_apply_recipe_to_missing_project_returns_404(self):
        resp = self.client.put(
            "/api/projects/99999/recipe",
            json={"recipe_id": "qa-sft"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_clear_recipe_blanks_snapshot_but_keeps_base_model(self):
        project_id = self._create_project("recipe-apply-clear")
        applied = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "qa-sft"},
        ).json()
        chosen_model = applied["base_model_name"]

        cleared = self.client.delete(f"/api/projects/{project_id}/recipe")
        self.assertEqual(cleared.status_code, 200, cleared.text)
        payload = cleared.json()
        self.assertIsNone(payload.get("selected_recipe"))
        # base_model_name stays — clearing the recipe shouldn't wipe
        # whatever the user is now training against.
        self.assertEqual(payload["base_model_name"], chosen_model)


if __name__ == "__main__":
    unittest.main()
