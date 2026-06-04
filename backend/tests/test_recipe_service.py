"""Tests for the task-shape recipe registry + header-based shape sniffer."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "recipe_service_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "recipe_service_data"

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
from app.services import recipe_service
from app.services.data_adapter_service import SUPPORTED_TASK_PROFILES

# Scoring modes are class attributes on EvalTaskHandlerService; mirror them as
# string literals here rather than depending on the class hierarchy.
SCORING_MODE_FIELD_MATCH = "field_match"
SCORING_MODE_SPAN_SET = "span_set"


class RecipeCatalogTests(unittest.TestCase):
    """Sanity checks on the built-in recipe catalog itself."""

    def test_ships_seven_built_in_recipes(self):
        # Arc R-1 added ``rag-protocol`` — the 7th built-in recipe.
        recipes = recipe_service.list_recipes()
        self.assertEqual(len(recipes), 7)
        ids = {r.id for r in recipes}
        self.assertEqual(
            ids,
            {
                "qa-sft",
                "classification",
                "span-extraction",
                "summarization",
                "code-review",
                "generic-sft",
                "rag-protocol",
            },
        )

    def test_every_recipe_references_real_task_profile(self):
        for recipe in recipe_service.list_recipes():
            self.assertIn(
                recipe.task_profile,
                SUPPORTED_TASK_PROFILES,
                f"recipe {recipe.id} references unknown task profile "
                f"{recipe.task_profile!r}",
            )

    def test_every_recipe_uses_a_known_scoring_mode(self):
        valid_modes = {SCORING_MODE_FIELD_MATCH, SCORING_MODE_SPAN_SET}
        for recipe in recipe_service.list_recipes():
            self.assertIn(
                recipe.scoring_mode,
                valid_modes,
                f"recipe {recipe.id} has unknown scoring_mode "
                f"{recipe.scoring_mode!r}",
            )

    def test_every_recipe_has_gold_template_and_eval_prompts(self):
        for recipe in recipe_service.list_recipes():
            self.assertTrue(
                recipe.gold_template.fields,
                f"recipe {recipe.id} ships an empty gold template",
            )
            self.assertGreaterEqual(
                len(recipe.sample_eval_prompts),
                1,
                f"recipe {recipe.id} ships no sample eval prompts",
            )

    def test_get_recipe_returns_none_for_unknown_id(self):
        self.assertIsNone(recipe_service.get_recipe("not-a-real-recipe"))

    def test_get_recipe_returns_matching_recipe(self):
        recipe = recipe_service.get_recipe("qa-sft")
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.id, "qa-sft")
        self.assertEqual(recipe.task_profile, "instruction_sft")
        self.assertEqual(recipe.scoring_mode, "field_match")

    def test_catalog_payload_carries_version_and_count(self):
        catalog = recipe_service.list_recipe_catalog()
        self.assertEqual(
            catalog["catalog_version"],
            recipe_service.RECIPE_CATALOG_VERSION,
        )
        self.assertEqual(catalog["catalog_source"], "builtin")
        self.assertEqual(catalog["recipe_count"], 7)
        self.assertEqual(len(catalog["recipes"]), 7)

    def test_supported_task_profiles_subset_of_data_adapter_set(self):
        for profile in recipe_service.list_supported_task_profiles_for_recipes():
            self.assertIn(profile, SUPPORTED_TASK_PROFILES)


class ShapeSnifferTests(unittest.TestCase):
    """Header-based recipe suggestion."""

    def test_qa_pair_headers_suggest_qa_sft(self):
        suggestions = recipe_service.sniff_recipe_from_headers(["question", "answer"])
        self.assertEqual(suggestions[0]["recipe_id"], "qa-sft")
        self.assertGreater(suggestions[0]["confidence"], 0.8)

    def test_text_label_headers_suggest_classification(self):
        suggestions = recipe_service.sniff_recipe_from_headers(["text", "label"])
        self.assertEqual(suggestions[0]["recipe_id"], "classification")
        self.assertGreater(suggestions[0]["confidence"], 0.8)

    def test_span_extraction_headers_suggest_span_extraction(self):
        suggestions = recipe_service.sniff_recipe_from_headers(["text", "entities"])
        self.assertEqual(suggestions[0]["recipe_id"], "span-extraction")
        self.assertGreater(suggestions[0]["confidence"], 0.85)

    def test_summarization_headers_suggest_summarization(self):
        suggestions = recipe_service.sniff_recipe_from_headers(["document", "summary"])
        self.assertEqual(suggestions[0]["recipe_id"], "summarization")
        self.assertGreater(suggestions[0]["confidence"], 0.8)

    def test_code_review_headers_suggest_code_review(self):
        suggestions = recipe_service.sniff_recipe_from_headers(["diff", "review"])
        self.assertEqual(suggestions[0]["recipe_id"], "code-review")
        self.assertGreater(suggestions[0]["confidence"], 0.8)

    def test_generic_fallback_when_nothing_matches(self):
        suggestions = recipe_service.sniff_recipe_from_headers(
            ["customer_id", "purchased_at"]
        )
        # Generic-sft is always present as the last-resort fallback so
        # the UI never has to render "no suggestion."
        self.assertTrue(
            any(s["recipe_id"] == "generic-sft" for s in suggestions),
            "generic-sft fallback should appear when no recipe matches cleanly",
        )

    def test_normalization_handles_case_and_punctuation(self):
        # "Question" + "Final Answer" — different case, space.
        suggestions = recipe_service.sniff_recipe_from_headers(
            ["Question", "Final Answer"]
        )
        self.assertEqual(suggestions[0]["recipe_id"], "qa-sft")

    def test_empty_headers_returns_only_generic_fallback(self):
        suggestions = recipe_service.sniff_recipe_from_headers([])
        # Empty input should not crash, and the generic fallback is the only
        # entry (or first entry if other recipes have no required columns).
        self.assertGreaterEqual(len(suggestions), 1)
        self.assertIn("generic-sft", {s["recipe_id"] for s in suggestions})


class RecipeApiTests(unittest.TestCase):
    """End-to-end checks via the FastAPI test client."""

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

    def test_list_endpoint_returns_catalog(self):
        resp = self.client.get("/api/recipes")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["recipe_count"], 7)
        self.assertEqual(len(payload["recipes"]), 7)
        self.assertEqual(
            payload["catalog_version"],
            recipe_service.RECIPE_CATALOG_VERSION,
        )

    def test_single_recipe_endpoint(self):
        resp = self.client.get("/api/recipes/span-extraction")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["id"], "span-extraction")
        self.assertEqual(payload["scoring_mode"], "span_set")
        self.assertEqual(payload["task_profile"], "structured_extraction")

    def test_single_recipe_endpoint_returns_404_for_unknown_id(self):
        resp = self.client.get("/api/recipes/not-a-real-recipe")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_sniff_endpoint_ranks_qa_pair_headers(self):
        resp = self.client.post(
            "/api/recipes/sniff",
            json={"headers": ["question", "answer"]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["top_recipe_id"], "qa-sft")
        self.assertGreater(payload["suggestions"][0]["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
