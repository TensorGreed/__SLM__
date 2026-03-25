"""Phase 42 tests: dynamic target/model catalog registries with plugin fallback."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase42_catalog_plugins_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase42_catalog_plugins_data"

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
from app.services import model_selection_service, target_profile_service


class Phase42CatalogPluginRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        cls._prev_target_modules = list(settings.TARGET_PROFILE_PLUGIN_MODULES or [])
        cls._prev_model_modules = list(settings.MODEL_CATALOG_PLUGIN_MODULES or [])

        settings.AUTH_ENABLED = False
        settings.TARGET_PROFILE_PLUGIN_MODULES = []
        settings.MODEL_CATALOG_PLUGIN_MODULES = []
        target_profile_service.clear_target_profile_plugins()
        model_selection_service.clear_model_catalog_plugins()

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
        settings.TARGET_PROFILE_PLUGIN_MODULES = cls._prev_target_modules
        settings.MODEL_CATALOG_PLUGIN_MODULES = cls._prev_model_modules
        target_profile_service.clear_target_profile_plugins()
        model_selection_service.clear_model_catalog_plugins()

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def setUp(self):
        self._prev_target_modules = list(settings.TARGET_PROFILE_PLUGIN_MODULES or [])
        self._prev_model_modules = list(settings.MODEL_CATALOG_PLUGIN_MODULES or [])
        settings.TARGET_PROFILE_PLUGIN_MODULES = []
        settings.MODEL_CATALOG_PLUGIN_MODULES = []
        target_profile_service.clear_target_profile_plugins()
        model_selection_service.clear_model_catalog_plugins()

    def tearDown(self):
        settings.TARGET_PROFILE_PLUGIN_MODULES = self._prev_target_modules
        settings.MODEL_CATALOG_PLUGIN_MODULES = self._prev_model_modules
        target_profile_service.clear_target_profile_plugins()
        model_selection_service.clear_model_catalog_plugins()

    def _create_project(self, name: str) -> int:
        resp = self.client.post("/api/projects", json={"name": name, "description": "phase42"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_targets_catalog_api_exposes_provenance_metadata(self):
        resp = self.client.get("/api/targets/catalog")
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = [item for item in resp.json() if isinstance(item, dict)]
        self.assertTrue(rows)

        first = rows[0]
        self.assertIn("catalog_source", first)
        self.assertIn("catalog_version", first)
        self.assertIn("is_builtin", first)

        meta_resp = self.client.get(
            "/api/targets/catalog",
            params={"include_registry_meta": True},
        )
        self.assertEqual(meta_resp.status_code, 200, meta_resp.text)
        payload = meta_resp.json()
        self.assertIn("catalog_version", payload)
        self.assertIn("targets", payload)
        self.assertIn("loaded_plugin_modules", payload)
        self.assertIn("plugin_load_errors", payload)

    def test_model_catalog_api_and_recommendations_expose_registry_metadata(self):
        project_id = self._create_project("phase42-model-catalog-api")

        catalog_resp = self.client.get(
            f"/api/projects/{project_id}/training/model-selection/catalog"
        )
        self.assertEqual(catalog_resp.status_code, 200, catalog_resp.text)
        catalog_payload = catalog_resp.json()
        self.assertEqual(int(catalog_payload.get("project_id") or 0), project_id)
        self.assertIn("catalog_version", catalog_payload)
        self.assertIn("entry_count", catalog_payload)
        self.assertIn("models", catalog_payload)

        recommend_resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "laptop",
                "primary_language": "english",
                "available_vram_gb": 10,
                "task_profile": "instruction_sft",
                "top_k": 2,
            },
        )
        self.assertEqual(recommend_resp.status_code, 200, recommend_resp.text)
        payload = recommend_resp.json()
        self.assertIn("catalog_metadata", payload)

        rows = [item for item in payload.get("recommendations", []) if isinstance(item, dict)]
        self.assertTrue(rows)
        first = rows[0]
        self.assertIn("catalog_source", first)
        self.assertIn("catalog_version", first)
        self.assertIn("catalog_entry_is_builtin", first)

    def test_target_profile_plugin_success_and_failure_fallback(self):
        module_name = "phase42.target_plugin_ok"
        plugin_module = ModuleType(module_name)

        def get_target_profiles() -> list[dict[str, object]]:
            return [
                {
                    "id": "phase42_plugin_target",
                    "name": "Phase42 Plugin Target",
                    "description": "Plugin-supplied target profile for tests.",
                    "device_class": "mobile",
                    "constraints": {
                        "max_parameters_billions": 1.2,
                        "preferred_formats": ["onnx"],
                    },
                    "inference_runner_default": "exporter.onnx",
                    "catalog_version": "phase42-targets-2026.03",
                }
            ]

        plugin_module.get_target_profiles = get_target_profiles  # type: ignore[attr-defined]

        settings.TARGET_PROFILE_PLUGIN_MODULES = [module_name]
        with patch(
            "app.services.target_profile_service.importlib.import_module",
            return_value=plugin_module,
        ):
            catalog = target_profile_service.list_target_catalog()

        rows = [item for item in catalog.get("targets", []) if isinstance(item, dict)]
        plugin_row = next((item for item in rows if item.get("id") == "phase42_plugin_target"), None)
        self.assertIsNotNone(plugin_row)
        self.assertEqual(plugin_row.get("catalog_source"), module_name)
        self.assertEqual(plugin_row.get("catalog_version"), "phase42-targets-2026.03")
        self.assertFalse(bool(plugin_row.get("is_builtin")))

        self.assertIn(module_name, list(catalog.get("loaded_plugin_modules") or []))
        self.assertFalse(bool(catalog.get("plugin_load_errors")))

        settings.TARGET_PROFILE_PLUGIN_MODULES = ["phase42.target_plugin_missing"]
        target_profile_service.clear_target_profile_plugins()
        with patch(
            "app.services.target_profile_service.importlib.import_module",
            side_effect=ModuleNotFoundError("missing target plugin"),
        ):
            fallback_catalog = target_profile_service.list_target_catalog()

        fallback_ids = {
            str(item.get("id"))
            for item in list(fallback_catalog.get("targets") or [])
            if isinstance(item, dict)
        }
        self.assertIn("vllm_server", fallback_ids)
        self.assertIn("phase42.target_plugin_missing", fallback_catalog.get("plugin_load_errors", {}))

    def test_model_catalog_plugin_success_and_failure_fallback(self):
        module_name = "phase42.model_plugin_ok"
        plugin_module = ModuleType(module_name)

        def get_model_catalog_entries() -> list[dict[str, object]]:
            return [
                {
                    "model_id": "phase42/Tiny-0.8B-Instruct",
                    "family": "phase42",
                    "params_b": 0.8,
                    "estimated_min_vram_gb": 3.0,
                    "estimated_ideal_vram_gb": 4.5,
                    "preferred_chat_template": "chatml",
                    "supported_languages": ["english", "coding"],
                    "strengths": ["Ultra-light model for plugin tests"],
                    "caveats": ["Synthetic test-only model"],
                    "catalog_version": "phase42-models-2026.03",
                }
            ]

        plugin_module.get_model_catalog_entries = get_model_catalog_entries  # type: ignore[attr-defined]

        settings.MODEL_CATALOG_PLUGIN_MODULES = [module_name]
        with patch(
            "app.services.model_selection_service.importlib.import_module",
            return_value=plugin_module,
        ):
            catalog = model_selection_service.list_model_catalog()

        rows = [item for item in catalog.get("models", []) if isinstance(item, dict)]
        plugin_row = next((item for item in rows if item.get("model_id") == "phase42/Tiny-0.8B-Instruct"), None)
        self.assertIsNotNone(plugin_row)
        self.assertEqual(plugin_row.get("catalog_source"), module_name)
        self.assertEqual(plugin_row.get("catalog_version"), "phase42-models-2026.03")
        self.assertFalse(bool(plugin_row.get("is_builtin")))

        self.assertIn(module_name, list(catalog.get("loaded_plugin_modules") or []))
        self.assertFalse(bool(catalog.get("plugin_load_errors")))

        settings.MODEL_CATALOG_PLUGIN_MODULES = ["phase42.model_plugin_missing"]
        model_selection_service.clear_model_catalog_plugins()
        with patch(
            "app.services.model_selection_service.importlib.import_module",
            side_effect=ModuleNotFoundError("missing model plugin"),
        ):
            fallback_catalog = model_selection_service.list_model_catalog()

        fallback_ids = {
            str(item.get("model_id"))
            for item in list(fallback_catalog.get("models") or [])
            if isinstance(item, dict)
        }
        self.assertIn("meta-llama/Llama-3.2-1B-Instruct", fallback_ids)
        self.assertIn("phase42.model_plugin_missing", fallback_catalog.get("plugin_load_errors", {}))


if __name__ == "__main__":
    unittest.main()
