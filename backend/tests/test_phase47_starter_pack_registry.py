"""Phase 47 tests: starter-pack dynamic registry, plugin fallback, and project defaults."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase47_starter_packs_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase47_starter_packs_data"

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
from app.services import starter_pack_service


class Phase47StarterPackRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        cls._prev_plugin_modules = list(settings.STARTER_PACK_PLUGIN_MODULES or [])

        settings.AUTH_ENABLED = False
        settings.STARTER_PACK_PLUGIN_MODULES = []
        starter_pack_service.clear_starter_pack_plugins()

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
        settings.STARTER_PACK_PLUGIN_MODULES = cls._prev_plugin_modules
        starter_pack_service.clear_starter_pack_plugins()

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def setUp(self):
        self._prev_plugin_modules = list(settings.STARTER_PACK_PLUGIN_MODULES or [])
        settings.STARTER_PACK_PLUGIN_MODULES = []
        starter_pack_service.clear_starter_pack_plugins()

    def tearDown(self):
        settings.STARTER_PACK_PLUGIN_MODULES = self._prev_plugin_modules
        starter_pack_service.clear_starter_pack_plugins()

    def test_catalog_api_exposes_builtin_packs_and_registry_metadata(self):
        resp = self.client.get("/api/starter-packs/catalog")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        self.assertIn("catalog_version", payload)
        self.assertIn("loaded_plugin_modules", payload)
        self.assertIn("plugin_load_errors", payload)
        self.assertIn("starter_packs", payload)

        rows = [item for item in payload.get("starter_packs", []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(rows), 4)

        row_ids = {str(item.get("id")) for item in rows}
        self.assertTrue({"legal", "customer_support", "healthcare_generic", "finance_generic"}.issubset(row_ids))

        legal = next((item for item in rows if item.get("id") == "legal"), None)
        self.assertIsNotNone(legal)
        self.assertEqual(legal.get("catalog_source"), "builtin")
        self.assertEqual(legal.get("catalog_version"), "builtin-v1")
        self.assertTrue(bool(legal.get("is_builtin")))
        self.assertIn("recommended_model_families", legal)
        self.assertIn("adapter_task_defaults", legal)
        self.assertIn("evaluation_gate_defaults", legal)
        self.assertIn("safety_compliance_reminders", legal)
        self.assertIn("target_profile_default", legal)

        detail_resp = self.client.get("/api/starter-packs/legal")
        self.assertEqual(detail_resp.status_code, 200, detail_resp.text)
        self.assertEqual(detail_resp.json().get("id"), "legal")

    def test_plugin_success_and_failure_fallback_behavior(self):
        module_name = "phase47.starter_pack_plugin_ok"
        plugin_module = ModuleType(module_name)

        def get_starter_packs() -> list[dict[str, object]]:
            return [
                {
                    "id": "retail_ops",
                    "display_name": "Retail Operations Starter",
                    "description": "Plugin-supplied starter pack for tests.",
                    "domain": "retail",
                    "recommended_model_families": ["qwen"],
                    "recommended_models": ["Qwen/Qwen2.5-1.5B-Instruct"],
                    "default_base_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "adapter_task_defaults": {
                        "adapter_id": "default-canonical",
                        "task_profile": "instruction_sft",
                    },
                    "evaluation_gate_defaults": {
                        "must_pass": True,
                        "min_score": 0.7,
                    },
                    "safety_compliance_reminders": ["No pricing commitments without approval."],
                    "target_profile_default": "edge_gpu",
                    "catalog_version": "phase47-starters-2026.03",
                }
            ]

        plugin_module.get_starter_packs = get_starter_packs  # type: ignore[attr-defined]

        settings.STARTER_PACK_PLUGIN_MODULES = [module_name]
        with patch(
            "app.services.starter_pack_service.importlib.import_module",
            return_value=plugin_module,
        ):
            catalog = starter_pack_service.list_starter_pack_catalog()

        rows = [item for item in catalog.get("starter_packs", []) if isinstance(item, dict)]
        plugin_row = next((item for item in rows if item.get("id") == "retail_ops"), None)
        self.assertIsNotNone(plugin_row)
        self.assertEqual(plugin_row.get("catalog_source"), module_name)
        self.assertEqual(plugin_row.get("catalog_version"), "phase47-starters-2026.03")
        self.assertFalse(bool(plugin_row.get("is_builtin")))
        self.assertIn(module_name, list(catalog.get("loaded_plugin_modules") or []))

        settings.STARTER_PACK_PLUGIN_MODULES = ["phase47.starter_pack_plugin_missing"]
        starter_pack_service.clear_starter_pack_plugins()
        with patch(
            "app.services.starter_pack_service.importlib.import_module",
            side_effect=ModuleNotFoundError("missing starter-pack plugin"),
        ):
            fallback_catalog = starter_pack_service.list_starter_pack_catalog()

        fallback_ids = {
            str(item.get("id"))
            for item in list(fallback_catalog.get("starter_packs") or [])
            if isinstance(item, dict)
        }
        self.assertIn("legal", fallback_ids)
        self.assertIn("phase47.starter_pack_plugin_missing", fallback_catalog.get("plugin_load_errors", {}))

    def test_project_creation_applies_starter_pack_defaults(self):
        resp = self.client.post(
            "/api/projects",
            json={
                "name": "phase47-starter-project",
                "description": "phase47",
                "starter_pack_id": "legal",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()

        self.assertEqual(payload.get("base_model_name"), "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(payload.get("target_profile_id"), "edge_gpu")

        gate_policy = payload.get("gate_policy") or {}
        self.assertTrue(bool(gate_policy.get("must_pass")))
        self.assertEqual(gate_policy.get("blocked_if_missing"), True)

    def test_project_creation_respects_explicit_overrides_with_starter_pack(self):
        resp = self.client.post(
            "/api/projects",
            json={
                "name": "phase47-starter-overrides",
                "description": "phase47",
                "starter_pack_id": "finance_generic",
                "base_model_name": "custom/model-1",
                "target_profile_id": "mobile_cpu",
                "gate_policy": {
                    "must_pass": False,
                    "min_score": 0.33,
                    "blocked_if_missing": False,
                },
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()

        self.assertEqual(payload.get("base_model_name"), "custom/model-1")
        self.assertEqual(payload.get("target_profile_id"), "mobile_cpu")
        self.assertEqual(
            payload.get("gate_policy"),
            {
                "must_pass": False,
                "min_score": 0.33,
                "blocked_if_missing": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
