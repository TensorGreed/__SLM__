"""Phase 17 tests: runtime settings API and persistence."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase17_runtime_settings_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase17_runtime_settings_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.main import app


class Phase17RuntimeSettingsApiTests(unittest.TestCase):
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

    def test_get_and_update_runtime_settings(self):
        res = self.client.get("/api/settings/runtime")
        self.assertEqual(res.status_code, 200, res.text)
        payload = res.json()
        fields = {str(item.get("key")): item for item in payload.get("fields", []) if isinstance(item, dict)}
        self.assertIn("TRAINING_BACKEND", fields)
        self.assertIn("ALLOW_SIMULATED_TRAINING", fields)

        update_res = self.client.put(
            "/api/settings/runtime",
            json={
                "updates": {
                    "ALLOW_SIMULATED_TRAINING": True,
                    "EXTERNAL_COMMAND_TIMEOUT_SECONDS": 12345,
                }
            },
        )
        self.assertEqual(update_res.status_code, 200, update_res.text)
        updated = update_res.json()
        self.assertIn("ALLOW_SIMULATED_TRAINING", updated.get("updated_keys", []))
        self.assertIn("EXTERNAL_COMMAND_TIMEOUT_SECONDS", updated.get("updated_keys", []))

        res2 = self.client.get("/api/settings/runtime")
        self.assertEqual(res2.status_code, 200, res2.text)
        fields2 = {
            str(item.get("key")): item
            for item in res2.json().get("fields", [])
            if isinstance(item, dict)
        }
        self.assertEqual(fields2["ALLOW_SIMULATED_TRAINING"]["value"], True)
        self.assertEqual(fields2["EXTERNAL_COMMAND_TIMEOUT_SECONDS"]["value"], 12345)
        self.assertEqual(fields2["ALLOW_SIMULATED_TRAINING"]["source"], "override")

        persisted_path = TEST_DATA_DIR / "system" / "runtime_overrides.json"
        self.assertTrue(persisted_path.exists())

    def test_update_rejects_unknown_keys(self):
        res = self.client.put(
            "/api/settings/runtime",
            json={"updates": {"UNSUPPORTED_KEY": "abc"}},
        )
        self.assertEqual(res.status_code, 400, res.text)
        self.assertIn("Unsupported setting keys", res.text)

