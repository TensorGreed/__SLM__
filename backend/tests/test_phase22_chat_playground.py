"""Phase 22 tests: chat playground API scaffold."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase22_playground_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase22_playground_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase22ChatPlaygroundTests(unittest.TestCase):
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
            json={"name": name, "description": "phase22"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_mock_playground_chat_returns_assistant_reply(self):
        project_id = self._create_project("phase22-playground-1")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/playground/chat",
            json={
                "provider": "mock",
                "model_name": "microsoft/phi-2",
                "messages": [
                    {"role": "user", "content": "Hello playground"},
                ],
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload.get("provider"), "mock")
        self.assertEqual(payload.get("message_count"), 1)
        self.assertIn("Hello playground", str(payload.get("reply", "")))
        self.assertTrue(bool(payload.get("latency_ms")) or payload.get("latency_ms") == 0)

    def test_playground_rejects_unsupported_provider(self):
        project_id = self._create_project("phase22-playground-2")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/playground/chat",
            json={
                "provider": "unknown_provider",
                "messages": [
                    {"role": "user", "content": "hello"},
                ],
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = str(resp.json().get("detail", ""))
        self.assertIn("Unsupported provider", detail)

