"""Phase 23 tests: playground sessions, model picker, and streaming chat."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase23_playground_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase23_playground_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase23PlaygroundSessionsStreamingTests(unittest.TestCase):
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

    def _create_project(self, name: str, base_model_name: str = "microsoft/phi-2") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": name,
                "description": "phase23",
                "base_model_name": base_model_name,
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_chat_persists_session_and_history(self):
        project_id = self._create_project("phase23-playground-1")
        chat_resp = self.client.post(
            f"/api/projects/{project_id}/training/playground/chat",
            json={
                "provider": "mock",
                "save_history": True,
                "session_title": "My Session",
                "messages": [{"role": "user", "content": "Hello there"}],
            },
        )
        self.assertEqual(chat_resp.status_code, 200, chat_resp.text)
        payload = chat_resp.json()
        session_id = int(payload.get("session_id") or 0)
        self.assertGreater(session_id, 0)
        self.assertEqual(payload.get("provider"), "mock")
        self.assertEqual(payload.get("message_count"), 1)

        list_resp = self.client.get(f"/api/projects/{project_id}/training/playground/sessions")
        self.assertEqual(list_resp.status_code, 200, list_resp.text)
        sessions = [item for item in list_resp.json().get("sessions", []) if isinstance(item, dict)]
        self.assertTrue(any(int(item.get("id", 0)) == session_id for item in sessions))

        detail_resp = self.client.get(
            f"/api/projects/{project_id}/training/playground/sessions/{session_id}"
        )
        self.assertEqual(detail_resp.status_code, 200, detail_resp.text)
        messages = [item for item in detail_resp.json().get("messages", []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(messages), 2)
        self.assertEqual(messages[-1].get("role"), "assistant")
        self.assertIn("Hello there", str(messages[-1].get("content", "")))

    def test_stream_endpoint_emits_delta_and_final(self):
        project_id = self._create_project("phase23-playground-2")
        events: list[dict] = []
        with self.client.stream(
            "POST",
            f"/api/projects/{project_id}/training/playground/chat/stream",
            json={
                "provider": "mock",
                "save_history": True,
                "messages": [{"role": "user", "content": "stream me"}],
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            for line in response.iter_lines():
                text = str(line or "").strip()
                if not text.startswith("data:"):
                    continue
                raw = text.removeprefix("data:").strip()
                if not raw:
                    continue
                events.append(json.loads(raw))

        self.assertTrue(any(item.get("type") == "delta" for item in events), events)
        finals = [item for item in events if item.get("type") == "final"]
        self.assertEqual(len(finals), 1, events)
        final_event = finals[0]
        self.assertEqual(final_event.get("provider"), "mock")
        self.assertGreater(int(final_event.get("session_id") or 0), 0)
        self.assertIn("stream me", str(final_event.get("reply", "")))

    def test_playground_models_includes_project_base_model(self):
        project_id = self._create_project("phase23-playground-3", base_model_name="TinyLlama/TinyLlama-1.1B")
        resp = self.client.get(f"/api/projects/{project_id}/training/playground/models")
        self.assertEqual(resp.status_code, 200, resp.text)
        models = [item for item in resp.json().get("models", []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(models), 1)
        names = {str(item.get("model_name", "")) for item in models}
        self.assertIn("TinyLlama/TinyLlama-1.1B", names)


if __name__ == "__main__":
    unittest.main()
