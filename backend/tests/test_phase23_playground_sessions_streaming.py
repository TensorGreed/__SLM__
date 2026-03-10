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

    def _create_experiment_with_output(self, project_id: int, name: str) -> Path:
        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": name,
                "description": "playground-models",
                "config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        output_dir = Path(str(resp.json().get("output_dir") or ""))
        self.assertTrue(output_dir.exists(), output_dir)
        return output_dir

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

    def test_playground_provider_catalog_includes_llama_cpp(self):
        project_id = self._create_project("phase23-playground-4")
        resp = self.client.get(f"/api/projects/{project_id}/training/playground/providers")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        providers = [item for item in payload.get("providers", []) if isinstance(item, dict)]
        provider_ids = {str(item.get("provider")) for item in providers}
        self.assertIn("mock", provider_ids)
        self.assertIn("openai_compatible", provider_ids)
        self.assertIn("llama_cpp", provider_ids)

    def test_playground_logs_persist_feedback_and_quality_summary(self):
        project_id = self._create_project("phase23-playground-5")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/playground/logs",
            json={
                "prompt": "Return JSON for this record",
                "reply": "I don't know",
                "rating": -1,
                "tags": ["quality", "refusal"],
                "notes": "Bad response.",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload.get("summary", {}).get("event_count"), 1)
        quality_checks = [item for item in payload.get("event", {}).get("quality_checks", []) if isinstance(item, dict)]
        self.assertTrue(any(str(item.get("code")) == "user_negative_feedback" for item in quality_checks))

        read_resp = self.client.get(f"/api/projects/{project_id}/training/playground/logs")
        self.assertEqual(read_resp.status_code, 200, read_resp.text)
        read_payload = read_resp.json()
        self.assertEqual(read_payload.get("count"), 1)
        summary = read_payload.get("summary", {})
        self.assertEqual(summary.get("negative_count"), 1)
        top_tags = [item for item in summary.get("top_tags", []) if isinstance(item, dict)]
        self.assertTrue(any(str(item.get("tag")) == "quality" for item in top_tags))

    def test_playground_models_include_runtime_hint_for_gguf_artifacts(self):
        project_id = self._create_project("phase23-playground-6")
        output_dir = self._create_experiment_with_output(project_id, "phase23-model-hints")
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = model_dir / "model-q4.gguf"
        gguf_path.write_bytes(b"gguf")

        resp = self.client.get(f"/api/projects/{project_id}/training/playground/models")
        self.assertEqual(resp.status_code, 200, resp.text)
        models = [item for item in resp.json().get("models", []) if isinstance(item, dict)]
        hinted = [
            item for item in models
            if isinstance(item.get("runtime_hint"), dict)
            and str(item.get("runtime_hint", {}).get("artifact_kind", "")).startswith("gguf")
        ]
        self.assertTrue(hinted)
        self.assertTrue(any(str(item.get("recommended_provider")) == "llama_cpp" for item in hinted))
        self.assertTrue(
            any(str(item.get("runtime_hint", {}).get("runtime_model_ref")) == str(gguf_path) for item in hinted)
        )


if __name__ == "__main__":
    unittest.main()
