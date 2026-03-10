"""Phase 25 tests: one-click serve plans from export and registry."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase25_serve_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase25_serve_data"

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


class Phase25OneClickServeTests(unittest.TestCase):
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
        runtime_data_dir = Path(str(settings.DATA_DIR))
        if runtime_data_dir != TEST_DATA_DIR and runtime_data_dir.exists():
            for path in sorted(runtime_data_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _create_project(self, name: str) -> int:
        resp = self.client.post("/api/projects", json={"name": name, "description": "phase25"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_completed_like_experiment(self, project_id: int) -> tuple[int, Path]:
        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "serve-exp",
                "description": "serve-plan",
                "config": {
                    "base_model": "microsoft/phi-2",
                },
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()
        experiment_id = int(payload["id"])
        output_dir = Path(str(payload.get("output_dir") or ""))
        self.assertTrue(output_dir.exists(), output_dir)

        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type":"phi"}', encoding="utf-8")
        (model_dir / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"model-bytes")
        return experiment_id, output_dir

    def test_export_serve_plan_contains_commands_and_curl(self):
        project_id = self._create_project("phase25-serve-1")
        experiment_id, _ = self._create_completed_like_experiment(project_id)

        create_resp = self.client.post(
            f"/api/projects/{project_id}/export/create",
            json={
                "experiment_id": experiment_id,
                "export_format": "huggingface",
                "quantization": "none",
            },
        )
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        export_id = int(create_resp.json()["id"])

        run_resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/run",
            json={
                "deployment_targets": ["exporter.huggingface"],
                "run_smoke_tests": False,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        self.assertEqual(str(run_resp.json().get("status")), "completed")

        serve_resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/serve-plan",
            json={
                "host": "127.0.0.1",
                "port": 8010,
                "smoke_test_prompt": "Ping serve endpoint",
            },
        )
        self.assertEqual(serve_resp.status_code, 200, serve_resp.text)
        serve_payload = serve_resp.json()
        templates = [item for item in serve_payload.get("templates", []) if isinstance(item, dict)]
        template_ids = {str(item.get("template_id")) for item in templates}
        self.assertIn("builtin.fastapi", template_ids)
        self.assertIn("runner.vllm", template_ids)
        builtin = next(item for item in templates if item.get("template_id") == "builtin.fastapi")
        smoke = builtin.get("smoke_test", {})
        self.assertIn("curl", smoke)
        self.assertIn("Ping serve endpoint", str(smoke.get("prompt", "")))

    def test_registry_serve_plan_uses_linked_export(self):
        project_id = self._create_project("phase25-serve-2")
        experiment_id, _ = self._create_completed_like_experiment(project_id)

        create_resp = self.client.post(
            f"/api/projects/{project_id}/export/create",
            json={
                "experiment_id": experiment_id,
                "export_format": "huggingface",
                "quantization": "none",
            },
        )
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        export_id = int(create_resp.json()["id"])

        run_resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/run",
            json={
                "deployment_targets": ["exporter.huggingface"],
                "run_smoke_tests": False,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)

        reg_resp = self.client.post(
            f"/api/projects/{project_id}/registry/models/register",
            json={
                "experiment_id": experiment_id,
                "export_id": export_id,
                "name": "serve-registry-model",
                "version": "v1",
            },
        )
        self.assertEqual(reg_resp.status_code, 201, reg_resp.text)
        model_id = int(reg_resp.json()["id"])

        serve_resp = self.client.post(
            f"/api/projects/{project_id}/registry/models/{model_id}/serve-plan",
            json={
                "host": "0.0.0.0",
                "port": 8020,
            },
        )
        self.assertEqual(serve_resp.status_code, 200, serve_resp.text)
        payload = serve_resp.json()
        self.assertEqual(payload.get("source"), "registry")
        self.assertEqual(int(payload.get("model_id", 0)), model_id)
        self.assertEqual(int(payload.get("export_id", 0)), export_id)
        self.assertTrue(len(payload.get("templates", [])) >= 1)


if __name__ == "__main__":
    unittest.main()
