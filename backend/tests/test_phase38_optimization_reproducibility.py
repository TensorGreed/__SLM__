"""Phase 38 tests: optimization evidence persistence and export manifest attachment."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase38_optimization_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase38_optimization_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Phase38OptimizationReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        cls._prev_data_dir = settings.DATA_DIR
        cls._prev_database_url = settings.DATABASE_URL
        cls._prev_db_require_alembic = settings.DB_REQUIRE_ALEMBIC_HEAD
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.DATABASE_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
        settings.DB_REQUIRE_ALEMBIC_HEAD = False
        settings.ensure_dirs()
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
        settings.DATA_DIR = cls._prev_data_dir
        settings.DATABASE_URL = cls._prev_database_url
        settings.DB_REQUIRE_ALEMBIC_HEAD = cls._prev_db_require_alembic
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
            json={"name": name, "description": "phase38"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_local_base_model(self, project_id: int) -> Path:
        model_dir = TEST_DATA_DIR / "projects" / str(project_id) / "local_base_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            """{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 512,
  "num_hidden_layers": 4,
  "vocab_size": 32000,
  "max_position_embeddings": 2048
}
""",
            encoding="utf-8",
        )
        return model_dir

    def _create_experiment(self, project_id: int, base_model_path: Path) -> dict:
        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase38-exp",
                "description": "phase38 experiment",
                "config": {"base_model": base_model_path.as_posix()},
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()
        self.assertTrue(payload.get("output_dir"))
        return payload

    def _seed_experiment_artifacts(self, output_dir: str) -> Path:
        model_dir = Path(output_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            '{"architectures":["LlamaForCausalLM"],"model_type":"llama"}',
            encoding="utf-8",
        )
        (model_dir / "tokenizer.json").write_text('{"tokenizer":"dummy"}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"phase38-weights")
        return model_dir

    def _fake_probe(self):
        async def _impl(*, project_id: int, candidate: dict):
            return (
                {
                    "model_size_bytes": int(candidate.get("size_bytes") or 0),
                    "benchmark_samples": 2,
                    "metrics": {"latency": {"p50_ms": 9.8, "avg_ms": 10.1}},
                    "runtime": {"engine": "transformers", "device": "cpu"},
                    "sample_outputs": [
                        {"output_preview": "Root cause summary and mitigation."},
                        {"output_preview": "Incident owner and timeline."},
                    ],
                    "prompt_set": {
                        "source": "file",
                        "prompt_set_hash": "abc123promptset",
                        "prompt_count": 5,
                    },
                    "report_path": "/tmp/phase38-probe.json",
                    "report_sha256": "deadbeef",
                },
                "",
            )

        return _impl

    def test_optimize_persists_reproducibility_run_metadata(self):
        project_id = self._create_project("phase38-optimize-persist")
        base_model_dir = self._seed_local_base_model(project_id)
        exp = self._create_experiment(project_id, base_model_dir)
        self._seed_experiment_artifacts(str(exp["output_dir"]))

        with patch("app.services.export_service._run_optimizer_benchmark_probe", side_effect=self._fake_probe()):
            resp = self.client.post(
                f"/api/projects/{project_id}/export/optimize",
                json={"target_id": "vllm_server"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        run_info = payload.get("optimization_run") or {}
        self.assertTrue(run_info.get("run_id"))
        self.assertTrue(run_info.get("run_hash"))
        self.assertTrue(run_info.get("prompt_set_id"))
        self.assertTrue(run_info.get("prompt_set_hash"))
        self.assertGreaterEqual(int(run_info.get("candidate_count") or 0), 1)

        run_path = Path(str(run_info.get("run_path") or ""))
        self.assertTrue(run_path.exists(), run_path)
        run_payload = json.loads(run_path.read_text(encoding="utf-8"))
        self.assertEqual(str(run_payload.get("run_hash") or ""), str(run_info.get("run_hash") or ""))
        self.assertEqual(
            str((run_payload.get("prompt_set") or {}).get("prompt_set_id") or ""),
            str(run_info.get("prompt_set_id") or ""),
        )
        rows = [item for item in list(run_payload.get("candidates") or []) if isinstance(item, dict)]
        self.assertTrue(rows)
        first = rows[0]
        self.assertTrue(str(first.get("artifact_identifier") or ""))
        self.assertIn(str(first.get("metric_source") or ""), {"measured", "mixed", "estimated"})
        self.assertIn("metrics", first)
        self.assertIn("measurement", first)

    def test_export_manifest_attaches_latest_optimization_evidence(self):
        project_id = self._create_project("phase38-export-evidence")
        base_model_dir = self._seed_local_base_model(project_id)
        exp = self._create_experiment(project_id, base_model_dir)
        self._seed_experiment_artifacts(str(exp["output_dir"]))

        with patch("app.services.export_service._run_optimizer_benchmark_probe", side_effect=self._fake_probe()):
            optimize_resp = self.client.post(
                f"/api/projects/{project_id}/export/optimize",
                json={"target_id": "vllm_server"},
            )
        self.assertEqual(optimize_resp.status_code, 200, optimize_resp.text)
        optimize_payload = optimize_resp.json()
        optimize_run = optimize_payload.get("optimization_run") or {}

        create_resp = self.client.post(
            f"/api/projects/{project_id}/export/create",
            json={
                "experiment_id": int(exp["id"]),
                "export_format": "huggingface",
                "quantization": "none",
            },
        )
        self.assertEqual(create_resp.status_code, 201, create_resp.text)
        export_id = int(create_resp.json()["id"])

        fake_deployment = {
            "summary": {"deployable_artifact": True},
            "artifact_validation": {"errors": []},
            "targets": [],
        }
        with patch("app.services.export_service.run_deployment_target_suite", return_value=fake_deployment):
            run_resp = self.client.post(
                f"/api/projects/{project_id}/export/{export_id}/run",
                json={},
            )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_payload = run_resp.json()
        manifest = dict(run_payload.get("manifest") or {})
        evidence = dict(manifest.get("optimization_evidence") or {})
        self.assertTrue(evidence, manifest)
        self.assertEqual(
            str(evidence.get("run_id") or ""),
            str(optimize_run.get("run_id") or ""),
        )
        self.assertEqual(
            str(evidence.get("run_hash") or ""),
            str(optimize_run.get("run_hash") or ""),
        )
        self.assertTrue(str(evidence.get("prompt_set_id") or ""))
        self.assertTrue(str(evidence.get("prompt_set_hash") or ""))
        self.assertTrue(isinstance(evidence.get("selected_candidate"), dict) or evidence.get("selected_candidate") is None)


if __name__ == "__main__":
    unittest.main()

