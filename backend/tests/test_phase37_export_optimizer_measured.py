"""Phase 37 tests: export optimizer uses measured probes with estimated fallback."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase37_export_optimizer_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase37_export_optimizer_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Phase37ExportOptimizerMeasuredTests(unittest.TestCase):
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
            json={"name": name, "description": "phase37 export optimizer"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_hf_artifact(self, project_id: int, experiment_id: int = 1) -> Path:
        model_dir = (
            TEST_DATA_DIR
            / "projects"
            / str(project_id)
            / "experiments"
            / str(experiment_id)
            / "model"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            '{"architectures":["LlamaForCausalLM"],"model_type":"llama"}',
            encoding="utf-8",
        )
        (model_dir / "tokenizer.json").write_text('{"model":"dummy"}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"phase37-weights")
        return model_dir

    def test_optimizer_returns_measured_metrics_when_probe_succeeds(self):
        project_id = self._create_project("phase37-optimize-measured")
        self._seed_hf_artifact(project_id=project_id, experiment_id=1)

        async def _fake_probe(*, project_id: int, candidate: dict):
            return (
                {
                    "model_size_bytes": int(candidate.get("size_bytes") or 0),
                    "benchmark_samples": 2,
                    "metrics": {"latency": {"p50_ms": 12.5, "avg_ms": 13.1}},
                    "runtime": {"engine": "transformers", "device": "cpu"},
                    "sample_outputs": [
                        {"output_preview": "Incident summary with remediation steps."},
                        {"output_preview": "Escalation policy with owner and ETA."},
                    ],
                    "report_path": "/tmp/phase37-measured.json",
                },
                "",
            )

        with patch("app.services.export_service._run_optimizer_benchmark_probe", side_effect=_fake_probe):
            resp = self.client.post(
                f"/api/projects/{project_id}/export/optimize",
                json={"target_id": "vllm_server"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()

        rows = [item for item in payload.get("candidates", []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(rows), 1)
        measured_rows = [item for item in rows if str(item.get("metric_source")) == "measured"]
        self.assertTrue(measured_rows, rows)

        top = measured_rows[0]
        self.assertEqual(
            str((top.get("metric_sources") or {}).get("latency_ms") or ""),
            "measured",
        )
        self.assertEqual(
            str((top.get("measurement") or {}).get("mode") or ""),
            "measured",
        )
        self.assertAlmostEqual(float((top.get("metrics") or {}).get("latency_ms") or 0.0), 12.5, places=3)

    def test_optimizer_keeps_candidate_with_estimated_fallback_when_probe_fails(self):
        project_id = self._create_project("phase37-optimize-fallback")
        self._seed_hf_artifact(project_id=project_id, experiment_id=1)

        async def _failing_probe(*, project_id: int, candidate: dict):
            return None, "Benchmark probe failed: mock runtime error"

        with patch("app.services.export_service._run_optimizer_benchmark_probe", side_effect=_failing_probe):
            resp = self.client.post(
                f"/api/projects/{project_id}/export/optimize",
                json={"target_id": "vllm_server"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        rows = [item for item in payload.get("candidates", []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(rows), 1)

        top = rows[0]
        self.assertEqual(
            str((top.get("measurement") or {}).get("mode") or ""),
            "estimated",
        )
        self.assertIn(
            "Benchmark probe failed",
            str((top.get("measurement") or {}).get("fallback_reason") or ""),
        )
        self.assertNotEqual(str(top.get("metric_source") or ""), "measured")
        reasons = [str(item) for item in list(top.get("reasons") or [])]
        self.assertTrue(any("Benchmark probe failed" in item for item in reasons), reasons)


if __name__ == "__main__":
    unittest.main()
