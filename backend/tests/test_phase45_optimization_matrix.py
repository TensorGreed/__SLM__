"""Phase 45 tests: optimization benchmark matrix APIs and reproducibility metadata."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase45_optimization_matrix_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase45_optimization_matrix_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Phase45OptimizationMatrixTests(unittest.TestCase):
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
            json={"name": name, "description": "phase45 optimization matrix"},
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
        (model_dir / "weights.safetensors").write_bytes(b"phase45-hf-weights")
        return model_dir

    def _seed_gguf_artifact(self, project_id: int, file_name: str = "model-q4.gguf") -> Path:
        artifact = TEST_DATA_DIR / "projects" / str(project_id) / "compressed" / file_name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"phase45-gguf-weights")
        return artifact

    def _measured_probe(self):
        async def _impl(*, project_id: int, candidate: dict):
            runtime = str(candidate.get("runtime_template") or "")
            latency = 8.0 if runtime == "gguf" else 12.0
            candidate_id = str(candidate.get("id") or "candidate")
            return (
                {
                    "model_size_bytes": int(candidate.get("size_bytes") or 0),
                    "benchmark_samples": 2,
                    "metrics": {"latency": {"p50_ms": latency, "avg_ms": latency + 0.25}},
                    "runtime": {"engine": runtime, "device": "cpu"},
                    "sample_outputs": [
                        {"output_preview": f"{candidate_id} output A"},
                        {"output_preview": f"{candidate_id} output B"},
                    ],
                    "report_path": f"/tmp/{candidate_id}.json",
                    "report_sha256": f"sha-{candidate_id[:8]}",
                },
                "",
            )

        return _impl

    def _failing_probe(self):
        async def _impl(*, project_id: int, candidate: dict):
            return None, "Benchmark probe failed: mock runtime error"

        return _impl

    def test_matrix_start_and_status_include_measured_provenance_and_runtime_fingerprint(self):
        project_id = self._create_project("phase45-matrix-measured")
        self._seed_hf_artifact(project_id=project_id, experiment_id=1)
        self._seed_gguf_artifact(project_id=project_id)

        with patch("app.services.export_service._run_optimizer_benchmark_probe", side_effect=self._measured_probe()):
            start_resp = self.client.post(
                f"/api/projects/{project_id}/export/optimize/matrix/start",
                json={
                    "target_ids": ["mobile_cpu", "vllm_server"],
                    "max_probe_candidates_per_target": 4,
                },
            )
        self.assertEqual(start_resp.status_code, 200, start_resp.text)
        payload = start_resp.json()

        self.assertEqual(str(payload.get("status") or ""), "completed")
        self.assertTrue(str(payload.get("run_id") or ""))
        self.assertTrue(str(payload.get("run_hash") or ""))
        self.assertTrue(str(payload.get("run_path") or ""))

        runtime = dict(payload.get("runtime_fingerprint") or {})
        self.assertTrue(runtime)
        self.assertIn("gpu", runtime)
        self.assertIn("toolchain", runtime)
        self.assertIn("host", runtime)

        targets = [item for item in list(payload.get("targets") or []) if isinstance(item, dict)]
        self.assertEqual(len(targets), 2)
        self.assertTrue(all(int(row.get("candidate_count") or 0) >= 1 for row in targets))

        measured_seen = False
        for target_row in targets:
            candidates = [item for item in list(target_row.get("candidates") or []) if isinstance(item, dict)]
            for row in candidates:
                if str(row.get("metric_source") or "") == "measured":
                    measured_seen = True
                self.assertTrue(isinstance(row.get("metric_sources"), dict))
                self.assertTrue(isinstance(row.get("confidence"), dict))
                self.assertTrue(str((row.get("confidence") or {}).get("band") or ""))
        self.assertTrue(measured_seen)

        run_id = str(payload.get("run_id") or "")
        status_resp = self.client.get(
            f"/api/projects/{project_id}/export/optimize/matrix/{run_id}",
        )
        self.assertEqual(status_resp.status_code, 200, status_resp.text)
        status_payload = status_resp.json()
        self.assertEqual(str(status_payload.get("run_id") or ""), run_id)
        self.assertEqual(str(status_payload.get("run_hash") or ""), str(payload.get("run_hash") or ""))
        self.assertEqual(str(status_payload.get("status") or ""), "completed")

    def test_matrix_fallback_marks_estimation_reason_and_remediation(self):
        project_id = self._create_project("phase45-matrix-fallback")
        self._seed_hf_artifact(project_id=project_id, experiment_id=1)

        with patch("app.services.export_service._run_optimizer_benchmark_probe", side_effect=self._failing_probe()):
            start_resp = self.client.post(
                f"/api/projects/{project_id}/export/optimize/matrix/start",
                json={"target_ids": ["vllm_server"], "max_probe_candidates_per_target": 2},
            )
        self.assertEqual(start_resp.status_code, 200, start_resp.text)
        payload = start_resp.json()
        self.assertEqual(str(payload.get("status") or ""), "completed")

        targets = [item for item in list(payload.get("targets") or []) if isinstance(item, dict)]
        self.assertEqual(len(targets), 1)
        candidates = [item for item in list(targets[0].get("candidates") or []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(candidates), 1)

        row = candidates[0]
        self.assertIn(str(row.get("metric_source") or ""), {"mixed", "estimated"})
        self.assertEqual(str((row.get("measurement") or {}).get("mode") or ""), "estimated")
        self.assertIn(
            "Benchmark probe failed",
            str((row.get("measurement") or {}).get("fallback_reason") or ""),
        )
        self.assertTrue(str((row.get("measurement") or {}).get("fallback_remediation_hint") or ""))
        confidence = dict(row.get("confidence") or {})
        self.assertTrue(confidence)
        self.assertTrue(str(confidence.get("band") or ""))

    def test_matrix_recommendations_are_deterministic_and_include_reproducibility_fields(self):
        project_id = self._create_project("phase45-matrix-ranking")
        self._seed_hf_artifact(project_id=project_id, experiment_id=1)
        self._seed_gguf_artifact(project_id=project_id, file_name="rank-q4.gguf")

        with patch("app.services.export_service._run_optimizer_benchmark_probe", side_effect=self._measured_probe()):
            start_resp = self.client.post(
                f"/api/projects/{project_id}/export/optimize/matrix/start",
                json={"target_ids": ["mobile_cpu"], "max_probe_candidates_per_target": 3},
            )
        self.assertEqual(start_resp.status_code, 200, start_resp.text)
        run_payload = start_resp.json()
        run_id = str(run_payload.get("run_id") or "")

        rec1 = self.client.get(
            f"/api/projects/{project_id}/export/optimize/matrix/{run_id}/recommendations",
            params={"target_id": "mobile_cpu", "top_k": 2},
        )
        rec2 = self.client.get(
            f"/api/projects/{project_id}/export/optimize/matrix/{run_id}/recommendations",
            params={"target_id": "mobile_cpu", "top_k": 2},
        )
        self.assertEqual(rec1.status_code, 200, rec1.text)
        self.assertEqual(rec2.status_code, 200, rec2.text)
        self.assertEqual(rec1.json(), rec2.json())

        rec_payload = rec1.json()
        by_target = [item for item in list(rec_payload.get("recommendations_by_target") or []) if isinstance(item, dict)]
        self.assertEqual(len(by_target), 1)
        recommendations = [item for item in list(by_target[0].get("recommendations") or []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(recommendations), 1)

        first = recommendations[0]
        self.assertTrue(str(first.get("artifact_identifier") or ""))
        self.assertTrue(isinstance(first.get("metric_sources"), dict))
        self.assertTrue(isinstance(first.get("confidence"), dict))
        self.assertTrue(isinstance(first.get("rank_score"), list))

        status_resp = self.client.get(f"/api/projects/{project_id}/export/optimize/matrix/{run_id}")
        self.assertEqual(status_resp.status_code, 200, status_resp.text)
        status_payload = status_resp.json()
        self.assertTrue(str(status_payload.get("run_hash") or ""))
        self.assertTrue(str(status_payload.get("run_path") or ""))
        prompt_set = dict(status_payload.get("prompt_set") or {})
        self.assertTrue(str(prompt_set.get("prompt_set_id") or ""))
        self.assertTrue(str(prompt_set.get("prompt_set_hash") or ""))
        runtime = dict(status_payload.get("runtime_fingerprint") or {})
        self.assertIn("gpu", runtime)
        self.assertIn("toolchain", runtime)


if __name__ == "__main__":
    unittest.main()
