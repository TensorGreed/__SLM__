"""Phase 28 tests: roadmap4 implementation slices (phases 1-5)."""

from __future__ import annotations

import json
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase28_roadmap4_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase28_roadmap4_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"
os.environ["ALLOW_SYNTHETIC_DEMO_FALLBACK"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Phase28Roadmap4Tests(unittest.TestCase):
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
        unique_name = f"{name}-{uuid.uuid4().hex[:8]}"
        resp = self.client.post("/api/projects", json={"name": unique_name, "description": "phase28"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _upload_text_doc(self, project_id: int, filename: str, content: str) -> int:
        resp = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={"file": (filename, content.encode("utf-8"), "text/plain")},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_completed_export(self, project_id: int) -> int:
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase28-exp",
                "description": "phase28",
                "config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        exp_payload = exp_resp.json()
        experiment_id = int(exp_payload["id"])
        output_dir = Path(str(exp_payload.get("output_dir") or ""))
        self.assertTrue(output_dir.exists(), output_dir)

        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type":"phi"}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"phase28-model")
        (model_dir / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")

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
            json={"deployment_targets": ["exporter.huggingface"], "run_smoke_tests": False},
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        self.assertEqual(str(run_resp.json().get("status")), "completed")
        return export_id

    def _create_training_experiment(self, project_id: int, name: str = "phase28-vibe-exp") -> int:
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:6]}",
                "description": "phase28-vibe",
                "config": {"base_model": "microsoft/phi-2", "num_epochs": 1},
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        return int(exp_resp.json()["id"])

    def test_phase1_magic_create_eda_and_outlier_cleanup(self):
        magic_resp = self.client.post(
            "/api/projects/magic-create",
            json={
                "prompt": (
                    "I have 500 PDFs of legal contracts and need extraction of liabilities. "
                    "My laptop has 8GB VRAM."
                )
            },
        )
        self.assertEqual(magic_resp.status_code, 201, magic_resp.text)
        project_id = int(magic_resp.json()["id"])

        self._upload_text_doc(
            project_id,
            "contracts.txt",
            "Liability clause A. Liability clause B. This is a harmless legal contract sample.",
        )
        eda_resp = self.client.get(f"/api/projects/{project_id}/ingestion/eda")
        self.assertEqual(eda_resp.status_code, 200, eda_resp.text)
        payload = eda_resp.json()
        self.assertIn("toxicity", payload)
        self.assertIn("topic_clusters", payload)
        self.assertIn("outlier_candidates", payload)

        prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        train_path = prepared_dir / "train.jsonl"

        rows = [
            {"text": "ok training sample with enough content"},
            {"text": "bad"},
            {"text": "I will kill and hate everyone here"},
        ]
        with open(train_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        prune_resp = self.client.post(f"/api/projects/{project_id}/ingestion/eda/remove-outliers", json={})
        self.assertEqual(prune_resp.status_code, 200, prune_resp.text)
        prune_payload = prune_resp.json()
        self.assertGreaterEqual(int(prune_payload.get("rows_removed", 0)), 1)
        report_path = Path(str(prune_payload.get("report_path") or ""))
        self.assertTrue(report_path.exists(), report_path)

    def test_phase3_data_governance_and_multimodal_catalog(self):
        project_id = self._create_project("phase28-safety")
        document_id = self._upload_text_doc(
            project_id,
            "safety.txt",
            (
                "Employee John Doe has SSN 123-45-6789. "
                "If this fails, I will kill the process."
            ),
        )

        clean_resp = self.client.post(
            f"/api/projects/{project_id}/cleaning/clean",
            json={
                "document_id": document_id,
                "chunk_size": 256,
                "chunk_overlap": 16,
                "redact_pii": True,
                "redact_toxicity": True,
            },
        )
        self.assertEqual(clean_resp.status_code, 200, clean_resp.text)
        clean_payload = clean_resp.json()
        self.assertGreaterEqual(len(clean_payload.get("pii_findings", [])), 1)
        self.assertGreaterEqual(len(clean_payload.get("toxicity_findings", [])), 1)

        catalog_resp = self.client.get(f"/api/projects/{project_id}/dataset/adapters/catalog")
        self.assertEqual(catalog_resp.status_code, 200, catalog_resp.text)
        adapters = catalog_resp.json().get("adapters", {})
        adapter_ids = set(adapters.keys()) if isinstance(adapters, dict) else set()
        self.assertIn("vision-language-pair", adapter_ids)
        self.assertIn("audio-transcript", adapter_ids)

    def test_phase2_model_benchmark_sweep_matrix(self):
        project_id = self._create_project("phase28-benchmark")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/benchmark-sweep",
            json={
                "target_device": "laptop",
                "primary_language": "english",
                "available_vram_gb": 12,
                "max_models": 3,
                "sample_size": 40,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        matrix = payload.get("matrix", [])
        self.assertEqual(len(matrix), 3)
        self.assertTrue(str(payload.get("run_id") or "").strip())
        self.assertIn(
            str(payload.get("benchmark_mode") or ""),
            {
                "real_sampled_heuristic",
                "real_sampled_tokenizer",
                "real_sampled_tokenizer_mixed",
            },
        )
        self.assertGreaterEqual(int(payload.get("sampled_row_count") or 0), 1)
        self.assertIn("sampled_avg_tokens", payload)
        self.assertIn("estimated_accuracy_percent", matrix[0])
        self.assertIn("estimated_latency_ms", matrix[0])

        history_resp = self.client.get(
            f"/api/projects/{project_id}/training/model-selection/benchmark-sweep/history",
        )
        self.assertEqual(history_resp.status_code, 200, history_resp.text)
        history_payload = history_resp.json()
        self.assertGreaterEqual(int(history_payload.get("count") or 0), 1)
        runs = [item for item in history_payload.get("runs", []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(runs), 1)
        self.assertEqual(str(runs[0].get("run_id") or ""), str(payload.get("run_id") or ""))

    def test_phase4_rag_compare_side_by_side(self):
        project_id = self._create_project("phase28-rag")
        self._upload_text_doc(
            project_id,
            "kb.txt",
            (
                "Policy 1: Rotate API keys every 90 days. "
                "Policy 2: Enable MFA for all admin users. "
                "Policy 3: Use TLS 1.2 or above for all endpoints."
            ),
        )
        resp = self.client.post(
            f"/api/projects/{project_id}/training/playground/rag-compare",
            json={
                "query": "How often should API keys be rotated?",
                "provider": "mock",
                "top_k": 3,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        snippets = payload.get("retrieved_snippets", [])
        self.assertGreaterEqual(len(snippets), 1)
        self.assertTrue(str(payload.get("base", {}).get("reply") or "").strip())
        self.assertTrue(str(payload.get("tuned", {}).get("reply") or "").strip())

    def test_phase4_vibe_check_timeline_snapshot(self):
        project_id = self._create_project("phase28-vibe")

        get_cfg = self.client.get(f"/api/projects/{project_id}/training/vibe-check/config")
        self.assertEqual(get_cfg.status_code, 200, get_cfg.text)
        self.assertGreaterEqual(len(get_cfg.json().get("prompts", [])), 1)

        put_cfg = self.client.put(
            f"/api/projects/{project_id}/training/vibe-check/config",
            json={
                "enabled": True,
                "interval_steps": 40,
                "provider": "mock",
                "prompts": [
                    "Prompt 1",
                    "Prompt 2",
                    "Prompt 3",
                    "Prompt 4",
                    "Prompt 5",
                ],
            },
        )
        self.assertEqual(put_cfg.status_code, 200, put_cfg.text)
        self.assertEqual(int(put_cfg.json().get("interval_steps", 0)), 40)

        experiment_id = self._create_training_experiment(project_id, name="phase28-vibe-exp")
        snap_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/{experiment_id}/vibe-check/snapshot",
            json={"step": 50, "epoch": 0.5},
        )
        self.assertEqual(snap_resp.status_code, 200, snap_resp.text)
        snap_payload = snap_resp.json()
        snapshot = snap_payload.get("snapshot") or {}
        self.assertEqual(int(snapshot.get("step", 0)), 50)
        self.assertGreaterEqual(len(snapshot.get("outputs", [])), 1)
        timeline_path = Path(str(snap_payload.get("timeline_path") or ""))
        self.assertTrue(timeline_path.exists(), timeline_path)

        timeline_resp = self.client.get(
            f"/api/projects/{project_id}/training/experiments/{experiment_id}/vibe-check/timeline"
        )
        self.assertEqual(timeline_resp.status_code, 200, timeline_resp.text)
        timeline_payload = timeline_resp.json()
        self.assertGreaterEqual(int(timeline_payload.get("snapshot_count", 0)), 1)
        self.assertGreaterEqual(len(timeline_payload.get("snapshots", [])), 1)

    def test_phase5_deploy_plan_and_mobile_sdk_stub(self):
        project_id = self._create_project("phase28-deploy")
        export_id = self._create_completed_export(project_id)

        deploy_resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/deploy-as-api",
            json={"target_id": "deployment.hf_inference_endpoint"},
        )
        self.assertEqual(deploy_resp.status_code, 200, deploy_resp.text)
        deploy_payload = deploy_resp.json()
        self.assertIn("curl_example", deploy_payload)

        execute_dry_run = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/deploy-as-api/execute",
            json={"target_id": "deployment.hf_inference_endpoint", "dry_run": True},
        )
        self.assertEqual(execute_dry_run.status_code, 200, execute_dry_run.text)
        execute_payload = execute_dry_run.json()
        self.assertEqual(str((execute_payload.get("execution") or {}).get("status")), "dry_run")

        sdk_resp = self.client.post(
            f"/api/projects/{project_id}/export/{export_id}/deploy-as-api",
            json={"target_id": "sdk.apple_coreml_stub"},
        )
        self.assertEqual(sdk_resp.status_code, 200, sdk_resp.text)
        sdk_payload = sdk_resp.json()
        zip_path = Path(str((sdk_payload.get("sdk_artifact") or {}).get("zip_path") or ""))
        self.assertTrue(zip_path.exists(), zip_path)


if __name__ == "__main__":
    unittest.main()
