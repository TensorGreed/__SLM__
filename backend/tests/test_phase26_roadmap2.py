"""Phase 26 tests: roadmap2 implementation slices (items 1-5)."""

from __future__ import annotations

import json
import os
import time
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase26_roadmap2_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase26_roadmap2_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"
os.environ["ALLOW_SYNTHETIC_DEMO_FALLBACK"] = "true"
os.environ["COMPRESSION_BACKEND"] = "stub"
os.environ["ALLOW_STUB_COMPRESSION"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Phase26Roadmap2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ALLOW_SYNTHETIC_DEMO_FALLBACK = True
        settings.COMPRESSION_BACKEND = "stub"
        settings.ALLOW_STUB_COMPRESSION = True
        settings.TRAINING_BACKEND = "simulate"
        settings.ALLOW_SIMULATED_TRAINING = True
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
        resp = self.client.post(
            "/api/projects",
            json={"name": unique_name, "description": "phase26"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _write_prepared_rows(self, project_id: int, rows: list[dict]) -> Path:
        prepared_dir = TEST_DATA_DIR / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        train_path = prepared_dir / "train.jsonl"
        with open(train_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return train_path

    def test_item1_multiturn_synthetic_generation_and_save(self):
        project_id = self._create_project("phase26-item1")
        source_text = (
            "Machine learning systems learn from historical data and improve over time. "
            "Feature engineering and clean labels improve downstream model quality. "
            "Evaluation should include both quality and safety checks before deployment. "
            "Monitoring in production helps detect drift and model regressions."
        )
        generate_resp = self.client.post(
            f"/api/projects/{project_id}/synthetic/generate-conversations",
            json={
                "source_text": source_text,
                "num_dialogues": 2,
                "min_turns": 3,
                "max_turns": 4,
            },
        )
        self.assertEqual(generate_resp.status_code, 200, generate_resp.text)
        payload = generate_resp.json()
        conversations = payload.get("conversations", [])
        self.assertEqual(int(payload.get("count", 0)), 2)
        self.assertEqual(len(conversations), 2)
        for conv in conversations:
            self.assertGreaterEqual(int(conv.get("turn_count", 0)), 3)
            self.assertLessEqual(int(conv.get("turn_count", 0)), 4)
            self.assertIsInstance(conv.get("messages"), list)

        save_resp = self.client.post(
            f"/api/projects/{project_id}/synthetic/save-conversations",
            json={
                "conversations": conversations,
                "min_confidence": 0.0,
            },
        )
        self.assertEqual(save_resp.status_code, 200, save_resp.text)
        save_payload = save_resp.json()
        self.assertEqual(int(save_payload.get("accepted", 0)), 2)

    def test_item2_semantic_dataset_intelligence(self):
        project_id = self._create_project("phase26-item2")
        self._write_prepared_rows(
            project_id,
            rows=[
                {"text": "Reset customer password with identity verification."},
                {"text": "Reset user password after verifying identity."},
                {"text": "Deploy Kubernetes service with rolling updates."},
                {"text": "Fine-tune language model with LoRA adapters."},
            ],
        )
        resp = self.client.post(
            f"/api/projects/{project_id}/dataset/semantic-intelligence/analyze",
            json={
                "target_split": "train",
                "sample_size": 200,
                "similarity_threshold": 0.8,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertIn("semantic_diversity_score", payload)
        self.assertIn("redundancy_ratio", payload)
        self.assertIn("clusters", payload)
        report_path = Path(str(payload.get("report_path") or ""))
        self.assertTrue(report_path.exists(), report_path)

    def test_item3_distillation_preflight(self):
        project_id = self._create_project("phase26-item3")
        self._write_prepared_rows(
            project_id,
            rows=[
                {
                    "text": "User: Explain TLS best practices.\nAssistant: Use TLS 1.2+ and rotate certs.",
                }
            ],
        )
        missing_teacher_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/preflight",
            json={
                "config": {
                    "base_model": "microsoft/phi-2",
                    "training_mode": "sft",
                    "task_type": "causal_lm",
                    "distillation_enabled": True,
                }
            },
        )
        self.assertEqual(missing_teacher_resp.status_code, 200, missing_teacher_resp.text)
        preflight_missing = missing_teacher_resp.json().get("preflight", {})
        errors = [str(item) for item in preflight_missing.get("errors", [])]
        self.assertTrue(
            any("distillation_teacher_model" in item for item in errors),
            errors,
        )

        valid_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/preflight",
            json={
                "config": {
                    "base_model": "microsoft/phi-2",
                    "training_mode": "sft",
                    "task_type": "causal_lm",
                    "distillation_enabled": True,
                    "distillation_teacher_model": "meta-llama/Llama-3.1-8B-Instruct",
                    "distillation_alpha": 0.6,
                    "distillation_temperature": 2.0,
                }
            },
        )
        self.assertEqual(valid_resp.status_code, 200, valid_resp.text)
        capability = valid_resp.json().get("preflight", {}).get("capability_summary", {})
        distill = capability.get("distillation", {})
        self.assertTrue(bool(distill.get("enabled")))
        self.assertEqual(
            str(distill.get("teacher_model") or ""),
            "meta-llama/Llama-3.1-8B-Instruct",
        )

    def test_item4_cloud_burst_catalog_quote_plan(self):
        project_id = self._create_project("phase26-item4")
        catalog_resp = self.client.get(f"/api/projects/{project_id}/training/cloud-burst/catalog")
        self.assertEqual(catalog_resp.status_code, 200, catalog_resp.text)
        catalog = catalog_resp.json()
        self.assertGreaterEqual(int(catalog.get("provider_count", 0)), 3)

        quote_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/quote",
            json={
                "provider_id": "runpod",
                "gpu_sku": "h100.80gb",
                "duration_hours": 1.5,
                "spot": True,
            },
        )
        self.assertEqual(quote_resp.status_code, 200, quote_resp.text)
        quote = quote_resp.json()
        total = float(quote.get("cost_breakdown_usd", {}).get("total", 0.0))
        self.assertGreater(total, 0.0)

        plan_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/launch-plan",
            json={
                "provider_id": "runpod",
                "gpu_sku": "a10g.24gb",
                "duration_hours": 2.0,
                "spot": True,
            },
        )
        self.assertEqual(plan_resp.status_code, 200, plan_resp.text)
        plan = plan_resp.json()
        credentials = plan.get("credentials", {})
        self.assertFalse(bool(credentials.get("ready")))
        self.assertIn("api_key", list(credentials.get("missing_keys", [])))
        self.assertIn("request_template", plan)
        record_path = Path(str(plan.get("record_path") or ""))
        self.assertTrue(record_path.exists(), record_path)

    def test_item5_model_merge_api_ties_and_dex(self):
        project_id = self._create_project("phase26-item5")
        for method in ("ties", "dex"):
            resp = self.client.post(
                f"/api/projects/{project_id}/compression/merge-models",
                json={
                    "model_paths": [
                        "/tmp/model-a",
                        "/tmp/model-b",
                    ],
                    "merge_method": method,
                    "weights": [0.6, 0.4],
                },
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            payload = resp.json()
            self.assertEqual(str(payload.get("status")), "simulated")
            self.assertEqual(str(payload.get("merge_method")), method)

    def test_item4b_cloud_burst_managed_job_lifecycle(self):
        project_id = self._create_project("phase26-item4b")
        create_exp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "cloud-burst-exp",
                "config": {
                    "base_model": "microsoft/phi-2",
                },
            },
        )
        self.assertEqual(create_exp.status_code, 201, create_exp.text)
        experiment_id = int(create_exp.json().get("id"))

        submit_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/submit",
            json={
                "provider_id": "runpod",
                "gpu_sku": "a10g.24gb",
                "duration_hours": 1.0,
                "experiment_id": experiment_id,
                "spot": True,
                "auto_artifact_sync": False,
                "artifact_sync_policy": "smart",
            },
        )
        self.assertEqual(submit_resp.status_code, 200, submit_resp.text)
        submitted = submit_resp.json()
        run_id = str(submitted.get("run_id") or "")
        self.assertTrue(run_id, submitted)

        list_resp = self.client.get(f"/api/projects/{project_id}/training/cloud-burst/jobs")
        self.assertEqual(list_resp.status_code, 200, list_resp.text)
        listed_runs = list_resp.json().get("runs", [])
        self.assertTrue(
            any(str(item.get("run_id")) == run_id for item in listed_runs if isinstance(item, dict)),
            listed_runs,
        )

        final_status = ""
        for _ in range(80):
            status_resp = self.client.get(
                f"/api/projects/{project_id}/training/cloud-burst/jobs/{run_id}?logs_tail=80"
            )
            self.assertEqual(status_resp.status_code, 200, status_resp.text)
            status_payload = status_resp.json()
            final_status = str(status_payload.get("status") or "").strip().lower()
            if final_status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        self.assertEqual(final_status, "completed")

        logs_resp = self.client.get(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/{run_id}/logs?tail=60"
        )
        self.assertEqual(logs_resp.status_code, 200, logs_resp.text)
        logs_payload = logs_resp.json()
        self.assertGreater(len(list(logs_payload.get("logs") or [])), 0)

        sync_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/{run_id}/sync-artifacts",
            json={
                "policy": "smart",
                "dry_run": False,
                "max_files": 2000,
            },
        )
        self.assertEqual(sync_resp.status_code, 200, sync_resp.text)
        sync_payload = sync_resp.json()
        sync_summary = dict(sync_payload.get("sync") or {})
        self.assertGreaterEqual(int(sync_summary.get("copied_count") or 0), 1, sync_summary)

        submit_cancel = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/submit",
            json={
                "provider_id": "runpod",
                "gpu_sku": "a10g.24gb",
                "duration_hours": 1.0,
                "spot": True,
            },
        )
        self.assertEqual(submit_cancel.status_code, 200, submit_cancel.text)
        cancel_run_id = str(submit_cancel.json().get("run_id") or "")
        self.assertTrue(cancel_run_id)

        cancel_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/{cancel_run_id}/cancel"
        )
        self.assertEqual(cancel_resp.status_code, 200, cancel_resp.text)

        cancel_final = ""
        for _ in range(40):
            status_resp = self.client.get(
                f"/api/projects/{project_id}/training/cloud-burst/jobs/{cancel_run_id}"
            )
            self.assertEqual(status_resp.status_code, 200, status_resp.text)
            status_payload = status_resp.json()
            cancel_final = str(status_payload.get("status") or "").strip().lower()
            if cancel_final in {"cancelled", "completed", "failed"}:
                break
            time.sleep(0.1)
        self.assertIn(cancel_final, {"cancelled", "completed"})

    def test_item4c_cloud_burst_idempotency_and_live_mode_guardrails(self):
        project_id = self._create_project("phase26-item4c")

        submit_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/submit",
            json={
                "provider_id": "runpod",
                "gpu_sku": "a10g.24gb",
                "duration_hours": 1.0,
                "execution_mode": "simulate",
                "idempotency_key": "phase26-item4c-unique-key",
            },
        )
        self.assertEqual(submit_resp.status_code, 200, submit_resp.text)
        first_payload = submit_resp.json()
        first_run_id = str(first_payload.get("run_id") or "").strip()
        self.assertTrue(first_run_id)
        self.assertEqual(str(first_payload.get("execution_mode_effective") or ""), "simulate")

        replay_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/submit",
            json={
                "provider_id": "runpod",
                "gpu_sku": "a10g.24gb",
                "duration_hours": 2.0,
                "execution_mode": "simulate",
                "idempotency_key": "phase26-item4c-unique-key",
            },
        )
        self.assertEqual(replay_resp.status_code, 200, replay_resp.text)
        replay_payload = replay_resp.json()
        replay_run_id = str(replay_payload.get("run_id") or "").strip()
        self.assertEqual(replay_run_id, first_run_id)
        self.assertTrue(bool(replay_payload.get("idempotent_replay")))

        live_required_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/submit",
            json={
                "provider_id": "runpod",
                "gpu_sku": "a10g.24gb",
                "duration_hours": 1.0,
                "execution_mode": "live",
                "allow_fallback_to_simulation": False,
            },
        )
        self.assertEqual(live_required_resp.status_code, 400, live_required_resp.text)
        self.assertIn("requires provider credentials", live_required_resp.text.lower())

    def test_item4d_cloud_burst_metrics_bridge_and_incremental_sync(self):
        project_id = self._create_project("phase26-item4d")
        create_exp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "cloud-burst-sync-exp",
                "config": {
                    "base_model": "microsoft/phi-2",
                },
            },
        )
        self.assertEqual(create_exp.status_code, 201, create_exp.text)
        exp_payload = create_exp.json()
        experiment_id = int(exp_payload.get("id"))
        output_dir = Path(str(exp_payload.get("output_dir") or ""))
        self.assertTrue(output_dir.exists(), output_dir)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / "training_report.json").write_text(
            json.dumps({"score": 0.81}, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "checkpoints" / "step-10.safetensors").write_text(
            "weights-v1",
            encoding="utf-8",
        )

        submit_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/submit",
            json={
                "provider_id": "runpod",
                "gpu_sku": "a10g.24gb",
                "duration_hours": 1.0,
                "experiment_id": experiment_id,
                "auto_artifact_sync": False,
                "execution_mode": "simulate",
            },
        )
        self.assertEqual(submit_resp.status_code, 200, submit_resp.text)
        run_id = str(submit_resp.json().get("run_id") or "").strip()
        self.assertTrue(run_id)

        metrics_count = 0
        for _ in range(40):
            status_resp = self.client.get(
                f"/api/projects/{project_id}/training/cloud-burst/jobs/{run_id}?logs_tail=100"
            )
            self.assertEqual(status_resp.status_code, 200, status_resp.text)
            status_payload = status_resp.json()
            metrics_count = int(status_payload.get("metrics_tail_count") or 0)
            if metrics_count > 0:
                break
            time.sleep(0.1)
        self.assertGreater(metrics_count, 0)

        first_sync = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/{run_id}/sync-artifacts",
            json={
                "policy": "smart",
                "dry_run": False,
                "max_files": 100,
            },
        )
        self.assertEqual(first_sync.status_code, 200, first_sync.text)
        first_summary = dict(first_sync.json().get("sync") or {})
        self.assertGreaterEqual(int(first_summary.get("copied_count") or 0), 1, first_summary)
        manifest_path = Path(str(first_summary.get("manifest_path") or ""))
        self.assertTrue(manifest_path.exists(), manifest_path)

        second_sync = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/{run_id}/sync-artifacts",
            json={
                "policy": "smart",
                "dry_run": False,
                "max_files": 100,
            },
        )
        self.assertEqual(second_sync.status_code, 200, second_sync.text)
        second_summary = dict(second_sync.json().get("sync") or {})
        self.assertEqual(int(second_summary.get("copied_count") or 0), 0, second_summary)
        self.assertGreaterEqual(int(second_summary.get("unchanged_count") or 0), 1, second_summary)

        (output_dir / "training_report.json").write_text(
            json.dumps({"score": 0.92}, ensure_ascii=False),
            encoding="utf-8",
        )
        third_sync = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/jobs/{run_id}/sync-artifacts",
            json={
                "policy": "smart",
                "dry_run": False,
                "max_files": 100,
            },
        )
        self.assertEqual(third_sync.status_code, 200, third_sync.text)
        third_summary = dict(third_sync.json().get("sync") or {})
        self.assertGreaterEqual(int(third_summary.get("copied_count") or 0), 1, third_summary)


if __name__ == "__main__":
    unittest.main()
