"""Phase 26 tests: roadmap2 implementation slices (items 1-5)."""

from __future__ import annotations

import json
import os
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
    def _cleanup_test_artifacts(cls):
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass

        if TEST_DB_PATH.exists():
            for _ in range(40):
                try:
                    TEST_DB_PATH.unlink()
                    break
                except PermissionError:
                    time.sleep(0.1)
                except FileNotFoundError:
                    break
            if TEST_DB_PATH.exists():
                try:
                    TEST_DB_PATH.unlink(missing_ok=True)
                except PermissionError:
                    pass

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
        cls._cleanup_test_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_test_artifacts()

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

    def _upload_and_process_raw_text(self, project_id: int, *, filename: str, content: str) -> int:
        upload_resp = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={"file": (filename, content.encode("utf-8"), "text/plain")},
            data={"source": "upload", "sensitivity": "internal", "license_info": ""},
        )
        self.assertEqual(upload_resp.status_code, 201, upload_resp.text)
        document_id = int(upload_resp.json()["id"])
        process_resp = self.client.post(f"/api/projects/{project_id}/ingestion/documents/{document_id}/process")
        self.assertEqual(process_resp.status_code, 200, process_resp.text)
        return document_id

    def _upload_raw_text_pending(self, project_id: int, *, filename: str, content: str) -> int:
        upload_resp = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={"file": (filename, content.encode("utf-8"), "text/plain")},
            data={"source": "upload", "sensitivity": "internal", "license_info": ""},
        )
        self.assertEqual(upload_resp.status_code, 201, upload_resp.text)
        return int(upload_resp.json()["id"])

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

    def test_item6_newbie_autopilot_intent_and_one_click_run(self):
        project_id = self._create_project("phase26-item6")
        self._write_prepared_rows(
            project_id,
            rows=[
                {
                    "text": (
                        "User: Summarize this support thread.\n"
                        "Assistant: Here is a short summary and next actions."
                    )
                },
                {
                    "text": (
                        "User: Extract contract liability clauses.\n"
                        "Assistant: Liability clauses: indemnity, caps, exceptions."
                    )
                },
            ],
        )

        resolve_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/intent-resolve",
            json={
                "intent": "I want a model that summarizes support tickets for my team.",
                "target_device": "laptop",
                "available_vram_gb": 8,
            },
        )
        self.assertEqual(resolve_resp.status_code, 200, resolve_resp.text)
        resolve_payload = resolve_resp.json()
        plan = dict(resolve_payload.get("plan") or {})
        safe_cfg = dict(resolve_payload.get("safe_training_config") or {})
        self.assertTrue(str(plan.get("preset_id") or "").strip(), plan)
        self.assertIn("base_model", safe_cfg)
        self.assertIn("task_type", safe_cfg)

        run_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/one-click-run",
            json={
                "intent": "I need structured extraction for invoice fields.",
                "target_device": "laptop",
                "available_vram_gb": 8,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_payload = run_resp.json()
        experiment = dict(run_payload.get("experiment") or {})
        experiment_id = int(experiment.get("id") or 0)
        self.assertGreater(experiment_id, 0, run_payload)

        started = bool(run_payload.get("started"))
        if started:
            terminal = ""
            for _ in range(120):
                status_resp = self.client.get(
                    f"/api/projects/{project_id}/training/experiments/{experiment_id}/status"
                )
                self.assertEqual(status_resp.status_code, 200, status_resp.text)
                status_payload = status_resp.json()
                terminal = str(status_payload.get("status") or "").strip().lower()
                if terminal in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.1)
            if terminal not in {"completed", "failed"}:
                self.assertIn(terminal, {"running", "pending"})
        else:
            self.assertTrue(str(run_payload.get("start_error") or "").strip(), run_payload)

    def test_item6b_newbie_autopilot_auto_applies_rewrite_suggestion(self):
        project_id = self._create_project("phase26-item6b")
        self._write_prepared_rows(
            project_id,
            rows=[
                {
                    "text": f"User: support request {idx}\nAssistant: concise response {idx}",
                }
                for idx in range(24)
            ],
        )

        resolve_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/intent-resolve",
            json={
                "intent": "help",
                "target_device": "laptop",
                "available_vram_gb": 8,
            },
        )
        self.assertEqual(resolve_resp.status_code, 200, resolve_resp.text)
        resolve_payload = resolve_resp.json()
        clarification = dict(resolve_payload.get("intent_clarification") or {})
        self.assertTrue(bool(clarification.get("required")), clarification)
        rewrite_rows = list(clarification.get("rewrite_suggestions") or [])
        self.assertGreater(len(rewrite_rows), 0, clarification)

        run_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/one-click-run",
            json={
                "intent": "help",
                "target_device": "laptop",
                "available_vram_gb": 8,
                "auto_apply_rewrite": True,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_payload = run_resp.json()
        applied = dict(run_payload.get("applied_intent_rewrite") or {})
        self.assertTrue(bool(applied.get("applied")), run_payload)
        original_intent = str(applied.get("original_intent") or "").strip().lower()
        rewritten_intent = str(applied.get("rewritten_intent") or "").strip().lower()
        self.assertTrue(rewritten_intent, applied)
        self.assertNotEqual(original_intent, rewritten_intent, applied)

    def test_item6c_newbie_autopilot_respects_base_model_override(self):
        project_id = self._create_project("phase26-item6c")
        self._write_prepared_rows(
            project_id,
            rows=[
                {
                    "text": f"User: classify ticket {idx}\nAssistant: label it as billing or technical {idx}",
                }
                for idx in range(28)
            ],
        )
        override_model = "Qwen/Qwen2.5-1.5B-Instruct"

        plan_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/plan-v2",
            json={
                "intent": "Classify incoming support tickets by queue.",
                "target_device": "laptop",
                "available_vram_gb": 8,
                "base_model": override_model,
            },
        )
        self.assertEqual(plan_resp.status_code, 200, plan_resp.text)
        plans = [dict(item) for item in list(plan_resp.json().get("plans") or []) if isinstance(item, dict)]
        self.assertGreater(len(plans), 0, plan_resp.json())
        for plan in plans:
            config = dict(plan.get("config") or {})
            self.assertEqual(str(config.get("base_model") or "").strip(), override_model)

        run_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/one-click-run",
            json={
                "intent": "Classify incoming support tickets by queue.",
                "target_device": "laptop",
                "available_vram_gb": 8,
                "base_model": override_model,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_payload = run_resp.json()
        experiment = dict(run_payload.get("experiment") or {})
        self.assertGreater(int(experiment.get("id") or 0), 0, run_payload)
        self.assertEqual(str(experiment.get("base_model") or "").strip(), override_model)

    def test_item6d_newbie_autopilot_resolves_target_device_from_profile(self):
        project_id = self._create_project("phase26-item6d")
        self._write_prepared_rows(
            project_id,
            rows=[
                {
                    "text": f"User: support ticket {idx}\nAssistant: concise answer {idx}",
                }
                for idx in range(28)
            ],
        )

        plan_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/plan-v2",
            json={
                "intent": "Build a support Q&A assistant.",
                "target_profile_id": "mobile_cpu",
                "target_device": "server",
                "available_vram_gb": 6,
            },
        )
        self.assertEqual(plan_resp.status_code, 200, plan_resp.text)
        plan_payload = plan_resp.json()
        self.assertEqual(str(plan_payload.get("resolved_target_device") or ""), "mobile", plan_payload)

        run_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/one-click-run",
            json={
                "intent": "Build a support Q&A assistant.",
                "target_profile_id": "mobile_cpu",
                "target_device": "server",
                "available_vram_gb": 6,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_payload = run_resp.json()
        plan_v2 = dict(run_payload.get("plan_v2") or {})
        self.assertEqual(str(plan_v2.get("resolved_target_device") or ""), "mobile", run_payload)

    def test_item6e_newbie_autopilot_one_click_auto_prepares_raw_data(self):
        project_id = self._create_project("phase26-item6e")
        raw_lines = "\n".join(
            [f"Ticket {idx}: customer cannot sign in. Response should include reset steps." for idx in range(36)]
        )
        self._upload_and_process_raw_text(
            project_id,
            filename="support_tickets.txt",
            content=raw_lines,
        )

        run_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/one-click-run",
            json={
                "intent": "Summarize support tickets with next steps.",
                "target_profile_id": "edge_gpu",
                "available_vram_gb": 8,
                "auto_prepare_data": True,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_payload = run_resp.json()
        auto_prepare = dict(run_payload.get("auto_prepare") or {})
        self.assertTrue(bool(auto_prepare.get("succeeded")), run_payload)

        prepared_train = TEST_DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"
        self.assertTrue(prepared_train.exists(), prepared_train)
        row_count = len([line for line in prepared_train.read_text(encoding="utf-8").splitlines() if line.strip()])
        self.assertGreater(row_count, 0, row_count)
        experiment = dict(run_payload.get("experiment") or {})
        self.assertGreater(int(experiment.get("id") or 0), 0, run_payload)

    def test_item6f_newbie_autopilot_auto_processes_pending_raw_docs(self):
        project_id = self._create_project("phase26-item6f")
        raw_lines = "\n".join(
            [f"Ticket {idx}: customer cannot sign in. Response should include reset steps." for idx in range(24)]
        )
        doc_id = self._upload_raw_text_pending(
            project_id,
            filename="support_pending.txt",
            content=raw_lines,
        )

        docs_before = self.client.get(f"/api/projects/{project_id}/ingestion/documents")
        self.assertEqual(docs_before.status_code, 200, docs_before.text)
        before_rows = [dict(item) for item in docs_before.json() if isinstance(item, dict)]
        target_before = next((item for item in before_rows if int(item.get("id") or 0) == doc_id), {})
        self.assertEqual(str(target_before.get("status") or "").strip().lower(), "pending", target_before)

        run_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/one-click-run",
            json={
                "intent": "Summarize support tickets with next steps.",
                "target_profile_id": "edge_gpu",
                "available_vram_gb": 8,
                "auto_prepare_data": True,
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_payload = run_resp.json()
        auto_prepare = dict(run_payload.get("auto_prepare") or {})
        self.assertTrue(bool(auto_prepare.get("succeeded")), run_payload)
        raw_docs = dict(auto_prepare.get("raw_documents") or {})
        self.assertTrue(bool(raw_docs.get("attempted")), raw_docs)
        self.assertGreaterEqual(int(raw_docs.get("processed_count") or 0), 1, raw_docs)
        self.assertGreaterEqual(int(raw_docs.get("accepted_after") or 0), 1, raw_docs)

        docs_after = self.client.get(f"/api/projects/{project_id}/ingestion/documents")
        self.assertEqual(docs_after.status_code, 200, docs_after.text)
        after_rows = [dict(item) for item in docs_after.json() if isinstance(item, dict)]
        target_after = next((item for item in after_rows if int(item.get("id") or 0) == doc_id), {})
        self.assertEqual(str(target_after.get("status") or "").strip().lower(), "accepted", target_after)

    def test_item6g_newbie_autopilot_auto_repairs_incompatible_override_in_plan_v2(self):
        project_id = self._create_project("phase26-item6g")
        self._write_prepared_rows(
            project_id,
            rows=[
                {
                    "text": f"User: route ticket {idx}\nAssistant: classify and route by intent {idx}",
                }
                for idx in range(30)
            ],
        )
        incompatible_override = "meta-llama/Llama-3.1-8B-Instruct"

        plan_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/plan-v2",
            json={
                "intent": "Classify support tickets for on-device assistant.",
                "target_profile_id": "mobile_cpu",
                "target_device": "server",
                "available_vram_gb": 8,
                "base_model": incompatible_override,
            },
        )
        self.assertEqual(plan_resp.status_code, 200, plan_resp.text)
        payload = plan_resp.json()

        compatibility = dict(payload.get("target_compatibility") or {})
        self.assertTrue(bool(compatibility.get("compatible")), compatibility)
        auto_repair = dict(compatibility.get("auto_repair") or {})
        self.assertTrue(bool(auto_repair.get("applied")), auto_repair)
        self.assertEqual(str(auto_repair.get("original_model_id") or "").strip(), incompatible_override)
        repaired_model = str(auto_repair.get("repaired_model_id") or "").strip()
        self.assertTrue(repaired_model, auto_repair)
        self.assertNotEqual(repaired_model, incompatible_override, auto_repair)

        plans = [dict(item) for item in list(payload.get("plans") or []) if isinstance(item, dict)]
        self.assertGreater(len(plans), 0, payload)
        for plan in plans:
            config = dict(plan.get("config") or {})
            self.assertEqual(str(config.get("base_model") or "").strip(), repaired_model)

        guardrails = dict(payload.get("guardrails") or {})
        warnings = [str(item) for item in list(guardrails.get("warnings") or [])]
        self.assertTrue(any("Autopilot switched" in item for item in warnings), warnings)

    @patch("app.api.training.start_training", new_callable=AsyncMock)
    def test_item6h_newbie_autopilot_one_click_uses_auto_repaired_model(self, mock_start_training: AsyncMock):
        project_id = self._create_project("phase26-item6h")
        self._write_prepared_rows(
            project_id,
            rows=[
                {
                    "text": f"User: resolve issue {idx}\nAssistant: provide support resolution {idx}",
                }
                for idx in range(30)
            ],
        )
        incompatible_override = "meta-llama/Llama-3.1-8B-Instruct"
        mock_start_training.return_value = {"ok": True, "task_id": "stub-autopilot-run"}

        run_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/one-click-run",
            json={
                "intent": "Build an on-device support assistant.",
                "target_profile_id": "mobile_cpu",
                "target_device": "server",
                "available_vram_gb": 8,
                "base_model": incompatible_override,
                "plan_profile": "balanced",
            },
        )
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        payload = run_resp.json()

        plan_v2 = dict(payload.get("plan_v2") or {})
        compatibility = dict(plan_v2.get("target_compatibility") or {})
        self.assertTrue(bool(compatibility.get("compatible")), compatibility)
        auto_repair = dict(compatibility.get("auto_repair") or {})
        self.assertTrue(bool(auto_repair.get("applied")), auto_repair)
        repaired_model = str(auto_repair.get("repaired_model_id") or "").strip()
        self.assertTrue(repaired_model, auto_repair)
        self.assertNotEqual(repaired_model, incompatible_override, auto_repair)

        experiment = dict(payload.get("experiment") or {})
        experiment_id = int(experiment.get("id") or 0)
        self.assertGreater(experiment_id, 0, payload)
        self.assertEqual(str(experiment.get("base_model") or "").strip(), repaired_model)
        self.assertTrue(bool(payload.get("started")), payload)


if __name__ == "__main__":
    unittest.main()
