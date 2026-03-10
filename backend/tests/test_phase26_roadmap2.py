"""Phase 26 tests: roadmap2 implementation slices (items 1-5)."""

from __future__ import annotations

import json
import os
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


if __name__ == "__main__":
    unittest.main()
