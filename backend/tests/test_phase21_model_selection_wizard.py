"""Phase 21 tests: model selection wizard recommendation API."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase21_model_selection_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase21_model_selection_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase21ModelSelectionWizardTests(unittest.TestCase):
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
        resp = self.client.post("/api/projects", json={"name": name, "description": "phase21"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_model_recommendation_returns_ranked_results(self):
        project_id = self._create_project("phase21-model-wizard-1")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "laptop",
                "primary_language": "coding",
                "available_vram_gb": 12,
                "task_profile": "instruction_sft",
                "top_k": 3,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(int(payload.get("project_id", 0)), project_id)

        request = payload.get("request", {})
        self.assertEqual(request.get("target_device"), "laptop")
        self.assertEqual(request.get("primary_language"), "coding")
        self.assertEqual(int(request.get("top_k", 0)), 3)

        rows = [item for item in payload.get("recommendations", []) if isinstance(item, dict)]
        self.assertEqual(len(rows), 3)
        scores = [float(item.get("match_score", 0.0)) for item in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))

        first = rows[0]
        self.assertTrue(bool(first.get("model_id")))
        self.assertIn("metadata_source", first)
        self.assertIn("architecture", first)
        self.assertIn("context_length", first)
        self.assertIn("license", first)
        defaults = first.get("suggested_defaults", {})
        self.assertIn(defaults.get("task_type"), {"causal_lm", "seq2seq", "classification"})
        self.assertTrue(bool(defaults.get("chat_template")))

    def test_model_recommendation_classification_profile_sets_task_type(self):
        project_id = self._create_project("phase21-model-wizard-2")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "server",
                "primary_language": "multilingual",
                "available_vram_gb": 24,
                "task_profile": "classification",
                "top_k": 2,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = [item for item in resp.json().get("recommendations", []) if isinstance(item, dict)]
        self.assertEqual(len(rows), 2)
        for item in rows:
            defaults = item.get("suggested_defaults", {})
            self.assertEqual(defaults.get("task_type"), "classification")

    def test_model_recommendation_warns_when_vram_is_too_low(self):
        project_id = self._create_project("phase21-model-wizard-3")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "mobile",
                "primary_language": "english",
                "available_vram_gb": 2,
                "task_profile": "chat_sft",
                "top_k": 2,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        warnings = [str(item) for item in payload.get("warnings", [])]
        self.assertTrue(any("available_vram_gb" in item for item in warnings), warnings)
        rows = [item for item in payload.get("recommendations", []) if isinstance(item, dict)]
        self.assertEqual(len(rows), 2)

    def test_model_introspection_endpoint_resolves_local_config(self):
        project_id = self._create_project("phase21-model-introspect-1")
        local_model_dir = TEST_DATA_DIR / "hf_local_models" / "phase21-local-introspect"
        local_model_dir.mkdir(parents=True, exist_ok=True)
        (local_model_dir / "config.json").write_text(
            """{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 2048,
  "num_hidden_layers": 18,
  "vocab_size": 32000,
  "max_position_embeddings": 8192
}
""",
            encoding="utf-8",
        )

        resp = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/introspect",
            json={
                "model_id": local_model_dir.as_posix(),
                "allow_network": False,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        introspection = payload.get("introspection", {})
        self.assertTrue(bool(introspection.get("resolved")))
        self.assertEqual(introspection.get("source"), "local_config")
        self.assertEqual(introspection.get("architecture"), "causal_lm")
        self.assertEqual(int(introspection.get("context_length") or 0), 8192)
        self.assertTrue(float(introspection.get("params_estimate_b") or 0.0) > 0.0)

    def test_model_wizard_telemetry_records_recommend_and_apply(self):
        project_id = self._create_project("phase21-model-wizard-telemetry-1")

        recommend = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/telemetry",
            json={
                "action": "recommend",
                "source": "training_setup_wizard",
                "auto_run": True,
                "target_device": "laptop",
                "primary_language": "english",
                "available_vram_gb": 8,
                "task_profile": "chat_sft",
                "top_k": 3,
                "recommendation_count": 3,
                "recommendation_model_ids": [
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "microsoft/phi-2",
                    "google/gemma-2b-it",
                ],
            },
        )
        self.assertEqual(recommend.status_code, 200, recommend.text)
        recommend_payload = recommend.json()
        self.assertEqual(recommend_payload.get("event", {}).get("action"), "recommend")
        self.assertEqual(recommend_payload.get("summary", {}).get("recommend_events"), 1)
        self.assertEqual(recommend_payload.get("summary", {}).get("apply_events"), 0)

        apply_event = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/telemetry",
            json={
                "action": "apply",
                "source": "training_setup_wizard",
                "selected_model_id": "microsoft/phi-2",
                "selected_rank": 2,
                "selected_score": 0.84,
            },
        )
        self.assertEqual(apply_event.status_code, 200, apply_event.text)
        apply_payload = apply_event.json()
        self.assertEqual(apply_payload.get("summary", {}).get("recommend_events"), 1)
        self.assertEqual(apply_payload.get("summary", {}).get("apply_events"), 1)
        self.assertEqual(apply_payload.get("summary", {}).get("apply_conversion_rate"), 1.0)

        summary = self.client.get(f"/api/projects/{project_id}/training/model-selection/telemetry")
        self.assertEqual(summary.status_code, 200, summary.text)
        summary_payload = summary.json()
        self.assertEqual(summary_payload.get("event_count"), 2)
        self.assertEqual(summary_payload.get("auto_recommend_events"), 1)
        self.assertTrue(bool(summary_payload.get("top_selected_models")))
        self.assertTrue(bool(summary_payload.get("path")))

    def test_model_recommendation_applies_adaptive_bias_from_telemetry(self):
        project_id = self._create_project("phase21-model-wizard-adaptive-1")

        baseline = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "laptop",
                "primary_language": "english",
                "available_vram_gb": 8,
                "task_profile": "chat_sft",
                "top_k": 3,
            },
        )
        self.assertEqual(baseline.status_code, 200, baseline.text)
        baseline_rows = [item for item in baseline.json().get("recommendations", []) if isinstance(item, dict)]
        baseline_score = float(
            next(
                (
                    item.get("match_score")
                    for item in baseline_rows
                    if item.get("model_id") == "microsoft/Phi-3-mini-4k-instruct"
                ),
                0.0,
            )
        )

        for _ in range(4):
            apply_event = self.client.post(
                f"/api/projects/{project_id}/training/model-selection/telemetry",
                json={
                    "action": "apply",
                    "source": "training_setup_wizard",
                    "target_device": "laptop",
                    "task_profile": "chat_sft",
                    "selected_model_id": "microsoft/Phi-3-mini-4k-instruct",
                    "selected_rank": 1,
                    "selected_score": 0.9,
                },
            )
            self.assertEqual(apply_event.status_code, 200, apply_event.text)

        adapted = self.client.post(
            f"/api/projects/{project_id}/training/model-selection/recommend",
            json={
                "target_device": "laptop",
                "primary_language": "english",
                "available_vram_gb": 8,
                "task_profile": "chat_sft",
                "top_k": 3,
            },
        )
        self.assertEqual(adapted.status_code, 200, adapted.text)
        adapted_payload = adapted.json()
        adaptive_meta = adapted_payload.get("adaptive_ranking", {})
        self.assertTrue(bool(adaptive_meta.get("enabled")))
        self.assertGreaterEqual(int(adaptive_meta.get("context_apply_events") or 0), 4)

        adapted_rows = [item for item in adapted_payload.get("recommendations", []) if isinstance(item, dict)]
        phi_row = next(
            (item for item in adapted_rows if item.get("model_id") == "microsoft/Phi-3-mini-4k-instruct"),
            None,
        )
        self.assertIsNotNone(phi_row)
        self.assertGreater(float(phi_row.get("adaptive_bias", 0.0)), 0.0)
        self.assertGreater(float(phi_row.get("match_score", 0.0)), baseline_score)
