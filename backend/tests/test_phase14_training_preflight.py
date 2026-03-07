"""Phase 14 tests: capability matrix + training preflight guardrails."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase14_preflight_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase14_preflight_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase14TrainingPreflightTests(unittest.TestCase):
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
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "phase14"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_prepared_split(self, project_id: int) -> None:
        prepared_dir = TEST_DATA_DIR / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        train_file = prepared_dir / "train.jsonl"
        val_file = prepared_dir / "val.jsonl"
        train_file.write_text('{"text":"hello world"}\n', encoding="utf-8")
        val_file.write_text('{"text":"hello eval"}\n', encoding="utf-8")

    def _create_experiment(self, project_id: int, config: dict) -> int:
        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={"name": "phase14-exp", "config": config},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_preflight_endpoint_blocks_seq2seq_on_causal_model(self):
        project_id = self._create_project("phase14-preflight-1")
        self._create_prepared_split(project_id)

        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/preflight",
            json={
                "config": {
                    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
                    "task_type": "seq2seq",
                    "trainer_backend": "hf_trainer",
                }
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        preflight = payload.get("preflight", {})
        self.assertFalse(preflight.get("ok"))
        errors = preflight.get("errors", [])
        self.assertTrue(any("task_type=seq2seq" in str(item) for item in errors))

    def test_start_training_rejects_failed_preflight(self):
        project_id = self._create_project("phase14-preflight-2")
        self._create_prepared_split(project_id)
        experiment_id = self._create_experiment(
            project_id,
            config={
                "base_model": "meta-llama/Llama-3.2-1B-Instruct",
                "task_type": "seq2seq",
                "trainer_backend": "hf_trainer",
            },
        )

        start = self.client.post(
            f"/api/projects/{project_id}/training/experiments/{experiment_id}/start"
        )
        self.assertEqual(start.status_code, 400, start.text)
        detail = start.json().get("detail", "")
        self.assertIn("Training preflight failed", detail)
        self.assertIn("task_type=seq2seq", detail)

    def test_preflight_endpoint_passes_for_compatible_config(self):
        project_id = self._create_project("phase14-preflight-3")
        self._create_prepared_split(project_id)

        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/preflight",
            json={
                "config": {
                    "base_model": "microsoft/phi-2",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "max_seq_length": 2048,
                }
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        preflight = payload.get("preflight", {})
        self.assertTrue(preflight.get("ok"), preflight)

    def test_preflight_plan_returns_profiles_and_fixes_precision_conflict(self):
        project_id = self._create_project("phase14-preflight-4")
        self._create_prepared_split(project_id)

        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/preflight/plan",
            json={
                "config": {
                    "base_model": "microsoft/phi-2",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "batch_size": 8,
                    "max_seq_length": 4096,
                    "fp16": True,
                    "bf16": True,
                }
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        plan = payload.get("plan", {})
        suggestions = plan.get("suggestions", [])
        self.assertGreaterEqual(len(suggestions), 3)
        profiles = {str(item.get("profile")) for item in suggestions if isinstance(item, dict)}
        self.assertTrue({"safe", "balanced", "max_quality"}.issubset(profiles))

        safe = next(item for item in suggestions if item.get("profile") == "safe")
        safe_cfg = safe.get("config", {})
        self.assertEqual(int(safe_cfg.get("batch_size", 0)), 1)
        self.assertLessEqual(int(safe_cfg.get("max_seq_length", 99999)), 1024)
        self.assertFalse(bool(safe_cfg.get("fp16", False)) and bool(safe_cfg.get("bf16", False)))

    def test_training_preferences_default_is_balanced(self):
        project_id = self._create_project("phase14-training-prefs-1")

        resp = self.client.get(f"/api/projects/{project_id}/training/preferences")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload.get("project_id"), project_id)
        self.assertEqual(payload.get("preferred_plan_profile"), "balanced")
        options = payload.get("profile_options", [])
        self.assertTrue({"safe", "balanced", "max_quality"}.issubset(set(options)))

    def test_training_preferences_can_be_updated(self):
        project_id = self._create_project("phase14-training-prefs-2")

        update_resp = self.client.put(
            f"/api/projects/{project_id}/training/preferences",
            json={"preferred_plan_profile": "safe"},
        )
        self.assertEqual(update_resp.status_code, 200, update_resp.text)
        updated = update_resp.json()
        self.assertEqual(updated.get("preferred_plan_profile"), "safe")
        self.assertEqual(updated.get("source"), "project")

        read_resp = self.client.get(f"/api/projects/{project_id}/training/preferences")
        self.assertEqual(read_resp.status_code, 200, read_resp.text)
        self.assertEqual(read_resp.json().get("preferred_plan_profile"), "safe")

    def test_training_preferences_reject_invalid_profile(self):
        project_id = self._create_project("phase14-training-prefs-3")

        resp = self.client.put(
            f"/api/projects/{project_id}/training/preferences",
            json={"preferred_plan_profile": "ultra"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = str(resp.json().get("detail", ""))
        self.assertIn("Invalid preferred_plan_profile", detail)

    def test_runtime_catalog_lists_builtin_runtimes(self):
        project_id = self._create_project("phase14-runtime-catalog-1")

        resp = self.client.get(f"/api/projects/{project_id}/training/runtimes")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        runtimes = payload.get("runtimes", [])
        runtime_ids = {str(item.get("runtime_id")) for item in runtimes if isinstance(item, dict)}
        self.assertIn("builtin.simulate", runtime_ids)
        self.assertIn("builtin.external_celery", runtime_ids)
        self.assertTrue(bool(payload.get("default_runtime_id")))

    def test_preflight_blocks_unknown_runtime_id(self):
        project_id = self._create_project("phase14-runtime-catalog-2")
        self._create_prepared_split(project_id)

        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/preflight",
            json={
                "config": {
                    "base_model": "microsoft/phi-2",
                    "training_runtime_id": "custom.missing-runtime",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                }
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        preflight = payload.get("preflight", {})
        self.assertFalse(preflight.get("ok"))
        errors = [str(item) for item in preflight.get("errors", [])]
        self.assertTrue(any("Unknown training_runtime_id" in item for item in errors), errors)

    def test_adapter_preference_defaults_then_updates(self):
        project_id = self._create_project("phase14-adapter-pref-1")

        initial = self.client.get(f"/api/projects/{project_id}/dataset/adapter-preference")
        self.assertEqual(initial.status_code, 200, initial.text)
        self.assertEqual(initial.json().get("source"), "default")
        self.assertEqual(initial.json().get("adapter_id"), "default-canonical")

        updated = self.client.put(
            f"/api/projects/{project_id}/dataset/adapter-preference",
            json={
                "adapter_id": "seq2seq-pair",
                "field_mapping": {"question": "prompt", "answer": "completion"},
            },
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        payload = updated.json()
        self.assertEqual(payload.get("source"), "project")
        self.assertEqual(payload.get("adapter_id"), "seq2seq-pair")
        self.assertEqual(payload.get("field_mapping", {}).get("question"), "prompt")

    def test_dataset_split_uses_saved_adapter_preference_when_request_omits_adapter(self):
        project_id = self._create_project("phase14-adapter-pref-2")

        save_pref = self.client.put(
            f"/api/projects/{project_id}/dataset/adapter-preference",
            json={"adapter_id": "qa-pair"},
        )
        self.assertEqual(save_pref.status_code, 200, save_pref.text)

        mocked_split = AsyncMock(
            return_value={"project_id": project_id, "total_entries": 10, "splits": {"train": 8, "val": 1, "test": 1}}
        )
        with patch("app.api.dataset.split_dataset", mocked_split):
            split_resp = self.client.post(
                f"/api/projects/{project_id}/dataset/split",
                json={},
            )
        self.assertEqual(split_resp.status_code, 200, split_resp.text)
        body = split_resp.json()
        self.assertEqual(body.get("adapter_preference_source"), "project")

        kwargs = mocked_split.await_args.kwargs
        self.assertEqual(kwargs.get("adapter_id"), "qa-pair")

    def test_preflight_blocks_classification_when_dataset_contract_is_incompatible(self):
        project_id = self._create_project("phase14-preflight-contract-1")
        self._create_prepared_split(project_id)  # text-only rows

        resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/preflight",
            json={
                "config": {
                    "base_model": "microsoft/phi-2",
                    "task_type": "classification",
                    "trainer_backend": "hf_trainer",
                }
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        preflight = payload.get("preflight", {})
        self.assertFalse(preflight.get("ok"))
        errors = [str(item) for item in preflight.get("errors", [])]
        self.assertTrue(any("Dataset contract mismatch" in item for item in errors), errors)
        hints = [str(item) for item in preflight.get("hints", [])]
        self.assertTrue(any("Dataset Prep split" in item or "save adapter preset" in item for item in hints), hints)


if __name__ == "__main__":
    unittest.main()
