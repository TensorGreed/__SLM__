"""Phase 24 tests: alignment contract and judge scoring scaffolds."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase24_alignment_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase24_alignment_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase24AlignmentScaffoldTests(unittest.TestCase):
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
            json={"name": name, "description": "phase24"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _import_preference_rows(self, project_id: int, rows: list[dict]) -> None:
        resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/preference-dataset/import",
            json={
                "rows": rows,
                "mode": "replace",
                "target": "prepared_train",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_alignment_recipe_resolve_returns_training_mode(self):
        project_id = self._create_project("phase24-alignment-1")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/recipes/resolve",
            json={
                "recipe_id": "recipe.alignment.dpo.fast",
                "base_config": {"base_model": "microsoft/phi-2"},
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        resolved = resp.json().get("resolved_config", {})
        self.assertEqual(resolved.get("training_mode"), "dpo")
        self.assertEqual(resolved.get("task_type"), "causal_lm")

    def test_contract_validation_reports_missing_fields(self):
        project_id = self._create_project("phase24-alignment-2")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/preference-contract/validate",
            json={
                "rows": [
                    {"prompt": "q1", "chosen": "a1", "rejected": "bad"},
                    {"prompt": "q2", "chosen": "a2"},
                ],
                "min_coverage": 0.9,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertFalse(bool(payload.get("ok")))
        self.assertEqual(int(payload.get("total_rows", 0)), 2)
        self.assertEqual(int(payload.get("valid_rows", 0)), 1)
        self.assertEqual(int(payload.get("invalid_rows", 0)), 1)
        self.assertGreater(len(payload.get("errors", [])), 0)

    def test_alignment_judge_scores_rows(self):
        project_id = self._create_project("phase24-alignment-3")
        resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/judge/score",
            json={
                "quality_threshold": 3.0,
                "rows": [
                    {
                        "prompt": "Summarize TLS best practices.",
                        "chosen": "Use TLS 1.2+ with strong ciphers and rotate certs.",
                        "rejected": "I like turtles.",
                    },
                    {
                        "prompt": "How to rotate keys?",
                        "chosen": "Rotate keys every 90 days.",
                        "rejected": "Rotate keys every 90 days.",
                    },
                ],
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(int(payload.get("scored_count", 0)), 1)
        self.assertGreaterEqual(int(payload.get("keep_count", 0)), 0)
        self.assertIn("contract", payload)

    def test_alignment_preference_dataset_import_and_summary(self):
        project_id = self._create_project("phase24-alignment-4")
        rows = [
            {
                "prompt": "Summarize key rotation policy.",
                "chosen": "Rotate keys every 90 days and after incidents.",
                "rejected": "Never rotate keys.",
            },
            {
                "question": "How should API traffic be secured?",
                "preferred": "Require TLS 1.2+ and reject plaintext.",
                "dispreferred": "HTTP is acceptable for production APIs.",
            },
        ]
        import_resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/preference-dataset/import",
            json={
                "rows": rows,
                "mode": "replace",
                "target": "prepared_train",
            },
        )
        self.assertEqual(import_resp.status_code, 200, import_resp.text)
        import_payload = import_resp.json()
        self.assertEqual(int(import_payload.get("rows_added", 0)), 2)
        self.assertEqual(int(import_payload.get("rows_dropped", 0)), 0)

        summary_resp = self.client.get(
            f"/api/projects/{project_id}/training/alignment/preference-dataset",
            params={"sample_size": 50, "quality_threshold": 3.0},
        )
        self.assertEqual(summary_resp.status_code, 200, summary_resp.text)
        summary = summary_resp.json()
        contract = summary.get("contract", {})
        self.assertTrue(summary.get("exists"))
        self.assertEqual(int(contract.get("valid_rows", 0)), 2)
        self.assertEqual(int(summary.get("quality", {}).get("scored_count", 0)), 2)

    def test_alignment_filter_can_apply_to_train_file(self):
        project_id = self._create_project("phase24-alignment-5")
        self._import_preference_rows(
            project_id,
            rows=[
                {
                    "prompt": "How to secure API keys?",
                    "chosen": "Store keys in a secret manager and rotate regularly.",
                    "rejected": "Post keys in public repos.",
                },
                {
                    "prompt": "How to greet users?",
                    "chosen": "hello",
                    "rejected": "goodbye",
                },
            ],
        )

        filter_resp = self.client.post(
            f"/api/projects/{project_id}/training/alignment/preference-dataset/filter",
            json={
                "quality_threshold": 2.5,
                "min_keep_ratio": 0.1,
                "apply_to_train_file": True,
            },
        )
        self.assertEqual(filter_resp.status_code, 200, filter_resp.text)
        payload = filter_resp.json()
        self.assertTrue(bool(payload.get("apply_to_train_file")))
        self.assertGreater(int(payload.get("keep_count", 0)), 0)
        self.assertTrue(str(payload.get("target_path", "")).endswith("train.jsonl"))

    def test_dpo_preflight_includes_alignment_quality_summary(self):
        project_id = self._create_project("phase24-alignment-6")
        self._import_preference_rows(
            project_id,
            rows=[
                {
                    "prompt": "How often should keys rotate?",
                    "chosen": "Rotate keys every 90 days and after incidents.",
                    "rejected": "Do not rotate keys.",
                },
                {
                    "prompt": "How to secure API traffic?",
                    "chosen": "Require TLS for all endpoints.",
                    "rejected": "Use plaintext HTTP.",
                },
            ],
        )

        preflight_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/preflight",
            json={
                "config": {
                    "base_model": "microsoft/phi-2",
                    "training_mode": "dpo",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "alignment_quality_threshold": 3.0,
                }
            },
        )
        self.assertEqual(preflight_resp.status_code, 200, preflight_resp.text)
        payload = preflight_resp.json()
        preflight = payload.get("preflight", {})
        self.assertTrue(preflight.get("ok"), preflight)
        capability = preflight.get("capability_summary", {})
        dataset = capability.get("dataset", {})
        alignment_quality = dataset.get("alignment_quality", {})
        self.assertGreater(int(alignment_quality.get("scored_count", 0)), 0)

    def test_dpo_start_training_auto_filter_wires_filtered_dataset(self):
        project_id = self._create_project("phase24-alignment-7")
        self._import_preference_rows(
            project_id,
            rows=[
                {
                    "prompt": "How often should keys rotate?",
                    "chosen": "Rotate keys every 90 days and after incidents.",
                    "rejected": "Do not rotate keys.",
                },
                {
                    "prompt": "What protocol for APIs?",
                    "chosen": "Use HTTPS/TLS for all endpoints.",
                    "rejected": "Use plaintext HTTP.",
                },
            ],
        )
        exp_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": "phase24-dpo-auto-filter",
                "config": {
                    "base_model": "microsoft/phi-2",
                    "training_mode": "dpo",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "alignment_auto_filter": True,
                    "alignment_quality_threshold": 3.0,
                    "alignment_min_keep_ratio": 0.1,
                },
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        experiment_id = int(exp_resp.json()["id"])

        start_resp = self.client.post(
            f"/api/projects/{project_id}/training/experiments/{experiment_id}/start"
        )
        self.assertEqual(start_resp.status_code, 200, start_resp.text)
        config = start_resp.json().get("config", {})
        runtime = config.get("_runtime", {})
        alignment_filter = runtime.get("alignment_filter", {})
        self.assertTrue(alignment_filter)
        self.assertGreater(int(alignment_filter.get("keep_count", 0)), 0)


if __name__ == "__main__":
    unittest.main()
