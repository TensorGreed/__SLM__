"""Phase 46 tests: closed-loop remediation plans from evaluation failures."""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase46_eval_remediation_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase46_eval_remediation_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"
os.environ["JUDGE_MODEL_API_URL"] = ""
os.environ["JUDGE_MODEL_API_KEY"] = ""

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Phase46EvaluationRemediationPlanTests(unittest.TestCase):
    @classmethod
    def _cleanup_artifacts(cls):
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                path.unlink()

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        cls._prev_data_dir = settings.DATA_DIR
        cls._prev_database_url = settings.DATABASE_URL
        cls._prev_db_require_alembic = settings.DB_REQUIRE_ALEMBIC_HEAD
        cls._prev_judge_url = settings.JUDGE_MODEL_API_URL
        cls._prev_judge_key = settings.JUDGE_MODEL_API_KEY

        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.DATABASE_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
        settings.DB_REQUIRE_ALEMBIC_HEAD = False
        settings.JUDGE_MODEL_API_URL = ""
        settings.JUDGE_MODEL_API_KEY = ""
        settings.ensure_dirs()

        cls._cleanup_artifacts()
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
        settings.JUDGE_MODEL_API_URL = cls._prev_judge_url
        settings.JUDGE_MODEL_API_KEY = cls._prev_judge_key

        cls._cleanup_artifacts()

    def _create_project(self, name: str) -> int:
        unique_name = f"{name}-{uuid.uuid4().hex[:8]}"
        response = self.client.post(
            "/api/projects",
            json={"name": unique_name, "description": "phase46 remediation"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return int(response.json()["id"])

    def _create_experiment(self, project_id: int, name: str = "phase46-exp") -> int:
        response = self.client.post(
            f"/api/projects/{project_id}/training/experiments",
            json={
                "name": name,
                "config": {
                    "base_model": "microsoft/phi-2",
                },
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return int(response.json()["id"])

    def _run_llm_judge_eval(
        self,
        *,
        project_id: int,
        experiment_id: int,
        predictions: list[dict[str, str]],
        dataset_name: str = "gold_test",
    ) -> int:
        response = self.client.post(
            f"/api/projects/{project_id}/evaluation/llm-judge",
            json={
                "experiment_id": experiment_id,
                "dataset_name": dataset_name,
                "judge_model": "meta-llama/Meta-Llama-3-70B-Instruct",
                "predictions": predictions,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return int(response.json()["id"])

    def test_root_cause_classification_is_stable_with_fixed_fixtures(self):
        project_id = self._create_project("phase46-remediation-stability")
        experiment_id = self._create_experiment(project_id)

        eval_id = self._run_llm_judge_eval(
            project_id=project_id,
            experiment_id=experiment_id,
            predictions=[
                {
                    "prompt": "Return only the order code.",
                    "reference": "ORDER-42",
                    "prediction": "```json\\n{\"code\": \"ORDER-42\"}\\n```",
                },
                {
                    "prompt": "What is the capital of France?",
                    "reference": "Paris",
                    "prediction": "London is the capital of France and Berlin is another likely answer.",
                },
                {
                    "prompt": "Status update",
                    "reference": "Escalated to L2",
                    "prediction": "N/A",
                },
            ],
        )

        first = self.client.post(
            f"/api/projects/{project_id}/evaluation/remediation-plans/generate",
            json={
                "experiment_id": experiment_id,
                "evaluation_result_id": eval_id,
            },
        )
        second = self.client.post(
            f"/api/projects/{project_id}/evaluation/remediation-plans/generate",
            json={
                "experiment_id": experiment_id,
                "evaluation_result_id": eval_id,
            },
        )
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 201, second.text)

        first_payload = first.json()
        second_payload = second.json()

        first_signature = [
            (
                str(item.get("root_cause") or ""),
                str(item.get("slice") or ""),
                int(item.get("failure_count") or 0),
            )
            for item in list(first_payload.get("clusters") or [])
            if isinstance(item, dict)
        ]
        second_signature = [
            (
                str(item.get("root_cause") or ""),
                str(item.get("slice") or ""),
                int(item.get("failure_count") or 0),
            )
            for item in list(second_payload.get("clusters") or [])
            if isinstance(item, dict)
        ]

        self.assertEqual(first_signature, second_signature)
        root_causes = {row[0] for row in first_signature}
        self.assertIn("formatting_mismatch", root_causes)
        self.assertIn("hallucination", root_causes)
        self.assertIn("coverage_gap", root_causes)

    def test_plan_generation_shape_and_persistence_retrieval(self):
        project_id = self._create_project("phase46-remediation-persistence")
        experiment_id = self._create_experiment(project_id)

        eval_id = self._run_llm_judge_eval(
            project_id=project_id,
            experiment_id=experiment_id,
            predictions=[
                {
                    "prompt": "Summarize this incident.",
                    "reference": "Service outage due to cache saturation.",
                    "prediction": "N/A",
                },
                {
                    "prompt": "Return only ticket id",
                    "reference": "TCK-9",
                    "prediction": "Answer: TCK-9",
                },
            ],
        )

        generate = self.client.post(
            f"/api/projects/{project_id}/evaluation/remediation-plans/generate",
            json={
                "experiment_id": experiment_id,
                "evaluation_result_id": eval_id,
            },
        )
        self.assertEqual(generate.status_code, 201, generate.text)
        payload = generate.json()

        self.assertTrue(str(payload.get("plan_id") or ""))
        self.assertEqual(int(payload.get("experiment_id") or 0), experiment_id)
        self.assertEqual(int(payload.get("evaluation_result_id") or 0), eval_id)

        recommendations = [item for item in list(payload.get("recommendations") or []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(recommendations), 1)
        first_rec = recommendations[0]
        self.assertTrue(isinstance(first_rec.get("data_operations"), list))
        self.assertTrue(isinstance(first_rec.get("training_config_changes"), list))
        self.assertTrue(isinstance(first_rec.get("expected_impact"), dict))
        self.assertTrue(isinstance(first_rec.get("confidence"), dict))

        linked = dict(payload.get("linked_artifacts") or {})
        self.assertTrue(isinstance(linked.get("evaluation_result_artifact"), dict))
        self.assertTrue(isinstance(linked.get("remediation_plan_artifact"), dict))

        plan_id = str(payload.get("plan_id") or "")

        index_resp = self.client.get(
            f"/api/projects/{project_id}/evaluation/remediation-plans",
            params={"experiment_id": experiment_id},
        )
        self.assertEqual(index_resp.status_code, 200, index_resp.text)
        index_payload = index_resp.json()
        entries = [item for item in list(index_payload.get("plans") or []) if isinstance(item, dict)]
        self.assertTrue(any(str(item.get("plan_id") or "") == plan_id for item in entries), entries)

        detail = self.client.get(
            f"/api/projects/{project_id}/evaluation/remediation-plans/{plan_id}",
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        detail_payload = detail.json()
        self.assertEqual(str(detail_payload.get("plan_id") or ""), plan_id)
        self.assertEqual(int(detail_payload.get("evaluation_result_id") or 0), eval_id)

    def test_no_failures_returns_structured_actionable_blocker(self):
        project_id = self._create_project("phase46-remediation-no-failures")
        experiment_id = self._create_experiment(project_id)

        eval_id = self._run_llm_judge_eval(
            project_id=project_id,
            experiment_id=experiment_id,
            predictions=[
                {
                    "prompt": "Reset password policy?",
                    "reference": "Reset passwords every 90 days.",
                    "prediction": "Reset passwords every 90 days.",
                }
            ],
        )

        response = self.client.post(
            f"/api/projects/{project_id}/evaluation/remediation-plans/generate",
            json={
                "experiment_id": experiment_id,
                "evaluation_result_id": eval_id,
            },
        )
        self.assertEqual(response.status_code, 409, response.text)
        detail = dict(response.json().get("detail") or {})

        self.assertEqual(str(detail.get("error_code") or ""), "REMEDIATION_NOT_REQUIRED")
        self.assertEqual(str(detail.get("stage") or ""), "evaluation")
        self.assertTrue(str(detail.get("actionable_fix") or ""))
        metadata = dict(detail.get("metadata") or {})
        self.assertEqual(int(metadata.get("evaluation_result_id") or 0), eval_id)


if __name__ == "__main__":
    unittest.main()
