"""Phase 7 reliability tests: structured error envelope and guardrail reason codes."""

from __future__ import annotations

import os
import time
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase7_reliability_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase7_reliability_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"
os.environ["ALLOW_SYNTHETIC_DEMO_FALLBACK"] = "true"

from fastapi.testclient import TestClient

from app.main import app


class Phase7ReliabilityEnvelopeTests(unittest.TestCase):
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
            json={"name": unique_name, "description": "phase7 reliability"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _assert_structured_error(self, payload: dict, stage: str):
        self.assertEqual(str(payload.get("stage") or ""), stage, payload)
        self.assertTrue(str(payload.get("error_code") or "").strip(), payload)
        self.assertTrue(str(payload.get("actionable_fix") or "").strip(), payload)
        self.assertTrue(str(payload.get("docs_url") or "").strip(), payload)
        # Backward compatibility with legacy clients that inspect `detail`.
        self.assertIn("detail", payload)

    def test_structured_error_envelope_for_ingestion_training_export(self):
        project_id = self._create_project("phase7-envelope")

        ingest_resp = self.client.post(
            f"/api/projects/{project_id}/ingestion/import-remote",
            json={
                "source_type": "invalid-source",
                "identifier": "owner/dataset",
            },
        )
        self.assertEqual(ingest_resp.status_code, 422, ingest_resp.text)
        self._assert_structured_error(ingest_resp.json(), "ingestion")

        training_resp = self.client.post(
            f"/api/projects/{project_id}/training/cloud-burst/quote",
            json={
                "provider_id": "unknown-provider",
                "gpu_sku": "h100.80gb",
            },
        )
        self.assertEqual(training_resp.status_code, 400, training_resp.text)
        self._assert_structured_error(training_resp.json(), "training")

        export_resp = self.client.post(
            f"/api/projects/{project_id}/export/999999/run",
            json={},
        )
        self.assertEqual(export_resp.status_code, 404, export_resp.text)
        self._assert_structured_error(export_resp.json(), "export")

    def test_autopilot_plan_v2_exposes_reason_codes_and_unblock_actions(self):
        project_id = self._create_project("phase7-guardrails")

        plan_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/plan-v2",
            json={
                "intent": "I want a model that summarizes support tickets.",
                "target_device": "laptop",
                "target_profile_id": "mobile_cpu",
                "available_vram_gb": 4,
            },
        )
        self.assertEqual(plan_resp.status_code, 200, plan_resp.text)
        payload = plan_resp.json()
        guardrails = dict(payload.get("guardrails") or {})
        self.assertEqual(str(payload.get("resolved_target_device") or ""), "mobile", payload)

        reason_codes = [str(item) for item in list(guardrails.get("reason_codes") or [])]
        self.assertGreaterEqual(len(reason_codes), 1, guardrails)

        unblock_actions = [item for item in list(guardrails.get("unblock_actions") or []) if isinstance(item, dict)]
        self.assertGreaterEqual(len(unblock_actions), 1, guardrails)
        for action in unblock_actions:
            self.assertTrue(str(action.get("reason_code") or "").strip(), action)
            self.assertTrue(str(action.get("label") or "").strip(), action)
            self.assertIn("one_click_available", action)

    def test_autopilot_plan_v2_surfaces_target_compatibility_warnings(self):
        project_id = self._create_project("phase7-target-warnings")

        plan_resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/plan-v2",
            json={
                "intent": "Build a domain support assistant.",
                "target_device": "laptop",
                "target_profile_id": "edge_gpu",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "available_vram_gb": 8,
            },
        )
        self.assertEqual(plan_resp.status_code, 200, plan_resp.text)
        payload = plan_resp.json()
        self.assertEqual(str(payload.get("resolved_target_device") or ""), "laptop", payload)
        compatibility = dict(payload.get("target_compatibility") or {})
        warnings = [str(item) for item in list(compatibility.get("warnings") or [])]
        self.assertGreaterEqual(len(warnings), 1, compatibility)
        self.assertTrue(any("VRAM" in item for item in warnings), warnings)

        guardrails = dict(payload.get("guardrails") or {})
        guardrail_warnings = [str(item) for item in list(guardrails.get("warnings") or [])]
        self.assertTrue(any("VRAM" in item for item in guardrail_warnings), guardrail_warnings)


if __name__ == "__main__":
    unittest.main()
