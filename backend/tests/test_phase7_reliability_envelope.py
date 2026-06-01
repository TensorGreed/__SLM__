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
        # Diagnostics Intervention A — every envelope must carry a
        # troubleshooting_id so the user can copy-paste it into a bug
        # report and the developer can grep logs for it.
        trace_id = str(payload.get("troubleshooting_id") or "")
        self.assertTrue(trace_id.startswith("err_"), payload)
        self.assertGreater(len(trace_id), 8, payload)
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

    # ── Diagnostics Intervention A — widened envelope coverage ────

    def test_envelope_now_wraps_synthetic_errors(self):
        """Pre-A: /synthetic/* errors returned raw {detail: '...'} so
        the frontend rendered a generic toast with no troubleshooting
        id, no error_code, no remediation copy. Post-A: every
        synthetic 4xx/5xx flows through the same envelope shape that
        ingestion/training/export already used."""
        project_id = self._create_project("phase-a-synth")
        resp = self.client.post(
            f"/api/projects/{project_id}/synthetic/run-playbook",
            json={"mode": "non-existent-mode", "target_count": 1},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self._assert_structured_error(resp.json(), "synthetic")

    def test_envelope_now_wraps_gold_errors(self):
        project_id = self._create_project("phase-a-gold")
        # 422: ``pairs`` must be a list of dicts. Send a string to
        # force a clean validation failure.
        resp = self.client.post(
            f"/api/projects/{project_id}/gold/import",
            json={"pairs": "not-a-list"},
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self._assert_structured_error(resp.json(), "gold")

    def test_envelope_now_wraps_data_health_autofix_errors(self):
        project_id = self._create_project("phase-a-data-health")
        resp = self.client.post(
            f"/api/projects/{project_id}/data-health/autofix",
            json={"fix_kind": "unknown-fix"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self._assert_structured_error(resp.json(), "data-health")

    def test_envelope_now_wraps_cleaning_errors(self):
        project_id = self._create_project("phase-a-cleaning")
        # Try to fetch a non-existent cleaning task.
        resp = self.client.get(
            f"/api/projects/{project_id}/cleaning/tasks/does-not-exist",
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self._assert_structured_error(resp.json(), "cleaning")

    def test_envelope_now_wraps_dataset_import_errors(self):
        project_id = self._create_project("phase-a-dataset-import")
        # Empty body fails validation.
        resp = self.client.post(
            f"/api/projects/{project_id}/dataset-import/run",
            json={},
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self._assert_structured_error(resp.json(), "dataset-import")

    def test_envelope_general_stage_for_unrecognised_paths(self):
        """``/api/...`` paths whose first segment isn't in the stage
        table get ``stage='general'`` rather than being skipped. The
        previous behavior dropped them through to legacy
        ``{detail: '...'}`` which is what we're trying to phase out."""
        # 404 on an unknown project hits /api/projects/{id} — generic
        # project route, no specific stage.
        resp = self.client.get("/api/projects/999999999")
        self.assertEqual(resp.status_code, 404, resp.text)
        payload = resp.json()
        self.assertEqual(payload.get("stage"), "general", payload)
        self.assertTrue(payload.get("troubleshooting_id"))

    def test_validation_errors_carry_envelope_shape(self):
        """422 validation errors also flow through the envelope so the
        frontend can render them the same way as 4xx/5xx. Backend
        preserves ``detail = validation_errors[]`` for legacy clients
        that consume that shape."""
        project_id = self._create_project("phase-a-validation")
        resp = self.client.post(
            f"/api/projects/{project_id}/synthetic/run-playbook",
            json={"target_count": "not-an-int"},  # bad type
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        payload = resp.json()
        self.assertEqual(payload.get("stage"), "synthetic", payload)
        self.assertTrue(payload.get("troubleshooting_id"))
        # Backward compat: detail is the original validation-errors list.
        self.assertIsInstance(payload.get("detail"), list)
        # New: metadata carries the same list under a named key.
        meta = payload.get("metadata") or {}
        self.assertIn("validation_errors", meta)

    def test_last_resort_wraps_unhandled_exceptions(self):
        """The Kaggle SDK's ``sys.exit(1)`` at import time used to
        crash request handlers with a bare 500 + opaque traceback.
        With the last-resort ``Exception`` handler registered, even
        SystemExit-style escapes flow through the envelope shape so
        the user gets a troubleshooting_id + the developer gets a
        log line keyed on the same id.

        TestClient's default ``raise_server_exceptions=True``
        propagates handler exceptions through to the test caller,
        which masks the production behavior we're trying to verify
        here. Build a dedicated client with the flag off so the
        envelope renders as it would for a real request."""
        from fastapi.testclient import TestClient as _TC
        from app.main import app as live_app

        async def _bomb():
            raise RuntimeError("boom from a buggy handler")

        live_app.add_api_route(
            "/api/_test/last_resort_bomb",
            _bomb,
            methods=["GET"],
        )
        try:
            with _TC(live_app, raise_server_exceptions=False) as bomb_client:
                resp = bomb_client.get("/api/_test/last_resort_bomb")
        finally:
            # Strip the route so other tests don't see it.
            live_app.router.routes = [
                r for r in live_app.router.routes
                if getattr(r, "path", "") != "/api/_test/last_resort_bomb"
            ]
        self.assertEqual(resp.status_code, 500, resp.text)
        payload = resp.json()
        self.assertEqual(payload.get("stage"), "general", payload)
        self.assertIn("RuntimeError", str(payload.get("message", "")), payload)
        self.assertTrue(str(payload.get("troubleshooting_id", "")).startswith("err_"))
        # Metadata carries the exception type + request path so logs
        # + frontend dispatchers can branch on the failure mode.
        meta = payload.get("metadata") or {}
        self.assertEqual(meta.get("exception_type"), "RuntimeError")
        self.assertIn("/api/_test/last_resort_bomb", str(meta.get("request_path", "")))


if __name__ == "__main__":
    unittest.main()
