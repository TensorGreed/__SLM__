"""Phase 44 tests: autopilot v2 orchestration with auto-repair and strict-mode guardrails."""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase44_autopilot_v2_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase44_autopilot_v2_data"

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


class Phase44AutopilotV2OrchestrationTests(unittest.TestCase):
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

        for suffix in ("", "-shm", "-wal"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.TRAINING_BACKEND = "simulate"
        settings.ALLOW_SIMULATED_TRAINING = True
        settings.STRICT_EXECUTION_MODE = False
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
            json={"name": unique_name, "description": "phase44 autopilot v2"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _write_prepared_rows(self, project_id: int, row_count: int = 24) -> Path:
        prepared_dir = TEST_DATA_DIR / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        train_path = prepared_dir / "train.jsonl"
        with open(train_path, "w", encoding="utf-8") as handle:
            for idx in range(row_count):
                handle.write(
                    '{"text": "User: support issue %d\\nAssistant: concise support response %d"}\n'
                    % (idx, idx)
                )
        return train_path

    def _upload_raw_text_pending(self, project_id: int, *, filename: str, content: str) -> int:
        upload_resp = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={"file": (filename, content.encode("utf-8"), "text/plain")},
            data={"source": "upload", "sensitivity": "internal", "license_info": ""},
        )
        self.assertEqual(upload_resp.status_code, 201, upload_resp.text)
        return int(upload_resp.json()["id"])

    def test_orchestrate_dry_run_reports_blockers_and_actionable_fixes(self):
        project_id = self._create_project("phase44-dry-run")

        resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/v2/orchestrate",
            json={
                "intent": "Build a support assistant for incoming tickets.",
                "target_profile_id": "mobile_cpu",
                "dry_run": True,
                "auto_prepare_data": False,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = dict(resp.json() or {})
        self.assertTrue(bool(payload.get("dry_run")), payload)

        guardrails = dict(payload.get("guardrails") or {})
        self.assertFalse(bool(guardrails.get("can_run")), guardrails)
        blockers = [str(item).strip() for item in list(guardrails.get("blockers") or []) if str(item).strip()]
        self.assertGreaterEqual(len(blockers), 1, guardrails)

        decision_log = [row for row in list(payload.get("decision_log") or []) if isinstance(row, dict)]
        self.assertGreaterEqual(len(decision_log), 1, payload)
        final_rows = [row for row in decision_log if str(row.get("step") or "") == "final_guardrails"]
        self.assertGreaterEqual(len(final_rows), 1, decision_log)
        self.assertEqual(str(final_rows[-1].get("status") or ""), "blocked", final_rows[-1])
        fixes = [row for row in list(final_rows[-1].get("fixes") or []) if isinstance(row, dict)]
        self.assertGreaterEqual(len(fixes), 1, final_rows[-1])

    @patch("app.api.training.start_training", new_callable=AsyncMock)
    def test_orchestrate_run_auto_prepares_data_and_starts_training(self, mock_start_training: AsyncMock):
        project_id = self._create_project("phase44-auto-prepare")
        raw_lines = "\n".join(
            [f"Ticket {idx}: customer cannot sign in. Include reset steps." for idx in range(36)]
        )
        self._upload_raw_text_pending(
            project_id,
            filename="support_pending.txt",
            content=raw_lines,
        )
        mock_start_training.return_value = {"status": "queued", "task_id": "phase44-autopilot-task"}

        resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/v2/orchestrate/run",
            json={
                "intent": "Summarize support tickets and suggest next actions.",
                "target_profile_id": "edge_gpu",
                "dry_run": False,
                "auto_prepare_data": True,
                "plan_profile": "balanced",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = dict(resp.json() or {})

        self.assertFalse(bool(payload.get("dry_run")), payload)
        self.assertTrue(bool(payload.get("started")), payload)
        experiment = dict(payload.get("experiment") or {})
        self.assertGreater(int(experiment.get("id") or 0), 0, payload)

        repairs = dict(payload.get("repairs") or {})
        data_repair = dict(repairs.get("dataset_auto_prepare") or {})
        self.assertTrue(bool(data_repair.get("succeeded")), repairs)

        decision_log = [row for row in list(payload.get("decision_log") or []) if isinstance(row, dict)]
        self.assertTrue(
            any(
                str(row.get("step") or "") == "dataset_auto_prepare"
                and str(row.get("status") or "") == "applied"
                for row in decision_log
            ),
            decision_log,
        )

    def test_orchestrate_dry_run_strict_mode_blocks_simulation_fallback(self):
        project_id = self._create_project("phase44-strict")
        self._write_prepared_rows(project_id, row_count=28)

        with patch.object(settings, "STRICT_EXECUTION_MODE", True), patch.object(
            settings,
            "TRAINING_BACKEND",
            "simulate",
        ), patch.object(settings, "ALLOW_SIMULATED_TRAINING", True):
            resp = self.client.post(
                f"/api/projects/{project_id}/training/autopilot/v2/orchestrate",
                json={
                    "intent": "Build a support assistant.",
                    "target_profile_id": "vllm_server",
                    "dry_run": True,
                    "auto_prepare_data": False,
                },
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        payload = dict(resp.json() or {})
        guardrails = dict(payload.get("guardrails") or {})
        self.assertFalse(bool(guardrails.get("can_run")), guardrails)
        reason_codes = [str(item) for item in list(guardrails.get("reason_codes") or [])]
        self.assertIn("STRICT_EXECUTION_MODE", reason_codes, guardrails)

        decision_log = [row for row in list(payload.get("decision_log") or []) if isinstance(row, dict)]
        strict_rows = [row for row in decision_log if str(row.get("step") or "") == "strict_mode_guardrail"]
        if strict_rows:
            self.assertEqual(str(strict_rows[-1].get("status") or ""), "blocked", strict_rows[-1])
        else:
            planning_rows = [row for row in decision_log if str(row.get("step") or "") == "initial_planning"]
            self.assertGreaterEqual(len(planning_rows), 1, decision_log)
            self.assertEqual(str(planning_rows[-1].get("status") or ""), "blocked", planning_rows[-1])


if __name__ == "__main__":
    unittest.main()
