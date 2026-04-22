"""Phase 53 tests — autopilot repair preview / apply separation (priority.md P3).

Exercises the two-step flow:

- `POST /autopilot/repair-preview` persists a preview and returns a plan_token.
- `POST /autopilot/repair-apply` re-validates state and executes non-dry.
- `GET  /autopilot/repair-previews/{plan_token}` retrieves a preview.

Error paths: unknown token, double-apply, expired preview, state drift,
expected_state_hash mismatch, force-override, and that `one-click-run` is
still present as the unchanged convenience wrapper.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase53_autopilot_repair_preview_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase53_autopilot_repair_preview_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"
os.environ["ALLOW_SYNTHETIC_DEMO_FALLBACK"] = "true"

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.autopilot_repair_preview import AutopilotRepairPreview
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project


class Phase53AutopilotRepairPreviewApplyTests(unittest.TestCase):
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

    # -- helpers ----------------------------------------------------------

    def _create_project(self, name: str) -> int:
        unique_name = f"{name}-{uuid.uuid4().hex[:8]}"
        resp = self.client.post(
            "/api/projects",
            json={"name": unique_name, "description": "phase53 repair preview/apply"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_prepared_rows(self, project_id: int, row_count: int = 32) -> None:
        prepared_dir = TEST_DATA_DIR / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        with open(prepared_dir / "train.jsonl", "w", encoding="utf-8") as handle:
            for idx in range(row_count):
                handle.write(
                    '{"text": "User: ticket %d\\nAssistant: short reply %d"}\n' % (idx, idx)
                )

    def _preview(self, project_id: int, **overrides) -> dict:
        payload = {
            "project_id": project_id,
            "intent": "Build a concise support assistant.",
            "target_profile_id": "edge_gpu",
            "auto_prepare_data": False,
            "plan_profile": "balanced",
        }
        payload.update(overrides)
        resp = self.client.post("/api/autopilot/repair-preview", json=payload)
        self.assertEqual(resp.status_code, 200, resp.text)
        return dict(resp.json() or {})

    # -- tests ------------------------------------------------------------

    def test_preview_returns_token_and_diff_without_creating_experiment(self):
        project_id = self._create_project("phase53-preview")
        self._seed_prepared_rows(project_id)

        before_count = self._experiment_count(project_id)
        payload = self._preview(project_id)
        after_count = self._experiment_count(project_id)

        # Dry-run must not mutate.
        self.assertEqual(after_count, before_count, "preview must not create experiments")

        preview = dict(payload.get("preview") or {})
        self.assertTrue(preview.get("plan_token"))
        self.assertEqual(preview["project_id"], project_id)
        self.assertIsNone(preview.get("applied_at"))

        diff = dict(payload.get("config_diff") or {})
        self.assertIn("summary", diff)
        self.assertIn("guardrails", diff)
        self.assertIn("repairs_planned", diff)
        self.assertIn("decision_log_preview", diff)

        dry_run = dict(payload.get("dry_run_response") or {})
        self.assertTrue(bool(dry_run.get("dry_run")))
        self.assertFalse(bool(dry_run.get("started")))

    def test_get_preview_by_token_roundtrips(self):
        project_id = self._create_project("phase53-get")
        self._seed_prepared_rows(project_id)
        payload = self._preview(project_id)
        token = str(dict(payload.get("preview") or {}).get("plan_token") or "")

        resp = self.client.get(f"/api/autopilot/repair-previews/{token}")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = dict(resp.json() or {})
        self.assertEqual(body.get("plan_token"), token)
        self.assertEqual(body.get("project_id"), project_id)

        # Unknown token returns 404.
        resp_404 = self.client.get("/api/autopilot/repair-previews/does-not-exist")
        self.assertEqual(resp_404.status_code, 404, resp_404.text)

    def test_apply_executes_plan_and_marks_preview_applied(self):
        project_id = self._create_project("phase53-apply")
        self._seed_prepared_rows(project_id)
        payload = self._preview(project_id)
        token = str(dict(payload.get("preview") or {}).get("plan_token") or "")

        before_count = self._experiment_count(project_id)
        with patch(
            "app.api.training.start_training",
            new_callable=AsyncMock,
        ) as mock_start_training:
            mock_start_training.return_value = {"status": "queued", "task_id": "phase53-task"}
            resp = self.client.post(
                "/api/autopilot/repair-apply",
                json={"plan_token": token, "actor": "phase53-test", "reason": "confirmed"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = dict(resp.json() or {})
        self.assertTrue(bool(body.get("ok")))

        response = dict(body.get("response") or {})
        self.assertFalse(bool(response.get("dry_run")))
        self.assertTrue(bool(response.get("started")))
        run_id = str(response.get("run_id") or "")
        self.assertTrue(run_id)

        # Experiment was created.
        after_count = self._experiment_count(project_id)
        self.assertEqual(after_count, before_count + 1)

        # Preview row is marked applied.
        applied = dict(body.get("preview") or {})
        self.assertIsNotNone(applied.get("applied_at"))
        self.assertEqual(applied.get("applied_by"), "phase53-test")
        self.assertEqual(applied.get("applied_run_id"), run_id)
        self.assertEqual(applied.get("applied_reason"), "confirmed")

    def test_double_apply_returns_409(self):
        project_id = self._create_project("phase53-double")
        self._seed_prepared_rows(project_id)
        payload = self._preview(project_id)
        token = str(dict(payload.get("preview") or {}).get("plan_token") or "")

        with patch(
            "app.api.training.start_training",
            new_callable=AsyncMock,
        ) as mock_start_training:
            mock_start_training.return_value = {"status": "queued", "task_id": "phase53-task-a"}
            first = self.client.post(
                "/api/autopilot/repair-apply", json={"plan_token": token}
            )
            self.assertEqual(first.status_code, 200, first.text)

            second = self.client.post(
                "/api/autopilot/repair-apply", json={"plan_token": token}
            )
        self.assertEqual(second.status_code, 409, second.text)
        detail = dict(second.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "already_applied")

    def test_unknown_plan_token_returns_404(self):
        resp = self.client.post(
            "/api/autopilot/repair-apply",
            json={"plan_token": "never-issued-token-xxxxxxxx"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        detail = dict(resp.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "preview_not_found")

    def test_expired_preview_returns_410(self):
        project_id = self._create_project("phase53-expired")
        self._seed_prepared_rows(project_id)
        payload = self._preview(project_id)
        token = str(dict(payload.get("preview") or {}).get("plan_token") or "")

        self._force_expire_preview(token)

        resp = self.client.post(
            "/api/autopilot/repair-apply", json={"plan_token": token}
        )
        self.assertEqual(resp.status_code, 410, resp.text)
        detail = dict(resp.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "preview_expired")

    def test_state_drift_blocks_apply_but_force_overrides(self):
        project_id = self._create_project("phase53-drift")
        self._seed_prepared_rows(project_id)
        payload = self._preview(project_id)
        token = str(dict(payload.get("preview") or {}).get("plan_token") or "")

        # Mutate the project so state_hash drifts. Directly bumping a hashed
        # field via the DB is simpler and more deterministic than depending
        # on project-update endpoints.
        self._mutate_project_description(project_id, "changed after preview")

        # Apply without force should refuse.
        resp = self.client.post(
            "/api/autopilot/repair-apply", json={"plan_token": token}
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        detail = dict(resp.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "state_drift")
        self.assertIn("preview_state_hash", detail)
        self.assertIn("current_state_hash", detail)

        # With force=true it should proceed.
        with patch(
            "app.api.training.start_training",
            new_callable=AsyncMock,
        ) as mock_start_training:
            mock_start_training.return_value = {"status": "queued", "task_id": "phase53-forced"}
            forced = self.client.post(
                "/api/autopilot/repair-apply",
                json={"plan_token": token, "force": True},
            )
        self.assertEqual(forced.status_code, 200, forced.text)
        self.assertTrue(bool(dict(forced.json()).get("ok")))

    def test_expected_state_hash_mismatch_returns_409(self):
        project_id = self._create_project("phase53-expected")
        self._seed_prepared_rows(project_id)
        payload = self._preview(project_id)
        token = str(dict(payload.get("preview") or {}).get("plan_token") or "")

        resp = self.client.post(
            "/api/autopilot/repair-apply",
            json={"plan_token": token, "expected_state_hash": "deadbeef" * 8},
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        detail = dict(resp.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "state_hash_mismatch")

    def test_one_click_run_still_works_as_convenience_wrapper(self):
        project_id = self._create_project("phase53-oneclick")
        self._seed_prepared_rows(project_id)

        with patch(
            "app.api.training.start_training",
            new_callable=AsyncMock,
        ) as mock_start_training:
            mock_start_training.return_value = {"status": "queued", "task_id": "phase53-one-click"}
            resp = self.client.post(
                f"/api/projects/{project_id}/training/autopilot/one-click-run",
                json={
                    "intent": "Train a support assistant",
                    "target_profile_id": "edge_gpu",
                    "auto_prepare_data": False,
                    "plan_profile": "balanced",
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)

    # -- sync helpers -----------------------------------------------------

    def _experiment_count(self, project_id: int) -> int:
        async def _count():
            async with async_session_factory() as db:
                rows = (
                    await db.execute(
                        select(Experiment).where(Experiment.project_id == int(project_id))
                    )
                ).scalars().all()
                return len(list(rows))

        return asyncio.new_event_loop().run_until_complete(_count())

    def _mutate_project_description(self, project_id: int, description: str) -> None:
        # Mutate `base_model_name` which is one of the hash inputs so the
        # state_hash drifts. Using description (not hashed) would not trigger
        # the drift — we want a clean signal in tests.
        async def _patch():
            async with async_session_factory() as db:
                row = (
                    await db.execute(
                        select(Project).where(Project.id == int(project_id))
                    )
                ).scalar_one_or_none()
                self.assertIsNotNone(row)
                row.base_model_name = f"mutation-{uuid.uuid4().hex[:6]}"
                await db.commit()

        asyncio.new_event_loop().run_until_complete(_patch())

    def _force_expire_preview(self, plan_token: str) -> None:
        async def _patch():
            async with async_session_factory() as db:
                row = (
                    await db.execute(
                        select(AutopilotRepairPreview).where(
                            AutopilotRepairPreview.plan_token == plan_token
                        )
                    )
                ).scalar_one_or_none()
                self.assertIsNotNone(row)
                row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
                await db.commit()

        asyncio.new_event_loop().run_until_complete(_patch())


if __name__ == "__main__":
    unittest.main()
