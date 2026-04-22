"""Phase 52 tests — autopilot snapshot + rollback (priority.md P2).

Exercises:
- Snapshot capture during orchestration (autopilot training launch).
- `GET /autopilot/snapshots` and `GET /autopilot/decisions/{id}/snapshot`.
- `POST /autopilot/rollback/{decision_id}/preview` (non-mutating).
- `POST /autopilot/rollback/{decision_id}` (executes + writes new decision).
- Error paths: unknown decision, no-snapshot decision, double rollback,
  expired snapshot.
"""

from __future__ import annotations

import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase52_autopilot_rollback_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase52_autopilot_rollback_data"

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
from app.models.autopilot_decision import AutopilotDecision
from app.models.autopilot_snapshot import AutopilotSnapshot
from app.models.experiment import Experiment, ExperimentStatus


class Phase52AutopilotRollbackTests(unittest.TestCase):
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
            json={"name": unique_name, "description": "phase52 autopilot rollback"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_prepared_rows(self, project_id: int, row_count: int = 32) -> None:
        prepared_dir = TEST_DATA_DIR / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        with open(prepared_dir / "train.jsonl", "w", encoding="utf-8") as handle:
            for idx in range(row_count):
                handle.write(
                    '{"text": "User: support ticket %d\\nAssistant: concise reply %d"}\n'
                    % (idx, idx)
                )

    def _orchestrate_training_launch(self, project_id: int) -> dict:
        with patch(
            "app.api.training.start_training",
            new_callable=AsyncMock,
        ) as mock_start_training:
            mock_start_training.return_value = {
                "status": "queued",
                "task_id": "phase52-autopilot-task",
            }
            resp = self.client.post(
                f"/api/projects/{project_id}/training/autopilot/v2/orchestrate/run",
                json={
                    "intent": "Train a concise support assistant.",
                    "target_profile_id": "edge_gpu",
                    "dry_run": False,
                    "auto_prepare_data": False,
                    "plan_profile": "balanced",
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        return dict(resp.json() or {})

    def _start_training_decision(self, run_id: str) -> dict:
        resp = self.client.get(
            f"/api/autopilot/runs/{run_id}/decisions"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        items = list(dict(resp.json() or {}).get("items") or [])
        for row in items:
            if row.get("stage") == "start_training" and row.get("status") == "completed":
                return row
        self.fail(f"No completed start_training decision in run {run_id}; items={items}")
        return {}  # unreachable

    # -- tests ------------------------------------------------------------

    def test_orchestration_captures_snapshot_for_training_launch(self):
        project_id = self._create_project("phase52-capture")
        self._seed_prepared_rows(project_id)
        payload = self._orchestrate_training_launch(project_id)

        run_id = str(payload.get("run_id") or "")
        self.assertTrue(run_id)
        self.assertTrue(bool(payload.get("started")))

        snapshots_resp = self.client.get(
            "/api/autopilot/snapshots",
            params={"project_id": project_id, "run_id": run_id},
        )
        self.assertEqual(snapshots_resp.status_code, 200, snapshots_resp.text)
        items = list(dict(snapshots_resp.json() or {}).get("items") or [])
        self.assertEqual(len(items), 1, items)
        snap = items[0]
        self.assertEqual(snap["run_id"], run_id)
        self.assertEqual(snap["project_id"], project_id)
        self.assertEqual(snap["snapshot_type"], "autopilot_training_launch")
        self.assertIsNone(snap["restored_at"])
        self.assertIn("project", dict(snap.get("pre_state") or {}))
        post_state = dict(snap.get("post_state") or {})
        self.assertGreater(int(post_state.get("experiment_id") or 0), 0)
        self.assertTrue(bool(post_state.get("training_started")))

        # Per-decision snapshot lookup matches.
        decision = self._start_training_decision(run_id)
        lookup = self.client.get(f"/api/autopilot/decisions/{decision['id']}/snapshot")
        self.assertEqual(lookup.status_code, 200, lookup.text)
        self.assertEqual(lookup.json()["id"], snap["id"])

    def test_rollback_preview_is_reversible_and_does_not_mutate(self):
        project_id = self._create_project("phase52-preview")
        self._seed_prepared_rows(project_id)
        payload = self._orchestrate_training_launch(project_id)
        decision = self._start_training_decision(str(payload.get("run_id") or ""))

        resp = self.client.post(
            f"/api/autopilot/rollback/{decision['id']}/preview"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = dict(resp.json() or {})
        self.assertTrue(bool(data.get("reversible")), data)
        steps = list(data.get("steps") or [])
        kinds = {step.get("kind") for step in steps}
        self.assertIn("cancel_experiment", kinds)

        # Experiment should still exist and NOT be cancelled.
        exp_id = int(
            dict(dict(data.get("snapshot") or {}).get("post_state") or {}).get(
                "experiment_id"
            )
            or 0
        )
        self.assertGreater(exp_id, 0)
        self.assertEqual(self._experiment_status(exp_id), ExperimentStatus.PENDING)

        # Snapshot is not marked restored.
        snap = dict(data.get("snapshot") or {})
        self.assertIsNone(snap.get("restored_at"))

    def test_rollback_cancels_experiment_and_records_new_decision(self):
        project_id = self._create_project("phase52-execute")
        self._seed_prepared_rows(project_id)
        payload = self._orchestrate_training_launch(project_id)
        run_id = str(payload.get("run_id") or "")
        decision = self._start_training_decision(run_id)

        before_decisions = self._count_decisions_for_run(run_id)

        resp = self.client.post(
            f"/api/autopilot/rollback/{decision['id']}",
            json={"reason": "test rollback", "actor": "phase52-test"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = dict(resp.json() or {})
        self.assertTrue(bool(data.get("ok")), data)

        # Experiment should be cancelled.
        snap = dict(data.get("snapshot") or {})
        exp_id = int(dict(snap.get("post_state") or {}).get("experiment_id") or 0)
        self.assertEqual(self._experiment_status(exp_id), ExperimentStatus.CANCELLED)

        # New rollback decision exists at the tail of the run.
        after_decisions = self._count_decisions_for_run(run_id)
        self.assertEqual(after_decisions, before_decisions + 1)

        rb = dict(data.get("rollback_decision") or {})
        self.assertEqual(rb.get("stage"), "rollback")
        self.assertEqual(rb.get("action"), "rolled_back")
        self.assertEqual(rb.get("run_id"), run_id)

        # Rollback decision is also reachable through the P1 query API.
        search = self.client.get(
            "/api/autopilot/decisions",
            params={"run_id": run_id, "stage": "rollback"},
        )
        self.assertEqual(search.status_code, 200)
        hits = list(dict(search.json() or {}).get("items") or [])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["action"], "rolled_back")
        self.assertEqual(hits[0]["actor"], "phase52-test")
        self.assertEqual(hits[0]["reason_code"], "AUTOPILOT_ROLLBACK")

    def test_double_rollback_returns_409(self):
        project_id = self._create_project("phase52-double")
        self._seed_prepared_rows(project_id)
        payload = self._orchestrate_training_launch(project_id)
        decision = self._start_training_decision(str(payload.get("run_id") or ""))

        first = self.client.post(f"/api/autopilot/rollback/{decision['id']}")
        self.assertEqual(first.status_code, 200, first.text)

        second = self.client.post(f"/api/autopilot/rollback/{decision['id']}")
        self.assertEqual(second.status_code, 409, second.text)
        detail = dict(second.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "already_rolled_back")

    def test_rollback_without_snapshot_returns_409(self):
        project_id = self._create_project("phase52-no-snapshot")
        # A dry-run orchestrate emits decisions but never the training-launch
        # snapshot (no create_experiment happens).
        resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/v2/orchestrate",
            json={
                "intent": "Build a support assistant.",
                "target_profile_id": "mobile_cpu",
                "dry_run": True,
                "auto_prepare_data": False,
            },
        )
        self.assertEqual(resp.status_code, 200)
        run_id = str(resp.json().get("run_id") or "")
        # Pick any decision id from this run.
        decisions = self.client.get(
            "/api/autopilot/decisions", params={"run_id": run_id, "limit": 1}
        )
        items = list(dict(decisions.json()).get("items") or [])
        self.assertTrue(items)
        decision_id = int(items[0]["id"])

        rollback = self.client.post(f"/api/autopilot/rollback/{decision_id}")
        self.assertEqual(rollback.status_code, 409, rollback.text)
        detail = dict(rollback.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "no_snapshot")

    def test_rollback_unknown_decision_returns_404(self):
        resp = self.client.post("/api/autopilot/rollback/999999999")
        self.assertEqual(resp.status_code, 404, resp.text)
        detail = dict(resp.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "decision_not_found")

    def test_expired_snapshot_returns_410(self):
        project_id = self._create_project("phase52-expired")
        self._seed_prepared_rows(project_id)
        payload = self._orchestrate_training_launch(project_id)
        decision = self._start_training_decision(str(payload.get("run_id") or ""))

        # Force-expire the snapshot.
        self._expire_snapshot_for_decision(decision["id"])

        resp = self.client.post(f"/api/autopilot/rollback/{decision['id']}")
        self.assertEqual(resp.status_code, 410, resp.text)
        detail = dict(resp.json().get("detail") or {})
        self.assertEqual(detail.get("reason"), "snapshot_expired")

    def test_purge_expired_snapshots_removes_only_expired(self):
        project_id = self._create_project("phase52-purge")
        self._seed_prepared_rows(project_id)

        # Fresh (non-expired) run.
        payload_fresh = self._orchestrate_training_launch(project_id)
        run_fresh = str(payload_fresh.get("run_id") or "")

        # Expired run.
        payload_stale = self._orchestrate_training_launch(project_id)
        run_stale = str(payload_stale.get("run_id") or "")
        decision_stale = self._start_training_decision(run_stale)
        self._expire_snapshot_for_decision(decision_stale["id"])

        purge = self.client.post("/api/autopilot/snapshots/purge-expired")
        self.assertEqual(purge.status_code, 200, purge.text)
        self.assertGreaterEqual(int(dict(purge.json()).get("removed") or 0), 1)

        # The fresh run's snapshot is still there.
        fresh_resp = self.client.get(
            "/api/autopilot/snapshots", params={"run_id": run_fresh}
        )
        fresh_items = list(dict(fresh_resp.json()).get("items") or [])
        self.assertEqual(len(fresh_items), 1, fresh_items)

        # The stale run's snapshot is gone.
        stale_resp = self.client.get(
            "/api/autopilot/snapshots", params={"run_id": run_stale}
        )
        stale_items = list(dict(stale_resp.json()).get("items") or [])
        self.assertEqual(len(stale_items), 0, stale_items)

    # -- sync DB helpers --------------------------------------------------

    def _experiment_status(self, experiment_id: int) -> ExperimentStatus:
        import asyncio

        async def _fetch():
            async with async_session_factory() as db:
                row = (
                    await db.execute(
                        select(Experiment).where(Experiment.id == int(experiment_id))
                    )
                ).scalar_one_or_none()
                return row.status if row else None

        status = asyncio.new_event_loop().run_until_complete(_fetch())
        self.assertIsNotNone(status)
        return status

    def _count_decisions_for_run(self, run_id: str) -> int:
        resp = self.client.get(
            "/api/autopilot/decisions", params={"run_id": run_id, "limit": 500}
        )
        return len(list(dict(resp.json()).get("items") or []))

    def _expire_snapshot_for_decision(self, decision_id: int) -> None:
        import asyncio

        async def _patch():
            async with async_session_factory() as db:
                dec = (
                    await db.execute(
                        select(AutopilotDecision).where(
                            AutopilotDecision.id == int(decision_id)
                        )
                    )
                ).scalar_one_or_none()
                self.assertIsNotNone(dec)
                snap = (
                    await db.execute(
                        select(AutopilotSnapshot).where(
                            AutopilotSnapshot.run_id == dec.run_id,
                            AutopilotSnapshot.decision_sequence == int(dec.sequence),
                        )
                    )
                ).scalar_one_or_none()
                self.assertIsNotNone(snap)
                snap.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
                await db.commit()

        asyncio.new_event_loop().run_until_complete(_patch())


if __name__ == "__main__":
    unittest.main()
