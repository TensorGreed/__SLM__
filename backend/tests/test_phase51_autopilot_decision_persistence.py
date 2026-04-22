"""Phase 51 tests — autopilot decision-log persistence (priority.md P1).

Exercises the new `autopilot_decisions` table, the `/autopilot/decisions`
query surface, and the orchestration wiring that writes a row per step on
every call to `/training/autopilot/v2/orchestrate`.
"""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase51_autopilot_decision_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase51_autopilot_decision_data"

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

from app.config import settings
from app.main import app


class Phase51AutopilotDecisionPersistenceTests(unittest.TestCase):
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
            json={"name": unique_name, "description": "phase51 autopilot persistence"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _orchestrate_dry_run(self, project_id: int, **overrides):
        payload = {
            "intent": "Summarize support tickets and suggest next actions.",
            "target_profile_id": "mobile_cpu",
            "dry_run": True,
            "auto_prepare_data": False,
        }
        payload.update(overrides)
        resp = self.client.post(
            f"/api/projects/{project_id}/training/autopilot/v2/orchestrate",
            json=payload,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return dict(resp.json() or {})

    # -- tests ------------------------------------------------------------

    def test_orchestrate_dry_run_returns_stable_run_id_and_persists_entries(self):
        project_id = self._create_project("phase51-dry-run")
        payload = self._orchestrate_dry_run(project_id)

        run_id = str(payload.get("run_id") or "").strip()
        self.assertTrue(run_id, "Orchestration response must include a non-empty run_id")

        # run_id should be a uuid-shaped string.
        self.assertEqual(len(run_id.replace("-", "")), 32, run_id)

        decision_log = [row for row in list(payload.get("decision_log") or []) if isinstance(row, dict)]
        self.assertGreater(len(decision_log), 0, payload)

        # Persisted entries should match the in-memory decision_log size.
        listing = self.client.get(
            "/api/autopilot/decisions",
            params={"run_id": run_id, "limit": 500},
        )
        self.assertEqual(listing.status_code, 200, listing.text)
        listing_payload = dict(listing.json() or {})
        items = list(listing_payload.get("items") or [])
        self.assertEqual(len(items), len(decision_log), listing_payload)

        # Every row carries the expected metadata.
        for row in items:
            self.assertEqual(row["run_id"], run_id)
            self.assertEqual(row["project_id"], project_id)
            self.assertTrue(row["dry_run"])
            self.assertEqual(row["actor"], "autopilot")
            self.assertIsNotNone(row.get("stage"))
            self.assertIsNotNone(row.get("status"))
            self.assertIn(
                row["action"],
                {"info", "applied", "blocked", "warned", "skipped", "rolled_back"},
            )

        # The per-run endpoint returns the same data ordered by sequence.
        run_resp = self.client.get(f"/api/autopilot/runs/{run_id}/decisions")
        self.assertEqual(run_resp.status_code, 200, run_resp.text)
        run_payload = dict(run_resp.json() or {})
        run_items = list(run_payload.get("items") or [])
        self.assertEqual(len(run_items), len(items))
        sequences = [int(item.get("sequence") or 0) for item in run_items]
        self.assertEqual(sequences, sorted(sequences), run_items)

    def test_list_filters_by_stage_and_status(self):
        project_id = self._create_project("phase51-filters")
        payload = self._orchestrate_dry_run(project_id)
        run_id = str(payload.get("run_id") or "")

        # Every orchestration writes a `final_guardrails` entry at the end.
        stage_resp = self.client.get(
            "/api/autopilot/decisions",
            params={"run_id": run_id, "stage": "final_guardrails"},
        )
        self.assertEqual(stage_resp.status_code, 200, stage_resp.text)
        stage_items = list(dict(stage_resp.json() or {}).get("items") or [])
        self.assertGreaterEqual(len(stage_items), 1, stage_items)
        for row in stage_items:
            self.assertEqual(row["stage"], "final_guardrails")

        # Dry-run against mobile_cpu with no data should surface blocked entries.
        blocked_resp = self.client.get(
            "/api/autopilot/decisions",
            params={"run_id": run_id, "status": "blocked"},
        )
        self.assertEqual(blocked_resp.status_code, 200, blocked_resp.text)
        blocked_items = list(dict(blocked_resp.json() or {}).get("items") or [])
        self.assertGreaterEqual(len(blocked_items), 1, blocked_items)
        for row in blocked_items:
            self.assertEqual(row["status"], "blocked")
            self.assertEqual(row["action"], "blocked")

    def test_two_orchestrations_produce_distinct_run_ids_and_stay_isolated(self):
        project_id = self._create_project("phase51-two-runs")
        first = self._orchestrate_dry_run(project_id)
        second = self._orchestrate_dry_run(project_id)

        run_a = str(first.get("run_id") or "")
        run_b = str(second.get("run_id") or "")
        self.assertTrue(run_a and run_b)
        self.assertNotEqual(run_a, run_b)

        # Project-scoped listing should contain both runs.
        resp = self.client.get(
            "/api/autopilot/decisions",
            params={"project_id": project_id, "limit": 1000},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        items = list(dict(resp.json() or {}).get("items") or [])
        ids_seen = {row["run_id"] for row in items}
        self.assertIn(run_a, ids_seen)
        self.assertIn(run_b, ids_seen)

        # Per-run filter isolates.
        resp_a = self.client.get("/api/autopilot/decisions", params={"run_id": run_a, "limit": 500})
        resp_b = self.client.get("/api/autopilot/decisions", params={"run_id": run_b, "limit": 500})
        items_a = list(dict(resp_a.json() or {}).get("items") or [])
        items_b = list(dict(resp_b.json() or {}).get("items") or [])
        self.assertTrue(items_a)
        self.assertTrue(items_b)
        self.assertTrue(all(row["run_id"] == run_a for row in items_a))
        self.assertTrue(all(row["run_id"] == run_b for row in items_b))

    def test_get_run_decisions_returns_404_for_unknown_run(self):
        resp = self.client.get(
            f"/api/autopilot/runs/{uuid.uuid4().hex}/decisions"
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_decision_log_persists_across_successful_run_and_training_start(self):
        project_id = self._create_project("phase51-run-start")
        prepared_dir = TEST_DATA_DIR / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        train_path = prepared_dir / "train.jsonl"
        with open(train_path, "w", encoding="utf-8") as handle:
            for idx in range(32):
                handle.write(
                    '{"text": "User: support %d\\nAssistant: reply %d"}\n' % (idx, idx)
                )

        with patch(
            "app.api.training.start_training",
            new_callable=AsyncMock,
        ) as mock_start_training:
            mock_start_training.return_value = {"status": "queued", "task_id": "phase51-task"}
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

        payload = dict(resp.json() or {})
        run_id = str(payload.get("run_id") or "")
        self.assertTrue(run_id)
        self.assertTrue(bool(payload.get("started")), payload)

        listing = self.client.get(
            f"/api/autopilot/runs/{run_id}/decisions"
        )
        self.assertEqual(listing.status_code, 200, listing.text)
        items = list(dict(listing.json() or {}).get("items") or [])
        self.assertGreaterEqual(len(items), 1, items)
        # The final step of a successful orchestrate/run is start_training=completed.
        self.assertTrue(
            any(
                row.get("stage") == "start_training" and row.get("status") == "completed"
                for row in items
            ),
            items,
        )
        # None of the rows should be marked dry_run.
        self.assertTrue(all(row["dry_run"] is False for row in items))


if __name__ == "__main__":
    unittest.main()
