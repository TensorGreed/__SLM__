"""Phase 80 — unified timeline service + endpoint (priority.md P32, Wave G).

Covers ``GET /api/projects/{id}/timeline`` + the underlying
:func:`build_timeline` service.

Test matrix:

- Empty project returns empty tree, totals zeroed.
- Multi-event single-run: rollup populates first/last/duration/
  severity_counts/highest_severity/event_count and ``stages_present``.
- Two-level tree (eval child → exp parent) joins via ``parent_run_id``.
- Three-level tree (autopilot → exp → eval).
- Orphan: a child whose parent isn't in the result set becomes a root
  with ``is_orphan=true``.
- Stage filter drops non-matching events and the now-empty run nodes.
- Severity filter retains only matching events; ``highest_severity``
  reflects the kept events.
- Anchor (``run_id``) restricts to that run + transitive descendants
  and forces it to be a root (parent pointer cleared when outside).
- Anchor for a missing run returns ``anchor_present=false`` + empty
  tree without error.
- ``since`` / ``until`` window filter trims correctly.
- Invalid filters surface stable 400 codes; unknown project → 404.
- ``truncated`` flag flips when event count hits ``limit``.
- Children are sorted by ``first_ts`` ASC at every level.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase80_timeline.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase80_timeline_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.run_event import (
    SEVERITY_CRITICAL,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    STAGE_AUTOPILOT,
    STAGE_DEPLOYMENT,
    STAGE_EVAL,
    STAGE_EXPORT,
    STAGE_TRAINING,
)
from app.services.run_event_service import emit_event


def _cleanup_artifacts() -> None:
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


class Phase80TimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        _cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        _cleanup_artifacts()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_project(self, name: str = "phase80") -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"{name}-{uuid.uuid4().hex[:8]}",
                "description": "phase80",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_events(
        self,
        project_id: int,
        events: list[dict],
    ) -> None:
        """Bulk-seed events. Each entry: ts_offset (seconds, >=0), run_id,
        stage, severity (default info), summary, parent_run_id, reason_code."""

        async def _runner():
            async with async_session_factory() as db:
                base = datetime.now(timezone.utc) - timedelta(hours=1)
                for index, entry in enumerate(events):
                    await emit_event(
                        db,
                        project_id=project_id,
                        run_id=entry["run_id"],
                        parent_run_id=entry.get("parent_run_id"),
                        stage=entry["stage"],
                        severity=entry.get("severity", SEVERITY_INFO),
                        summary=entry.get(
                            "summary", f"event-{index}"
                        ),
                        reason_code=entry.get("reason_code"),
                        ts=base + timedelta(seconds=entry["ts_offset"]),
                    )
                await db.commit()

        asyncio.run(_runner())

    def _get_timeline(
        self, project_id: int, **params: object
    ) -> dict:
        resp = self.client.get(
            f"/api/projects/{project_id}/timeline", params=params
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_empty_project_returns_empty_tree(self):
        project_id = self._create_project("empty")
        body = self._get_timeline(project_id)
        self.assertEqual(body["tree"], [])
        self.assertEqual(body["total_events"], 0)
        self.assertEqual(body["total_runs"], 0)
        self.assertEqual(body["orphaned_count"], 0)
        self.assertFalse(body["truncated"])

    def test_single_run_multiple_events_rollup(self):
        project_id = self._create_project("rollup")
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                    "severity": SEVERITY_INFO,
                    "summary": "Training started",
                },
                {
                    "ts_offset": 5,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                    "severity": SEVERITY_WARNING,
                    "summary": "low gpu memory",
                },
                {
                    "ts_offset": 30,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                    "severity": SEVERITY_ERROR,
                    "summary": "Training failed",
                    "reason_code": "training_runtime_error",
                },
            ],
        )
        body = self._get_timeline(project_id)
        self.assertEqual(len(body["tree"]), 1)
        node = body["tree"][0]
        self.assertEqual(node["run_id"], "exp-1")
        self.assertEqual(node["stage"], "training")
        self.assertEqual(node["event_count"], 3)
        self.assertEqual(
            node["severity_counts"],
            {"info": 1, "warning": 1, "error": 1},
        )
        self.assertEqual(node["highest_severity"], "error")
        self.assertEqual(node["latest_reason_code"], "training_runtime_error")
        self.assertEqual(node["summary"], "Training failed")
        self.assertEqual(node["stages_present"], ["training"])
        self.assertAlmostEqual(node["duration_seconds"], 30.0, places=1)
        self.assertEqual(node["children"], [])

    def test_two_level_tree_links_via_parent_run_id(self):
        project_id = self._create_project("twolevel")
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                    "summary": "training started",
                },
                {
                    "ts_offset": 60,
                    "run_id": "eval-1",
                    "parent_run_id": "exp-1",
                    "stage": STAGE_EVAL,
                    "summary": "eval completed",
                },
                {
                    "ts_offset": 65,
                    "run_id": "eval-2",
                    "parent_run_id": "exp-1",
                    "stage": STAGE_EVAL,
                    "summary": "second eval completed",
                },
            ],
        )
        body = self._get_timeline(project_id)
        self.assertEqual(len(body["tree"]), 1)
        root = body["tree"][0]
        self.assertEqual(root["run_id"], "exp-1")
        children = root["children"]
        self.assertEqual(len(children), 2)
        self.assertEqual(
            [c["run_id"] for c in children], ["eval-1", "eval-2"]
        )
        self.assertTrue(all(c["children"] == [] for c in children))

    def test_three_level_tree_assembled_in_order(self):
        project_id = self._create_project("threelevel")
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "autopilot-X",
                    "stage": STAGE_AUTOPILOT,
                },
                {
                    "ts_offset": 10,
                    "run_id": "exp-1",
                    "parent_run_id": "autopilot-X",
                    "stage": STAGE_TRAINING,
                },
                {
                    "ts_offset": 50,
                    "run_id": "eval-1",
                    "parent_run_id": "exp-1",
                    "stage": STAGE_EVAL,
                },
            ],
        )
        body = self._get_timeline(project_id)
        self.assertEqual(len(body["tree"]), 1)
        root = body["tree"][0]
        self.assertEqual(root["run_id"], "autopilot-X")
        self.assertEqual(len(root["children"]), 1)
        exp_node = root["children"][0]
        self.assertEqual(exp_node["run_id"], "exp-1")
        self.assertEqual(len(exp_node["children"]), 1)
        self.assertEqual(exp_node["children"][0]["run_id"], "eval-1")

    def test_orphan_appears_as_root_with_flag(self):
        project_id = self._create_project("orphan")
        # Only the child is seeded — the parent ``exp-99`` is referenced
        # but never emitted, so the child should be promoted to a root
        # with is_orphan=true.
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "eval-99",
                    "parent_run_id": "exp-99",
                    "stage": STAGE_EVAL,
                    "summary": "eval-only",
                },
            ],
        )
        body = self._get_timeline(project_id)
        self.assertEqual(len(body["tree"]), 1)
        node = body["tree"][0]
        self.assertEqual(node["run_id"], "eval-99")
        self.assertTrue(node["is_orphan"])
        self.assertEqual(node["parent_run_id"], "exp-99")
        self.assertEqual(body["orphaned_count"], 1)

    def test_stage_filter_drops_runs_without_matching_events(self):
        project_id = self._create_project("stagefilter")
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                },
                {
                    "ts_offset": 5,
                    "run_id": "deploy-1",
                    "stage": STAGE_DEPLOYMENT,
                },
            ],
        )
        body = self._get_timeline(project_id, stage="deployment")
        self.assertEqual(len(body["tree"]), 1)
        self.assertEqual(body["tree"][0]["run_id"], "deploy-1")

    def test_severity_filter_retains_only_matching_events(self):
        project_id = self._create_project("sevfilter")
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                    "severity": SEVERITY_INFO,
                },
                {
                    "ts_offset": 5,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                    "severity": SEVERITY_CRITICAL,
                    "reason_code": "training_oom",
                },
            ],
        )
        body = self._get_timeline(project_id, severity="critical")
        self.assertEqual(len(body["tree"]), 1)
        node = body["tree"][0]
        self.assertEqual(node["event_count"], 1)
        self.assertEqual(node["highest_severity"], "critical")
        self.assertEqual(node["latest_reason_code"], "training_oom")

    def test_anchor_restricts_to_subtree(self):
        project_id = self._create_project("anchor")
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                },
                {
                    "ts_offset": 1,
                    "run_id": "exp-2",
                    "stage": STAGE_TRAINING,
                },
                {
                    "ts_offset": 5,
                    "run_id": "eval-1",
                    "parent_run_id": "exp-1",
                    "stage": STAGE_EVAL,
                },
                {
                    "ts_offset": 6,
                    "run_id": "eval-2",
                    "parent_run_id": "exp-2",
                    "stage": STAGE_EVAL,
                },
            ],
        )
        body = self._get_timeline(project_id, run_id="exp-1")
        self.assertTrue(body["anchor_present"])
        self.assertEqual(body["anchor_run_id"], "exp-1")
        self.assertEqual(len(body["tree"]), 1)
        root = body["tree"][0]
        self.assertEqual(root["run_id"], "exp-1")
        # The anchor's parent pointer is cleared so it isn't an orphan
        # even when its real parent is filtered out.
        self.assertFalse(root["is_orphan"])
        self.assertEqual(len(root["children"]), 1)
        self.assertEqual(root["children"][0]["run_id"], "eval-1")
        # exp-2 / eval-2 are out of subtree.
        self.assertNotIn("exp-2", _all_run_ids(root))
        self.assertNotIn("eval-2", _all_run_ids(root))

    def test_anchor_missing_returns_anchor_absent_empty_tree(self):
        project_id = self._create_project("anchormiss")
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                },
            ],
        )
        body = self._get_timeline(project_id, run_id="exp-9999")
        self.assertFalse(body["anchor_present"])
        self.assertEqual(body["tree"], [])
        self.assertEqual(body["total_runs"], 0)

    def test_since_until_window_trims_events(self):
        project_id = self._create_project("window")
        # ``_seed_events`` sets base = now − 1h. Offsets:
        #   old-run @ base + 0      → 1h ago
        #   new-run @ base + 3300s  → 5 minutes ago
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "old-run",
                    "stage": STAGE_TRAINING,
                },
                {
                    "ts_offset": 3300,
                    "run_id": "new-run",
                    "stage": STAGE_TRAINING,
                },
            ],
        )
        # Window from 10 minutes ago: only new-run lands inside.
        since = (
            datetime.now(timezone.utc) - timedelta(minutes=10)
        ).isoformat()
        body = self._get_timeline(project_id, since=since)
        run_ids = {n["run_id"] for n in body["tree"]}
        self.assertIn("new-run", run_ids)
        self.assertNotIn("old-run", run_ids)

    def test_invalid_window_400(self):
        project_id = self._create_project("badwin")
        now = datetime.now(timezone.utc)
        resp = self.client.get(
            f"/api/projects/{project_id}/timeline",
            params={
                "since": now.isoformat(),
                "until": (now - timedelta(hours=1)).isoformat(),
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("invalid_window", resp.json()["detail"])

    def test_invalid_stage_400(self):
        project_id = self._create_project("badstage")
        resp = self.client.get(
            f"/api/projects/{project_id}/timeline",
            params={"stage": "ghost-stage"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("invalid_stage", resp.json()["detail"])

    def test_invalid_severity_400(self):
        project_id = self._create_project("badsev")
        resp = self.client.get(
            f"/api/projects/{project_id}/timeline",
            params={"severity": "not-real"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("invalid_severity", resp.json()["detail"])

    def test_unknown_project_404(self):
        resp = self.client.get("/api/projects/999999/timeline")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "project_not_found")

    def test_truncation_flag_flips_when_limit_hit(self):
        project_id = self._create_project("trunc")
        # Seed 10 events; query with limit=5.
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": i,
                    "run_id": f"r-{i}",
                    "stage": STAGE_TRAINING,
                }
                for i in range(10)
            ],
        )
        body = self._get_timeline(project_id, limit=5)
        self.assertTrue(body["truncated"])
        self.assertEqual(body["total_events"], 5)

    def test_children_sorted_by_first_ts_ascending(self):
        project_id = self._create_project("childorder")
        self._seed_events(
            project_id,
            [
                {
                    "ts_offset": 0,
                    "run_id": "exp-1",
                    "stage": STAGE_TRAINING,
                },
                {
                    "ts_offset": 100,
                    "run_id": "eval-late",
                    "parent_run_id": "exp-1",
                    "stage": STAGE_EVAL,
                },
                {
                    "ts_offset": 50,
                    "run_id": "export-mid",
                    "parent_run_id": "exp-1",
                    "stage": STAGE_EXPORT,
                },
                {
                    "ts_offset": 25,
                    "run_id": "eval-early",
                    "parent_run_id": "exp-1",
                    "stage": STAGE_EVAL,
                },
            ],
        )
        body = self._get_timeline(project_id)
        root = body["tree"][0]
        child_ids = [c["run_id"] for c in root["children"]]
        self.assertEqual(
            child_ids, ["eval-early", "export-mid", "eval-late"]
        )


def _all_run_ids(node: dict) -> set[str]:
    out = {node["run_id"]}
    for child in node.get("children", []):
        out |= _all_run_ids(child)
    return out


if __name__ == "__main__":
    unittest.main()
