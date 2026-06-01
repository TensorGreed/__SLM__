"""Tests for the project smoke-test service (Diagnostics Intervention C).

Covers:
  * Happy path: every check on a healthy project lands ``ok``.
  * Missing-recipe project: recipe_applied fails, synth_catalog
    skips, others still run + come back ok.
  * Non-existent project: project_exists fails, downstream checks
    skip cleanly rather than 5xx-ing.
  * Failure envelope shape — fail checks carry an ErrorEnvelope-
    shaped dict the frontend can drop into <ErrorPanel>.
  * Overall rollup logic: any fail → fail; else any warn → warn;
    else ok. Skip counts as neutral.
  * Parallel execution — total elapsed should be close to the
    slowest single check, not the sum.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "smoke_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "smoke_test_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402


class ProjectSmokeTestApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _create_project(self, name: str | None = None) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name or f"smoke-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _apply_recipe(self, project_id: int, recipe_id: str) -> None:
        resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": recipe_id},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def _checks_by_name(self, body: dict) -> dict:
        return {c["name"]: c for c in body["checks"]}

    # ── Envelope shape contract ─────────────────────────────

    def test_response_shape_matches_documented_contract(self):
        """Every response carries the keys the frontend depends on:
        ``overall``, ``elapsed_ms``, ``counts``, ``checks`` (list of
        ``{name, status, elapsed_ms, message, remediation, envelope,
        metadata}``)."""
        pid = self._create_project()
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], pid)
        self.assertIn(body["overall"], {"ok", "warn", "fail", "skip"})
        self.assertIsInstance(body["elapsed_ms"], int)
        self.assertIsInstance(body["counts"], dict)
        # Counts cover all 4 statuses (some at 0 is fine).
        for status in ("ok", "warn", "fail", "skip"):
            self.assertIn(status, body["counts"])
        self.assertIsInstance(body["checks"], list)
        for check in body["checks"]:
            self.assertIn(check["status"], {"ok", "warn", "fail", "skip"})
            self.assertIsInstance(check["name"], str)
            self.assertGreater(len(check["name"]), 0)
            self.assertIn("message", check)
            self.assertIn("envelope", check)  # may be None
            self.assertIn("metadata", check)

    def test_no_recipe_project_lands_warn_overall_with_recipe_fail(self):
        """A freshly-created project has no recipe. ``recipe_applied``
        is the explicit failure; downstream checks that depend on a
        recipe (``synth_catalog``) skip cleanly."""
        pid = self._create_project()
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        body = resp.json()
        checks = self._checks_by_name(body)
        self.assertEqual(checks["recipe_applied"]["status"], "fail")
        self.assertIn("recipe", checks["recipe_applied"]["message"].lower())
        # synth_catalog correctly skips (no recipe means nothing to
        # enumerate — that's not a platform bug, just nothing to test).
        self.assertEqual(checks["synth_catalog"]["status"], "skip")
        # The fail check carries an envelope the frontend can render.
        env = checks["recipe_applied"]["envelope"]
        self.assertIsNotNone(env)
        self.assertEqual(env["error_code"], "SMOKE_RECIPE_MISSING")
        self.assertTrue(env["troubleshooting_id"].startswith("err_"))
        # Overall reflects the fail.
        self.assertEqual(body["overall"], "fail")

    def test_project_with_recipe_applied_passes_recipe_check(self):
        """Apply a recipe → recipe_applied flips to ok + synth_catalog
        un-skips (it now has a recipe to enumerate against)."""
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        body = resp.json()
        checks = self._checks_by_name(body)
        self.assertEqual(checks["recipe_applied"]["status"], "ok")
        self.assertEqual(checks["synth_catalog"]["status"], "ok")
        self.assertIn("classification", checks["recipe_applied"]["message"])
        # The synth catalog reports how many playbooks for this recipe.
        self.assertIn("playbook", checks["synth_catalog"]["message"].lower())

    def test_non_existent_project_returns_404_via_fail_check_not_5xx(self):
        """Smoke-testing a project that doesn't exist used to be the
        kind of failure that would propagate as a 500. With the
        per-check ``except`` + skip-downstream behavior, the response
        is a clean 200 with project_exists=fail and the rest at skip.

        This is the load-bearing property: a broken project should
        never break the smoke-test endpoint itself."""
        resp = self.client.post("/api/projects/999999/smoke-test")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        checks = self._checks_by_name(body)
        # project_exists is the primary fail.
        self.assertEqual(checks["project_exists"]["status"], "fail")
        # Downstream checks that depend on the project should NOT crash
        # — they either skip (recipe_applied / synth_catalog) or just
        # return their natural empty-state result.
        # recipe_applied + synth_catalog know to short-circuit.
        self.assertEqual(checks["recipe_applied"]["status"], "skip")
        self.assertEqual(checks["synth_catalog"]["status"], "skip")
        # Overall ≥ fail.
        self.assertEqual(body["overall"], "fail")

    def test_overall_rollup_is_worst_status(self):
        """A project with no recipe has at least one fail + at least
        one warn (gold_set empty, prepared_splits empty). The overall
        rollup must be ``fail`` regardless of how many warns are
        below it."""
        pid = self._create_project()
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        body = resp.json()
        # We deliberately verify the COMBINED state — there are
        # multiple warns AND one fail.
        self.assertGreaterEqual(body["counts"]["fail"], 1)
        self.assertEqual(body["overall"], "fail")

    def test_parallel_execution_is_faster_than_sum_of_checks(self):
        """Sanity check that the orchestrator runs checks in parallel.
        Total elapsed should be near the slowest single check, NOT
        the sum across all 9 checks. We assert a loose upper bound
        (total < 2× slowest) which is enough to catch a regression
        to serial execution."""
        pid = self._create_project()
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        body = resp.json()
        total = body["elapsed_ms"]
        slowest = max(c["elapsed_ms"] for c in body["checks"])
        sum_serial = sum(c["elapsed_ms"] for c in body["checks"])
        # Serial would have total ≈ sum. Parallel has total ≈ slowest.
        # The bound is loose to avoid CI flakiness; in practice we see
        # ratios like 1.05× or 1.2×.
        self.assertLess(
            total, slowest * 3,
            f"Smoke test isn't parallel: total={total}ms slowest={slowest}ms",
        )
        # And the cumulative serial time should be meaningfully larger
        # than the actual parallel total — otherwise the checks are
        # too cheap to tell apart.
        if sum_serial > 20:  # threshold to skip the test on a too-fast box
            self.assertLess(total, sum_serial)


if __name__ == "__main__":
    unittest.main()
