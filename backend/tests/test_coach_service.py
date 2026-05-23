"""Tests for the Coach Mode service (USER-SUCCESS Epic 4 Phase 1).

Covers:
- ``_topup_count`` clamp + threshold math.
- ``_data_stage_suggestions`` against synthetic projects: empty gold,
  thin gold, mid-tier gold, comfortable gold.
- ``suggest_for_stage`` payload shape + unknown-stage fallback.
- End-to-end via the FastAPI test client: 404 on unknown project,
  400 on invalid stage, 200 on data stage with suggestion list.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "coach_service_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "coach_service_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.coach_service import (  # noqa: E402
    GOLD_ROW_COMFORTABLE_MIN,
    GOLD_ROW_THIN_MAX,
    SUGGESTED_TOPUP_CEILING,
    SUGGESTED_TOPUP_FLOOR,
    _data_stage_suggestions,
    _topup_count,
)


class CoachServicePureFunctionTests(unittest.TestCase):
    def test_topup_count_floors_at_minimum_when_already_comfortable(self):
        # Project already past the comfortable threshold — delta would
        # be negative, but the floor keeps the suggestion meaningful.
        self.assertEqual(
            _topup_count(GOLD_ROW_COMFORTABLE_MIN + 50),
            SUGGESTED_TOPUP_FLOOR,
        )

    def test_topup_count_clamps_at_ceiling_for_thin_projects(self):
        # The synth playbook endpoint caps target_count at 500;
        # _topup_count must not exceed that or the request 422s.
        self.assertLessEqual(_topup_count(0), SUGGESTED_TOPUP_CEILING)
        self.assertGreater(_topup_count(0), SUGGESTED_TOPUP_FLOOR)

    def test_topup_count_returns_delta_in_between(self):
        # 200 gold rows → 300 - 200 = 100 topup (between floor + ceiling).
        # This is a sanity check that the typical mid-range case
        # produces a sensible delta.
        result = _topup_count(200)
        self.assertEqual(result, GOLD_ROW_COMFORTABLE_MIN - 200)


class CoachServiceApiTests(unittest.TestCase):
    """End-to-end via FastAPI TestClient — instantiates a project
    template + calls the coach endpoint, asserts payload shape."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

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
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_endpoint_404s_on_unknown_project(self):
        resp = self.client.get("/api/projects/999999/coach/data")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_endpoint_400s_on_invalid_stage(self):
        project = self._instantiate_template(
            "ticket-router", "Coach Invalid Stage"
        )
        resp = self.client.get(
            f"/api/projects/{project['id']}/coach/bogus-stage"
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        # Error message lists valid stages so the client can self-
        # correct without reading server logs.
        self.assertIn("data", resp.json()["detail"])

    def test_data_stage_returns_payload_shape(self):
        project = self._instantiate_template(
            "ticket-router", "Coach Data Stage Shape"
        )
        resp = self.client.get(f"/api/projects/{project['id']}/coach/data")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        # Shape contract.
        self.assertEqual(payload["project_id"], project["id"])
        self.assertEqual(payload["stage"], "data")
        self.assertIsInstance(payload["suggestions"], list)
        self.assertTrue(payload["handler_available"])

    def test_data_stage_emits_gold_row_count_suggestion_when_below_comfortable(self):
        # The ticket-router template ships with a 200-row gold set,
        # which is past the thin floor (100) but below the comfortable
        # threshold (300) — so the coach must emit a warning-severity
        # suggestion proposing a top-up via the recipe's
        # positives_paraphrase playbook.
        project = self._instantiate_template(
            "ticket-router", "Coach Below Comfortable"
        )
        resp = self.client.get(f"/api/projects/{project['id']}/coach/data")
        self.assertEqual(resp.status_code, 200, resp.text)
        suggestions = resp.json()["suggestions"]
        suggestion_ids = {s["id"] for s in suggestions}
        self.assertIn("data:gold-row-count", suggestion_ids)
        gold_row_suggestion = next(
            s for s in suggestions if s["id"] == "data:gold-row-count"
        )
        # 200 rows is between thin (99) and comfortable (300) → warning.
        self.assertEqual(gold_row_suggestion["severity"], "warning")
        # Project carries a selected_recipe → action must be
        # run_playbook with positives_paraphrase.
        self.assertEqual(gold_row_suggestion["action"]["kind"], "run_playbook")
        self.assertEqual(
            gold_row_suggestion["action"]["params"]["mode"],
            "positives_paraphrase",
        )
        target = gold_row_suggestion["action"]["params"]["target_count"]
        self.assertGreaterEqual(target, SUGGESTED_TOPUP_FLOOR)
        self.assertLessEqual(target, SUGGESTED_TOPUP_CEILING)
        # Context carries the row count + thresholds so the UI can
        # render numeric framing inline if it wants to.
        ctx = gold_row_suggestion["context"]
        self.assertEqual(ctx["gold_row_count"], 200)
        self.assertEqual(ctx["comfortable_threshold"], GOLD_ROW_COMFORTABLE_MIN)
        self.assertEqual(ctx["thin_threshold"], GOLD_ROW_THIN_MAX)


class CoachServiceDirectCallTests(unittest.IsolatedAsyncioTestCase):
    """Direct calls to ``_data_stage_suggestions`` with a stubbed
    project + patched row-count loader. Lets us exercise each
    severity tier (thin / mid / comfortable) without standing up a
    different template per case."""

    async def _suggestions_for_row_count(
        self,
        row_count: int,
        *,
        with_recipe: bool = True,
    ) -> list[dict]:
        from unittest.mock import patch

        class _StubProject:
            id = 42
            selected_recipe = (
                {"recipe_id": "classification"} if with_recipe else None
            )

        with patch(
            "app.services.coach_service._read_gold_row_count",
            return_value=row_count,
        ) as patched:
            async def _async_return(*_a, **_k):
                return patched.return_value

            patched.side_effect = _async_return
            return await _data_stage_suggestions(db=None, project=_StubProject())  # type: ignore[arg-type]

    async def test_thin_gold_triggers_critical_severity(self):
        suggestions = await self._suggestions_for_row_count(30)
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["severity"], "critical")
        self.assertEqual(s["context"]["gold_row_count"], 30)
        # Body framing for thin sets emphasizes the 100-row floor.
        self.assertIn("100", s["body"])

    async def test_mid_tier_gold_triggers_warning_severity(self):
        suggestions = await self._suggestions_for_row_count(180)
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["severity"], "warning")
        self.assertEqual(s["context"]["gold_row_count"], 180)

    async def test_comfortable_gold_emits_no_suggestion(self):
        suggestions = await self._suggestions_for_row_count(
            GOLD_ROW_COMFORTABLE_MIN
        )
        # Exactly at the threshold → no suggestion. We don't badger
        # users who are already in the comfort zone.
        self.assertEqual(suggestions, [])

    async def test_no_recipe_yet_falls_back_to_navigate_action(self):
        # Project without a selected_recipe can't trigger run-playbook
        # (the endpoint requires a recipe). The coach must degrade to
        # a navigate action so the click is still useful.
        suggestions = await self._suggestions_for_row_count(
            30, with_recipe=False
        )
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["action"]["kind"], "navigate")
        self.assertEqual(s["action"]["params"]["target"], "recipe-picker")


if __name__ == "__main__":
    unittest.main()
