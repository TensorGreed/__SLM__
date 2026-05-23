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
    CLASS_BALANCE_TOPUP_DEFAULT,
    DIVERSITY_TOPUP_DEFAULT,
    DOC_ERROR_MIN_TOTAL,
    DOC_ERROR_RATE_WARN,
    GOLD_ROW_COMFORTABLE_MIN,
    GOLD_ROW_THIN_MAX,
    SUGGESTED_TOPUP_CEILING,
    SUGGESTED_TOPUP_FLOOR,
    _cleaning_stage_suggestions,
    _data_stage_suggestions,
    _gold_set_stage_suggestions,
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


class CoachServiceCleaningStageTests(unittest.IsolatedAsyncioTestCase):
    """Phase 2: ``_cleaning_stage_suggestions`` direct calls with
    patched data loaders. Mirrors the patched-loader pattern from the
    data-stage tests."""

    async def _suggestions(
        self,
        *,
        pii_stats: dict,
        status_counts: dict,
    ) -> list[dict]:
        from unittest.mock import patch

        class _StubProject:
            id = 7
            selected_recipe = {"recipe_id": "classification"}

        async def _async_pii(*_a, **_k):
            return pii_stats

        async def _async_status(*_a, **_k):
            return status_counts

        with (
            patch(
                "app.services.coach_service._read_pii_stats",
                side_effect=_async_pii,
            ),
            patch(
                "app.services.coach_service._read_doc_status_breakdown",
                side_effect=_async_status,
            ),
        ):
            return await _cleaning_stage_suggestions(
                db=None,  # type: ignore[arg-type]
                project=_StubProject(),
            )

    async def test_no_pii_and_no_errors_emits_no_suggestions(self):
        suggestions = await self._suggestions(
            pii_stats={"total_pii": 0, "docs_with_pii": 0, "pii_types": []},
            status_counts={"accepted": 50},
        )
        self.assertEqual(suggestions, [])

    async def test_pii_findings_emit_warning_with_navigate_action(self):
        suggestions = await self._suggestions(
            pii_stats={
                "total_pii": 12,
                "docs_with_pii": 4,
                "pii_types": ["email", "phone"],
            },
            status_counts={"accepted": 50},
        )
        ids = {s["id"] for s in suggestions}
        self.assertIn("cleaning:pii-findings", ids)
        s = next(s for s in suggestions if s["id"] == "cleaning:pii-findings")
        self.assertEqual(s["severity"], "warning")
        self.assertEqual(s["action"]["kind"], "navigate")
        self.assertEqual(s["action"]["params"]["target"], "cleaning-pii-review")
        # Title carries the counts so the user gets a one-glance answer
        # without expanding the body.
        self.assertIn("12", s["title"])
        self.assertIn("4", s["title"])
        self.assertEqual(s["context"]["pii_types"], ["email", "phone"])

    async def test_doc_error_rate_below_floor_emits_no_failure_suggestion(self):
        # 5 docs total — below DOC_ERROR_MIN_TOTAL — even with all
        # failures we suppress the alarm. Avoids flagging tiny test
        # corpora.
        self.assertEqual(DOC_ERROR_MIN_TOTAL, 10)
        suggestions = await self._suggestions(
            pii_stats={"total_pii": 0, "docs_with_pii": 0, "pii_types": []},
            status_counts={"error": 5},
        )
        ids = {s["id"] for s in suggestions}
        self.assertNotIn("cleaning:doc-error-rate", ids)

    async def test_doc_error_rate_at_warning_level_emits_warning(self):
        # 10% error rate over 30 docs (3 errors / 27 accepted) — past
        # the 5% threshold but under the 20% critical cutoff.
        suggestions = await self._suggestions(
            pii_stats={"total_pii": 0, "docs_with_pii": 0, "pii_types": []},
            status_counts={"accepted": 27, "error": 3},
        )
        ids = {s["id"] for s in suggestions}
        self.assertIn("cleaning:doc-error-rate", ids)
        s = next(s for s in suggestions if s["id"] == "cleaning:doc-error-rate")
        self.assertEqual(s["severity"], "warning")
        self.assertEqual(s["action"]["kind"], "navigate")
        self.assertEqual(s["context"]["total_docs"], 30)
        self.assertEqual(s["context"]["error_count"], 3)
        self.assertAlmostEqual(s["context"]["error_rate"], 0.1)
        self.assertAlmostEqual(s["context"]["warn_threshold"], DOC_ERROR_RATE_WARN)

    async def test_doc_error_rate_above_critical_threshold_escalates(self):
        # 25% error rate — past the 20% critical cutoff.
        suggestions = await self._suggestions(
            pii_stats={"total_pii": 0, "docs_with_pii": 0, "pii_types": []},
            status_counts={"accepted": 30, "error": 10},
        )
        s = next(s for s in suggestions if s["id"] == "cleaning:doc-error-rate")
        self.assertEqual(s["severity"], "critical")


class CoachServiceGoldSetStageTests(unittest.IsolatedAsyncioTestCase):
    """Phase 2: ``_gold_set_stage_suggestions`` — exercises class
    imbalance + diversity translations + the no-recipe fallback."""

    async def _suggestions(
        self,
        *,
        gold_rows: list[dict],
        task_profile: str | None = "classification",
        with_recipe: bool = True,
    ) -> list[dict]:
        from unittest.mock import MagicMock, patch

        class _StubProject:
            id = 11
            selected_recipe = (
                {"recipe_id": "classification"} if with_recipe else None
            )

        recipe_stub = MagicMock()
        recipe_stub.task_profile = task_profile

        async def _async_rows(*_a, **_k):
            return gold_rows

        with (
            patch(
                "app.services.trainability_forecast_service._load_gold_rows",
                side_effect=_async_rows,
            ),
            patch(
                "app.services.recipe_service.get_recipe",
                return_value=recipe_stub if with_recipe else None,
            ),
        ):
            return await _gold_set_stage_suggestions(
                db=None,  # type: ignore[arg-type]
                project=_StubProject(),
            )

    async def test_no_recipe_returns_navigate_to_recipe_picker(self):
        suggestions = await self._suggestions(
            gold_rows=[], with_recipe=False
        )
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["id"], "gold_set:no-recipe")
        self.assertEqual(s["action"]["kind"], "navigate")
        self.assertEqual(s["action"]["params"]["target"], "recipe-picker")

    async def test_severe_class_imbalance_triggers_critical(self):
        # 90/10 split → very low Shannon entropy → "block" severity
        # in trainability terms → maps to coach "critical".
        gold_rows = (
            [{"question": f"q{i}", "answer": "a", "label": "billing"} for i in range(90)]
            + [{"question": f"q{i + 90}", "answer": "a", "label": "technical"} for i in range(10)]
        )
        suggestions = await self._suggestions(gold_rows=gold_rows)
        ids = {s["id"] for s in suggestions}
        self.assertIn("gold_set:class-imbalance", ids)
        s = next(s for s in suggestions if s["id"] == "gold_set:class-imbalance")
        self.assertEqual(s["severity"], "critical")
        # Action targets the under-represented label specifically + uses
        # the class_balance_fill playbook (Epic 2b).
        self.assertEqual(s["action"]["kind"], "run_playbook")
        self.assertEqual(s["action"]["params"]["mode"], "class_balance_fill")
        self.assertEqual(s["action"]["params"]["target_class"], "technical")
        self.assertEqual(
            s["action"]["params"]["target_count"],
            CLASS_BALANCE_TOPUP_DEFAULT,
        )

    async def test_balanced_classes_emit_no_imbalance_suggestion(self):
        # 3 balanced classes → Shannon entropy ≈ ln(3) ≈ 1.10, above
        # the trainability service's CLASS_ENTROPY_WARN (1.0) → "ok"
        # → coach emits no suggestion.
        # (Binary 50/50 has entropy ≈ 0.69 which the trainability
        # signal still classifies as "warn" by its own threshold —
        # Coach mirrors that semantic faithfully.)
        labels = ["billing", "technical", "shipping"]
        gold_rows = []
        for label_idx, label in enumerate(labels):
            for i in range(15):
                gold_rows.append({
                    "question": (
                        f"label-{label_idx} q{i} "
                        f"{'token' * (i % 7)} {'word' * (i % 11)} unique{label_idx}-{i}"
                    ),
                    "answer": "a",
                    "label": label,
                })
        suggestions = await self._suggestions(gold_rows=gold_rows)
        ids = {s["id"] for s in suggestions}
        self.assertNotIn("gold_set:class-imbalance", ids)

    async def test_low_diversity_emits_paraphrase_suggestion(self):
        # Identical rows → mean pairwise Jaccard = 1.0 → diversity
        # warn.
        gold_rows = [
            {"question": "what is the refund policy?", "answer": "30 days", "label": "billing"}
            for _ in range(8)
        ]
        suggestions = await self._suggestions(gold_rows=gold_rows)
        ids = {s["id"] for s in suggestions}
        self.assertIn("gold_set:diversity-low", ids)
        s = next(s for s in suggestions if s["id"] == "gold_set:diversity-low")
        self.assertEqual(s["severity"], "warning")
        self.assertEqual(s["action"]["kind"], "run_playbook")
        self.assertEqual(s["action"]["params"]["mode"], "positives_paraphrase")
        self.assertEqual(
            s["action"]["params"]["target_count"], DIVERSITY_TOPUP_DEFAULT
        )

    async def test_class_imbalance_and_diversity_can_coexist(self):
        # 95/5 skew + identical texts inside each class → both
        # signals fire. Coach surfaces both suggestions side-by-side.
        gold_rows = (
            [{"question": "billing q?", "answer": "a", "label": "billing"} for _ in range(19)]
            + [{"question": "tech q?", "answer": "a", "label": "technical"} for _ in range(1)]
        )
        suggestions = await self._suggestions(gold_rows=gold_rows)
        ids = {s["id"] for s in suggestions}
        self.assertIn("gold_set:class-imbalance", ids)
        self.assertIn("gold_set:diversity-low", ids)


if __name__ == "__main__":
    unittest.main()
