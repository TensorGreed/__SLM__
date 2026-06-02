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
    CLUSTER_AUGMENT_DEFAULT,
    DIVERSITY_TOPUP_DEFAULT,
    DOC_ERROR_MIN_TOTAL,
    DOC_ERROR_RATE_WARN,
    EVAL_PASS_RATE_CRITICAL,
    EVAL_PASS_RATE_HEALTHY,
    GOLD_ROW_COMFORTABLE_MIN,
    GOLD_ROW_THIN_MAX,
    SUGGESTED_TOPUP_CEILING,
    SUGGESTED_TOPUP_FLOOR,
    _cleaning_stage_suggestions,
    _data_stage_suggestions,
    _eval_stage_suggestions,
    _gold_set_stage_suggestions,
    _topup_count,
    _training_stage_suggestions,
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
        # Arc 4 — rule_id distinguishes which side of the rule
        # tripped. Thin gold lands on ``.thin``.
        self.assertEqual(s["rule_id"], "gold-row-count.thin")
        # Topup target threaded through the context for the trace.
        self.assertIn("topup_target", s["context"])
        self.assertGreater(s["context"]["topup_target"], 0)

    async def test_mid_tier_gold_triggers_warning_severity(self):
        suggestions = await self._suggestions_for_row_count(180)
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["severity"], "warning")
        self.assertEqual(s["context"]["gold_row_count"], 180)
        # Mid-tier lands on the ``.below-comfortable`` branch.
        self.assertEqual(s["rule_id"], "gold-row-count.below-comfortable")

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
        review_queue: dict | None = None,
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

        # Default to an empty queue — most existing tests don't care
        # about the synth-review-pending suggestion, so they should
        # not see it.
        queue_payload = review_queue if review_queue is not None else {
            "project_id": 11,
            "total_pending": 0,
            "total_accepted": 0,
            "groups": [],
            "accepted_groups": [],
        }

        async def _async_queue(*_a, **_k):
            return queue_payload

        with (
            patch(
                "app.services.trainability_forecast_service._load_gold_rows",
                side_effect=_async_rows,
            ),
            patch(
                "app.services.recipe_service.get_recipe",
                return_value=recipe_stub if with_recipe else None,
            ),
            patch(
                "app.services.synth_review_queue_service.list_review_queue",
                side_effect=_async_queue,
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
        # Arc 4 — rule_id stamps the "block" branch + the trace
        # context carries the underrepresented target_class +
        # topup_target so the UI can render the full decision.
        self.assertEqual(s["rule_id"], "class-imbalance.block")
        self.assertEqual(s["context"]["target_class"], "technical")
        self.assertEqual(
            s["context"]["topup_target"], CLASS_BALANCE_TOPUP_DEFAULT,
        )

    async def test_class_imbalance_auto_pins_schema_aware_backend_when_available(self):
        """Phase 5c: when a schema-aware backend (vllm or nemo) is
        configured + reachable, Coach's class_balance_fill suggestion
        stamps a ``backend`` pin on the action so the click-to-execute
        flow gets constrained decoding for free. Without that pin, the
        orchestrator would auto-pick Ollama (the registry's first
        available) and silently lose the schema constraint."""
        from unittest.mock import patch

        from app.services.synth_backends import (
            NemoBackend,
            OllamaBackend,
            TeacherModelBackend,
            VllmBackend,
        )

        gold_rows = (
            [{"question": f"q{i}", "answer": "a", "label": "billing"} for i in range(90)]
            + [{"question": f"q{i + 90}", "answer": "a", "label": "technical"} for i in range(10)]
        )
        # Both vllm + nemo are reachable; vllm wins by preference.
        with (
            patch.object(VllmBackend, "is_available", return_value=True),
            patch.object(VllmBackend, "describe", return_value="vllm:llama-3.1-8b"),
            patch.object(NemoBackend, "is_available", return_value=True),
            patch.object(OllamaBackend, "is_available", return_value=True),
            patch.object(TeacherModelBackend, "is_available", return_value=False),
        ):
            suggestions = await self._suggestions(gold_rows=gold_rows)
        s = next(s for s in suggestions if s["id"] == "gold_set:class-imbalance")
        # The pin is on the action params (so CoachSuggestion.tsx
        # forwards it to runPlaybook) AND surfaced on context (so the
        # UI can explain the auto-upgrade if it ever wants to).
        self.assertEqual(s["action"]["params"]["backend"], "vllm:llama-3.1-8b")
        self.assertEqual(
            s["context"]["schema_aware_backend"], "vllm:llama-3.1-8b"
        )

    async def test_class_imbalance_omits_backend_pin_when_no_schema_aware_available(self):
        """Backwards-compat: existing installs with only Ollama keep
        the same suggestion shape — no backend pin → orchestrator
        auto-pick (Ollama) runs the playbook unchanged."""
        from unittest.mock import patch

        from app.services.synth_backends import (
            NemoBackend,
            OllamaBackend,
            VllmBackend,
        )

        gold_rows = (
            [{"question": f"q{i}", "answer": "a", "label": "billing"} for i in range(90)]
            + [{"question": f"q{i + 90}", "answer": "a", "label": "technical"} for i in range(10)]
        )
        with (
            patch.object(OllamaBackend, "is_available", return_value=True),
            patch.object(NemoBackend, "is_available", return_value=False),
            patch.object(VllmBackend, "is_available", return_value=False),
        ):
            suggestions = await self._suggestions(gold_rows=gold_rows)
        s = next(s for s in suggestions if s["id"] == "gold_set:class-imbalance")
        self.assertNotIn("backend", s["action"]["params"])
        # Context still carries the field for UI consistency, just null.
        self.assertIsNone(s["context"].get("schema_aware_backend"))

    async def test_synth_review_pending_emits_navigate_action(self):
        """When the synth review queue has pending rows, gold_set
        Coach surfaces a one-click ``navigate`` action so the user
        doesn't lose work to the dataset-prep gate."""
        # Balanced gold so no imbalance/diversity suggestions fire +
        # we can isolate the pending-review-queue case.
        gold_rows = (
            [{"question": f"q{i}", "answer": "a", "label": "billing"} for i in range(15)]
            + [{"question": f"q{i + 15}", "answer": "a", "label": "technical"} for i in range(15)]
            + [{"question": f"q{i + 30}", "answer": "a", "label": "shipping"} for i in range(15)]
        )
        # Add unique tokens so diversity warn doesn't trip.
        for i, row in enumerate(gold_rows):
            row["question"] = f"unique{i} " + row["question"]
        queue = {
            "project_id": 11,
            "total_pending": 12,
            "total_accepted": 0,
            "groups": [
                {
                    "synth_source": "playbook:classification:class_balance_fill:class=technical",
                    "count": 12,
                    "rows": [],
                }
            ],
            "accepted_groups": [],
        }
        suggestions = await self._suggestions(
            gold_rows=gold_rows, review_queue=queue
        )
        s = next(
            (s for s in suggestions if s["id"] == "gold_set:synth-review-pending"),
            None,
        )
        self.assertIsNotNone(s, "expected synth-review-pending suggestion")
        self.assertEqual(s["action"]["kind"], "navigate")
        self.assertEqual(
            s["action"]["params"]["target"], "synthetic-review-queue"
        )
        # Phase 5c — the top source bucket is stamped onto the action
        # params so CoachSuggestion.tsx can build a focused URL
        # (?focus_synth_source=...) and SynthReviewQueue can render
        # its one-click "Accept all N" banner.
        self.assertEqual(
            s["action"]["params"]["synth_source"],
            "playbook:classification:class_balance_fill:class=technical",
        )
        # 12 pending → severity escalates to "warning" (threshold ≥ 5).
        self.assertEqual(s["severity"], "warning")
        # Title surfaces the count so the user knows the scale.
        self.assertIn("12", s["title"])
        # Body mentions the class_balance_fill source so the user
        # connects this back to the suggestion they ran.
        self.assertIn("class-imbalance", s["body"].lower())
        # Context carries the raw fields for the UI / future signals.
        self.assertEqual(s["context"]["total_pending"], 12)

    async def test_synth_review_low_count_emits_info_severity(self):
        """1-4 pending rows is just a nudge, not a warning."""
        gold_rows = (
            [{"question": f"unique{i} q{i}", "answer": "a", "label": "billing"} for i in range(15)]
            + [{"question": f"unique{i + 15} q{i + 15}", "answer": "a", "label": "technical"} for i in range(15)]
            + [{"question": f"unique{i + 30} q{i + 30}", "answer": "a", "label": "shipping"} for i in range(15)]
        )
        queue = {
            "project_id": 11,
            "total_pending": 3,
            "total_accepted": 0,
            "groups": [
                {"synth_source": "playbook:qa-sft:positives_paraphrase", "count": 3, "rows": []}
            ],
            "accepted_groups": [],
        }
        suggestions = await self._suggestions(
            gold_rows=gold_rows, review_queue=queue
        )
        s = next(
            s for s in suggestions if s["id"] == "gold_set:synth-review-pending"
        )
        self.assertEqual(s["severity"], "info")
        # Body should reference the actual source bucket since it's
        # not a class_balance_fill run.
        self.assertIn("positives_paraphrase", s["body"])

    async def test_synth_review_empty_queue_emits_no_suggestion(self):
        """Empty queue → no suggestion (the default queue mock in
        ``_suggestions`` exercises this case)."""
        gold_rows = (
            [{"question": f"unique{i} q{i}", "answer": "a", "label": "billing"} for i in range(15)]
            + [{"question": f"unique{i + 15} q{i + 15}", "answer": "a", "label": "technical"} for i in range(15)]
            + [{"question": f"unique{i + 30} q{i + 30}", "answer": "a", "label": "shipping"} for i in range(15)]
        )
        suggestions = await self._suggestions(gold_rows=gold_rows)
        ids = {s["id"] for s in suggestions}
        self.assertNotIn("gold_set:synth-review-pending", ids)

    async def test_synth_review_pending_omits_synth_source_when_no_groups(self):
        """Edge: queue reports total_pending > 0 but groups is empty
        (e.g. legacy rows without synth_source). The action stays
        actionable but params.synth_source is omitted; SynthReviewQueue
        won't render the focused banner."""
        gold_rows = (
            [{"question": f"unique{i} q{i}", "answer": "a", "label": "billing"} for i in range(15)]
            + [{"question": f"unique{i + 15} q{i + 15}", "answer": "a", "label": "technical"} for i in range(15)]
            + [{"question": f"unique{i + 30} q{i + 30}", "answer": "a", "label": "shipping"} for i in range(15)]
        )
        queue = {
            "project_id": 11,
            "total_pending": 4,
            "total_accepted": 0,
            "groups": [],  # legacy / source-less pending rows
            "accepted_groups": [],
        }
        suggestions = await self._suggestions(
            gold_rows=gold_rows, review_queue=queue
        )
        s = next(
            s for s in suggestions if s["id"] == "gold_set:synth-review-pending"
        )
        # Action still fires + still navigates (the queue UI itself
        # can handle a focus-less landing) but synth_source is absent.
        self.assertEqual(
            s["action"]["params"]["target"], "synthetic-review-queue"
        )
        self.assertNotIn("synth_source", s["action"]["params"])
        self.assertIsNone(s["context"]["top_source"])

    async def test_synth_review_queue_read_failure_does_not_block_other_suggestions(self):
        """A queue read that throws must not crash the gold_set strip.
        Other suggestions (imbalance, diversity) still emit."""
        from unittest.mock import MagicMock, patch

        class _StubProject:
            id = 11
            selected_recipe = {"recipe_id": "classification"}

        recipe_stub = MagicMock()
        recipe_stub.task_profile = "classification"
        gold_rows = (
            [{"question": f"q{i}", "answer": "a", "label": "billing"} for i in range(90)]
            + [{"question": f"q{i + 90}", "answer": "a", "label": "technical"} for i in range(10)]
        )

        async def _async_rows(*_a, **_k):
            return gold_rows

        async def _boom(*_a, **_k):
            raise RuntimeError("disk error")

        with (
            patch(
                "app.services.trainability_forecast_service._load_gold_rows",
                side_effect=_async_rows,
            ),
            patch(
                "app.services.recipe_service.get_recipe",
                return_value=recipe_stub,
            ),
            patch(
                "app.services.synth_review_queue_service.list_review_queue",
                side_effect=_boom,
            ),
        ):
            suggestions = await _gold_set_stage_suggestions(
                db=None, project=_StubProject()  # type: ignore[arg-type]
            )
        # The imbalance suggestion still fires; the pending-review
        # suggestion is silently skipped.
        ids = {s["id"] for s in suggestions}
        self.assertIn("gold_set:class-imbalance", ids)
        self.assertNotIn("gold_set:synth-review-pending", ids)

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


class CoachServiceTrainingStageTests(unittest.IsolatedAsyncioTestCase):
    """Phase 3: ``_training_stage_suggestions`` — exercises the
    likely_fail / borderline / likely_pass branches + the
    heavier-alt-base-model picker."""

    async def _suggestions(
        self,
        *,
        overall: str,
        confidence_pct: int = 30,
        signals: list[dict] | None = None,
        with_recipe: bool = True,
        alt_base_models: list[str] | None = None,
        current_base: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    ) -> list[dict]:
        from unittest.mock import MagicMock, patch

        class _StubProject:
            # Use a deliberately-high project_id so other tests in
            # the same pytest run can't accidentally seed a
            # ``prepared/train.jsonl`` at the same path and trigger
            # the Phase 6d curriculum nudge (which reads from
            # ``settings.DATA_DIR/projects/<id>/prepared/train.jsonl``
            # for thin-classification projects). Project IDs assigned
            # by TestClient template-instantiation start at 1 and
            # auto-increment; 999021 is comfortably above any
            # realistic test-run count.
            id = 999021
            base_model_name = current_base
            selected_recipe = (
                {"recipe_id": "classification"} if with_recipe else None
            )

        recipe_stub = MagicMock()
        recipe_stub.suggested_base_model = current_base
        recipe_stub.alt_base_models = list(alt_base_models or [])

        async def _async_forecast(*_a, **_k):
            return {
                "overall": overall,
                "confidence_pct": confidence_pct,
                "signals": signals or [],
                "computed_at": "2026-05-23T00:00:00",
                "cache_key": "stub",
                "cache_hit": False,
            }

        async def _no_sweep_nudge(*_a, **_k):
            # Default off for the focused forecast tests in this class —
            # the sweep nudge has its own dedicated test class below.
            # Without this, every training-stage test would have to seed
            # a sweep to keep the assertion counts honest.
            return None

        with (
            patch(
                "app.services.trainability_forecast_service.forecast_training",
                side_effect=_async_forecast,
            ),
            patch(
                "app.services.recipe_service.get_recipe",
                return_value=recipe_stub if with_recipe else None,
            ),
            patch(
                "app.services.coach_service._inconclusive_sweep_nudge",
                side_effect=_no_sweep_nudge,
            ),
        ):
            return await _training_stage_suggestions(
                db=None,  # type: ignore[arg-type]
                project=_StubProject(),
            )

    async def test_no_recipe_returns_navigate_to_recipe_picker(self):
        suggestions = await self._suggestions(
            overall="likely_fail", with_recipe=False
        )
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["id"], "training:no-recipe")
        self.assertEqual(s["action"]["kind"], "navigate")
        self.assertEqual(s["action"]["params"]["target"], "recipe-picker")

    async def test_likely_pass_emits_no_suggestion(self):
        suggestions = await self._suggestions(
            overall="likely_pass", confidence_pct=85
        )
        self.assertEqual(suggestions, [])

    async def test_likely_fail_emits_critical_with_base_model_hint(self):
        suggestions = await self._suggestions(
            overall="likely_fail",
            confidence_pct=18,
            alt_base_models=[
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
            ],
            signals=[{
                "id": "row_count_below_minimum",
                "severity": "block",
                "headline": "Train corpus has 45 rows; recipe needs ≥ 100.",
                "detail": "Add rows or move to a recipe with a lower floor.",
                "suggested_action": None,
            }],
        )
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["id"], "training:trainability-forecast")
        self.assertEqual(s["severity"], "critical")
        # Picks the lighter heavier-alt first (0.5B before 3B).
        self.assertEqual(
            s["context"]["recommended_base_model"],
            "Qwen/Qwen2.5-0.5B-Instruct",
        )
        self.assertEqual(s["action"]["kind"], "navigate")
        self.assertEqual(
            s["action"]["params"]["target"], "training-base-model-picker"
        )
        # Body carries the dominant blocker's headline so the user
        # sees the "why" without expanding the forecast panel.
        self.assertIn("45 rows", s["body"])
        # Confidence_pct round-trips into the context bag for the UI.
        self.assertEqual(s["context"]["confidence_pct"], 18)
        self.assertEqual(s["context"]["blocker_signal_id"], "row_count_below_minimum")

    async def test_borderline_emits_warning_severity(self):
        suggestions = await self._suggestions(
            overall="borderline",
            confidence_pct=52,
            alt_base_models=["Qwen/Qwen2.5-0.5B-Instruct"],
        )
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["severity"], "warning")
        self.assertEqual(suggestions[0]["context"]["overall"], "borderline")

    async def test_no_heavier_alt_falls_back_to_forecast_navigation(self):
        # Current base is already the heaviest alt — Coach must still
        # surface a suggestion (the forecast is bad) but skip the
        # base-model hint and route to the forecast panel instead.
        suggestions = await self._suggestions(
            overall="likely_fail",
            current_base="Qwen/Qwen2.5-3B-Instruct",
            alt_base_models=[
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "Qwen/Qwen2.5-0.5B-Instruct",
            ],
        )
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertIsNone(s["context"]["recommended_base_model"])
        self.assertEqual(s["action"]["params"]["target"], "trainability-forecast")

    async def test_forecast_raises_value_error_returns_empty(self):
        # The forecast service raises ValueError if the project
        # disappears mid-call. Coach must swallow it cleanly so the
        # 500 doesn't leak to the UI.
        from unittest.mock import patch

        class _StubProject:
            id = 99
            base_model_name = "x"
            selected_recipe = {"recipe_id": "classification"}

        async def _raise(*_a, **_k):
            raise ValueError("project gone")

        with patch(
            "app.services.trainability_forecast_service.forecast_training",
            side_effect=_raise,
        ):
            result = await _training_stage_suggestions(
                db=None,  # type: ignore[arg-type]
                project=_StubProject(),
            )
        self.assertEqual(result, [])


class CoachServiceSweepInconclusiveNudgeTests(unittest.IsolatedAsyncioTestCase):
    """Direct calls to ``_inconclusive_sweep_nudge``: surfaces a coach
    card when the latest sweep's verdict is inconclusive, stays silent
    on promote / pending / no-sweep."""

    def _fake_db_with_latest_sweep(self, sweep_id: str | None):
        """Build a fake AsyncSession whose ``execute(...)`` returns a
        result yielding one experiment carrying ``sweep_id`` in its
        ``_sweep`` config (or yielding nothing when sweep_id is None).

        The coach nudge first runs a ``select(Sweep).limit(1)`` query
        (which calls ``.scalar_one_or_none()`` on the result); we return
        ``None`` from that so the code falls through to the legacy
        breadcrumb scan, which is what these tests are written against.
        Once the Sweep table is populated everywhere, the same stub
        can return a real Sweep record instead.
        """
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        rows: list[SimpleNamespace] = []
        if sweep_id:
            rows.append(SimpleNamespace(
                id=1,
                config={"_sweep": {"sweep_id": sweep_id, "label": "r8", "cell_index": 0}},
            ))

        class _ScalarsResult:
            def __init__(self, items): self._items = items
            def __iter__(self): return iter(self._items)

        class _ExecuteResult:
            def __init__(self, items): self._items = items
            def scalars(self): return _ScalarsResult(self._items)
            # First call (the Sweep lookup) reads scalar_one_or_none ->
            # None forces the legacy fallback path.
            def scalar_one_or_none(self): return None

        db = SimpleNamespace()
        db.execute = AsyncMock(return_value=_ExecuteResult(rows))
        return db

    async def _run(self, *, sweep_id: str | None, pareto: dict | None):
        from unittest.mock import patch
        from app.services.coach_service import _inconclusive_sweep_nudge

        async def _stub_pareto(_db, _pid, _sid):
            if pareto is None:
                raise ValueError("No sweep")
            return pareto

        with patch(
            "app.services.hyperparameter_sweep_service.get_sweep_pareto",
            side_effect=_stub_pareto,
        ):
            return await _inconclusive_sweep_nudge(
                self._fake_db_with_latest_sweep(sweep_id),
                project_id=42,
            )

    async def test_no_sweep_returns_none(self):
        result = await self._run(sweep_id=None, pareto=None)
        self.assertIsNone(result)

    async def test_inconclusive_surfaces_warning_with_failure_clusters_navigate(self):
        pareto = {
            "sweep_id": "sweepabc",
            "verdict": "inconclusive",
            "verdict_reason": "No completed cell cleared the project gate.",
            "cell_count": 4,
            "gate_summary": {"pack_id": "evalpack.demo", "measurable_count": 4},
            "cells": [
                {"label": "r8", "gate_passed": False, "gate_failed_ids": ["acc_gte_0.8"]},
                {"label": "r16", "gate_passed": False, "gate_failed_ids": ["acc_gte_0.8", "f1_gte_0.7"]},
                {"label": "r32", "gate_passed": False, "gate_failed_ids": ["acc_gte_0.8"]},
                {"label": "r64", "gate_passed": False, "gate_failed_ids": ["acc_gte_0.8"]},
            ],
        }
        result = await self._run(sweep_id="sweepabc", pareto=pareto)
        self.assertIsNotNone(result)
        assert result is not None  # for type narrowing
        self.assertEqual(result["id"], "training:sweep-inconclusive")
        self.assertEqual(result["severity"], "warning")
        self.assertEqual(result["action"]["kind"], "navigate")
        self.assertEqual(result["action"]["params"]["target"], "failure-clusters-panel")
        # Body names the dominant gate + cell count (4 cells missed acc_gte_0.8).
        self.assertIn("acc_gte_0.8 (4 cells)", result["body"])
        self.assertIn("4/4 measurable", result["body"])
        # Context payload carries the sweep id + dedup'd gate counts for downstream telemetry.
        self.assertEqual(result["context"]["sweep_id"], "sweepabc")
        failed = dict(result["context"]["failed_gates"])
        self.assertEqual(failed["acc_gte_0.8"], 4)
        self.assertEqual(failed["f1_gte_0.7"], 1)

    async def test_promote_verdict_returns_none(self):
        pareto = {
            "sweep_id": "sweepabc",
            "verdict": "promote",
            "cell_count": 2,
            "gate_summary": {"pack_id": "evalpack.demo", "measurable_count": 2},
            "cells": [{"label": "r8", "gate_passed": True, "gate_failed_ids": []}],
        }
        result = await self._run(sweep_id="sweepabc", pareto=pareto)
        # Promote is success — the sweep panel already shows it. Coach stays silent.
        self.assertIsNone(result)

    async def test_pending_verdict_returns_none(self):
        pareto = {
            "sweep_id": "sweepabc",
            "verdict": "pending",
            "cell_count": 4,
            "gate_summary": {"pack_id": "evalpack.demo", "measurable_count": 0},
            "cells": [],
        }
        result = await self._run(sweep_id="sweepabc", pareto=pareto)
        # Pending = still training; nothing actionable yet.
        self.assertIsNone(result)

    async def test_pareto_lookup_failure_returns_none(self):
        # get_sweep_pareto raised — the latest sweep id may have been
        # deleted between the DB query and the Pareto call. Surface
        # nothing rather than 500ing the coach response.
        result = await self._run(sweep_id="ghost", pareto=None)
        self.assertIsNone(result)

    async def test_uses_first_class_sweep_row_when_available(self):
        """Primary path: when a Sweep row exists, the nudge looks it up
        in one query instead of scanning experiments. This test stubs
        ``scalar_one_or_none`` to return a Sweep so the legacy fallback
        branch is provably bypassed."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from app.services.coach_service import _inconclusive_sweep_nudge

        fake_sweep = SimpleNamespace(sweep_id="sweep-from-table", id=99)

        # First execute call (the Sweep lookup) yields the fake sweep
        # row. Every other execute call also returns it, but the nudge
        # only inspects scalar_one_or_none for the Sweep query and
        # scalars() for the legacy fallback — neither of which should
        # fire because the primary path returns the Sweep up front.
        class _ExecuteResult:
            def __init__(self): self._sweep = fake_sweep
            def scalar_one_or_none(self): return self._sweep
            def scalars(self):
                # If this fires we've gone down the legacy path — test
                # would observe by checking the sweep_id passed to
                # get_sweep_pareto below.
                return iter([])

        db = SimpleNamespace()
        db.execute = AsyncMock(return_value=_ExecuteResult())

        observed_sweep_ids: list[str] = []

        async def _stub_pareto(_db, _pid, sid):
            observed_sweep_ids.append(sid)
            return {
                "sweep_id": sid,
                "verdict": "inconclusive",
                "verdict_reason": "No completed cell cleared the project gate.",
                "cell_count": 1,
                "gate_summary": {"pack_id": "evalpack.demo", "measurable_count": 1},
                "cells": [{"label": "r8", "gate_passed": False, "gate_failed_ids": ["acc"]}],
            }

        with patch(
            "app.services.hyperparameter_sweep_service.get_sweep_pareto",
            side_effect=_stub_pareto,
        ):
            result = await _inconclusive_sweep_nudge(db, project_id=42)

        self.assertIsNotNone(result)
        # Hit the Sweep-table token, not a legacy breadcrumb.
        self.assertEqual(observed_sweep_ids, ["sweep-from-table"])


class CoachServiceEvalStageTests(unittest.IsolatedAsyncioTestCase):
    """Phase 3: ``_eval_stage_suggestions`` — exercises the no-eval /
    healthy / critical-with-cluster / critical-without-cluster
    branches."""

    async def _suggestions(
        self,
        *,
        latest_eval: Any | None,
        clusters: list[dict] | None = None,
        cluster_raises: bool = False,
    ) -> list[dict]:
        from unittest.mock import patch

        class _StubProject:
            id = 31
            selected_recipe = {"recipe_id": "classification"}

        async def _async_latest(*_a, **_k):
            return latest_eval

        async def _async_clusters(*_a, **_k):
            if cluster_raises:
                raise ValueError("eval_result_not_found")
            return {
                "eval_result_id": getattr(latest_eval, "id", 0),
                "experiment_id": None,
                "dataset_name": "test",
                "eval_type": "f1",
                "total_failures_analyzed": sum(
                    int(c.get("failure_count") or 0)
                    for c in (clusters or [])
                ),
                "reason_code_totals": {},
                "dominant_reason_code": None,
                "clusters": list(clusters or []),
                "remediation_plans": [],
            }

        with (
            patch(
                "app.services.coach_service._read_latest_eval_result",
                side_effect=_async_latest,
            ),
            patch(
                "app.services.failure_cluster_service.cluster_eval_result_failures",
                side_effect=_async_clusters,
            ),
        ):
            return await _eval_stage_suggestions(
                db=None,  # type: ignore[arg-type]
                project=_StubProject(),
            )

    def _make_eval_row(self, *, eval_id: int, pass_rate: float | None):
        # Lightweight stand-in for the EvalResult ORM row — the coach
        # handler only touches ``id`` + ``pass_rate``. ``SimpleNamespace``
        # gives us attribute access without the Python class-body
        # closure trap (class bodies don't capture method locals).
        from types import SimpleNamespace
        return SimpleNamespace(id=eval_id, pass_rate=pass_rate)

    async def test_no_eval_yet_emits_nothing(self):
        result = await self._suggestions(latest_eval=None)
        self.assertEqual(result, [])

    async def test_healthy_pass_rate_emits_nothing(self):
        result = await self._suggestions(
            latest_eval=self._make_eval_row(
                eval_id=1, pass_rate=EVAL_PASS_RATE_HEALTHY + 0.02
            ),
        )
        self.assertEqual(result, [])

    async def test_warning_when_below_healthy_but_above_critical(self):
        # Mid pass rate: warning severity + cluster-targeted action.
        clusters = [
            {
                "cluster_id": "cluster-1",
                "reason_code": "hallucination",
                "failure_count": 12,
                "share_of_total": 0.45,
                "classifier_confidence": 0.8,
                "classifier_reason": "Model invents facts not in source.",
                "output_pattern": "p1",
                "exemplars": [],
            },
            {
                "cluster_id": "cluster-2",
                "reason_code": "coverage_gap",
                "failure_count": 5,
                "share_of_total": 0.20,
                "classifier_confidence": 0.6,
                "classifier_reason": "Model doesn't know concept.",
                "output_pattern": "p2",
                "exemplars": [],
            },
        ]
        result = await self._suggestions(
            latest_eval=self._make_eval_row(eval_id=42, pass_rate=0.75),
            clusters=clusters,
        )
        self.assertEqual(len(result), 1)
        s = result[0]
        self.assertEqual(s["id"], "eval:top-failure-cluster")
        self.assertEqual(s["severity"], "warning")
        self.assertEqual(s["action"]["kind"], "augment_from_cluster")
        # Largest cluster wins (cluster-1 with 12 failures).
        self.assertEqual(s["action"]["params"]["eval_result_id"], 42)
        self.assertEqual(s["action"]["params"]["cluster_id"], "cluster-1")
        self.assertEqual(
            s["action"]["params"]["target_count"], CLUSTER_AUGMENT_DEFAULT
        )
        self.assertEqual(s["context"]["reason_code"], "hallucination")
        self.assertEqual(s["context"]["failure_count"], 12)

    async def test_critical_when_pass_rate_below_critical_threshold(self):
        result = await self._suggestions(
            latest_eval=self._make_eval_row(
                eval_id=7, pass_rate=EVAL_PASS_RATE_CRITICAL - 0.1
            ),
            clusters=[{
                "cluster_id": "cluster-X",
                "reason_code": "formatting_mismatch",
                "failure_count": 30,
                "share_of_total": 0.80,
                "classifier_confidence": 0.9,
                "classifier_reason": "Wrong output shape.",
                "output_pattern": "p",
                "exemplars": [],
            }],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["severity"], "critical")

    async def test_no_clusters_falls_back_to_navigate_predictions(self):
        # Below-threshold pass rate but clustering returned nothing —
        # Coach softens to a navigate suggestion rather than nothing.
        result = await self._suggestions(
            latest_eval=self._make_eval_row(eval_id=99, pass_rate=0.55),
            clusters=[],
        )
        self.assertEqual(len(result), 1)
        s = result[0]
        self.assertEqual(s["id"], "eval:low-pass-rate-no-clusters")
        self.assertEqual(s["action"]["kind"], "navigate")
        self.assertEqual(s["action"]["params"]["target"], "eval-predictions")

    async def test_cluster_lookup_value_error_returns_empty(self):
        # Eval row deleted between the read + the cluster call — Coach
        # swallows the ValueError and emits nothing.
        result = await self._suggestions(
            latest_eval=self._make_eval_row(eval_id=5, pass_rate=0.3),
            cluster_raises=True,
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
