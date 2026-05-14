"""Gamification (Lab Journal) progression service.

Pins:
- XP curve at L1, L5, L10 anchors.
- ``award_xp`` increments balance + level on crossing the threshold.
- ``check_and_unlock`` is idempotent — firing twice grants once.
- ``process_run_event`` translates a ``dataset_import_run`` row into
  a ``first_ingest`` unlock + per-event XP drip.
- F1 ladder: a single 0.95 eval event unlocks all three f1_* tiers.
- Toast-spam buffer suppresses repeat continuous-event toasts within
  the 30s window.
- API endpoint round-trip via TestClient.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.services import gamification_service as gs  # noqa: E402
from app.services.gamification.achievements import (  # noqa: E402
    ACHIEVEMENTS,
    ACHIEVEMENT_BY_ID,
    level_title,
)


TEST_DATA_DIR = Path(tempfile.gettempdir()) / f"brewslm-gamification-{uuid.uuid4().hex[:8]}"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class GamificationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def setUp(self):
        # Each test gets a fresh project so the JSON column starts clean.
        resp = self.client.post(
            "/api/projects",
            json={"name": f"gamification-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.project_id = int(resp.json()["id"])
        # Toast-spam buffer is process-local; clear between tests so
        # repeated continuous events still produce toasts.
        with gs._RECENT_GRANTS_LOCK:
            gs._RECENT_GRANTS.clear()

    # ── Curve ───────────────────────────────────────────────────────

    def test_xp_curve_anchors(self):
        # L1→2 costs 100; L2→3 ≈ 282; L5→6 ≈ 1118; L10→11 ≈ 3162.
        self.assertEqual(gs.xp_to_next(1), 100)
        self.assertEqual(gs.xp_to_next(2), 282)
        self.assertEqual(gs.xp_to_next(5), 1118)
        self.assertEqual(gs.xp_to_next(10), 3162)

    def test_level_for_total_xp_walks_thresholds(self):
        # 0 XP → L1 with 100 to next.
        self.assertEqual(gs.level_for_total_xp(0), (1, 0, 100))
        # Just below the L1→L2 threshold.
        self.assertEqual(gs.level_for_total_xp(99), (1, 99, 100))
        # Exactly at the threshold lifts to L2 with 0 into level.
        self.assertEqual(gs.level_for_total_xp(100), (2, 0, 282))
        # 100 + 282 = 382 lifts to L3.
        level, into, to_next = gs.level_for_total_xp(382)
        self.assertEqual(level, 3)
        self.assertEqual(into, 0)
        self.assertGreater(to_next, 0)

    # ── award_xp + level transitions ────────────────────────────────

    async def _award(self, project_id, amount, reason="t"):
        async with async_session_factory() as db:
            result = await gs.award_xp(db, project_id, amount, reason)
            await db.commit()
            return result

    async def _state(self, project_id):
        async with async_session_factory() as db:
            return await gs.get_progression(db, project_id)

    def test_award_xp_increments_balance_and_crosses_level(self):
        # Start at 0/L1; +100 XP lifts to L2 with the level-up entry in
        # recent_unlocks.
        result = _run(self._award(self.project_id, 100, "test_award"))
        self.assertEqual(result["xp_balance"], 100)
        self.assertEqual(result["level"], 2)
        # recent_unlocks gets a level_up entry on level change.
        kinds = [entry.get("kind") for entry in result["recent_unlocks"]]
        self.assertIn("level_up", kinds)
        self.assertEqual(result["level_title"], level_title(2))

    def test_award_xp_zero_amount_is_no_op(self):
        before = _run(self._state(self.project_id))
        _run(self._award(self.project_id, 0, "zero"))
        after = _run(self._state(self.project_id))
        self.assertEqual(before["xp_balance"], after["xp_balance"])

    # ── check_and_unlock idempotency ────────────────────────────────

    async def _unlock(self, project_id, achievement_id):
        async with async_session_factory() as db:
            result = await gs.check_and_unlock(db, project_id, achievement_id)
            await db.commit()
            return result

    def test_check_and_unlock_idempotent(self):
        first = _run(self._unlock(self.project_id, "first_train"))
        self.assertIsNotNone(first)
        self.assertEqual(first["achievement_id"], "first_train")
        first_xp = ACHIEVEMENT_BY_ID["first_train"].xp
        state_after_first = _run(self._state(self.project_id))
        self.assertEqual(state_after_first["xp_balance"], first_xp)

        # Second call is a no-op.
        second = _run(self._unlock(self.project_id, "first_train"))
        self.assertIsNone(second)
        state_after_second = _run(self._state(self.project_id))
        self.assertEqual(state_after_second["xp_balance"], first_xp)
        self.assertEqual(
            state_after_second["achievements_unlocked"].count("first_train"), 1
        )

    def test_check_and_unlock_unknown_id_is_no_op(self):
        result = _run(self._unlock(self.project_id, "totally_fake_id"))
        self.assertIsNone(result)
        state = _run(self._state(self.project_id))
        self.assertEqual(state["xp_balance"], 0)

    # ── process_run_event dispatch ──────────────────────────────────

    async def _process(self, event):
        async with async_session_factory() as db:
            await gs.process_run_event(db, event)
            await db.commit()

    def _fake_event(self, **overrides):
        # Minimal duck-typed RunEvent — process_run_event reads
        # attributes via getattr so a SimpleNamespace is sufficient.
        defaults = {
            "project_id": self.project_id,
            "stage": "ingestion",
            "severity": "info",
            "reason_code": None,
            "run_id": "",
            "payload": {},
            "ts": datetime.now(timezone.utc),
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_dataset_import_event_unlocks_first_ingest_and_drips_xp(self):
        event = self._fake_event(
            reason_code="dataset_import_run",
            payload={
                "source_id": "jsonl",
                "locator": "jsonl:/tmp/x.jsonl",
                "mapper_id": "label_to_classification",
                "accepted_count": 12,
            },
        )
        _run(self._process(event))
        state = _run(self._state(self.project_id))
        self.assertIn("first_ingest", state["achievements_unlocked"])
        # 20 XP drip + 100 from first_ingest unlock = 120 total.
        self.assertEqual(state["xp_balance"], 120)

    def test_multi_dataset_unlocks_after_three_distinct_sources(self):
        for src in ("jsonl", "csv", "hf"):
            event = self._fake_event(
                reason_code="dataset_import_run",
                payload={"source_id": src, "mapper_id": "text_only"},
            )
            _run(self._process(event))
        state = _run(self._state(self.project_id))
        self.assertIn("multi_dataset", state["achievements_unlocked"])

    def test_saved_mapping_reused_unlocks_on_config_id(self):
        event = self._fake_event(
            reason_code="dataset_import_run",
            payload={
                "source_id": "jsonl",
                "mapper_id": "text_only",
                "config_id": 42,
            },
        )
        _run(self._process(event))
        state = _run(self._state(self.project_id))
        self.assertIn("saved_mapping_reused", state["achievements_unlocked"])

    def test_training_event_unlocks_first_train_and_counts(self):
        event = self._fake_event(
            stage="training",
            severity="info",
            run_id="exp-7",
            payload={"experiment_id": 7, "base_model": "llama3-8b"},
        )
        _run(self._process(event))
        state = _run(self._state(self.project_id))
        self.assertIn("first_train", state["achievements_unlocked"])
        self.assertEqual(state["counters"]["successful_training_runs"], 1)
        self.assertIn("llama3-8b", state["counters"]["base_models_trained"])

    def test_ten_trainings_milestone(self):
        for i in range(10):
            event = self._fake_event(
                stage="training",
                severity="info",
                run_id=f"exp-{i}",
                payload={"experiment_id": i, "base_model": "llama3-8b"},
            )
            _run(self._process(event))
        state = _run(self._state(self.project_id))
        self.assertIn("ten_trainings", state["achievements_unlocked"])

    def test_multi_model_after_three_distinct_base_models(self):
        for idx, model in enumerate(("llama3-8b", "qwen2-7b", "mistral-7b")):
            event = self._fake_event(
                stage="training",
                severity="info",
                run_id=f"exp-{idx}",
                payload={"experiment_id": idx, "base_model": model},
            )
            _run(self._process(event))
        state = _run(self._state(self.project_id))
        self.assertIn("multi_model", state["achievements_unlocked"])

    def test_eval_f1_ladder_unlocks_all_tiers_on_high_pass_rate(self):
        # A single 0.95 eval should unlock f1_above_80, f1_above_90,
        # and f1_above_95 — every threshold the score satisfies.
        event = self._fake_event(
            stage="eval",
            severity="info",
            run_id="eval-1",
            payload={"eval_result_id": 1, "pass_rate": 0.95, "total": 100},
        )
        _run(self._process(event))
        state = _run(self._state(self.project_id))
        unlocked = set(state["achievements_unlocked"])
        self.assertIn("first_eval", unlocked)
        self.assertIn("f1_above_80", unlocked)
        self.assertIn("f1_above_90", unlocked)
        self.assertIn("f1_above_95", unlocked)

    def test_eval_mid_tier_only_unlocks_matching_thresholds(self):
        event = self._fake_event(
            stage="eval",
            severity="info",
            run_id="eval-2",
            payload={"eval_result_id": 2, "pass_rate": 0.82, "total": 100},
        )
        _run(self._process(event))
        unlocked = set(_run(self._state(self.project_id))["achievements_unlocked"])
        self.assertIn("f1_above_80", unlocked)
        self.assertNotIn("f1_above_90", unlocked)
        self.assertNotIn("f1_above_95", unlocked)

    def test_export_with_gguf_format_unlocks_compression(self):
        event = self._fake_event(
            stage="export",
            severity="info",
            run_id="export-1",
            payload={"export_id": 1, "format": "gguf-q4_k_m"},
        )
        _run(self._process(event))
        unlocked = set(_run(self._state(self.project_id))["achievements_unlocked"])
        self.assertIn("first_export", unlocked)
        self.assertIn("compression_used", unlocked)

    def test_deployment_promote_unlocks_first_deploy(self):
        event = self._fake_event(
            stage="deployment",
            severity="info",
            run_id="deploy-1",
            payload={"action": "promote", "deployment_version_id": 1, "version": 1},
        )
        _run(self._process(event))
        unlocked = set(_run(self._state(self.project_id))["achievements_unlocked"])
        self.assertIn("first_deploy", unlocked)

    def test_deployment_create_does_not_unlock_first_deploy(self):
        # Only promote counts as "shipped" — create is a staging step.
        event = self._fake_event(
            stage="deployment",
            severity="info",
            run_id="deploy-1",
            payload={"action": "create", "deployment_version_id": 1, "version": 1},
        )
        _run(self._process(event))
        unlocked = set(_run(self._state(self.project_id))["achievements_unlocked"])
        self.assertNotIn("first_deploy", unlocked)

    def test_night_owl_unlocks_on_late_night_training_ts(self):
        midnight_ts = datetime(2026, 5, 14, 2, 30, tzinfo=timezone.utc)
        event = self._fake_event(
            stage="training",
            severity="info",
            run_id="exp-99",
            payload={"experiment_id": 99, "base_model": "llama3-8b"},
            ts=midnight_ts,
        )
        _run(self._process(event))
        unlocked = set(_run(self._state(self.project_id))["achievements_unlocked"])
        self.assertIn("night_owl", unlocked)

    # ── Toast-spam buffer ───────────────────────────────────────────

    def test_toast_suppress_returns_true_after_recent_grant(self):
        # First call records a timestamp + returns False.
        self.assertFalse(
            gs._should_suppress_toast(self.project_id, "dataset_import_run")
        )
        # Immediate repeat is suppressed.
        self.assertTrue(
            gs._should_suppress_toast(self.project_id, "dataset_import_run")
        )

    def test_toast_suppress_pretends_outside_window(self):
        # Seed the buffer with an old timestamp.
        with gs._RECENT_GRANTS_LOCK:
            gs._RECENT_GRANTS[(self.project_id, "training_complete")] = datetime(
                2020, 1, 1, tzinfo=timezone.utc
            )
        self.assertFalse(
            gs._should_suppress_toast(self.project_id, "training_complete")
        )

    # ── API ─────────────────────────────────────────────────────────

    def test_api_progression_returns_default_for_fresh_project(self):
        resp = self.client.get(
            f"/api/projects/{self.project_id}/gamification"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["xp_balance"], 0)
        self.assertEqual(body["level"], 1)
        self.assertEqual(body["xp_to_next_level"], 100)
        self.assertEqual(body["level_title"], "Intern")
        self.assertEqual(body["achievements_unlocked"], [])

    def test_api_achievements_lists_full_catalog_with_unlocked_flag(self):
        resp = self.client.get(
            f"/api/projects/{self.project_id}/gamification/achievements"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["summary"]["total"], len(ACHIEVEMENTS))
        self.assertEqual(body["summary"]["unlocked"], 0)
        ids = [a["id"] for a in body["achievements"]]
        self.assertIn("first_ingest", ids)
        self.assertIn("first_train", ids)
        for entry in body["achievements"]:
            self.assertFalse(entry["unlocked"])
            self.assertIsNone(entry["unlocked_at"])

    def test_api_achievements_marks_unlocked_after_event(self):
        _run(
            self._process(
                self._fake_event(
                    reason_code="dataset_import_run",
                    payload={"source_id": "jsonl", "mapper_id": "text_only"},
                )
            )
        )
        resp = self.client.get(
            f"/api/projects/{self.project_id}/gamification/achievements"
        )
        body = resp.json()
        unlocked = {a["id"] for a in body["achievements"] if a["unlocked"]}
        self.assertIn("first_ingest", unlocked)

    def test_api_404_for_unknown_project(self):
        resp = self.client.get("/api/projects/999999/gamification")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
