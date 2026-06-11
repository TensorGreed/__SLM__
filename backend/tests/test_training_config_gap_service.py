"""Training Config Gap scanner — Coach-stage-2 phase 1.

Covers each signal's threshold logic + the no-recipe fallback. The
service is read-only so all tests just exercise the scan + endpoint
shape; no autofix surface to round-trip in phase 1.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "training_config_gap.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "training_config_gap_data"

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
from app.models.dataset import Dataset, DatasetType  # noqa: E402


def _signal_by_id(body: dict, signal_id: str) -> dict | None:
    for group in body["groups"]:
        for sig in group["signals"]:
            if sig["id"] == signal_id:
                return sig
    return None


class TrainingConfigGapTests(unittest.TestCase):
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

    # ── helpers ─────────────────────────────────────────────────────

    def _create_project(
        self,
        *,
        recipe_id: str | None = None,
        base_model: str | None = None,
    ) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"tcg-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])

        async def _set():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, pid)
                if recipe_id is not None:
                    proj.selected_recipe = {"recipe_id": recipe_id}
                if base_model is not None:
                    proj.base_model_name = base_model
                await db.commit()
        if recipe_id is not None or base_model is not None:
            asyncio.run(_set())
        return pid

    def _seed_labelled_rows(self, project_id: int, count: int) -> None:
        """Stamp a CLEANED Dataset whose record_count = N. The gap
        scanner reads ``record_count`` directly — no real row content
        needed.
        """
        async def _add():
            async with async_session_factory() as db:
                ds = Dataset(
                    project_id=project_id,
                    name="cleaned",
                    dataset_type=DatasetType.CLEANED,
                    file_path="",
                    record_count=count,
                )
                db.add(ds)
                await db.commit()
        asyncio.run(_add())

    # ── Tests ───────────────────────────────────────────────────────

    def test_no_recipe_emits_block_fallback(self):
        pid = self._create_project()
        resp = self.client.get(f"/api/projects/{pid}/training-config-gaps")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["overall"], "block")
        # Only the no-recipe signal fires; no per-config signals can
        # be computed without a task profile.
        no_recipe = _signal_by_id(body, "training_config.no_recipe_selected")
        self.assertIsNotNone(no_recipe)
        assert no_recipe is not None
        self.assertEqual(no_recipe["severity"], "block")
        # Plain-English carries through.
        self.assertTrue(no_recipe["plain_english"])
        self.assertTrue(no_recipe["why_it_matters"])

    def test_endpoint_404s_for_missing_project(self):
        resp = self.client.get("/api/projects/9999999/training-config-gaps")
        self.assertEqual(resp.status_code, 404)

    def test_small_base_on_structured_extraction_warns(self):
        """135M base on a structured-extraction recipe with ~1k rows
        should fire ``training_config.base_model_undersized``.
        """
        pid = self._create_project(
            recipe_id="recipe.structured_extraction.entity",
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        self._seed_labelled_rows(pid, 1000)
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.base_model_undersized")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertIn(sig["severity"], ("warn", "block"))
        ctx = sig["context"]
        self.assertEqual(ctx["current_params_m"], 135)
        # Floor should be ≥360M on structured_extraction + 1000 rows.
        self.assertGreaterEqual(ctx["params_floor_m"], 360)
        # Action points at the base-model picker.
        action = sig["suggested_action"]
        self.assertEqual(action["target"], "training-base-model-picker")

    def test_well_sized_base_on_classification_returns_ok(self):
        """135M default on a classification recipe with a small gold
        set is the calibrated sweet-spot — the signal should be ``ok``.
        """
        pid = self._create_project(
            recipe_id="recipe.classification.sentiment",
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        self._seed_labelled_rows(pid, 80)
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.base_model_undersized")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")

    def test_tiny_dataset_makes_eval_cadence_fire(self):
        """20 rows × 3 epochs ÷ (4 × 4) = 3 total steps; eval_steps=100
        means 0 eval observations. Should be at least warn (block for 0
        / ≤1 observations)."""
        pid = self._create_project(
            recipe_id="recipe.classification.sentiment",
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        self._seed_labelled_rows(pid, 20)
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.eval_cadence_too_sparse")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertIn(sig["severity"], ("warn", "block"))
        self.assertLessEqual(sig["context"]["eval_observations"], 1)
        # Recommendation surfaces with the right action target.
        action = sig["suggested_action"]
        self.assertEqual(action["target"], "training-config")
        self.assertIn("recommended_eval_steps", action["params"])

    def test_large_dataset_eval_cadence_is_ok(self):
        """50000 rows × 3 epochs ÷ (4 × 4) = 9375 steps ÷ eval_steps 100
        → ~93 eval observations. Plenty."""
        pid = self._create_project(
            recipe_id="recipe.classification.sentiment",
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        self._seed_labelled_rows(pid, 50000)
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.eval_cadence_too_sparse")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertGreaterEqual(sig["context"]["eval_observations"], 3)

    def test_severity_summary_and_overall_rolls_up_correctly(self):
        """When at least one warn + no blocks, overall = warn. Counts in
        ``severity_summary`` match the signals."""
        pid = self._create_project(
            recipe_id="recipe.classification.sentiment",
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        # Tiny dataset triggers eval_cadence + epochs-vs-data signals.
        self._seed_labelled_rows(pid, 20)
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        summary = body["severity_summary"]
        # Count must match the actual signal severities.
        all_signals = [
            sig for g in body["groups"] for sig in g["signals"]
        ]
        self.assertEqual(
            summary["ok"],
            sum(1 for s in all_signals if s["severity"] == "ok"),
        )
        self.assertEqual(
            summary["warn"],
            sum(1 for s in all_signals if s["severity"] == "warn"),
        )
        self.assertEqual(
            summary["block"],
            sum(1 for s in all_signals if s["severity"] == "block"),
        )
        # Overall obeys the block→warn→ok priority.
        if summary["block"] > 0:
            self.assertEqual(body["overall"], "block")
        elif summary["warn"] > 0:
            self.assertEqual(body["overall"], "warn")
        else:
            self.assertEqual(body["overall"], "ok")
        # total_signals matches signal count.
        self.assertEqual(body["total_signals"], len(all_signals))
