"""Tests for the Phase 6d default-on heuristic + Coach Mode nudge.

Covers:
  * ``_decide_curriculum_default`` truth table over the 5 paths:
    - no recipe selected → no default
    - non-curriculum recipe (qa-sft, generic-sft, …) → no default
    - classification + missing prepared train file → no default
    - classification + thin (≤ 200 rows) → DEFAULT ON
    - classification + thick (> 200 rows) → no default
  * ``create_experiment`` integration:
    - explicit ``curriculum=False`` is preserved (opt-out wins)
    - explicit ``curriculum=True`` is preserved (opt-in wins)
    - missing key → heuristic fires
    - missing key + thick dataset → no auto-default
  * ``_curriculum_training_suggestion`` returns a navigate-action
    suggestion only when the project would benefit (classification
    + thin), with the A/B numbers in the body so users aren't asked
    to take it on faith.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

# Override DATA_DIR before importing the services so the heuristic's
# file-probe targets a sandbox we control per test.
_TEST_DATA_DIR = Path(__file__).resolve().parent / "curriculum_phase6d_data"
os.environ["DATA_DIR"] = _TEST_DATA_DIR.as_posix()

from app.config import settings  # noqa: E402
from app.services.training_service import (  # noqa: E402
    CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS,
    _decide_curriculum_default,
)


def _project_stub(recipe_id: str | None):
    p = MagicMock()
    p.selected_recipe = {"recipe_id": recipe_id} if recipe_id else None
    return p


def _seed_train_file(project_id: int, row_count: int) -> Path:
    """Write a synthetic prepared train.jsonl with ``row_count`` rows
    under the test DATA_DIR."""
    path = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(row_count):
            f.write(json.dumps({"text": f"row{i}", "label": "L"}) + "\n")
    return path


class DecideCurriculumDefaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _TEST_DATA_DIR.exists():
            for p in sorted(_TEST_DATA_DIR.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
        _TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if _TEST_DATA_DIR.exists():
            for p in sorted(_TEST_DATA_DIR.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()

    def test_no_recipe_returns_no_default(self):
        decision = _decide_curriculum_default(
            project_obj=_project_stub(None),
            project_id=1001,
        )
        self.assertFalse(decision["should_default_on"])
        self.assertEqual(decision["reason"], "no_recipe_selected")

    def test_non_curriculum_recipe_returns_no_default(self):
        # qa-sft, generic-sft, … all have no scoring mode in Phase 6a.
        for recipe in ("qa-sft", "generic-sft", "span-extraction", "summarization"):
            with self.subTest(recipe=recipe):
                decision = _decide_curriculum_default(
                    project_obj=_project_stub(recipe),
                    project_id=2000 + hash(recipe) % 100,
                )
                self.assertFalse(decision["should_default_on"])
                self.assertIn("recipe_has_no_curriculum", decision["reason"])
                self.assertIn(recipe, decision["reason"])

    def test_classification_without_prepared_file_returns_no_default(self):
        decision = _decide_curriculum_default(
            project_obj=_project_stub("classification"),
            project_id=3001,
        )
        self.assertFalse(decision["should_default_on"])
        self.assertEqual(decision["reason"], "no_prepared_train_file")

    def test_classification_thin_dataset_defaults_on(self):
        """≤ CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS rows → on by default."""
        _seed_train_file(project_id=4001, row_count=150)
        decision = _decide_curriculum_default(
            project_obj=_project_stub("classification"),
            project_id=4001,
        )
        self.assertTrue(decision["should_default_on"])
        self.assertIn("thin_classification", decision["reason"])
        self.assertIn("150", decision["reason"])

    def test_classification_thick_dataset_does_not_default(self):
        """> CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS rows → no auto-default
        (we don't have an empirical mandate above 200 rows)."""
        _seed_train_file(project_id=4002, row_count=CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS + 1)
        decision = _decide_curriculum_default(
            project_obj=_project_stub("classification"),
            project_id=4002,
        )
        self.assertFalse(decision["should_default_on"])
        self.assertIn("thick_dataset", decision["reason"])

    def test_boundary_at_exactly_threshold_defaults_on(self):
        """Threshold is inclusive — at exactly 200 rows we still
        default on. Phase 6c A/B trained on 144 rows; the boundary
        gives headroom for slightly thicker thin-data projects."""
        _seed_train_file(project_id=4003, row_count=CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS)
        decision = _decide_curriculum_default(
            project_obj=_project_stub("classification"),
            project_id=4003,
        )
        self.assertTrue(decision["should_default_on"])


class CreateExperimentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercises create_experiment's interaction with the heuristic
    without spinning up the full FastAPI app — patches the bits the
    helper needs."""

    @classmethod
    def setUpClass(cls):
        if _TEST_DATA_DIR.exists():
            for p in sorted(_TEST_DATA_DIR.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
        _TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    async def _create_experiment(
        self, *, project_id: int, recipe_id: str | None, config: dict,
        train_row_count: int | None = None,
    ) -> tuple[dict, bool]:
        """Return (resolved_config, auto_defaulted_flag) — the parts
        of the experiment that matter for these tests."""
        from app.models.experiment import Experiment, TrainingMode
        from app.services import training_service

        if train_row_count is not None:
            _seed_train_file(project_id, train_row_count)

        # Mock the project fetch.
        project = _project_stub(recipe_id)

        captured: dict[str, dict] = {}

        async def _fake_db_execute(stmt):
            result = MagicMock()
            result.scalar_one_or_none = MagicMock(return_value=project)
            return result

        async def _noop_flush():
            return None

        async def _noop_refresh(obj):
            return None

        db = MagicMock()
        db.execute = AsyncMock(side_effect=_fake_db_execute)
        db.add = MagicMock(side_effect=lambda exp: captured.setdefault("exp", exp))
        db.flush = AsyncMock(side_effect=_noop_flush)
        db.refresh = AsyncMock(side_effect=_noop_refresh)

        # Force Experiment.id population since we skip a real DB flush.
        original_add = db.add
        def _add_with_id(exp):
            original_add(exp)
            if exp.id is None:
                exp.id = project_id * 10  # arbitrary unique id

        db.add = MagicMock(side_effect=_add_with_id)

        # Avoid the base-model validator (which hits a config store).
        with patch.object(
            training_service,
            "evaluate_training_base_model_compatibility",
            return_value={"ok": True, "errors": []},
        ):
            exp = await training_service.create_experiment(
                db=db,
                project_id=project_id,
                name="t",
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                config=dict(config),
                training_mode=TrainingMode.SFT,
            )
        return exp.config, "_curriculum_auto_defaulted" in (exp.config or {})

    async def test_explicit_false_is_respected_even_on_thin_classification(self):
        cfg, auto = await self._create_experiment(
            project_id=5001,
            recipe_id="classification",
            config={"curriculum": False},
            train_row_count=100,
        )
        self.assertEqual(cfg["curriculum"], False)
        self.assertFalse(auto)

    async def test_explicit_true_is_respected(self):
        cfg, auto = await self._create_experiment(
            project_id=5002,
            recipe_id="classification",
            config={"curriculum": True},
            train_row_count=100,
        )
        self.assertEqual(cfg["curriculum"], True)
        self.assertFalse(auto)  # not "auto", just respected as-is

    async def test_unset_on_thin_classification_auto_defaults_on(self):
        cfg, auto = await self._create_experiment(
            project_id=5003,
            recipe_id="classification",
            config={},
            train_row_count=80,
        )
        self.assertEqual(cfg["curriculum"], True)
        self.assertTrue(auto)
        # Reason is recorded for the UI to show.
        self.assertIn("thin_classification", cfg["_curriculum_auto_defaulted"])

    async def test_unset_on_thick_classification_does_not_default(self):
        cfg, auto = await self._create_experiment(
            project_id=5004,
            recipe_id="classification",
            config={},
            train_row_count=500,
        )
        self.assertNotIn("curriculum", cfg)
        self.assertFalse(auto)

    async def test_unset_on_non_classification_does_not_default(self):
        cfg, auto = await self._create_experiment(
            project_id=5005,
            recipe_id="qa-sft",
            config={},
            train_row_count=50,
        )
        self.assertNotIn("curriculum", cfg)
        self.assertFalse(auto)


class CurriculumTrainingSuggestionTests(unittest.TestCase):
    """``_curriculum_training_suggestion`` returns either a nudge
    suggestion or None — gated on the same eligibility check as the
    default-on heuristic."""

    @classmethod
    def setUpClass(cls):
        if _TEST_DATA_DIR.exists():
            for p in sorted(_TEST_DATA_DIR.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
        _TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def test_returns_none_without_recipe(self):
        from app.services.coach_service import _curriculum_training_suggestion
        self.assertIsNone(_curriculum_training_suggestion(6001, None))

    def test_returns_none_for_non_classification_recipe(self):
        from app.services.coach_service import _curriculum_training_suggestion
        # No file probe needed — the recipe gate short-circuits.
        self.assertIsNone(_curriculum_training_suggestion(6002, "qa-sft"))

    def test_returns_none_when_no_prepared_train_file(self):
        from app.services.coach_service import _curriculum_training_suggestion
        # Classification recipe but no train file → no nudge.
        self.assertIsNone(_curriculum_training_suggestion(6003, "classification"))

    def test_returns_nudge_for_thin_classification(self):
        from app.services.coach_service import _curriculum_training_suggestion
        _seed_train_file(project_id=6004, row_count=120)
        nudge = _curriculum_training_suggestion(6004, "classification")
        self.assertIsNotNone(nudge)
        # Action is a navigate to training config.
        self.assertEqual(nudge["action"]["kind"], "navigate")
        self.assertEqual(
            nudge["action"]["params"]["target"], "training-config"
        )
        # Body cites the A/B numbers explicitly so the user can audit
        # the recommendation against the roadmap.
        self.assertIn("Phase 6c", nudge["body"])
        self.assertIn("ticket-router", nudge["body"])
        self.assertIn("log-triage", nudge["body"])
        # Context carries the row count + threshold + lift numbers.
        self.assertEqual(nudge["context"]["train_row_count"], 120)
        self.assertEqual(nudge["context"]["threshold"], CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS)
        self.assertIn("ticket-router", nudge["context"]["ab_lift_pct"])
        self.assertIn("log-triage", nudge["context"]["ab_lift_pct"])
        # info severity — this isn't a problem, just a heads-up.
        self.assertEqual(nudge["severity"], "info")

    def test_returns_none_for_thick_classification(self):
        from app.services.coach_service import _curriculum_training_suggestion
        _seed_train_file(project_id=6005, row_count=CURRICULUM_AUTO_ON_MAX_TRAIN_ROWS + 50)
        # Above the threshold → no auto-on AND no nudge (we don't
        # claim curriculum helps in the thick-data regime).
        self.assertIsNone(_curriculum_training_suggestion(6005, "classification"))


if __name__ == "__main__":
    unittest.main()
