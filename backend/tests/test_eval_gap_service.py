"""Eval Gap scanner — Coach-stage-2 phase 3.

Covers the 3 eval-side signals end-to-end:
- archetype_coverage_low: stubbed archetype comparison to drive each branch.
- no_regression_baseline: completion + promotion combinations.
- train_eval_label_kl_high: train/eval label distribution matching.

Plus the public scan + 404 fallback + coach eval roll-up nudge.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "eval_gap.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "eval_gap_data"

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
from app.models.experiment import (  # noqa: E402
    Checkpoint, Experiment, ExperimentStatus,
)


def _signal_by_id(body: dict, signal_id: str) -> dict | None:
    for group in body["groups"]:
        for sig in group["signals"]:
            if sig["id"] == signal_id:
                return sig
    return None


class EvalGapTests(unittest.TestCase):
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

    def _make_project(self, *, recipe_id: str | None = None) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"egap-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        if recipe_id is not None:
            async def _set():
                async with async_session_factory() as db:
                    from app.models.project import Project
                    proj = await db.get(Project, pid)
                    proj.selected_recipe = {"recipe_id": recipe_id}
                    await db.commit()
            asyncio.run(_set())
        return pid

    def _seed_labels(
        self,
        project_id: int,
        *,
        train_labels: list[str],
        eval_labels: list[str],
    ) -> None:
        """Drop train + gold JSONL files and register Datasets that
        point at them. Each row carries a ``label`` field so the
        gap service's _row_to_label picks it up.
        """
        ds_dir = TEST_DATA_DIR / f"project-{project_id}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        train_path = ds_dir / "train.jsonl"
        eval_path = ds_dir / "gold_dev.jsonl"
        with open(train_path, "w", encoding="utf-8") as f:
            for label in train_labels:
                f.write(json.dumps({"label": label}) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            for label in eval_labels:
                f.write(json.dumps({"label": label}) + "\n")

        async def _add():
            async with async_session_factory() as db:
                db.add(Dataset(
                    project_id=project_id,
                    name="train",
                    dataset_type=DatasetType.TRAIN,
                    file_path=str(train_path),
                    record_count=len(train_labels),
                ))
                db.add(Dataset(
                    project_id=project_id,
                    name="gold_dev",
                    dataset_type=DatasetType.GOLD_DEV,
                    file_path=str(eval_path),
                    record_count=len(eval_labels),
                ))
                await db.commit()
        asyncio.run(_add())

    def _add_experiment(
        self,
        project_id: int,
        *,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
        with_promoted_checkpoint: bool = False,
    ) -> int:
        async def _add() -> int:
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=project_id,
                    name=f"exp-{uuid.uuid4().hex[:6]}",
                    description="",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config={},
                    status=status,
                )
                db.add(exp)
                await db.flush()
                if with_promoted_checkpoint:
                    db.add(Checkpoint(
                        experiment_id=exp.id,
                        epoch=1,
                        step=100,
                        train_loss=0.5,
                        eval_loss=0.6,
                        file_path="/tmp/ckpt",
                        is_best=True,
                        metrics={},
                        promoted_at=datetime.now(timezone.utc),
                    ))
                await db.commit()
                return int(exp.id)
        return asyncio.run(_add())

    # ── Tests ───────────────────────────────────────────────────────

    def test_no_recipe_returns_block_fallback(self):
        pid = self._make_project()
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        self.assertEqual(body["overall"], "block")
        sig = _signal_by_id(body, "eval_gaps.no_recipe_selected")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "block")

    def test_endpoint_404s_for_missing_project(self):
        resp = self.client.get("/api/projects/9999999/eval-gaps")
        self.assertEqual(resp.status_code, 404)

    # ── Archetype coverage ──────────────────────────────────────────

    def test_archetype_coverage_degrades_ok_when_comparison_raises(self):
        # When the archetype service raises (e.g. archetype not computed
        # for this recipe, gold rows missing, etc.) the signal degrades
        # to ok with a skipped_reason rather than blocking the report.
        from unittest.mock import patch

        async def _boom(*_a, **_k):
            raise RuntimeError("archetype not available")

        pid = self._make_project(recipe_id="classification")
        with patch(
            "app.services.archetype_service.compare_project_to_archetype",
            side_effect=_boom,
        ):
            body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.archetype_coverage_low")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertIn("skipped_reason", sig["context"])

    def test_archetype_coverage_blocks_when_three_below_features(self):
        # Stub the comparison to return 3 below-band features → block.
        from unittest.mock import patch

        async def _stub_comparison(*_a, **_k):
            return {
                "project_id": 1,
                "recipe_id": "classification",
                "archetype": {},
                "features": [
                    {"feature_id": "row_count", "label": "Rows",
                     "status": "below", "your_value": 10},
                    {"feature_id": "class_entropy", "label": "Class entropy",
                     "status": "below", "your_value": 0.1},
                    {"feature_id": "goldset_diversity", "label": "Diversity",
                     "status": "below", "your_value": 0.05},
                    {"feature_id": "hard_negative_ratio",
                     "label": "Hard negatives",
                     "status": "ok", "your_value": 0.5},
                ],
                "summary": {},
            }

        pid = self._make_project(recipe_id="classification")
        with patch(
            "app.services.archetype_service.compare_project_to_archetype",
            side_effect=_stub_comparison,
        ):
            body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.archetype_coverage_low")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "block")
        self.assertEqual(sig["context"]["below_count"], 3)
        self.assertEqual(sig["context"]["feature_count"], 4)
        self.assertEqual(
            sig["suggested_action"]["target"],
            "archetype-comparison-panel",
        )

    # ── Regression baseline ─────────────────────────────────────────

    def test_regression_baseline_ok_when_no_runs_completed(self):
        pid = self._make_project(recipe_id="classification")
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.no_regression_baseline")
        self.assertIsNotNone(sig)
        assert sig is not None
        # No completed runs → baseline check is deferred (ok).
        self.assertEqual(sig["severity"], "ok")
        self.assertFalse(sig["context"]["has_completed_runs"])

    def test_regression_baseline_warns_when_completed_but_unpromoted(self):
        pid = self._make_project(recipe_id="classification")
        self._add_experiment(pid, with_promoted_checkpoint=False)
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.no_regression_baseline")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "warn")
        self.assertTrue(sig["context"]["has_completed_runs"])
        self.assertEqual(
            sig["suggested_action"]["target"], "checkpoints-panel"
        )

    def test_regression_baseline_ok_when_checkpoint_promoted(self):
        pid = self._make_project(recipe_id="classification")
        self._add_experiment(pid, with_promoted_checkpoint=True)
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.no_regression_baseline")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertIn("promoted_checkpoint_id", sig["context"])

    # ── Train/eval label KL ─────────────────────────────────────────

    def test_label_kl_skips_non_classification(self):
        pid = self._make_project(
            recipe_id="span-extraction"
        )
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.train_eval_label_kl_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertEqual(
            sig["context"]["task_profile"], "structured_extraction"
        )

    def test_label_kl_skips_when_below_minimum_sample_sizes(self):
        pid = self._make_project(recipe_id="classification")
        # Only 5 train labels, 3 eval — below the 20/10 floors.
        self._seed_labels(
            pid,
            train_labels=["pos"] * 3 + ["neg"] * 2,
            eval_labels=["pos"] * 2 + ["neg"] * 1,
        )
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.train_eval_label_kl_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertIn("below minimum sample sizes", sig["context"]["skipped_reason"])

    def test_label_kl_ok_when_distributions_match(self):
        pid = self._make_project(recipe_id="classification")
        # 50/50 train, 50/50 eval → KL ≈ 0.
        self._seed_labels(
            pid,
            train_labels=["pos"] * 25 + ["neg"] * 25,
            eval_labels=["pos"] * 10 + ["neg"] * 10,
        )
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.train_eval_label_kl_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")
        self.assertLess(sig["context"]["kl_nats"], 0.10)

    def test_label_kl_warns_when_distributions_diverge(self):
        pid = self._make_project(recipe_id="classification")
        # 90% pos in train, 50/50 in eval → meaningful KL.
        self._seed_labels(
            pid,
            train_labels=["pos"] * 45 + ["neg"] * 5,
            eval_labels=["pos"] * 10 + ["neg"] * 10,
        )
        body = self.client.get(f"/api/projects/{pid}/eval-gaps").json()
        sig = _signal_by_id(body, "eval_gaps.train_eval_label_kl_high")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertIn(sig["severity"], ("warn", "block"))
        self.assertGreaterEqual(sig["context"]["kl_nats"], 0.10)
        self.assertEqual(
            sig["suggested_action"]["target"], "data-studio-splits"
        )

    # ── Coach roll-up ───────────────────────────────────────────────

    def test_coach_eval_stage_emits_rollup_when_gaps_present(self):
        # Use the unpromoted-checkpoint path: a completed run with no
        # promotion fires the no_regression_baseline warn signal,
        # which should bubble up into the coach eval rollup.
        pid = self._make_project(recipe_id="classification")
        self._add_experiment(pid, with_promoted_checkpoint=False)
        resp = self.client.get(f"/api/projects/{pid}/coach/eval")
        self.assertEqual(resp.status_code, 200, resp.text)
        suggestions = resp.json()
        if isinstance(suggestions, dict) and "suggestions" in suggestions:
            suggestions = suggestions["suggestions"]
        rollups = [s for s in suggestions if s.get("id") == "eval:gaps-rollup"]
        self.assertEqual(len(rollups), 1)
        card = rollups[0]
        self.assertIn(card["severity"], ("warning", "critical"))
        self.assertEqual(
            card["action"]["params"]["target"], "eval-gaps-panel"
        )

    def test_coach_eval_stage_silent_when_no_eval_gaps(self):
        # No recipe — the eval gap scanner emits the no-recipe block,
        # but the coach eval rollup still rolls it up (block_count=1).
        # To prove silence, give it a promoted baseline + a
        # non-classification recipe so all three signals are ok.
        pid = self._make_project(
            recipe_id="span-extraction"
        )
        self._add_experiment(pid, with_promoted_checkpoint=True)
        # Need to also stub archetype to return all-ok features so
        # the archetype signal doesn't fire.
        from unittest.mock import patch

        async def _all_ok(*_a, **_k):
            return {
                "project_id": pid,
                "recipe_id": "span-extraction",
                "archetype": {},
                "features": [
                    {"feature_id": "row_count", "label": "Rows",
                     "status": "ok", "your_value": 200},
                ],
                "summary": {},
            }

        with patch(
            "app.services.archetype_service.compare_project_to_archetype",
            side_effect=_all_ok,
        ):
            resp = self.client.get(f"/api/projects/{pid}/coach/eval")
        self.assertEqual(resp.status_code, 200, resp.text)
        suggestions = resp.json()
        if isinstance(suggestions, dict) and "suggestions" in suggestions:
            suggestions = suggestions["suggestions"]
        rollups = [s for s in suggestions if s.get("id") == "eval:gaps-rollup"]
        self.assertEqual(rollups, [])
