"""Tests for the "Did SFT help?" lift summary (Theme 8 Epic 4)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401
from app.database import Base
from app.models.experiment import (
    EvalResult,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.models.project import Project
from app.services.sft_lift_summary_service import compute_sft_lift_summary


class SftLiftSummaryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "sft_lift.db"
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}", future=True,
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def _new_project(self, *, recipe: dict | None = None) -> int:
        async with self.session_factory() as db:
            project = Project(
                name=f"lift-{uuid4().hex[:8]}",
                description="sft lift tests",
                base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                selected_recipe=recipe,
            )
            db.add(project)
            await db.commit()
            return project.id

    async def _add_experiment(
        self,
        project_id: int,
        *,
        name: str,
        is_baseline: bool,
        metrics: dict[str, float] | None,
    ) -> int:
        async with self.session_factory() as db:
            config: dict = {"is_baseline": True} if is_baseline else {"base_model": "x"}
            exp = Experiment(
                project_id=project_id,
                name=name,
                status=ExperimentStatus.COMPLETED,
                training_mode=TrainingMode.SFT,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                output_dir=None if is_baseline else "/tmp/out",
                config=config,
            )
            db.add(exp)
            await db.flush()
            if metrics is not None:
                er = EvalResult(
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="exact_match",
                    metrics=metrics,
                    details={},
                )
                db.add(er)
            await db.commit()
            return exp.id

    # ── Status branches ──────────────────────────────────────────

    async def test_no_baseline_returns_explanatory_payload(self):
        project_id = await self._new_project()
        await self._add_experiment(
            project_id, name="trained-1", is_baseline=False,
            metrics={"exact_match": 0.5, "f1": 0.6},
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)
        self.assertEqual(payload["status"], "no_baseline")
        self.assertIsNone(payload["baseline"])
        self.assertIsNotNone(payload["trained"])
        self.assertEqual(payload["metric_lifts"], [])
        self.assertIn("Baseline", payload["message"])

    async def test_no_trained_returns_explanatory_payload(self):
        project_id = await self._new_project()
        await self._add_experiment(
            project_id, name="Baseline · SmolLM2", is_baseline=True,
            metrics={"exact_match": 0.2, "f1": 0.3},
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)
        self.assertEqual(payload["status"], "no_trained")
        self.assertIsNotNone(payload["baseline"])
        self.assertIsNone(payload["trained"])

    async def test_project_not_found_raises(self):
        async with self.session_factory() as db:
            with self.assertRaisesRegex(ValueError, "project_not_found:99999"):
                await compute_sft_lift_summary(db, 99999)

    async def test_no_overlap_when_metric_keys_disjoint(self):
        project_id = await self._new_project()
        await self._add_experiment(
            project_id, name="Baseline · SmolLM2", is_baseline=True,
            metrics={"exact_match": 0.2},
        )
        await self._add_experiment(
            project_id, name="trained-1", is_baseline=False,
            metrics={"groundedness": 0.7},  # no overlap with baseline
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)
        self.assertEqual(payload["status"], "no_overlap")
        self.assertEqual(payload["metric_lifts"], [])
        self.assertIn("comparable", payload["message"])

    # ── Happy-path lift computation ──────────────────────────────

    async def test_metric_lifts_computed_correctly_with_absolute_and_relative(self):
        project_id = await self._new_project()
        await self._add_experiment(
            project_id, name="Baseline · SmolLM2", is_baseline=True,
            metrics={"exact_match": 0.2, "f1": 0.3, "accuracy": 0.5},
        )
        await self._add_experiment(
            project_id, name="trained-1", is_baseline=False,
            metrics={"exact_match": 0.5, "f1": 0.6, "accuracy": 0.45},
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["metric_lifts"]), 3)

        by_id = {row["metric_id"]: row for row in payload["metric_lifts"]}
        # exact_match: 0.2 → 0.5 = +0.30 absolute, +150% relative.
        self.assertAlmostEqual(by_id["exact_match"]["absolute_delta"], 0.3, places=2)
        self.assertEqual(by_id["exact_match"]["relative_delta_pct"], 150.0)
        self.assertEqual(by_id["exact_match"]["direction"], "improved")
        # accuracy: 0.5 → 0.45 = regression.
        self.assertEqual(by_id["accuracy"]["direction"], "regressed")
        self.assertAlmostEqual(by_id["accuracy"]["absolute_delta"], -0.05, places=2)

    async def test_headline_metrics_sort_to_top(self):
        project_id = await self._new_project()
        await self._add_experiment(
            project_id, name="Baseline", is_baseline=True,
            metrics={"f1": 0.3, "schema_mismatch": 0.0, "custom_metric": 0.1},
        )
        await self._add_experiment(
            project_id, name="trained-1", is_baseline=False,
            metrics={"f1": 0.6, "schema_mismatch": 0.0, "custom_metric": 0.2},
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)
        # First lift is f1 (a preferred headline metric); custom_metric
        # sorts after. schema_mismatch is filtered out.
        ids = [row["metric_id"] for row in payload["metric_lifts"]]
        self.assertEqual(ids[0], "f1")
        self.assertIn("custom_metric", ids)
        self.assertNotIn("schema_mismatch", ids)

    # ── Gate evaluation ──────────────────────────────────────────

    async def test_gates_bucket_cleared_still_failing_always_passed(self):
        recipe = {
            "recipe_id": "qa-sft",
            "task_profile": "instruction_sft",
        }
        project_id = await self._new_project(recipe=recipe)
        # baseline below thresholds, trained above some.
        # evalpack.general.default for instruction_sft (Theme 1 Epic 3
        # recalibration): exact_match 0.4, f1 0.5, llm_judge 0.65 (non-req).
        await self._add_experiment(
            project_id, name="Baseline", is_baseline=True,
            metrics={"exact_match": 0.2, "f1": 0.3, "safety_pass_rate": 0.95},
        )
        await self._add_experiment(
            project_id, name="trained-1", is_baseline=False,
            metrics={"exact_match": 0.55, "f1": 0.45, "safety_pass_rate": 0.92},
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)

        statuses = {row["metric_id"]: row["status"] for row in payload["gate_status"]}
        # exact_match: baseline 0.2 < 0.4 fails, trained 0.55 >= 0.4 passes → cleared
        self.assertEqual(statuses.get("exact_match"), "cleared")
        # f1: baseline 0.3 < 0.5 fails, trained 0.45 < 0.5 still fails → still_failing
        self.assertEqual(statuses.get("f1"), "still_failing")
        # safety_pass_rate: both above 0.9 → always_passed
        self.assertEqual(statuses.get("safety_pass_rate"), "always_passed")

    async def test_gate_regression_status_when_baseline_passed_but_trained_fails(self):
        recipe = {"recipe_id": "classification", "task_profile": "classification"}
        project_id = await self._new_project(recipe=recipe)
        # classification gates: accuracy 0.5, macro_f1 0.5.
        await self._add_experiment(
            project_id, name="Baseline", is_baseline=True,
            metrics={"accuracy": 0.6, "macro_f1": 0.55},
        )
        await self._add_experiment(
            project_id, name="trained-1", is_baseline=False,
            metrics={"accuracy": 0.45, "macro_f1": 0.6},
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)
        statuses = {row["metric_id"]: row["status"] for row in payload["gate_status"]}
        # accuracy regressed (0.6 → 0.45 below 0.5 threshold).
        self.assertEqual(statuses.get("accuracy"), "regressed")
        # macro_f1 still_failing → wait, both 0.55 and 0.6 are above 0.5;
        # that's always_passed.
        self.assertEqual(statuses.get("macro_f1"), "always_passed")

    async def test_only_latest_baseline_and_trained_are_used(self):
        """When multiple experiments exist, the comparison uses the
        most recent baseline + most recent non-baseline by id desc."""
        project_id = await self._new_project()
        await self._add_experiment(
            project_id, name="Baseline · old", is_baseline=True,
            metrics={"f1": 0.1},
        )
        await self._add_experiment(
            project_id, name="trained-old", is_baseline=False,
            metrics={"f1": 0.4},
        )
        # Newer entries — these should be picked.
        await self._add_experiment(
            project_id, name="Baseline · new", is_baseline=True,
            metrics={"f1": 0.3},
        )
        await self._add_experiment(
            project_id, name="trained-new", is_baseline=False,
            metrics={"f1": 0.7},
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)
        self.assertEqual(payload["baseline"]["experiment_name"], "Baseline · new")
        self.assertEqual(payload["trained"]["experiment_name"], "trained-new")
        by_id = {row["metric_id"]: row for row in payload["metric_lifts"]}
        self.assertAlmostEqual(by_id["f1"]["baseline_value"], 0.3, places=2)
        self.assertAlmostEqual(by_id["f1"]["trained_value"], 0.7, places=2)

    async def test_skips_failed_and_cancelled_trained_experiments(self):
        project_id = await self._new_project()
        await self._add_experiment(
            project_id, name="Baseline", is_baseline=True,
            metrics={"f1": 0.3},
        )
        # Failed experiments should NOT count as the "latest trained."
        async with self.session_factory() as db:
            failed = Experiment(
                project_id=project_id,
                name="trained-failed",
                status=ExperimentStatus.FAILED,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                output_dir="/tmp/x",
                config={"base_model": "x"},
            )
            db.add(failed)
            await db.flush()
            db.add(
                EvalResult(
                    experiment_id=failed.id,
                    dataset_name="test",
                    eval_type="exact_match",
                    metrics={"f1": 0.55},  # would mislead the summary
                    details={},
                )
            )
            await db.commit()
        # Add a healthy newer one.
        await self._add_experiment(
            project_id, name="trained-ok", is_baseline=False,
            metrics={"f1": 0.65},
        )
        async with self.session_factory() as db2:
            payload = await compute_sft_lift_summary(db2, project_id)
        self.assertEqual(payload["trained"]["experiment_name"], "trained-ok")

    async def test_zero_baseline_value_yields_null_relative_delta(self):
        """Going from 0 to anything is mathematically undefined / infinite
        — surface as null relative_delta_pct so the UI can render
        'new' instead of a misleading number."""
        project_id = await self._new_project()
        await self._add_experiment(
            project_id, name="Baseline", is_baseline=True,
            metrics={"f1": 0.0, "exact_match": 0.0},
        )
        await self._add_experiment(
            project_id, name="trained-1", is_baseline=False,
            metrics={"f1": 0.5, "exact_match": 0.0},  # f1 jumps from 0, EM stays
        )
        async with self.session_factory() as db:
            payload = await compute_sft_lift_summary(db, project_id)
        by_id = {row["metric_id"]: row for row in payload["metric_lifts"]}
        self.assertIsNone(by_id["f1"]["relative_delta_pct"])
        # exact_match stayed at 0 — relative is 0%, direction unchanged.
        self.assertEqual(by_id["exact_match"]["relative_delta_pct"], 0.0)
        self.assertEqual(by_id["exact_match"]["direction"], "unchanged")


if __name__ == "__main__":
    unittest.main()
