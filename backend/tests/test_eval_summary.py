"""Eval tab summary card (Wave 3b): verdict vs the base model, headline
metric, and real failing rows (not the first rows, which are mostly passes).
Private engine per test (like test_post_training_lift_eval)."""

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
from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode
from app.models.project import Project
from app.services.eval_summary_service import build_eval_summary
from app.services.evaluation_service import _prediction_failed

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


class PredictionFailedTests(unittest.TestCase):
    def test_uses_row_exact_match_then_normalized_compare(self):
        self.assertTrue(_prediction_failed({"row_exact_match": 0.0, "prediction": "x", "reference": "x"}))
        self.assertFalse(_prediction_failed({"row_exact_match": 1.0}))
        self.assertFalse(_prediction_failed({"prediction": "Billing.", "reference": "billing"}))
        self.assertTrue(_prediction_failed({"prediction": "shipping", "reference": "billing"}))
        self.assertFalse(_prediction_failed({"prediction": "anything", "reference": ""}))


class EvalSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{Path(self._tmp.name) / 's.db'}", future=True)
        self.sf = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with self.sf() as db:
            project = Project(name=f"sum-{uuid4().hex[:6]}", description="")
            db.add(project)
            await db.commit()
            self.pid = project.id

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def _exp(self, *, baseline=False, status=ExperimentStatus.COMPLETED) -> int:
        async with self.sf() as db:
            exp = Experiment(
                project_id=self.pid, name="base" if baseline else "run", status=status,
                training_mode=TrainingMode.SFT, base_model=BASE,
                config={"is_baseline": True} if baseline else {},
            )
            db.add(exp)
            await db.commit()
            return exp.id

    async def _result(self, exp_id: int, em: float, details: dict | None = None) -> None:
        async with self.sf() as db:
            db.add(EvalResult(experiment_id=exp_id, dataset_name="test", eval_type="exact_match",
                              metrics={"exact_match": em, "evaluated_samples": 40}, details=details or {}))
            await db.commit()

    async def _summary(self, experiment_id=None):
        async with self.sf() as db:
            return await build_eval_summary(db, self.pid, experiment_id=experiment_id)

    async def test_no_trained_run_and_not_evaluated(self):
        self.assertEqual((await self._summary())["verdict"], "no_trained_run")
        await self._exp(baseline=True)
        run = await self._exp()
        out = await self._summary()
        self.assertEqual((out["verdict"], out["experiment_id"]), ("not_evaluated", run))

    async def test_better_than_base_with_real_failures(self):
        base = await self._exp(baseline=True)
        await self._result(base, 0.1)
        run = await self._exp()
        failures = [{"prompt": f"q{i}", "reference": "a", "prediction": "b", "row_exact_match": 0.0} for i in range(8)]
        await self._result(run, 0.8, {"failures_preview": failures, "failed_count": 8})
        out = await self._summary()
        self.assertEqual(out["verdict"], "better")
        self.assertEqual(out["headline"]["metric_id"], "exact_match")
        self.assertEqual((out["headline"]["baseline_value"], out["headline"]["trained_value"]), (0.1, 0.8))
        self.assertEqual(len(out["failures"]), 5)
        self.assertEqual(out["failed_count"], 8)

    async def test_worse_and_no_baseline(self):
        run = await self._exp()
        await self._result(run, 0.3)
        self.assertEqual((await self._summary(run))["verdict"], "no_baseline")
        base = await self._exp(baseline=True)
        await self._result(base, 0.6)
        self.assertEqual((await self._summary(run))["verdict"], "worse")

    async def test_legacy_results_fall_back_to_preview_failures(self):
        base = await self._exp(baseline=True)
        await self._result(base, 0.1)
        run = await self._exp()
        preview = [
            {"prompt": "q1", "reference": "billing", "prediction": "billing"},
            {"prompt": "q2", "reference": "billing", "prediction": "shipping"},
        ]
        await self._result(run, 0.5, {"predictions_preview": preview})
        out = await self._summary(run)
        self.assertEqual([f["prompt"] for f in out["failures"]], ["q2"])


if __name__ == "__main__":
    unittest.main()
