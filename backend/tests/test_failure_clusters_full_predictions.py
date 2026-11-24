"""Phase 7k: failure clusters read the failing rows the Verdict counts.

Before: clusters came from ``predictions_preview`` (the first 5 rows, cut at 160 characters) re-scored by the
clusterer's own rule, so a judged result with 21 failing rows clustered 0-5 rows that the Verdict may not even
call failures. Now: the full ``eval_predictions/<id>.jsonl`` (rows with ``correct`` = 0, the same rule as
``evaluation_service._prediction_failed``), else ``failures_preview``, else the old path.

Private engine per test (like test_post_training_lift_eval).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401,E402
from app.database import Base  # noqa: E402
from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.services import eval_predictions_service  # noqa: E402
from app.services.evaluation_remediation_service import extract_failing_rows  # noqa: E402
from app.services.failure_cluster_service import cluster_eval_result_failures  # noqa: E402


def _record(i: int, correct: int | None, *, judged: bool = True) -> dict:
    rec = {
        "row_key": f"k{i}", "row_index": i,
        "prompt": f"Question {i}: explain the retention rule for record type {i}." + " detail" * 40,
        "reference": f"Records of type {i} are kept for {i + 1} years.",
        "prediction": f"They are kept for {i + 3} years." if correct == 0 else f"Records of type {i} are kept for {i + 1} years.",
        "row_exact_match": 0.0, "row_f1": 0.4,
    }
    if correct is not None:
        rec["correct"] = correct
    if judged:
        rec["judge"] = {"scale": "verdict", "task": "answer", "label": "judge-x",
                        "verdict": "wrong" if correct == 0 else "correct", "score": 0.0 if correct == 0 else 1.0,
                        "reason": "states the wrong period" if correct == 0 else "", "content_key": f"c{i}"}
    return rec


class FullPredictionsClusterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{Path(self._tmp.name) / 'c.db'}", future=True)
        self.sf = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def _result(self, details: dict, metrics: dict | None = None) -> tuple[int, int]:
        async with self.sf() as db:
            project = Project(name=f"c-{uuid4().hex[:6]}", description="", base_model_name="m")
            db.add(project)
            await db.flush()
            exp = Experiment(project_id=project.id, name="run", base_model="m", training_mode=TrainingMode.SFT,
                             status=ExperimentStatus.COMPLETED, config={})
            db.add(exp)
            await db.flush()
            er = EvalResult(experiment_id=exp.id, dataset_name="test", eval_type="exact_match",
                            metrics=metrics or {"exact_match": 0.0, "f1": 0.4, "judge_correct": 0.3},
                            pass_rate=0.4, details=details)
            db.add(er)
            await db.commit()
            return project.id, er.id

    async def test_judged_result_clusters_every_failing_row_from_the_predictions_file(self):
        # 30 rows: 21 the judge calls wrong (correct 0), 8 right, 1 unjudged (no ``correct``: not a failure).
        records = [_record(i, 0) for i in range(21)] + [_record(21 + i, 1) for i in range(8)] + [_record(29, None, judged=False)]
        preview = [{"prompt": "p", "reference": "r", "prediction": "r", "row_exact_match": 1.0}]  # first rows: passes
        pid, rid = await self._result({"predictions_preview": preview, "failed_count": 21,
                                       "failures_preview": [{"prompt": "x", "reference": "y", "prediction": "z"}] * 20})
        pointer = eval_predictions_service.write_file(pid, rid, records)
        async with self.sf() as db:
            er = await db.get(EvalResult, rid)
            er.details = {**er.details, "eval_predictions_file": pointer}
            await db.commit()
            payload = await cluster_eval_result_failures(db, eval_result_id=rid)
        self.assertEqual(payload["total_failures_analyzed"], 21)
        self.assertEqual(payload["failed_count"], 21)
        self.assertEqual(payload["failures_source"], "eval_predictions_file")
        self.assertEqual(payload["scored_as"], "free text vs answer key, graded by a judge model")
        self.assertEqual(sum(c["failure_count"] for c in payload["clusters"]), 21)
        # Full text, not the 160-character preview cut; the judge's reason rides along.
        exemplar = payload["clusters"][0]["exemplars"][0]
        self.assertGreater(len(exemplar["prompt"]), 160)
        self.assertEqual(exemplar.get("judge_rationale"), "states the wrong period")
        # Stable cluster ids for one result (the remediation-plan linkage).
        async with self.sf() as db:
            again = await cluster_eval_result_failures(db, eval_result_id=rid)
        self.assertEqual([c["cluster_id"] for c in again["clusters"]], [c["cluster_id"] for c in payload["clusters"]])
        self.assertEqual([c["failure_count"] for c in again["clusters"]], [c["failure_count"] for c in payload["clusters"]])

    async def test_without_the_file_it_falls_back_to_failures_preview(self):
        failing = [{"prompt": f"q{i}", "reference": "a", "prediction": "b", "row_exact_match": 0.0} for i in range(20)]
        pid, rid = await self._result({"failures_preview": failing, "failed_count": 57,
                                       "eval_predictions_file": {"path": "eval_predictions/missing.jsonl"}})
        async with self.sf() as db:
            payload = await cluster_eval_result_failures(db, eval_result_id=rid)
        self.assertEqual(payload["failures_source"], "failures_preview")
        self.assertEqual(payload["total_failures_analyzed"], 20)
        self.assertEqual(payload["failed_count"], 57)

    async def test_an_older_result_keeps_the_old_path(self):
        preview = [
            {"prompt": "q1", "reference": "Paris", "prediction": "Paris"},
            {"prompt": "q2", "reference": "Rome", "prediction": "Madrid"},
        ]
        pid, rid = await self._result({"predictions_preview": preview}, metrics={"exact_match": 0.5})
        async with self.sf() as db:
            er = await db.get(EvalResult, rid)
        failures, meta = extract_failing_rows(er, max_failures=200, project_id=pid)
        self.assertEqual(meta["source"], "legacy")
        self.assertIsNone(meta["failed_count"])
        self.assertEqual([f["prompt"] for f in failures], ["q2"])

    async def test_file_without_per_row_scores_falls_back(self):
        records = [_record(i, None, judged=False) for i in range(5)]
        pid, rid = await self._result({"failures_preview": [{"prompt": "q", "reference": "a", "prediction": "b"}], "failed_count": 1})
        pointer = eval_predictions_service.write_file(pid, rid, records)
        async with self.sf() as db:
            er = await db.get(EvalResult, rid)
            er.details = {**er.details, "eval_predictions_file": pointer}
            await db.commit()
        failures, meta = extract_failing_rows(er, max_failures=200, project_id=pid)
        self.assertEqual(meta["source"], "failures_preview")
        self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()
