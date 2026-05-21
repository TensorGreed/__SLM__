"""Tests for the active-learning recommender (Theme 8 Epic 2)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401
from app.config import settings
from app.database import Base
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import (
    EvalResult,
    Experiment,
    ExperimentStatus,
)
from app.models.project import Project
from app.services.active_learning_service import (
    promote_active_learning_batch,
    propose_active_learning_batch,
)


SAMPLE_GOLD_ROWS = [
    {
        "key": "g001",
        "input": {"question": "How do I reset my password?"},
        "expected": {"answer": "Go to Settings → Security and click Reset password."},
    },
    {
        "key": "g002",
        "input": {"question": "How do I export my invoice?"},
        "expected": {"answer": "Billing → History → Download all."},
    },
    {
        "key": "g003",
        "input": {"question": "Can I close my account?"},
        "expected": {"answer": "Yes, Account → Close. Data is purged after 30 days."},
    },
    {
        "key": "g004",
        "input": {"question": "Where can I download my data?"},
        "expected": {"answer": "Settings → Privacy → Export my data."},
    },
]


def _build_predictions_preview(passes: list[bool]) -> list[dict]:
    """Build a `predictions_preview` list that mirrors what eval
    handlers emit. `passes[i]` decides whether row i is treated as
    a pass (row_exact_match=1) or fail (row_exact_match=0)."""
    out: list[dict] = []
    for idx, ok in enumerate(passes):
        gold = SAMPLE_GOLD_ROWS[idx]
        ref = gold["expected"]["answer"]
        pred = ref if ok else f"WRONG prediction for row {idx}"
        out.append(
            {
                "prompt": gold["input"]["question"],
                "reference": ref,
                "prediction": pred,
                "row_exact_match": 1.0 if ok else 0.0,
                "row_f1": 1.0 if ok else 0.0,
            }
        )
    return out


class ActiveLearningServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmp.name)
        self._prev_data_dir = settings.DATA_DIR
        settings.DATA_DIR = self.tmp_root  # type: ignore[assignment]

        db_path = self.tmp_root / "active_learning.db"
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}", future=True
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        settings.DATA_DIR = self._prev_data_dir
        self._tmp.cleanup()

    async def _seed_project_with_eval_result(
        self,
        passes: list[bool],
    ) -> tuple[int, int, int]:
        """Create a project + experiment + eval result wired to a
        gold JSONL on disk. Returns (project_id, experiment_id,
        eval_result_id)."""
        async with self.session_factory() as db:
            project = Project(
                name=f"al-{uuid4().hex[:8]}",
                description="active-learning tests",
                base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            )
            db.add(project)
            await db.flush()

            # Write the gold JSONL where `propose` will read it.
            gold_path = self.tmp_root / "projects" / str(project.id) / "gold" / "gold.jsonl"
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            with gold_path.open("w", encoding="utf-8") as handle:
                for row in SAMPLE_GOLD_ROWS:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            gold_dataset = Dataset(
                project_id=project.id,
                name="Gold (dev)",
                dataset_type=DatasetType.GOLD_DEV,
                file_path=str(gold_path),
                record_count=len(SAMPLE_GOLD_ROWS),
            )
            db.add(gold_dataset)
            await db.flush()

            exp = Experiment(
                project_id=project.id,
                name="al-test-exp",
                status=ExperimentStatus.COMPLETED,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                output_dir=str(self.tmp_root / "experiments" / "1"),
            )
            db.add(exp)
            await db.flush()

            preview = _build_predictions_preview(passes)
            eval_result = EvalResult(
                experiment_id=exp.id,
                dataset_name="test",
                eval_type="exact_match",
                metrics={"exact_match": sum(passes) / len(passes)},
                details={
                    "dataset": {
                        "id": gold_dataset.id,
                        "name": gold_dataset.name,
                        "dataset_type": gold_dataset.dataset_type.value,
                        "file_path": str(gold_path),
                    },
                    "predictions_preview": preview,
                },
            )
            db.add(eval_result)
            await db.commit()
            return project.id, exp.id, eval_result.id

    # ── propose ───────────────────────────────────────────────────

    async def test_propose_returns_only_failed_rows(self):
        project_id, exp_id, _ = await self._seed_project_with_eval_result(
            passes=[True, False, False, True],
        )
        async with self.session_factory() as db:
            payload = await propose_active_learning_batch(
                db, project_id=project_id, experiment_id=exp_id, max_rows=10,
            )
        self.assertEqual(payload["total_failures"], 2)
        self.assertEqual(payload["total_predictions"], 4)
        indexes = [c["row_index"] for c in payload["candidates"]]
        # Row indexes 1 and 3 are the failures (passes=[T,F,F,T]).
        self.assertEqual(sorted(indexes), [1, 2])

    async def test_propose_respects_max_rows_cap(self):
        project_id, exp_id, _ = await self._seed_project_with_eval_result(
            passes=[False, False, False, False],
        )
        async with self.session_factory() as db:
            payload = await propose_active_learning_batch(
                db, project_id=project_id, experiment_id=exp_id, max_rows=2,
            )
        self.assertEqual(len(payload["candidates"]), 2)
        self.assertEqual(payload["total_failures"], 4)

    async def test_propose_returns_full_source_row_not_truncated_preview(self):
        """Candidates should include the un-truncated reference text
        from the gold JSONL, not just the 160-char preview snippet."""
        project_id, exp_id, _ = await self._seed_project_with_eval_result(
            passes=[True, False, True, True],
        )
        async with self.session_factory() as db:
            payload = await propose_active_learning_batch(
                db, project_id=project_id, experiment_id=exp_id, max_rows=10,
            )
        self.assertEqual(len(payload["candidates"]), 1)
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["row_index"], 1)
        # The reference should match the gold-row reference exactly.
        expected_answer = SAMPLE_GOLD_ROWS[1]["expected"]["answer"]
        self.assertEqual(candidate["reference"], expected_answer)

    async def test_propose_returns_empty_for_unknown_experiment(self):
        project_id, _, _ = await self._seed_project_with_eval_result(passes=[True])
        async with self.session_factory() as db:
            payload = await propose_active_learning_batch(
                db, project_id=project_id, experiment_id=99999, max_rows=10,
            )
        self.assertEqual(payload["candidates"], [])
        self.assertEqual(payload["total_failures"], 0)

    # ── promote ───────────────────────────────────────────────────

    async def test_promote_appends_to_synthetic_dataset_jsonl(self):
        project_id, exp_id, eval_id = await self._seed_project_with_eval_result(
            passes=[True, False, False, True],
        )
        async with self.session_factory() as db:
            result = await promote_active_learning_batch(
                db, project_id=project_id, experiment_id=exp_id,
                row_indexes=[1, 2],
            )
            await db.commit()

        self.assertEqual(result["promoted_count"], 2)
        self.assertEqual(result["skipped_already_promoted"], 0)
        written = Path(result["target_dataset_path"])
        self.assertTrue(written.exists())
        lines = [
            json.loads(line)
            for line in written.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), 2)
        sources = {row.get("source_row_index") for row in lines}
        self.assertEqual(sources, {1, 2})
        self.assertTrue(all(row.get("source") == "active_learning" for row in lines))
        # The answer field carries the full reference text, not a
        # truncated preview.
        answers = {row.get("answer") for row in lines}
        expected = {
            SAMPLE_GOLD_ROWS[1]["expected"]["answer"],
            SAMPLE_GOLD_ROWS[2]["expected"]["answer"],
        }
        self.assertEqual(answers, expected)

        # EvalResult.details["active_learning"]["promoted_indexes"] tracks state.
        async with self.session_factory() as db2:
            from sqlalchemy import select
            er = (
                await db2.execute(select(EvalResult).where(EvalResult.id == eval_id))
            ).scalar_one()
            al_state = (er.details or {}).get("active_learning") or {}
            self.assertEqual(sorted(al_state.get("promoted_indexes") or []), [1, 2])

    async def test_promote_is_idempotent(self):
        project_id, exp_id, _ = await self._seed_project_with_eval_result(
            passes=[True, False, False, True],
        )
        async with self.session_factory() as db:
            first = await promote_active_learning_batch(
                db, project_id=project_id, experiment_id=exp_id,
                row_indexes=[1, 2],
            )
            second = await promote_active_learning_batch(
                db, project_id=project_id, experiment_id=exp_id,
                row_indexes=[1, 2, 0],  # 0 is a passing row; still "valid" index.
            )
            await db.commit()
        self.assertEqual(first["promoted_count"], 2)
        # Re-promoting rows 1+2 is a no-op; index 0 is requested
        # for the first time so it does get promoted (passing rows
        # are an unusual choice but the service doesn't filter — the
        # propose endpoint is what filters; promote trusts the
        # caller).
        self.assertEqual(second["promoted_count"], 1)
        self.assertEqual(second["skipped_already_promoted"], 2)

    async def test_promote_ignores_out_of_range_indexes(self):
        project_id, exp_id, _ = await self._seed_project_with_eval_result(
            passes=[False, False, True, True],
        )
        async with self.session_factory() as db:
            result = await promote_active_learning_batch(
                db, project_id=project_id, experiment_id=exp_id,
                row_indexes=[0, 999, -1, 1],
            )
            await db.commit()
        self.assertEqual(result["promoted_count"], 2)  # 0 and 1
        self.assertEqual(result["skipped_invalid_indexes"], 2)  # 999, -1

    async def test_propose_marks_already_promoted_rows(self):
        project_id, exp_id, _ = await self._seed_project_with_eval_result(
            passes=[True, False, False, True],
        )
        async with self.session_factory() as db:
            await promote_active_learning_batch(
                db, project_id=project_id, experiment_id=exp_id,
                row_indexes=[1],
            )
            await db.commit()
        async with self.session_factory() as db2:
            payload = await propose_active_learning_batch(
                db2, project_id=project_id, experiment_id=exp_id, max_rows=10,
            )
        flagged = {
            c["row_index"]: c["already_promoted"]
            for c in payload["candidates"]
        }
        self.assertTrue(flagged.get(1))
        self.assertFalse(flagged.get(2))
        # Already-promoted rows sort to the bottom of the list.
        # First candidate should be index 2 (not yet promoted).
        self.assertEqual(payload["candidates"][0]["row_index"], 2)

    async def test_promote_raises_for_unknown_project(self):
        async with self.session_factory() as db:
            with self.assertRaisesRegex(ValueError, "project_not_found:99999"):
                await promote_active_learning_batch(
                    db, project_id=99999, experiment_id=1,
                    row_indexes=[0],
                )

    async def test_promote_raises_for_unknown_experiment(self):
        project_id, _, _ = await self._seed_project_with_eval_result(passes=[False])
        async with self.session_factory() as db:
            with self.assertRaisesRegex(ValueError, "eval_result_not_found"):
                await promote_active_learning_batch(
                    db, project_id=project_id, experiment_id=99999,
                    row_indexes=[0],
                )


if __name__ == "__main__":
    unittest.main()
