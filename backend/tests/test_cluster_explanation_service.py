"""Tests for the per-cluster failure explanation service
(Theme 8 Epic 3)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401
from app.config import settings
from app.database import Base
from app.models.experiment import (
    EvalResult,
    Experiment,
    ExperimentStatus,
)
from app.models.project import Project
from app.services.cluster_explanation_service import (
    JudgeUnavailableError,
    explain_failure_cluster,
)


def _predictions_preview(rows: list[tuple[str, str, str]]) -> list[dict]:
    """Build a predictions-preview list with `row_exact_match=0` so
    the failure clusterer treats every row as a failure."""
    return [
        {
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "row_exact_match": 0.0,
            "row_f1": 0.0,
        }
        for prompt, reference, prediction in rows
    ]


class ClusterExplanationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev_data_dir = settings.DATA_DIR
        self._prev_api_url = settings.JUDGE_MODEL_API_URL
        self._prev_api_key = settings.JUDGE_MODEL_API_KEY
        settings.DATA_DIR = Path(self._tmp.name)  # type: ignore[assignment]
        # Default to no judge — individual tests opt in via patching.
        settings.JUDGE_MODEL_API_URL = ""
        settings.JUDGE_MODEL_API_KEY = ""

        db_path = Path(self._tmp.name) / "cluster_explain.db"
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
        settings.DATA_DIR = self._prev_data_dir
        settings.JUDGE_MODEL_API_URL = self._prev_api_url
        settings.JUDGE_MODEL_API_KEY = self._prev_api_key
        self._tmp.cleanup()

    async def _seed_eval_result(
        self,
        preview: list[dict],
    ) -> tuple[int, int]:
        async with self.session_factory() as db:
            project = Project(
                name=f"explain-{uuid4().hex[:8]}",
                description="cluster explain tests",
                base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            )
            db.add(project)
            await db.flush()

            exp = Experiment(
                project_id=project.id,
                name="explain-exp",
                status=ExperimentStatus.COMPLETED,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                output_dir=str(Path(self._tmp.name) / "experiments" / "1"),
            )
            db.add(exp)
            await db.flush()

            eval_result = EvalResult(
                experiment_id=exp.id,
                dataset_name="test",
                eval_type="exact_match",
                metrics={"exact_match": 0.0},
                details={"predictions_preview": preview},
            )
            db.add(eval_result)
            await db.commit()
            return project.id, eval_result.id

    async def test_returns_judge_unavailable_when_no_judge_configured(self):
        preview = _predictions_preview([
            ("Q1?", "A1", "wrong1"),
            ("Q2?", "A2", "wrong2"),
            ("Q3?", "A3", "wrong3"),
        ])
        project_id, eval_id = await self._seed_eval_result(preview)
        async with self.session_factory() as db:
            result = await explain_failure_cluster(
                db,
                project_id=project_id,
                eval_result_id=eval_id,
                cluster_id="cluster-1",
            )
        self.assertEqual(result["status"], "judge_unavailable")
        self.assertEqual(result["explanation"], "")
        self.assertIn("judge model", result["note"].lower())

    async def test_cluster_not_found_returns_soft_error(self):
        preview = _predictions_preview([("Q1?", "A1", "wrong1")])
        project_id, eval_id = await self._seed_eval_result(preview)
        settings.JUDGE_MODEL_API_URL = "http://fake-judge.local/v1/chat/completions"
        async with self.session_factory() as db:
            result = await explain_failure_cluster(
                db,
                project_id=project_id,
                eval_result_id=eval_id,
                cluster_id="cluster-999",
            )
        self.assertEqual(result["status"], "cluster_not_found")
        self.assertEqual(result["explanation"], "")

    async def test_unknown_eval_result_raises(self):
        async with self.session_factory() as db:
            with self.assertRaisesRegex(ValueError, "eval_result_not_found:99999"):
                await explain_failure_cluster(
                    db,
                    project_id=1,
                    eval_result_id=99999,
                    cluster_id="cluster-1",
                )

    async def test_happy_path_writes_explanation_to_details_cache(self):
        preview = _predictions_preview([
            ("Is order urgent?", "urgent", "not urgent"),
            ("Is request urgent?", "urgent", "not urgent"),
            ("Is item urgent?", "urgent", "not urgent"),
        ])
        project_id, eval_id = await self._seed_eval_result(preview)
        settings.JUDGE_MODEL_API_URL = "http://fake-judge.local/v1/chat/completions"
        settings.JUDGE_MODEL_API_KEY = "test-key"

        mock_explain = AsyncMock(
            return_value="Model is dropping the negation marker."
        )
        with patch(
            "app.services.cluster_explanation_service._call_judge_freeform",
            new=mock_explain,
        ):
            async with self.session_factory() as db:
                result = await explain_failure_cluster(
                    db,
                    project_id=project_id,
                    eval_result_id=eval_id,
                    cluster_id="cluster-1",
                )
                await db.commit()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            result["explanation"],
            "Model is dropping the negation marker.",
        )
        self.assertFalse(result["cached"])
        self.assertGreater(result["exemplar_count"], 0)
        self.assertEqual(mock_explain.await_count, 1)

        # Reread → comes from cache, no new judge call.
        with patch(
            "app.services.cluster_explanation_service._call_judge_freeform",
            new=mock_explain,
        ):
            async with self.session_factory() as db2:
                cached = await explain_failure_cluster(
                    db2,
                    project_id=project_id,
                    eval_result_id=eval_id,
                    cluster_id="cluster-1",
                )
        self.assertTrue(cached["cached"])
        self.assertEqual(
            cached["explanation"],
            "Model is dropping the negation marker.",
        )
        # No additional judge call happened on the cached read.
        self.assertEqual(mock_explain.await_count, 1)

    async def test_force_refresh_re_invokes_the_judge(self):
        preview = _predictions_preview([
            ("Q1?", "expected", "wrong"),
            ("Q2?", "expected", "wrong"),
        ])
        project_id, eval_id = await self._seed_eval_result(preview)
        settings.JUDGE_MODEL_API_URL = "http://fake-judge.local/v1/chat/completions"

        first_call = AsyncMock(return_value="first explanation")
        with patch(
            "app.services.cluster_explanation_service._call_judge_freeform",
            new=first_call,
        ):
            async with self.session_factory() as db:
                first = await explain_failure_cluster(
                    db,
                    project_id=project_id,
                    eval_result_id=eval_id,
                    cluster_id="cluster-1",
                )
                await db.commit()

        second_call = AsyncMock(return_value="second explanation")
        with patch(
            "app.services.cluster_explanation_service._call_judge_freeform",
            new=second_call,
        ):
            async with self.session_factory() as db2:
                second = await explain_failure_cluster(
                    db2,
                    project_id=project_id,
                    eval_result_id=eval_id,
                    cluster_id="cluster-1",
                    force_refresh=True,
                )
                await db2.commit()

        self.assertEqual(first["explanation"], "first explanation")
        self.assertEqual(second["explanation"], "second explanation")
        self.assertFalse(second["cached"])
        self.assertEqual(second_call.await_count, 1)

    async def test_judge_call_failure_returns_error_status_no_cache(self):
        preview = _predictions_preview([("Q?", "ref", "pred")])
        project_id, eval_id = await self._seed_eval_result(preview)
        settings.JUDGE_MODEL_API_URL = "http://fake-judge.local/v1/chat/completions"

        boom = AsyncMock(side_effect=RuntimeError("judge endpoint exploded"))
        with patch(
            "app.services.cluster_explanation_service._call_judge_freeform",
            new=boom,
        ):
            async with self.session_factory() as db:
                result = await explain_failure_cluster(
                    db,
                    project_id=project_id,
                    eval_result_id=eval_id,
                    cluster_id="cluster-1",
                )
                await db.commit()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["explanation"], "")
        self.assertIn("judge endpoint exploded", result["note"])

        # A subsequent (non-force) call should retry — errors are
        # NOT cached.
        retry = AsyncMock(return_value="retry succeeded")
        with patch(
            "app.services.cluster_explanation_service._call_judge_freeform",
            new=retry,
        ):
            async with self.session_factory() as db2:
                retried = await explain_failure_cluster(
                    db2,
                    project_id=project_id,
                    eval_result_id=eval_id,
                    cluster_id="cluster-1",
                )
                await db2.commit()
        self.assertEqual(retried["explanation"], "retry succeeded")
        self.assertEqual(retry.await_count, 1)

    async def test_judge_unavailable_does_not_pollute_cache(self):
        """If no judge is configured today, the explanation should
        be regeneratable once the user wires a judge later — i.e.
        the unavailable verdict must NOT land in the cache."""
        preview = _predictions_preview([("Q?", "ref", "pred")])
        project_id, eval_id = await self._seed_eval_result(preview)
        async with self.session_factory() as db:
            first = await explain_failure_cluster(
                db,
                project_id=project_id,
                eval_result_id=eval_id,
                cluster_id="cluster-1",
            )
            await db.commit()
        self.assertEqual(first["status"], "judge_unavailable")

        # Now the user wires a judge — second call should attempt
        # generation instead of returning the cached "unavailable".
        settings.JUDGE_MODEL_API_URL = "http://fake-judge.local/v1/chat/completions"
        mock = AsyncMock(return_value="explanation after judge wired")
        with patch(
            "app.services.cluster_explanation_service._call_judge_freeform",
            new=mock,
        ):
            async with self.session_factory() as db2:
                second = await explain_failure_cluster(
                    db2,
                    project_id=project_id,
                    eval_result_id=eval_id,
                    cluster_id="cluster-1",
                )
                await db2.commit()
        self.assertEqual(second["status"], "ok")
        self.assertEqual(
            second["explanation"], "explanation after judge wired"
        )


if __name__ == "__main__":
    unittest.main()
