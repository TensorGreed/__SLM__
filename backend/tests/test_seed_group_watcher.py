"""Phase 7k: the training watcher Job of a seed-group leader reads its seed runs.

The leader never trains itself (no checkpoints, ``total_steps`` 0), so the bell said "initializing" for the
whole group and "0 steps" at the end. The watcher now reports "seed 2 of 3 · step 40/216" while the seeds
train, and its result carries ``seed_runs`` with ``total_steps: null``.

Private engine per test (like test_post_training_lift_eval), so the shared StaticPool engine caveat doesn't
apply; no Job is started (``start_job`` is captured and the runner driven directly).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401,E402
from app.api import training as training_api  # noqa: E402
from app.database import Base  # noqa: E402
from app.models.experiment import Checkpoint, Experiment, ExperimentStatus, TrainingMode  # noqa: E402
from app.models.project import Project  # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


class _Handle:
    def __init__(self):
        self.progress: list[tuple[float | None, str]] = []

    async def set_progress(self, fraction=None, message=None):
        self.progress.append((fraction, message))

    async def check_cancelled(self):
        return False


class SeedGroupWatcherTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{Path(self._tmp.name) / 'w.db'}", future=True)
        self.sf = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def _group(self, statuses: list[ExperimentStatus], *, leader_status=ExperimentStatus.RUNNING,
                     steps: list[int | None] | None = None) -> tuple[int, int, list[int]]:
        group = uuid4().hex
        async with self.sf() as db:
            project = Project(name=f"w-{uuid4().hex[:6]}", description="", base_model_name=BASE)
            db.add(project)
            await db.flush()
            leader = Experiment(project_id=project.id, name="smol · 3 seeds", base_model=BASE,
                                training_mode=TrainingMode.SFT, status=leader_status, seed_group_id=group,
                                config={"num_seeds": len(statuses)})
            db.add(leader)
            await db.flush()
            children = []
            for i, status in enumerate(statuses):
                child = Experiment(project_id=project.id, name=f"smol (seed={i + 1})", base_model=BASE,
                                   training_mode=TrainingMode.SFT, status=status, seed_group_id=group,
                                   seed_value=i + 1, total_steps=216, config={})
                db.add(child)
                await db.flush()
                children.append(child.id)
                step = (steps or [None] * len(statuses))[i]
                if step is not None:
                    db.add(Checkpoint(experiment_id=child.id, step=step, epoch=1, train_loss=2.0, file_path=""))
            await db.commit()
            return project.id, leader.id, children

    async def _capture_runner(self, project_id: int, leader_id: int):
        captured: dict = {}

        async def _start_job(db, *, kind, title, runner, project_id, params):
            captured.update(kind=kind, title=title, runner=runner)

        # The watcher binds ``async_session_factory`` when it is spawned: point it at this test's engine.
        async with self.sf() as db:
            with mock.patch("app.services.jobs_service.start_job", _start_job), \
                    mock.patch("app.database.async_session_factory", self.sf):
                await training_api._spawn_training_watcher_job(
                    db, project_id=project_id, experiment_id=leader_id, base_model=BASE,
                )
        return captured

    async def test_leader_progress_names_the_seed_and_its_step(self):
        pid, leader, children = await self._group(
            [ExperimentStatus.COMPLETED, ExperimentStatus.RUNNING, ExperimentStatus.PENDING], steps=[216, 40, None],
        )
        captured = await self._capture_runner(pid, leader)
        self.assertEqual(captured["title"], f"Train · seed group #{leader} (3 seeds) · {BASE}")
        async with self.sf() as db:
            exp = await db.get(Experiment, leader)
            rows = await training_api._seed_group_watch_rows(db, exp)
        self.assertEqual([r["experiment_id"] for r in rows], children)
        fraction, message = training_api._seed_group_progress(rows, elapsed=12)
        self.assertTrue(message.startswith("seed 2 of 3 · step 40/216"), message)
        self.assertAlmostEqual(fraction, (1 + 40 / 216) / 3)

    async def test_finished_group_result_has_seed_runs_and_no_zero_steps(self):
        pid, leader, children = await self._group(
            [ExperimentStatus.COMPLETED] * 3, leader_status=ExperimentStatus.COMPLETED, steps=[216, 216, 216],
        )
        captured = await self._capture_runner(pid, leader)
        handle = _Handle()

        async def _no_lift(db, *, project_id, experiment_id):
            return None

        with mock.patch("app.database.async_session_factory", self.sf), \
                mock.patch("app.services.post_training_eval_service.start_post_training_lift_job", _no_lift):
            result = await captured["runner"](handle)
        self.assertIsNone(result["total_steps"])
        self.assertEqual(result["n_seeds"], 3)
        self.assertEqual([r["experiment_id"] for r in result["seed_runs"]], children)
        self.assertEqual({r["total_steps"] for r in result["seed_runs"]}, {216})
        self.assertTrue(handle.progress)
        self.assertTrue(all("initializing" not in (m or "") for _, m in handle.progress))
        self.assertTrue(handle.progress[-1][1].startswith("3 of 3 seeds done"))

    async def test_a_single_run_is_unchanged(self):
        async with self.sf() as db:
            project = Project(name="single", description="", base_model_name=BASE)
            db.add(project)
            await db.flush()
            exp = Experiment(project_id=project.id, name="one", base_model=BASE, training_mode=TrainingMode.SFT,
                             status=ExperimentStatus.COMPLETED, total_steps=120, config={})
            db.add(exp)
            await db.commit()
            self.assertIsNone(await training_api._seed_group_watch_rows(db, exp))
            self.assertIsNone(training_api._seed_group_size(exp))
        captured = await self._capture_runner(project.id, exp.id)
        self.assertEqual(captured["title"], f"Train · exp #{exp.id} · {BASE}")

        async def _no_lift(db, *, project_id, experiment_id):
            return None

        with mock.patch("app.database.async_session_factory", self.sf), \
                mock.patch("app.services.post_training_eval_service.start_post_training_lift_job", _no_lift):
            result = await captured["runner"](_Handle())
        self.assertEqual(result["total_steps"], 120)
        self.assertNotIn("seed_runs", result)


if __name__ == "__main__":
    unittest.main()
