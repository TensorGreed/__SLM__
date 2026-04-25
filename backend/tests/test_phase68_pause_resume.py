"""Phase 68 — Pause / resume training (priority.md P17, RM3).

Covers:
- ``POST /training/runs/{id}/pause`` on a RUNNING experiment flips
  ``pause_requested`` to true and returns immediately. Non-RUNNING runs
  return 409 ``not_running``; unknown runs return 404
  ``experiment_not_found``.
- The runtime hook helper ``_record_pause_checkpoint`` (called by the
  simulate loop / runtime plugin when it observes ``pause_requested``)
  writes a resume-capable checkpoint at the current step, transitions
  status to PAUSED, clears ``pause_requested``, and stamps a ``_pause``
  block in config with the on-disk checkpoint path.
- ``POST /training/runs/{id}/resume`` on a PAUSED experiment loads the
  latest checkpoint, stamps ``_resume_from`` (parent_experiment_id =
  self, checkpoint_step, checkpoint_path), drops status back through
  PENDING, and re-dispatches via ``start_training`` (patched in tests so
  the simulate background task isn't observed).
- Resume 409s: non-PAUSED → ``not_paused``; PAUSED-without-checkpoint →
  ``no_resume_checkpoint``. Resume 404 ``experiment_not_found`` for
  unknown ids.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


TEST_DB_PATH = Path(__file__).resolve().parent / "phase68_pause_resume.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase68_pause_resume_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Checkpoint, Experiment, ExperimentStatus, TrainingMode
from app.services.training_service import (
    _record_pause_checkpoint,
    pause_training,
    resume_training,
)


_BASE_CONFIG = {
    "recipe": "recipe.pipeline.sft_default",
    "training_mode": "sft",
    "training_runtime_id": "runtime.hf_trainer",
    "seed": 42,
    "learning_rate": 2e-4,
    "num_epochs": 2,
    "batch_size": 4,
}


class Phase68PauseResumeTests(unittest.TestCase):
    @classmethod
    def _cleanup_artifacts(cls):
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        for suffix in ("", "-shm", "-wal"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_artifacts()

    # -- helpers ------------------------------------------------------------

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"phase68-{uuid.uuid4().hex[:8]}",
                "description": "phase68 pause/resume",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _create_experiment(
        self,
        project_id: int,
        *,
        status: ExperimentStatus = ExperimentStatus.RUNNING,
        config: dict | None = None,
    ) -> int:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=project_id,
                name=f"exp-{uuid.uuid4().hex[:6]}",
                description="phase68",
                status=status,
                base_model="microsoft/phi-2",
                training_mode=TrainingMode.SFT,
                config=dict(config or _BASE_CONFIG),
            )
            db.add(exp)
            await db.commit()
            await db.refresh(exp)
            # output_dir is needed by _record_pause_checkpoint.
            exp.output_dir = str(TEST_DATA_DIR / f"exp-{exp.id}")
            await db.commit()
            return int(exp.id)

    async def _seed_checkpoint(self, exp_id: int, *, step: int, epoch: int = 1) -> None:
        async with async_session_factory() as db:
            ckpt = Checkpoint(
                experiment_id=exp_id,
                epoch=epoch,
                step=step,
                train_loss=1.0,
                eval_loss=1.1,
                file_path=f"/tmp/phase68/ckpt-{exp_id}-step-{step}",
                metrics={"step": step},
            )
            db.add(ckpt)
            await db.commit()

    async def _read_experiment(self, exp_id: int) -> Experiment:
        async with async_session_factory() as db:
            return (
                await db.execute(
                    Experiment.__table__.select().where(Experiment.id == exp_id)
                )
            ).fetchone()

    async def _refetch(self, exp_id: int) -> dict:
        from sqlalchemy import select

        async with async_session_factory() as db:
            row = (
                await db.execute(select(Experiment).where(Experiment.id == exp_id))
            ).scalar_one()
            return {
                "status": row.status.value,
                "pause_requested": bool(row.pause_requested),
                "config": dict(row.config or {}),
                "output_dir": row.output_dir,
            }

    # -- pause endpoint ----------------------------------------------------

    def test_pause_running_run_sets_flag_and_returns_200(self):
        project_id = self._create_project()
        async def _setup():
            return await self._create_experiment(project_id, status=ExperimentStatus.RUNNING)
        exp_id = asyncio.run(_setup())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{exp_id}/pause"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["pause_requested"])
        self.assertEqual(body["status"], "running")  # not yet paused — runtime hasn't observed it
        self.assertIn("pause_requested_at", body)

        state = asyncio.run(self._refetch(exp_id))
        self.assertTrue(state["pause_requested"])
        self.assertEqual(state["status"], "running")
        self.assertIn("pause_requested_at", state["config"].get("_pause", {}))

    def test_pause_non_running_returns_409(self):
        project_id = self._create_project()
        async def _setup():
            return await self._create_experiment(project_id, status=ExperimentStatus.PENDING)
        exp_id = asyncio.run(_setup())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{exp_id}/pause"
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["detail"], "not_running")

    def test_pause_unknown_experiment_returns_404(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/424242/pause"
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "experiment_not_found")

    # -- runtime hook ------------------------------------------------------

    def test_record_pause_checkpoint_writes_checkpoint_and_paused_status(self):
        project_id = self._create_project()
        async def _run():
            exp_id = await self._create_experiment(
                project_id, status=ExperimentStatus.RUNNING
            )
            # Simulate the pause endpoint setting the flag.
            async with async_session_factory() as db:
                from sqlalchemy import select

                exp = (
                    await db.execute(select(Experiment).where(Experiment.id == exp_id))
                ).scalar_one()
                exp.pause_requested = True
                await db.commit()
                output_dir = Path(str(exp.output_dir))
            # The runtime hook the simulate loop calls when it observes
            # pause_requested mid-iteration.
            await _record_pause_checkpoint(
                experiment_id=exp_id,
                project_id=project_id,
                output_dir=output_dir,
                step=42,
                epoch=1,
                train_loss=0.97,
                eval_loss=None,
            )
            return exp_id, output_dir

        exp_id, output_dir = asyncio.run(_run())

        # Status flipped to PAUSED, pause_requested cleared, _pause stamp written.
        state = asyncio.run(self._refetch(exp_id))
        self.assertEqual(state["status"], "paused")
        self.assertFalse(state["pause_requested"])
        pause_block = state["config"].get("_pause") or {}
        self.assertEqual(pause_block.get("paused_at_step"), 42)
        self.assertEqual(pause_block.get("paused_at_epoch"), 1)
        self.assertTrue(str(pause_block.get("checkpoint_path") or "").endswith("step-42.json"))

        # Checkpoint row + on-disk file both exist at the pause step.
        from sqlalchemy import select

        async def _ckpts():
            async with async_session_factory() as db:
                rows = (
                    await db.execute(
                        select(Checkpoint).where(Checkpoint.experiment_id == exp_id)
                    )
                ).scalars().all()
                return [(r.step, r.train_loss, r.file_path) for r in rows]

        ckpts = asyncio.run(_ckpts())
        self.assertEqual(len(ckpts), 1)
        step, train_loss, file_path = ckpts[0]
        self.assertEqual(step, 42)
        self.assertEqual(train_loss, 0.97)
        self.assertTrue(Path(file_path).exists(), file_path)

    # -- resume endpoint ---------------------------------------------------

    def test_resume_paused_run_redispatches_with_resume_marker(self):
        project_id = self._create_project()
        async def _setup():
            exp_id = await self._create_experiment(
                project_id, status=ExperimentStatus.PAUSED
            )
            await self._seed_checkpoint(exp_id, step=80, epoch=1)
            return exp_id
        exp_id = asyncio.run(_setup())

        # Patch start_training so resume's re-dispatch returns a deterministic
        # payload without spawning a real simulate task.
        async def _fake_start(_db, project_id_arg, exp_id_arg):
            return {
                "experiment_id": exp_id_arg,
                "status": "running",
                "message": "fake-redispatch",
                "backend": "simulate",
                "runtime_id": "builtin.simulate",
                "task_id": "fake-task-id",
            }

        with patch(
            "app.services.training_service.start_training",
            side_effect=_fake_start,
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/training/runs/{exp_id}/resume"
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["resumed_from_step"], 80)
        self.assertEqual(body["message"], "fake-redispatch")

        # The pre-dispatch state changes are persisted: pause_requested
        # cleared, status moved to PENDING (start_training would normally
        # flip it to RUNNING, but our fake doesn't), _resume_from stamped.
        state = asyncio.run(self._refetch(exp_id))
        self.assertFalse(state["pause_requested"])
        resume_block = state["config"].get("_resume_from") or {}
        self.assertEqual(resume_block.get("parent_experiment_id"), exp_id)
        self.assertEqual(resume_block.get("checkpoint_step"), 80)
        self.assertEqual(resume_block.get("reason"), "resume-paused-run")
        # resume_history was appended on the _pause block.
        history = (state["config"].get("_pause") or {}).get("resume_history") or []
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["from_step"], 80)

    def test_resume_non_paused_returns_409(self):
        project_id = self._create_project()
        async def _setup():
            return await self._create_experiment(project_id, status=ExperimentStatus.RUNNING)
        exp_id = asyncio.run(_setup())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{exp_id}/resume"
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["detail"], "not_paused")

    def test_resume_paused_without_checkpoint_returns_409(self):
        project_id = self._create_project()
        async def _setup():
            return await self._create_experiment(project_id, status=ExperimentStatus.PAUSED)
        exp_id = asyncio.run(_setup())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{exp_id}/resume"
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["detail"], "no_resume_checkpoint")

    def test_resume_unknown_experiment_returns_404(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/777777/resume"
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "experiment_not_found")


if __name__ == "__main__":
    unittest.main()
