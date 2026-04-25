"""Phase 67 — Checkpoint browser backend (priority.md P16, RM3).

Covers:
- ``GET /training/runs/{id}/checkpoints`` returns every checkpoint row for
  a run, ordered by step ascending, with the spec's column shape
  (``run_id, step, epoch, loss, metrics_blob, path, is_best, promoted_at``).
- ``POST /training/runs/{id}/checkpoints/{step}/promote`` flips ``is_best``
  to true on the chosen step + stamps ``promoted_at``, while clearing the
  flag on every sibling checkpoint of the same run (exclusive promotion).
- Re-promoting the same step is idempotent — the flag stays true and
  ``promoted_at`` refreshes.
- 404s with stable reason codes: ``experiment_not_found`` for unknown
  runs, ``checkpoint_not_found`` for unknown steps.
- ``POST /training/runs/{id}/resume-from/{step}`` creates a **new**
  PENDING experiment whose config is the parent manifest's resolved
  config + a ``_resume_from`` block carrying the parent run id, manifest
  id, step, epoch, and on-disk file path. Non-overridden config keys are
  preserved verbatim, and the row's training_mode + base_model flow
  through from the manifest.
- Resume returns ``manifest_not_captured`` when the parent run has no
  P14 manifest yet, and ``checkpoint_not_found`` when the requested step
  was never written.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase67_checkpoint_browser.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase67_checkpoint_browser_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Checkpoint, Experiment, TrainingMode
from app.services.training_manifest_service import capture_training_manifest


_PARENT_CONFIG = {
    "recipe": "recipe.pipeline.sft_default",
    "training_mode": "sft",
    "training_runtime_id": "runtime.hf_trainer",
    "seed": 42,
    "learning_rate": 2e-4,
    "num_epochs": 2,
    "batch_size": 4,
    "chat_template": "llama3",
    "max_seq_length": 2048,
}


class Phase67CheckpointBrowserTests(unittest.TestCase):
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
                "name": f"phase67-{uuid.uuid4().hex[:8]}",
                "description": "phase67 checkpoint browser",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _create_experiment(
        self,
        project_id: int,
        *,
        config: dict | None = None,
        base_model: str = "microsoft/phi-2",
        mode: TrainingMode = TrainingMode.SFT,
    ) -> int:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=project_id,
                name=f"exp-{uuid.uuid4().hex[:6]}",
                description="phase67",
                status="completed",
                base_model=base_model,
                training_mode=mode,
                config=dict(config or _PARENT_CONFIG),
            )
            db.add(exp)
            await db.commit()
            await db.refresh(exp)
            return int(exp.id)

    async def _add_checkpoints(
        self,
        experiment_id: int,
        *,
        steps: list[tuple[int, int, float | None, float | None, bool]],
    ) -> None:
        """Write a list of ``(step, epoch, train_loss, eval_loss, is_best)`` rows."""
        async with async_session_factory() as db:
            for step, epoch, train_loss, eval_loss, is_best in steps:
                ckpt = Checkpoint(
                    experiment_id=experiment_id,
                    step=step,
                    epoch=epoch,
                    train_loss=train_loss,
                    eval_loss=eval_loss,
                    file_path=f"/tmp/phase67/ckpt-{experiment_id}-step-{step}",
                    is_best=is_best,
                    metrics={"step": step, "epoch": epoch, "train_loss": train_loss},
                )
                db.add(ckpt)
            await db.commit()

    async def _capture_manifest(self, project_id: int, experiment_id: int) -> None:
        async with async_session_factory() as db:
            await capture_training_manifest(
                db,
                project_id=project_id,
                experiment_id=experiment_id,
                collect_env=False,
            )

    def _bootstrap_run_with_checkpoints(
        self,
        *,
        capture_manifest: bool = True,
        steps: list[tuple[int, int, float | None, float | None, bool]] | None = None,
    ) -> tuple[int, int]:
        steps = steps or [
            (50, 1, 1.42, 1.55, False),
            (100, 1, 1.18, 1.31, False),
            (150, 2, 0.95, 1.09, False),
            (200, 2, 0.81, 0.93, False),
        ]
        project_id = self._create_project()

        async def _run():
            exp_id = await self._create_experiment(project_id)
            await self._add_checkpoints(exp_id, steps=steps)
            if capture_manifest:
                await self._capture_manifest(project_id, exp_id)
            return exp_id

        return project_id, asyncio.run(_run())

    # -- list ---------------------------------------------------------------

    def test_list_returns_every_checkpoint_in_step_order_with_spec_shape(self):
        project_id, exp_id = self._bootstrap_run_with_checkpoints()

        resp = self.client.get(
            f"/api/projects/{project_id}/training/runs/{exp_id}/checkpoints"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["experiment_id"], exp_id)
        self.assertEqual(body["count"], 4)
        steps = [c["step"] for c in body["checkpoints"]]
        self.assertEqual(steps, [50, 100, 150, 200])
        first = body["checkpoints"][0]
        # Spec column shape — every required key is present.
        for key in (
            "run_id",
            "step",
            "epoch",
            "loss",
            "metrics_blob",
            "path",
            "is_best",
            "promoted_at",
        ):
            self.assertIn(key, first, f"missing field {key} on checkpoint payload")
        self.assertEqual(first["run_id"], exp_id)
        self.assertEqual(first["loss"], 1.42)
        self.assertFalse(first["is_best"])
        self.assertIsNone(first["promoted_at"])
        self.assertTrue(first["path"].endswith("step-50"))

    def test_list_returns_404_for_unknown_experiment(self):
        project_id = self._create_project()
        resp = self.client.get(
            f"/api/projects/{project_id}/training/runs/999999/checkpoints"
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "experiment_not_found")

    def test_list_returns_empty_array_for_run_without_checkpoints(self):
        project_id = self._create_project()
        async def _run():
            return await self._create_experiment(project_id)
        exp_id = asyncio.run(_run())
        resp = self.client.get(
            f"/api/projects/{project_id}/training/runs/{exp_id}/checkpoints"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["count"], 0)
        self.assertEqual(body["checkpoints"], [])

    # -- promote ------------------------------------------------------------

    def test_promote_marks_best_clears_other_best_and_stamps_promoted_at(self):
        # Seed run with one already-best step at 50; we'll promote 150 instead.
        steps = [
            (50, 1, 1.42, 1.55, True),
            (100, 1, 1.18, 1.31, False),
            (150, 2, 0.95, 1.09, False),
            (200, 2, 0.81, 0.93, False),
        ]
        project_id, exp_id = self._bootstrap_run_with_checkpoints(steps=steps)

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{exp_id}/checkpoints/150/promote",
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["step"], 150)
        self.assertTrue(body["is_best"])
        self.assertIsNotNone(body["promoted_at"])

        # Sibling check via list — exactly one checkpoint flagged best.
        list_resp = self.client.get(
            f"/api/projects/{project_id}/training/runs/{exp_id}/checkpoints"
        )
        bests = [c for c in list_resp.json()["checkpoints"] if c["is_best"]]
        self.assertEqual(len(bests), 1)
        self.assertEqual(bests[0]["step"], 150)
        # The previously-best step 50 had its flag cleared.
        step_50 = next(c for c in list_resp.json()["checkpoints"] if c["step"] == 50)
        self.assertFalse(step_50["is_best"])
        self.assertIsNone(step_50["promoted_at"])

    def test_promote_is_idempotent_and_refreshes_promoted_at(self):
        project_id, exp_id = self._bootstrap_run_with_checkpoints()

        first = self.client.post(
            f"/api/projects/{project_id}/training/runs/{exp_id}/checkpoints/100/promote"
        )
        self.assertEqual(first.status_code, 200, first.text)
        first_promoted_at = first.json()["promoted_at"]
        self.assertIsNotNone(first_promoted_at)

        second = self.client.post(
            f"/api/projects/{project_id}/training/runs/{exp_id}/checkpoints/100/promote"
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertTrue(second.json()["is_best"])
        # Re-promotion still leaves it the only best, and refreshes the
        # timestamp (>= the first one).
        self.assertGreaterEqual(second.json()["promoted_at"], first_promoted_at)

    def test_promote_returns_404_for_unknown_experiment(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/999999/checkpoints/50/promote"
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "experiment_not_found")

    def test_promote_returns_404_for_unknown_step(self):
        project_id, exp_id = self._bootstrap_run_with_checkpoints()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{exp_id}/checkpoints/9999/promote"
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "checkpoint_not_found")

    # -- resume-from --------------------------------------------------------

    def test_resume_creates_pending_experiment_with_resume_from_block(self):
        project_id, parent_id = self._bootstrap_run_with_checkpoints()

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{parent_id}/resume-from/150",
            json={"run_name": "resumed at 150"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        new_id = body["experiment_id"]
        self.assertNotEqual(new_id, parent_id)
        self.assertEqual(body["name"], "resumed at 150")
        self.assertEqual(body["status"], "pending")
        # Manifest config flows through verbatim for non-resume keys.
        cfg = body["config"]
        for key, value in _PARENT_CONFIG.items():
            self.assertEqual(cfg.get(key), value, f"{key} drifted on resume")
        # Resume parentage block points at the parent + checkpoint we asked for.
        resume = body["resume_from"]
        self.assertEqual(resume["parent_experiment_id"], parent_id)
        self.assertEqual(resume["checkpoint_step"], 150)
        self.assertEqual(resume["checkpoint_epoch"], 2)
        self.assertEqual(resume["reason"], "resume-from-checkpoint")
        self.assertTrue(resume["checkpoint_path"].endswith("step-150"))
        # base_model + training_mode flow through from the parent manifest.
        self.assertEqual(body["base_model"], "microsoft/phi-2")
        self.assertEqual(body["training_mode"], "sft")

    def test_resume_returns_404_when_parent_manifest_missing(self):
        project_id, parent_id = self._bootstrap_run_with_checkpoints(capture_manifest=False)
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{parent_id}/resume-from/150",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "manifest_not_captured")

    def test_resume_returns_404_for_unknown_step(self):
        project_id, parent_id = self._bootstrap_run_with_checkpoints()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{parent_id}/resume-from/77777",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "checkpoint_not_found")

    def test_resume_returns_404_for_unknown_experiment(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/424242/resume-from/100",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "experiment_not_found")


if __name__ == "__main__":
    unittest.main()
