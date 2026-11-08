"""Resume a real (external-runtime) run from one of its Trainer checkpoints.

Covers:
- ``checkpoint_dir_details`` reads step / losses / best flag / resumability from a
  ``checkpoint-<step>`` dir's ``trainer_state.json``.
- A run that ends with no training report (killed, timed out, crashed) still gets a
  Checkpoint row per ``checkpoint-<step>`` dir, so it can be resumed from.
- ``POST /training/runs/{id}/resume-from/{step}`` on a real run sets
  ``resume_from_checkpoint`` to the checkpoint's absolute dir (what ``train.py`` reads);
  a missing or incomplete checkpoint is refused with 409 ``checkpoint_not_resumable``
  instead of a fork that would silently restart at step 0.
- ``train._resolve_resume_checkpoint`` accepts that absolute dir from another run's
  output dir.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Checkpoint, Experiment, ExperimentStatus, TrainingMode
from app.services.training_manifest_service import capture_training_manifest
from app.services.training_service import _monitor_external_training, checkpoint_dir_details
from sqlalchemy import select

_CLIENT_CM: TestClient | None = None
_CLIENT: TestClient | None = None

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = {"training_mode": "sft", "num_epochs": 2, "lora_r": 8, "learning_rate": 1e-4, "seed": 7}


def setUpModule():
    global _CLIENT_CM, _CLIENT
    _CLIENT_CM = TestClient(app)
    _CLIENT = _CLIENT_CM.__enter__()


def tearDownModule():
    if _CLIENT_CM is not None:
        _CLIENT_CM.__exit__(None, None, None)


def write_checkpoint(root: Path, step: int, *, state: bool = True, weights: bool = True,
                     optimizer: bool = True, best: bool = False, epoch: float = 0.4) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    if weights:
        (path / "adapter_model.safetensors").write_bytes(b"w")
        (path / "adapter_config.json").write_text(json.dumps({"r": 8, "base_model_name_or_path": BASE}))
    if optimizer:
        (path / "optimizer.pt").write_bytes(b"o")
        (path / "scheduler.pt").write_bytes(b"s")
    if state:
        (path / "trainer_state.json").write_text(json.dumps({
            "global_step": step,
            "epoch": epoch,
            "best_model_checkpoint": str(path) if best else None,
            "log_history": [
                {"step": step - 10, "loss": 0.9},
                {"step": step, "loss": 0.5},
                {"step": step, "eval_loss": 0.6},
            ],
        }))
    return path


class CheckpointDirDetailsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(settings.DATA_DIR) / "ckpt-details" / uuid.uuid4().hex[:8]
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_full_checkpoint_is_resumable_with_losses_and_best_flag(self):
        d = checkpoint_dir_details(write_checkpoint(self.root, 50, best=True))
        self.assertTrue(d["resumable"])
        self.assertTrue(d["has_optimizer_state"])
        self.assertEqual(d["global_step"], 50)
        self.assertEqual(d["train_loss"], 0.5)
        self.assertEqual(d["eval_loss"], 0.6)
        self.assertTrue(d["is_best"])

    def test_missing_trainer_state_or_weights_is_not_resumable(self):
        self.assertFalse(checkpoint_dir_details(write_checkpoint(self.root, 10, state=False))["resumable"])
        self.assertFalse(checkpoint_dir_details(write_checkpoint(self.root, 20, weights=False))["resumable"])

    def test_weights_without_optimizer_still_resumable_but_flagged(self):
        d = checkpoint_dir_details(write_checkpoint(self.root, 30, optimizer=False))
        self.assertTrue(d["resumable"])
        self.assertFalse(d["has_optimizer_state"])

    def test_unreadable_trainer_state_is_not_resumable(self):
        path = write_checkpoint(self.root, 40)
        (path / "trainer_state.json").write_text("{not json")
        d = checkpoint_dir_details(path)
        self.assertFalse(d["resumable"])
        self.assertIn("trainer_state_error", d)


class RealRunResumeTests(unittest.TestCase):
    def _project(self) -> int:
        resp = _CLIENT.post("/api/projects", json={"name": f"resume-real-{uuid.uuid4().hex[:8]}", "description": "t"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _run(self, project_id: int, *, status=ExperimentStatus.RUNNING, real: bool = True) -> tuple[int, Path]:
        async def go():
            async with async_session_factory() as db:
                cfg = dict(CONFIG)
                if real:
                    cfg["_runtime"] = {"backend": "external", "runtime_kind": "external_celery"}
                exp = Experiment(project_id=project_id, name=f"r-{uuid.uuid4().hex[:6]}", status=status,
                                 base_model=BASE, training_mode=TrainingMode.SFT, config=cfg)
                db.add(exp)
                await db.commit()
                await db.refresh(exp)
                out = Path(settings.DATA_DIR) / "projects" / str(project_id) / "experiments" / str(exp.id)
                exp.output_dir = str(out)
                await db.commit()
                return int(exp.id), out
        exp_id, out = asyncio.run(go())
        out.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        return exp_id, out

    def _checkpoints(self, exp_id: int) -> list[Checkpoint]:
        async def go():
            async with async_session_factory() as db:
                rows = await db.execute(select(Checkpoint).where(Checkpoint.experiment_id == exp_id).order_by(Checkpoint.step))
                return list(rows.scalars().all())
        return asyncio.run(go())

    def _manifest(self, project_id: int, exp_id: int) -> None:
        async def go():
            async with async_session_factory() as db:
                await capture_training_manifest(db, project_id=project_id, experiment_id=exp_id, collect_env=False)
        asyncio.run(go())

    def _killed(self, exp_id: int, out: Path) -> str:
        return asyncio.run(_monitor_external_training(
            exp_id, SimpleNamespace(returncode=-15), "train.py", out / "external_training.log", out,
            captured_stdout="SLM_METRIC {\"step\": 60}", captured_stderr=""))

    def test_killed_run_without_report_registers_its_checkpoints(self):
        project_id = self._project()
        exp_id, out = self._run(project_id)
        write_checkpoint(out, 50, epoch=0.4)
        write_checkpoint(out, 100, epoch=0.8, best=True)
        (out / "checkpoint-tmp").mkdir()  # not a step dir: ignored
        self.assertEqual(self._killed(exp_id, out), "failed")
        rows = self._checkpoints(exp_id)
        self.assertEqual([r.step for r in rows], [50, 100])
        self.assertEqual(rows[0].epoch, 1)
        self.assertEqual(rows[1].train_loss, 0.5)
        self.assertEqual(rows[1].eval_loss, 0.6)
        self.assertTrue(rows[1].is_best)
        self.assertTrue(rows[0].metrics["resumable"])
        self.assertEqual(rows[0].file_path, str(out / "checkpoint-50"))
        # A second pass (e.g. a re-delivered monitor) adds nothing.
        self._killed(exp_id, out)
        self.assertEqual(len(self._checkpoints(exp_id)), 2)

    def test_fork_of_real_run_sets_resume_from_checkpoint_to_absolute_dir(self):
        project_id = self._project()
        exp_id, out = self._run(project_id)
        write_checkpoint(out, 50)
        self._killed(exp_id, out)
        self._manifest(project_id, exp_id)
        resp = _CLIENT.post(f"/api/projects/{project_id}/training/runs/{exp_id}/resume-from/50", json={})
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["config"]["resume_from_checkpoint"], str((out / "checkpoint-50").resolve()))
        self.assertTrue(body["resume_from"]["resumes_training_state"])
        self.assertTrue(body["resume_from"]["has_optimizer_state"])
        self.assertEqual(body["resume_from"]["checkpoint_step"], 50)
        for key, value in CONFIG.items():
            self.assertEqual(body["config"].get(key), value)

    def test_fork_refused_when_checkpoint_lacks_training_state(self):
        project_id = self._project()
        exp_id, out = self._run(project_id)
        write_checkpoint(out, 50, state=False)
        self._killed(exp_id, out)
        self._manifest(project_id, exp_id)
        resp = _CLIENT.post(f"/api/projects/{project_id}/training/runs/{exp_id}/resume-from/50", json={})
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["detail"], "checkpoint_not_resumable")

    def test_fork_refused_when_real_runs_checkpoint_dir_is_gone(self):
        project_id = self._project()
        exp_id, out = self._run(project_id)
        write_checkpoint(out, 50)
        self._killed(exp_id, out)
        self._manifest(project_id, exp_id)
        shutil.rmtree(out / "checkpoint-50")
        resp = _CLIENT.post(f"/api/projects/{project_id}/training/runs/{exp_id}/resume-from/50", json={})
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["detail"], "checkpoint_not_resumable")

    def test_simulated_run_fork_is_unchanged(self):
        project_id = self._project()
        exp_id, _ = self._run(project_id, status=ExperimentStatus.COMPLETED, real=False)

        async def add():
            async with async_session_factory() as db:
                db.add(Checkpoint(experiment_id=exp_id, epoch=1, step=10, file_path="/nonexistent/sim/step-10",
                                  is_best=False, metrics={}))
                await db.commit()
        asyncio.run(add())
        self._manifest(project_id, exp_id)
        resp = _CLIENT.post(f"/api/projects/{project_id}/training/runs/{exp_id}/resume-from/10", json={})
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertNotIn("resume_from_checkpoint", body["config"])
        self.assertFalse(body["resume_from"]["resumes_training_state"])


class TrainScriptResolvesForkedCheckpointTests(unittest.TestCase):
    def test_absolute_checkpoint_in_another_runs_dir_is_used(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import train  # noqa: E402

        root = Path(settings.DATA_DIR) / "ckpt-train" / uuid.uuid4().hex[:8]
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        parent = write_checkpoint(root / "parent", 50)
        new_out = root / "child"
        new_out.mkdir(parents=True)
        warnings: list[str] = []
        got = train._resolve_resume_checkpoint(
            output_dir=new_out, resume_value=str(parent.resolve()), warnings=warnings,
            current_config={"lora_r": 8}, current_base_model=BASE)
        self.assertEqual(Path(got), parent.resolve())
        self.assertEqual(warnings, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
