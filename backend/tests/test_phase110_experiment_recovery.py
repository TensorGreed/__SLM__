"""Story 1.7 — experiment lifecycle recovery + checkpoint-compat gate.

Pins the two surfaces that close the 9/10/11 incident series:

- ``reset_experiment`` / ``delete_experiment`` / ``bulk_archive_failed``
  let an operator recover from a chain of FAILED experiments without
  hand-crafted SQL + ``mv`` commands. All idempotent. All refuse to
  touch RUNNING rows.
- ``_check_checkpoint_compatibility`` in ``backend/scripts/train.py``
  refuses to resume from a checkpoint whose adapter_config doesn't
  match the current run's LoRA / base-model config — the exact failure
  mode that ate experiments 9, 10, and 11 in May 2026.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import (  # noqa: E402
    Checkpoint,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.services.experiment_recovery_service import (  # noqa: E402
    bulk_archive_failed,
    delete_experiment,
    reset_experiment,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-phase110-{uuid.uuid4().hex[:8]}"
)


def _load_train_module():
    """train.py isn't an installed module — load it the same way the
    runtime does so we can poke its private helpers in tests."""
    here = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", here)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_script"] = mod
    spec.loader.exec_module(mod)
    return mod


class CheckpointCompatibilityGateTests(unittest.TestCase):
    """Pin the gate that prevents the experiment-9/10/11 failure mode."""

    @classmethod
    def setUpClass(cls):
        cls.train = _load_train_module()

    def _make_checkpoint(
        self, *, base_model: str, r: int, target_modules: list[str]
    ) -> Path:
        ckpt = Path(tempfile.mkdtemp()) / "checkpoint-4"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": base_model,
                    "r": r,
                    "lora_alpha": r * 2,
                    "target_modules": target_modules,
                    "peft_type": "LORA",
                }
            ),
            encoding="utf-8",
        )
        return ckpt

    def test_matching_config_passes(self):
        ckpt = self._make_checkpoint(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            r=8,
            target_modules=["q_proj", "v_proj"],
        )
        ok, msg = self.train._check_checkpoint_compatibility(
            ckpt,
            current_config={"lora_r": 8, "target_modules": ["q_proj", "v_proj"]},
            current_base_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertTrue(ok)
        self.assertIsNone(msg)

    def test_lora_r_mismatch_blocks(self):
        """The exact failure mode of experiments 10 and 11: rank=16
        checkpoint, rank=8 current config."""
        ckpt = self._make_checkpoint(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            r=16,
            target_modules=["q_proj", "v_proj"],
        )
        ok, msg = self.train._check_checkpoint_compatibility(
            ckpt,
            current_config={"lora_r": 8, "target_modules": ["q_proj", "v_proj"]},
            current_base_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertFalse(ok)
        self.assertIn("lora_r", msg)
        self.assertIn("16", msg)
        self.assertIn("8", msg)

    def test_base_model_mismatch_blocks(self):
        """Exp 9 hit this: phi-2 checkpoint, Qwen current."""
        ckpt = self._make_checkpoint(
            base_model="microsoft/phi-2",
            r=8,
            target_modules=["q_proj", "v_proj"],
        )
        ok, msg = self.train._check_checkpoint_compatibility(
            ckpt,
            current_config={"lora_r": 8, "target_modules": ["q_proj", "v_proj"]},
            current_base_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertFalse(ok)
        self.assertIn("base_model", msg)
        self.assertIn("phi-2", msg)

    def test_target_modules_mismatch_blocks(self):
        ckpt = self._make_checkpoint(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        ok, msg = self.train._check_checkpoint_compatibility(
            ckpt,
            current_config={"lora_r": 8, "target_modules": ["q_proj", "v_proj"]},
            current_base_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertFalse(ok)
        self.assertIn("target_modules", msg)

    def test_missing_adapter_config_does_not_block(self):
        """Non-LoRA checkpoints or older PEFT versions may not write
        an adapter_config.json. Conservative default: pass."""
        ckpt = Path(tempfile.mkdtemp()) / "checkpoint-1"
        ckpt.mkdir(parents=True)
        ok, msg = self.train._check_checkpoint_compatibility(
            ckpt,
            current_config={"lora_r": 8},
            current_base_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertTrue(ok)
        self.assertIsNone(msg)

    def test_resolve_resume_no_explicit_value_does_not_auto_pick(self):
        """Stricter default: when ``resume_from_checkpoint`` is missing
        from config, we no longer scan output_dir for a stale checkpoint.
        Prevents the brand-new-experiment-in-recycled-dir failure mode."""
        out_dir = Path(tempfile.mkdtemp())
        (out_dir / "checkpoint-9").mkdir()
        result = self.train._resolve_resume_checkpoint(
            output_dir=out_dir,
            resume_value=None,
            warnings=[],
            current_config={"lora_r": 8},
            current_base_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertIsNone(result)

    def test_resolve_resume_explicit_auto_with_incompat_raises(self):
        """Caller opted into ``auto`` resume and the latest checkpoint
        is incompatible — gate raises before torch sees the weights."""
        out_dir = Path(tempfile.mkdtemp())
        ckpt_dir = out_dir / "checkpoint-9"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "r": 16,
                    "target_modules": ["q_proj", "v_proj"],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(self.train.CheckpointCompatibilityError) as ctx:
            self.train._resolve_resume_checkpoint(
                output_dir=out_dir,
                resume_value="auto",
                warnings=[],
                current_config={
                    "lora_r": 8,
                    "target_modules": ["q_proj", "v_proj"],
                },
                current_base_model="Qwen/Qwen2.5-1.5B-Instruct",
            )
        self.assertIn("lora_r", str(ctx.exception))


class ExperimentRecoveryServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_project(self, label: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{label}-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_failed_experiment(
        self,
        project_id: int,
        *,
        with_output_dir: bool = True,
        with_checkpoints: int = 0,
    ) -> int:
        """Materialize a failed experiment row + optional on-disk
        artifacts so we can verify cleanup semantics end-to-end."""

        async def _go() -> int:
            output_path = None
            if with_output_dir:
                d = (
                    TEST_DATA_DIR
                    / "projects"
                    / str(project_id)
                    / "experiments"
                    / f"failed-{uuid.uuid4().hex[:6]}"
                )
                d.mkdir(parents=True, exist_ok=True)
                (d / "training_report.json").write_text("{}", encoding="utf-8")
                output_path = str(d)

            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=project_id,
                    name=f"failed-{uuid.uuid4().hex[:6]}",
                    status=ExperimentStatus.FAILED,
                    training_mode=TrainingMode.SFT,
                    base_model="Qwen/Qwen2.5-1.5B-Instruct",
                    output_dir=output_path,
                    started_at=datetime.now(timezone.utc),
                    final_train_loss=1.5,
                    final_eval_loss=1.6,
                    config={"lora_r": 8, "_runtime": {"task_id": "old"}},
                )
                session.add(exp)
                await session.commit()
                await session.refresh(exp)
                for i in range(with_checkpoints):
                    session.add(
                        Checkpoint(
                            experiment_id=exp.id,
                            epoch=1,
                            step=i,
                            eval_loss=1.6,
                            file_path=f"{output_path}/checkpoint-{i}"
                            if output_path
                            else "",
                            is_best=(i == 0),
                        )
                    )
                await session.commit()
                return int(exp.id)

        return asyncio.run(_go())

    def _run_async(self, coro_fn):
        async def _go():
            async with async_session_factory() as session:
                result = await coro_fn(session)
                await session.commit()
                return result

        return asyncio.run(_go())

    def _get_status(self, exp_id: int) -> str:
        async def _go():
            async with async_session_factory() as session:
                exp = await session.get(Experiment, exp_id)
                return exp.status.value if exp else "missing"

        return asyncio.run(_go())

    def _count_checkpoints(self, exp_id: int) -> int:
        async def _go():
            async with async_session_factory() as session:
                exp = await session.get(Experiment, exp_id)
                if exp is None:
                    return -1
                from sqlalchemy import select

                rows = await session.execute(
                    select(Checkpoint).where(Checkpoint.experiment_id == exp_id)
                )
                return len(list(rows.scalars().all()))

        return asyncio.run(_go())

    # ── reset_experiment ────────────────────────────────────────────

    def test_reset_archives_output_dir_and_flips_status(self):
        pid = self._create_project("recovery-reset")
        exp_id = self._create_failed_experiment(
            pid, with_output_dir=True, with_checkpoints=3
        )
        # Pre-condition: output dir exists.
        async def _orig_output():
            async with async_session_factory() as session:
                exp = await session.get(Experiment, exp_id)
                return exp.output_dir

        orig_dir = asyncio.run(_orig_output())
        self.assertTrue(Path(orig_dir).exists())
        self.assertEqual(self._count_checkpoints(exp_id), 3)

        report = self._run_async(
            lambda s: reset_experiment(
                s, project_id=pid, experiment_id=exp_id
            )
        )

        self.assertEqual(report["new_status"], "pending")
        self.assertEqual(report["previous_status"], "failed")
        self.assertEqual(report["checkpoints_deleted"], 3)
        # Output dir moved aside.
        self.assertFalse(Path(orig_dir).exists())
        self.assertTrue(Path(report["archived_output_dir"]).exists())
        # Status flipped.
        self.assertEqual(self._get_status(exp_id), "pending")
        # Checkpoint rows cleared.
        self.assertEqual(self._count_checkpoints(exp_id), 0)

    def test_reset_is_idempotent(self):
        pid = self._create_project("recovery-idempotent")
        exp_id = self._create_failed_experiment(pid, with_checkpoints=2)
        first = self._run_async(
            lambda s: reset_experiment(s, project_id=pid, experiment_id=exp_id)
        )
        second = self._run_async(
            lambda s: reset_experiment(s, project_id=pid, experiment_id=exp_id)
        )
        self.assertEqual(first["checkpoints_deleted"], 2)
        self.assertEqual(second["checkpoints_deleted"], 0)
        # Status stays PENDING; second call is a safe no-op effect-
        # wise on the DB row (still pending, no checkpoints, no extra
        # archive needed because output_dir was already nuked).
        self.assertEqual(self._get_status(exp_id), "pending")

    def test_reset_refuses_running_experiment(self):
        pid = self._create_project("recovery-running")

        async def _make_running() -> int:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=pid,
                    name="live-run",
                    status=ExperimentStatus.RUNNING,
                    training_mode=TrainingMode.SFT,
                    base_model="Qwen/Qwen2.5-1.5B-Instruct",
                )
                session.add(exp)
                await session.commit()
                await session.refresh(exp)
                return int(exp.id)

        exp_id = asyncio.run(_make_running())

        with self.assertRaises(ValueError) as ctx:
            self._run_async(
                lambda s: reset_experiment(
                    s, project_id=pid, experiment_id=exp_id
                )
            )
        self.assertEqual(str(ctx.exception), "experiment_running")

    def test_reset_missing_experiment_raises(self):
        pid = self._create_project("recovery-missing")
        with self.assertRaises(ValueError) as ctx:
            self._run_async(
                lambda s: reset_experiment(
                    s, project_id=pid, experiment_id=999_999
                )
            )
        self.assertEqual(str(ctx.exception), "experiment_not_found")

    # ── delete_experiment ───────────────────────────────────────────

    def test_delete_removes_db_row_and_output_dir(self):
        pid = self._create_project("recovery-delete")
        exp_id = self._create_failed_experiment(
            pid, with_output_dir=True, with_checkpoints=2
        )

        async def _orig_output():
            async with async_session_factory() as session:
                exp = await session.get(Experiment, exp_id)
                return exp.output_dir

        orig_dir = asyncio.run(_orig_output())

        report = self._run_async(
            lambda s: delete_experiment(
                s, project_id=pid, experiment_id=exp_id
            )
        )

        self.assertTrue(report["output_dir_removed"])
        self.assertEqual(report["checkpoints_deleted"], 2)
        self.assertFalse(Path(orig_dir).exists())
        self.assertEqual(self._get_status(exp_id), "missing")

    def test_delete_refuses_running_experiment(self):
        pid = self._create_project("recovery-delete-running")

        async def _make_running() -> int:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=pid,
                    name="live-run",
                    status=ExperimentStatus.RUNNING,
                    training_mode=TrainingMode.SFT,
                    base_model="Qwen/Qwen2.5-1.5B-Instruct",
                )
                session.add(exp)
                await session.commit()
                await session.refresh(exp)
                return int(exp.id)

        exp_id = asyncio.run(_make_running())
        with self.assertRaises(ValueError) as ctx:
            self._run_async(
                lambda s: delete_experiment(
                    s, project_id=pid, experiment_id=exp_id
                )
            )
        self.assertEqual(str(ctx.exception), "experiment_running")

    # ── bulk_archive_failed ─────────────────────────────────────────

    def test_bulk_archive_resets_every_failed_row(self):
        """The 9/10/11 recovery story in one call — three FAILED
        experiments in a project, archive sweep flips all to PENDING."""
        pid = self._create_project("recovery-bulk")
        exp_ids = [
            self._create_failed_experiment(pid, with_checkpoints=1)
            for _ in range(3)
        ]

        report = self._run_async(
            lambda s: bulk_archive_failed(s, project_id=pid)
        )

        self.assertEqual(report["total_failed"], 3)
        self.assertEqual(report["reset_count"], 3)
        self.assertEqual(report["skipped_count"], 0)
        for exp_id in exp_ids:
            self.assertEqual(self._get_status(exp_id), "pending")

    def test_bulk_archive_no_failed_returns_empty_report(self):
        pid = self._create_project("recovery-bulk-empty")
        report = self._run_async(
            lambda s: bulk_archive_failed(s, project_id=pid)
        )
        self.assertEqual(report["total_failed"], 0)
        self.assertEqual(report["reset_count"], 0)


class ExperimentRecoveryApiTests(unittest.TestCase):
    """Quick end-to-end on the three new endpoints — just enough to
    pin the URL contracts + error translations. The service-level
    semantics are pinned by ExperimentRecoveryServiceTests above."""

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_project(self, label: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{label}-api-{uuid.uuid4().hex[:6]}"},
        )
        return int(resp.json()["id"])

    def _create_failed(self, pid: int) -> int:
        async def _go() -> int:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=pid,
                    name=f"api-{uuid.uuid4().hex[:6]}",
                    status=ExperimentStatus.FAILED,
                    training_mode=TrainingMode.SFT,
                    base_model="Qwen/Qwen2.5-1.5B-Instruct",
                    output_dir=None,
                )
                session.add(exp)
                await session.commit()
                await session.refresh(exp)
                return int(exp.id)

        return asyncio.run(_go())

    def test_reset_endpoint_happy_path(self):
        pid = self._create_project("api-reset")
        exp_id = self._create_failed(pid)
        resp = self.client.post(
            f"/api/projects/{pid}/training/experiments/{exp_id}/reset"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["new_status"], "pending")

    def test_delete_endpoint_happy_path(self):
        pid = self._create_project("api-delete")
        exp_id = self._create_failed(pid)
        resp = self.client.delete(
            f"/api/projects/{pid}/training/experiments/{exp_id}"
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_reset_missing_returns_404(self):
        pid = self._create_project("api-reset-404")
        resp = self.client.post(
            f"/api/projects/{pid}/training/experiments/999999/reset"
        )
        self.assertEqual(resp.status_code, 404)

    def test_bulk_archive_endpoint(self):
        pid = self._create_project("api-bulk")
        for _ in range(2):
            self._create_failed(pid)
        resp = self.client.post(
            f"/api/projects/{pid}/training/experiments/bulk-archive-failed"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["total_failed"], 2)
        self.assertEqual(resp.json()["reset_count"], 2)


if __name__ == "__main__":
    unittest.main()
