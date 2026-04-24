"""Phase 66 — rerun-from-manifest + clone-from-run (priority.md P15).

Covers:
- ``POST /training/runs/{id}/rerun-from-manifest`` creates a new experiment
  whose config equals the parent manifest's ``resolved_config`` verbatim,
  with a ``_rerun_of`` parentage block pointing at the parent + manifest.
- ``POST /training/runs/{id}/clone`` applies shallow ``config_overrides``
  on top of the manifest config; the parentage block records what was
  overridden; unrelated fields are preserved byte-for-byte.
- ``training_mode`` and ``base_model`` overrides in a clone flow through
  to the new Experiment row (not just the config blob).
- 404s: unknown experiment → ``experiment_not_found``; experiment without
  a captured manifest → ``manifest_not_captured``.
- ``POST /training/runs/compare`` tolerance check: identical losses report
  ``reproduced=true``; drift beyond tolerance reports ``reproduced=false``
  and names the failing metric; metrics present on only one side land in
  ``baseline_only`` / ``candidate_only``.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase66_rerun_from_manifest.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase66_rerun_from_manifest_data"

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
from app.models.experiment import Experiment, TrainingMode
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


class Phase66RerunTests(unittest.TestCase):
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
                "name": f"phase66-{uuid.uuid4().hex[:8]}",
                "description": "phase66 rerun",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _create_experiment(
        self,
        project_id: int,
        *,
        config: dict,
        base_model: str = "microsoft/phi-2",
        mode: TrainingMode = TrainingMode.SFT,
        final_train_loss: float | None = None,
        final_eval_loss: float | None = None,
    ) -> int:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=project_id,
                name=f"exp-{uuid.uuid4().hex[:6]}",
                description="phase66",
                status="completed",
                base_model=base_model,
                training_mode=mode,
                config=config,
                final_train_loss=final_train_loss,
                final_eval_loss=final_eval_loss,
            )
            db.add(exp)
            await db.commit()
            await db.refresh(exp)
            return int(exp.id)

    async def _capture_manifest(self, project_id: int, experiment_id: int) -> None:
        async with async_session_factory() as db:
            await capture_training_manifest(
                db,
                project_id=project_id,
                experiment_id=experiment_id,
                collect_env=False,
            )

    def _bootstrap_parent_run(self, config: dict | None = None) -> tuple[int, int]:
        project_id = self._create_project()
        async def run():
            exp_id = await self._create_experiment(
                project_id, config=dict(config or _PARENT_CONFIG)
            )
            await self._capture_manifest(project_id, exp_id)
            return exp_id
        return project_id, asyncio.run(run())

    # -- rerun -------------------------------------------------------------

    def test_rerun_copies_manifest_config_verbatim_and_stamps_parentage(self):
        project_id, parent_id = self._bootstrap_parent_run()

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{parent_id}/rerun-from-manifest",
            json={"run_name": "rerun of phi-2"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        new_id = body["experiment_id"]
        self.assertNotEqual(new_id, parent_id)
        self.assertEqual(body["name"], "rerun of phi-2")
        # Config: every authored key is preserved byte-for-byte.
        cfg = body["config"]
        for key, value in _PARENT_CONFIG.items():
            self.assertEqual(cfg.get(key), value, f"{key} drifted on rerun")
        # Parentage stamp.
        parentage = body["parentage"]
        self.assertEqual(parentage["parent_experiment_id"], parent_id)
        self.assertIn("manifest_id", parentage)
        self.assertEqual(parentage["reason"], "rerun-from-manifest")
        # The new run starts as PENDING — rerun creates, it does not start.
        self.assertEqual(body["status"], "pending")
        # base_model + training_mode flow through from the manifest.
        self.assertEqual(body["base_model"], "microsoft/phi-2")
        self.assertEqual(body["training_mode"], "sft")

    def test_rerun_returns_404_for_unknown_experiment(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/999999/rerun-from-manifest",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "experiment_not_found")

    def test_rerun_returns_404_when_manifest_not_captured(self):
        project_id = self._create_project()
        async def run():
            return await self._create_experiment(project_id, config={"seed": 7})
        no_manifest_exp = asyncio.run(run())
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{no_manifest_exp}/rerun-from-manifest",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "manifest_not_captured")

    # -- clone -------------------------------------------------------------

    def test_clone_applies_shallow_config_overrides_and_records_them(self):
        project_id, parent_id = self._bootstrap_parent_run()

        overrides = {
            "seed": 99,
            "learning_rate": 3e-4,
            "num_epochs": 5,
        }
        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{parent_id}/clone",
            json={"config_overrides": overrides, "run_name": "lr-bump"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()

        cfg = body["config"]
        self.assertEqual(cfg["seed"], 99)
        self.assertEqual(cfg["learning_rate"], 3e-4)
        self.assertEqual(cfg["num_epochs"], 5)
        # Non-overridden fields survive verbatim.
        self.assertEqual(cfg["recipe"], _PARENT_CONFIG["recipe"])
        self.assertEqual(cfg["chat_template"], _PARENT_CONFIG["chat_template"])
        self.assertEqual(cfg["batch_size"], _PARENT_CONFIG["batch_size"])
        # Parentage records the exact overrides applied.
        parentage = body["parentage"]
        self.assertEqual(parentage["reason"], "clone-from-run")
        self.assertEqual(parentage["parent_experiment_id"], parent_id)
        self.assertEqual(parentage["config_overrides"], overrides)

    def test_clone_can_switch_training_mode_and_base_model(self):
        project_id, parent_id = self._bootstrap_parent_run()

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{parent_id}/clone",
            json={
                "config_overrides": {
                    "training_mode": "dpo",
                    "base_model": "google/gemma-2-2b-it",
                },
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["training_mode"], "dpo")
        self.assertEqual(body["base_model"], "google/gemma-2-2b-it")
        self.assertEqual(body["config"]["training_mode"], "dpo")
        self.assertEqual(body["config"]["base_model"], "google/gemma-2-2b-it")

    def test_clone_ignores_attempts_to_override_parentage_block(self):
        project_id, parent_id = self._bootstrap_parent_run()

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/{parent_id}/clone",
            json={
                "config_overrides": {
                    "_rerun_of": {"parent_experiment_id": 999, "reason": "spoofed"},
                    "seed": 7,
                },
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        # The real parentage is preserved; the spoofed block was discarded.
        self.assertEqual(body["parentage"]["parent_experiment_id"], parent_id)
        self.assertEqual(body["parentage"]["reason"], "clone-from-run")
        self.assertEqual(body["parentage"]["config_overrides"], {"seed": 7})
        self.assertEqual(body["config"]["seed"], 7)

    # -- tolerance comparator ---------------------------------------------

    def test_compare_runs_reports_reproduced_on_bit_identical_losses(self):
        project_id = self._create_project()
        async def run():
            baseline = await self._create_experiment(
                project_id,
                config={"seed": 1},
                final_train_loss=1.2813,
                final_eval_loss=1.2818,
            )
            candidate = await self._create_experiment(
                project_id,
                config={"seed": 1},
                final_train_loss=1.2813,
                final_eval_loss=1.2818,
            )
            return baseline, candidate
        baseline, candidate = asyncio.run(run())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/compare",
            json={
                "baseline_experiment_id": baseline,
                "candidate_experiment_id": candidate,
                "tolerance": 0.01,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["reproduced"])
        self.assertEqual(body["failures"], [])
        for comp in body["comparisons"]:
            self.assertTrue(comp["within_tolerance"])

    def test_compare_runs_flags_metric_drift_beyond_tolerance(self):
        project_id = self._create_project()
        async def run():
            baseline = await self._create_experiment(
                project_id,
                config={"seed": 1},
                final_train_loss=1.0,
                final_eval_loss=1.0,
            )
            candidate = await self._create_experiment(
                project_id,
                config={"seed": 1},
                final_train_loss=1.0,
                final_eval_loss=1.2,  # 20% worse — fails 5% tolerance.
            )
            return baseline, candidate
        baseline, candidate = asyncio.run(run())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/compare",
            json={
                "baseline_experiment_id": baseline,
                "candidate_experiment_id": candidate,
                "tolerance": 0.05,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["reproduced"])
        self.assertIn("final_eval_loss", body["failures"])
        eval_loss = next(c for c in body["comparisons"] if c["metric"] == "final_eval_loss")
        self.assertFalse(eval_loss["within_tolerance"])

    def test_compare_runs_lists_baseline_only_metrics_when_candidate_is_bare(self):
        project_id = self._create_project()
        async def run():
            baseline = await self._create_experiment(
                project_id,
                config={"seed": 1},
                final_train_loss=1.28,
                final_eval_loss=1.29,
            )
            # Candidate has no metrics populated — e.g., PENDING run.
            candidate = await self._create_experiment(
                project_id,
                config={"seed": 1},
                final_train_loss=None,
                final_eval_loss=None,
            )
            return baseline, candidate
        baseline, candidate = asyncio.run(run())

        resp = self.client.post(
            f"/api/projects/{project_id}/training/runs/compare",
            json={
                "baseline_experiment_id": baseline,
                "candidate_experiment_id": candidate,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["reproduced"])
        self.assertIn("final_train_loss", body["baseline_only"])
        self.assertIn("final_eval_loss", body["baseline_only"])
        self.assertEqual(body["comparisons"], [])


if __name__ == "__main__":
    unittest.main()
