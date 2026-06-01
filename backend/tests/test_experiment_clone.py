"""Tests for the experiment-clone endpoint
(``POST /api/projects/{p}/training/experiments/{e}/clone``).

Used by the bell's kill-switch flow: after cancelling a diverging
run, the user can immediately re-launch the same setup via a
PENDING clone.

Pins:
  * Clone copies the config + base_model + training_mode but lands
    in PENDING (not RUNNING).
  * Name suffix progression: first clone → ``"foo (retry)"``,
    second → ``"foo (retry 2)"``, third → ``"foo (retry 3)"``…
  * Cloning an already-cloned experiment doesn't accumulate
    ``(retry) (retry)`` — stem extraction strips the existing
    suffix.
  * Runtime-stamped fields (``_runtime``, ``_warm_start``,
    ``_curriculum_auto_defaulted``, ``_auto_rag_auto_defaulted``)
    are stripped from the cloned config so the clone re-resolves
    them at launch.
  * 404 for non-existent source experiment.
  * 404 for cross-project clone attempt.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import Experiment, TrainingMode  # noqa: E402


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-clone-{uuid.uuid4().hex[:8]}"
)


class ExperimentCloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"clone-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_experiment(
        self,
        project_id: int,
        *,
        name: str,
        config: dict | None = None,
    ) -> int:
        async def _go() -> int:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=project_id,
                    name=name,
                    description="seed",
                    status="completed",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    training_mode=TrainingMode.SFT,
                    config=config or {
                        "task_type": "classification",
                        "learning_rate": 2e-4,
                        "num_epochs": 3,
                    },
                )
                session.add(exp)
                await session.commit()
                return int(exp.id)
        return asyncio.run(_go())

    def test_clone_copies_config_into_pending_experiment(self):
        pid = self._create_project()
        src = self._seed_experiment(pid, name="original-run", config={
            "task_type": "classification",
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 32,
        })
        resp = self.client.post(
            f"/api/projects/{pid}/training/experiments/{src}/clone"
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "pending")
        self.assertEqual(
            body["base_model"], "HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        # Config fields preserved.
        cloned_config = body["config"]
        for key in ("task_type", "learning_rate", "num_epochs", "batch_size"):
            self.assertEqual(
                cloned_config[key],
                {
                    "task_type": "classification",
                    "learning_rate": 2e-4,
                    "num_epochs": 3,
                    "batch_size": 32,
                }[key],
                f"config.{key} not preserved",
            )

    def test_clone_name_progression(self):
        # First clone → "(retry)"; subsequent clones increment a
        # numeric suffix so a series of failed runs stays grouped.
        pid = self._create_project()
        src = self._seed_experiment(pid, name="my-run")

        r1 = self.client.post(
            f"/api/projects/{pid}/training/experiments/{src}/clone"
        )
        self.assertEqual(r1.status_code, 201, r1.text)
        self.assertEqual(r1.json()["name"], "my-run (retry)")

        r2 = self.client.post(
            f"/api/projects/{pid}/training/experiments/{src}/clone"
        )
        self.assertEqual(r2.status_code, 201, r2.text)
        self.assertEqual(r2.json()["name"], "my-run (retry 2)")

        r3 = self.client.post(
            f"/api/projects/{pid}/training/experiments/{src}/clone"
        )
        self.assertEqual(r3.status_code, 201, r3.text)
        self.assertEqual(r3.json()["name"], "my-run (retry 3)")

    def test_cloning_a_clone_strips_existing_retry_suffix(self):
        # Source name is already "foo (retry)" — cloning that
        # shouldn't produce "foo (retry) (retry)". The stem
        # extractor strips the suffix and re-applies progression.
        pid = self._create_project()
        src = self._seed_experiment(pid, name="foo (retry)")
        resp = self.client.post(
            f"/api/projects/{pid}/training/experiments/{src}/clone"
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        # Since "foo (retry)" already exists, the suffix scanner
        # bumps to "(retry 2)".
        self.assertEqual(resp.json()["name"], "foo (retry 2)")

    def test_clone_strips_runtime_stamped_config_fields(self):
        # Runtime-stamped fields would pin the clone to the
        # original's launch artifacts — they must be dropped so
        # the clone re-resolves at its own launch.
        pid = self._create_project()
        src = self._seed_experiment(pid, name="with-runtime", config={
            "task_type": "classification",
            "_runtime": {"some": "field"},
            "_warm_start": {"checkpoint_name": "x"},
            "_curriculum_auto_defaulted": "test-reason",
            "_auto_rag_auto_defaulted": "test-reason",
            "learning_rate": 2e-4,
        })
        resp = self.client.post(
            f"/api/projects/{pid}/training/experiments/{src}/clone"
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        cloned_config = resp.json()["config"]
        for key in (
            "_runtime",
            "_warm_start",
            "_curriculum_auto_defaulted",
            "_auto_rag_auto_defaulted",
        ):
            self.assertNotIn(
                key, cloned_config,
                f"runtime field {key!r} should be stripped from clone",
            )
        # Non-runtime fields survive.
        self.assertEqual(cloned_config["task_type"], "classification")
        self.assertEqual(cloned_config["learning_rate"], 2e-4)

    def test_clone_nonexistent_experiment_returns_404(self):
        pid = self._create_project()
        resp = self.client.post(
            f"/api/projects/{pid}/training/experiments/999999/clone"
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_clone_refuses_cross_project_experiment(self):
        # Experiment in project A; caller tries to clone via
        # project B's path. The query filters on project_id so
        # the lookup returns None → 404.
        pid_a = self._create_project()
        pid_b = self._create_project()
        src = self._seed_experiment(pid_a, name="x")
        resp = self.client.post(
            f"/api/projects/{pid_b}/training/experiments/{src}/clone"
        )
        self.assertEqual(resp.status_code, 404, resp.text)


if __name__ == "__main__":
    unittest.main()
