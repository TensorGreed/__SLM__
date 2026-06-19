"""Epic G phase G1.5 — classification samples must be trainable.

Guards the fix for a real newbie-facing defect the golden-path E2E
surfaced: a classification project resolved ``task_type=causal_lm`` and
then its label-shaped data was rejected by the training data gate
("zero rows with any target field [answer, …]"). Two layers:

  1. Effective-config now resolves ``task_type=classification`` for a
     classification project (api/training._resolve_project_task_profile).
  2. The data gate accepts ``label`` as a target field for classification
     (services/training_service).

This is the deterministic guard; the browser E2E stays single-demo to
keep the CI gate non-flaky.
"""

from __future__ import annotations

import json
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "clf_launch.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "clf_launch_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


class DataGateTaskAwarenessTests(unittest.TestCase):
    """Fix 2 — the gate recognizes ``label`` for classification."""

    def _write(self, rows: list[dict]) -> Path:
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = TEST_DATA_DIR / f"clf-{uuid.uuid4().hex[:8]}.jsonl"
        with path.open("w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        return path

    def test_label_rows_pass_when_label_is_a_target_field(self):
        from app.models.experiment import TrainingMode
        from app.services.training_data_gate import (
            DEFAULT_TARGET_FIELDS,
            verify_training_data_has_targets,
        )
        rows = [{"input": f"review {i}", "label": "positive"} for i in range(5)]
        train_file = self._write(rows)

        # Without label in the field list (the old behavior) → blocked.
        blocked = verify_training_data_has_targets(
            train_file, training_mode=TrainingMode.SFT,
            target_fields=DEFAULT_TARGET_FIELDS,
        )
        self.assertFalse(blocked["ok"])

        # With label added (the classification path) → passes.
        ok = verify_training_data_has_targets(
            train_file, training_mode=TrainingMode.SFT,
            target_fields=["label", *DEFAULT_TARGET_FIELDS],
        )
        self.assertTrue(ok["ok"])


class EffectiveConfigTaskTypeTests(unittest.TestCase):
    """Fix 1 — a classification project resolves task_type=classification."""

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def test_sentiment_demo_resolves_classification_task_type(self):
        # Seed the sentiment-classifier showcase (the one that used to fail).
        seed = self.client.post("/api/demo-projects/sentiment-classifier")
        self.assertEqual(seed.status_code, 200, seed.text)
        summary = seed.json()["summary"]
        self.assertEqual(summary["task_profile"], "classification")
        pid = summary["project_id"]

        # Effective config must now carry task_type=classification.
        resp = self.client.post(
            f"/api/projects/{pid}/training/experiments/effective-config", json={}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        task_type = resp.json()["resolved_training_config"]["task_type"]
        self.assertEqual(task_type, "classification")


if __name__ == "__main__":
    unittest.main()
