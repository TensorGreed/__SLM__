"""Pipeline plan refinement — Phase 1 tests.

The load-bearing test is the PRIVACY invariant: the cloud-safe profile must
carry only aggregates — never ingested row text, gold answers, or label names.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(tempfile.gettempdir()) / f"brewslm-refine-{uuid.uuid4().hex[:8]}.db"
TEST_DATA_DIR = Path(tempfile.gettempdir()) / f"brewslm-refine-{uuid.uuid4().hex[:8]}"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

import asyncio  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


# Distinctive strings we'll assert never leak into the cloud-safe profile.
SECRET_TEXT = "ACME_PROPRIETARY_INVOICE_PAYLOAD_zzz"
SECRET_LABELS = ["billing_internal", "technical_internal", "refund_internal"]


class PipelineRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _seed_project(self) -> int:
        resp = self.client.post("/api/projects", json={"name": f"refine-{uuid.uuid4().hex[:6]}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])

        gold_path = settings.DATA_DIR / "projects" / str(pid) / "gold_dev.jsonl"
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            [{"text": f"{SECRET_TEXT} {i}", "label": SECRET_LABELS[0]} for i in range(6)]
            + [{"text": f"{SECRET_TEXT} t", "label": SECRET_LABELS[1]}]   # below floor
            + [{"text": f"{SECRET_TEXT} r", "label": SECRET_LABELS[2]}]   # below floor
        )
        gold_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        async def _configure():
            from app.database import async_session_factory
            from app.models.dataset import Dataset, DatasetType
            from app.models.project import Project
            async with async_session_factory() as db:
                project = await db.get(Project, pid)
                project.selected_recipe = {"recipe_id": "classification", "task_profile": "classification"}
                project.base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
                project.target_profile_id = "mobile_cpu"
                db.add(Dataset(
                    project_id=pid, name="Gold Dev", dataset_type=DatasetType.GOLD_DEV,
                    file_path=str(gold_path), record_count=len(rows),
                ))
                await db.commit()

        asyncio.run(_configure())
        return pid

    def test_refine_plan_returns_profile_and_plan_health(self):
        pid = self._seed_project()
        resp = self.client.get(f"/api/projects/{pid}/refine-plan")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()

        # Plan echoes the project config.
        self.assertEqual(body["plan"]["recipe_id"], "classification")
        self.assertEqual(body["plan"]["base_model_name"], "Qwen/Qwen1.5-1.8B-Chat")
        self.assertEqual(body["plan"]["target_profile_id"], "mobile_cpu")

        # Aggregate profile: distribution SHAPE, not names.
        profile = body["cloud_safe_profile"]
        shape = profile["label_distribution_shape"]
        self.assertEqual(shape["num_classes"], 3)
        self.assertEqual(shape["min_class_count"], 1)
        self.assertEqual(shape["max_class_count"], 6)
        self.assertEqual(shape["classes_below_floor"], 2)  # the two singletons
        self.assertGreaterEqual(profile["labelled_row_count"], 1)

        # Plan-fit roll-up flags the below-floor classes (attention, not ready).
        self.assertIn(body["plan_health"]["verdict"], ("attention", "mismatch"))
        sids = {s["id"] for s in body["plan_health"]["signals"]}
        self.assertIn("plan.classes_below_floor", sids)

        # Phase-1 framing + provider support kept in mind.
        self.assertFalse(body["cloud_refinement"]["available"])
        self.assertIn("deepseek", body["cloud_refinement"]["supported_providers"])
        self.assertIn("qwen", body["cloud_refinement"]["supported_providers"])
        self.assertEqual(body["privacy"]["cloud_sharing"], "aggregate_only")

    def test_cloud_safe_profile_never_leaks_raw_data(self):
        # THE invariant: no ingested text, no gold answers, and no label NAMES
        # may appear anywhere in the cloud-safe profile.
        pid = self._seed_project()
        body = self.client.get(f"/api/projects/{pid}/refine-plan").json()
        serialized = json.dumps(body["cloud_safe_profile"])
        self.assertNotIn(SECRET_TEXT, serialized)
        for label in SECRET_LABELS:
            self.assertNotIn(label, serialized)

    def test_no_recipe_project_is_a_mismatch(self):
        resp = self.client.post("/api/projects", json={"name": f"refine-norecipe-{uuid.uuid4().hex[:6]}"})
        pid = int(resp.json()["id"])
        body = self.client.get(f"/api/projects/{pid}/refine-plan").json()
        self.assertEqual(body["plan_health"]["verdict"], "mismatch")
        self.assertIn("plan.no_recipe", {s["id"] for s in body["plan_health"]["signals"]})

    def test_unknown_project_404(self):
        resp = self.client.get("/api/projects/999999/refine-plan")
        self.assertEqual(resp.status_code, 404, resp.text)


if __name__ == "__main__":
    unittest.main()
