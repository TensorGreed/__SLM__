"""UI simplify phase 7g (gap G6): Continue to Train advances past Cleaning when a prepared split exists.

An own-split import writes train / validation splits straight from the raw rows (no cleaned dataset), so
``/pipeline/advance`` at the Cleaning stage used to answer 400 "no cleaned dataset entries available". A
prepared train split now satisfies Cleaning; a project with neither still gets the 400 (unchanged).
"""

from __future__ import annotations

import asyncio
import os
import unittest
from uuid import uuid4

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

import app.database as database_module  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import Dataset, DatasetType  # noqa: E402
from app.models.project import PipelineStage, Project  # noqa: E402

_CLIENT_CM = TestClient(app)
client: TestClient


def setUpModule() -> None:  # noqa: N802
    global client
    client = _CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _CLIENT_CM.__exit__(None, None, None)


def _project_at_cleaning(datasets: list[tuple[DatasetType, int]]) -> int:
    created = client.post("/api/projects", json={"name": f"g6-{uuid4().hex[:8]}", "base_model_name": "microsoft/phi-2"})
    assert created.status_code == 201, created.text
    pid = created.json()["id"]

    async def seed() -> None:
        async with database_module.async_session_factory() as db:
            project = await db.get(Project, pid)
            project.pipeline_stage = PipelineStage.CLEANING
            for dtype, count in datasets:
                db.add(Dataset(project_id=pid, name=f"{dtype.value}-{uuid4().hex[:6]}", dataset_type=dtype, record_count=count))
            await db.commit()

    asyncio.run(seed())
    return pid


class CleaningAdvanceTests(unittest.TestCase):
    def test_a_prepared_train_split_advances_past_cleaning(self):
        pid = _project_at_cleaning([(DatasetType.RAW, 10), (DatasetType.TRAIN, 8), (DatasetType.VALIDATION, 2)])
        res = client.post(f"/api/projects/{pid}/pipeline/advance")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["current_stage"], "gold_set")

    def test_no_cleaned_and_no_prepared_split_still_blocks(self):
        pid = _project_at_cleaning([(DatasetType.RAW, 10), (DatasetType.TRAIN, 0)])
        res = client.post(f"/api/projects/{pid}/pipeline/advance")
        self.assertEqual(res.status_code, 400, res.text)
        self.assertIn("no cleaned dataset entries", res.text)

    def test_a_cleaned_dataset_still_advances(self):
        pid = _project_at_cleaning([(DatasetType.CLEANED, 5)])
        res = client.post(f"/api/projects/{pid}/pipeline/advance")
        self.assertEqual(res.status_code, 200, res.text)
