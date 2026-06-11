"""Coach-stage-2 phase 1 — Coach training-stage roll-up nudge.

End-to-end test: the training-stage Coach endpoint should surface a
single rolled-up nudge linking to ``TrainingConfigGapsPanel`` when the
gap scanner detects warn/block signals, and stay silent when everything
is ok or no recipe is selected.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "tcg_coach.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "tcg_coach_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import Dataset, DatasetType  # noqa: E402


class CoachTrainingConfigGapsRollupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _make_project(
        self,
        *,
        recipe_id: str | None,
        base_model: str | None = None,
        labelled_rows: int = 0,
    ) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"tcg-coach-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])

        async def _set():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, pid)
                if recipe_id is not None:
                    proj.selected_recipe = {"recipe_id": recipe_id}
                if base_model is not None:
                    proj.base_model_name = base_model
                if labelled_rows > 0:
                    db.add(Dataset(
                        project_id=pid,
                        name="cleaned",
                        dataset_type=DatasetType.CLEANED,
                        file_path="",
                        record_count=labelled_rows,
                    ))
                await db.commit()
        asyncio.run(_set())
        return pid

    def _coach_training(self, pid: int) -> list[dict]:
        resp = self.client.get(f"/api/projects/{pid}/coach/training")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Coach endpoint returns {suggestions: [...]} per the locked
        # contract; tolerate either bare list or wrapped envelope.
        if isinstance(body, dict) and "suggestions" in body:
            return list(body["suggestions"])
        return list(body) if isinstance(body, list) else []

    def test_rollup_fires_when_gap_scanner_warns(self):
        # Tiny dataset on classification → eval cadence + epochs both
        # fire warn/block at default config. Expect the roll-up card.
        pid = self._make_project(
            recipe_id="recipe.classification.sentiment",
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            labelled_rows=20,
        )
        suggestions = self._coach_training(pid)
        rollups = [
            s for s in suggestions
            if s.get("id") == "training:config-gaps-rollup"
        ]
        self.assertEqual(len(rollups), 1)
        card = rollups[0]
        # Severity bubbles up from underlying signals.
        self.assertIn(card["severity"], ("warning", "critical"))
        # Action points at the panel via the registered navigate target.
        self.assertEqual(card["action"]["kind"], "navigate")
        self.assertEqual(
            card["action"]["params"]["target"],
            "training-config-gaps-panel",
        )
        # Context carries the rolled-up counts so the UI can verify.
        ctx = card["context"]
        self.assertGreaterEqual(
            int(ctx.get("warn_count", 0)) + int(ctx.get("block_count", 0)),
            1,
        )

    def test_rollup_silent_when_no_recipe(self):
        # No recipe → gap scanner emits the no-recipe block signal,
        # which would technically count as a block, BUT the existing
        # "training:no-recipe" coach card already covers that case and
        # short-circuits _training_stage_suggestions before our roll-up
        # is reached. So the assertion is that the roll-up card is not
        # present (the no-recipe card is).
        pid = self._make_project(recipe_id=None)
        suggestions = self._coach_training(pid)
        rollups = [
            s for s in suggestions
            if s.get("id") == "training:config-gaps-rollup"
        ]
        self.assertEqual(rollups, [])
        # Sanity: the no-recipe card IS there.
        no_recipe = [
            s for s in suggestions if s.get("id") == "training:no-recipe"
        ]
        self.assertEqual(len(no_recipe), 1)
