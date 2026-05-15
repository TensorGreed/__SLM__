"""``resolve_training_dataset_types`` contract.

Pins the auto-exclusion rule that prevents the prep step from dumping
70k+ text-only CLEANED rows on top of a few thousand structured
SYNTHETIC rows for SFT — the failure mode that produced eval F1 = 0%
in experiment 10 (Qwen-PII-V2 / commit 222bc5d).

The rule:
- Fires only when BOTH ``cleaned`` and ``synthetic`` are in the
  requested types AND the project has ≥1 synthetic row.
- Drops ``cleaned`` from the resolved list. Other types (gold, raw,
  validation, etc.) are untouched.
- Returns a report describing what was excluded and why, so the
  split manifest can surface it to the operator.
- Is a no-op when synthetic is empty (legitimate fresh-project case)
  or when only one of the two types was requested (caller knows
  what they want).
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
from app.models.dataset import Dataset, DatasetType  # noqa: E402
from app.services.dataset_service import (  # noqa: E402
    resolve_training_dataset_types,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-autoexclude-{uuid.uuid4().hex[:8]}"
)


def _seed_dataset(
    project_id: int, dataset_type: DatasetType, record_count: int
) -> None:
    async def _go():
        async with async_session_factory() as session:
            ds = Dataset(
                project_id=project_id,
                name=f"{dataset_type.value}-fixture",
                dataset_type=dataset_type,
                record_count=record_count,
            )
            session.add(ds)
            await session.commit()

    asyncio.run(_go())


def _resolve(project_id: int, requested):
    async def _go():
        async with async_session_factory() as session:
            return await resolve_training_dataset_types(
                session, project_id, requested
            )

    return asyncio.run(_go())


class ResolveTrainingDatasetTypesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
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

    # ── Happy path ──────────────────────────────────────────────────

    def test_excludes_cleaned_when_synthetic_has_rows(self):
        pid = self._create_project("auto-exclude")
        _seed_dataset(pid, DatasetType.SYNTHETIC, 2000)

        resolved, report = _resolve(
            pid,
            [
                DatasetType.CLEANED.value,
                DatasetType.SYNTHETIC.value,
                DatasetType.GOLD_DEV.value,
            ],
        )
        self.assertNotIn(DatasetType.CLEANED.value, resolved)
        self.assertIn(DatasetType.SYNTHETIC.value, resolved)
        self.assertIn(DatasetType.GOLD_DEV.value, resolved)
        self.assertEqual(report["auto_excluded"], [DatasetType.CLEANED.value])
        self.assertEqual(report["synthetic_rows"], 2000)
        self.assertIn("2000", report["reason"])

    def test_default_trio_uses_auto_exclusion(self):
        """When the caller passes None (signal: use defaults), the
        resolver still applies — that's the path the training API
        hits."""
        pid = self._create_project("auto-exclude-default")
        _seed_dataset(pid, DatasetType.SYNTHETIC, 50)

        resolved, report = _resolve(pid, None)
        self.assertNotIn(DatasetType.CLEANED.value, resolved)
        self.assertEqual(report["synthetic_rows"], 50)

    # ── No-ops ──────────────────────────────────────────────────────

    def test_no_op_when_synthetic_empty(self):
        pid = self._create_project("auto-exclude-empty")
        # No synthetic dataset seeded.

        resolved, report = _resolve(
            pid,
            [
                DatasetType.CLEANED.value,
                DatasetType.SYNTHETIC.value,
            ],
        )
        # Both kept; no exclusion happened.
        self.assertIn(DatasetType.CLEANED.value, resolved)
        self.assertIn(DatasetType.SYNTHETIC.value, resolved)
        self.assertEqual(report["auto_excluded"], [])
        self.assertEqual(report["synthetic_rows"], 0)

    def test_no_op_when_synthetic_exists_but_zero_rows(self):
        pid = self._create_project("auto-exclude-zero-rows")
        _seed_dataset(pid, DatasetType.SYNTHETIC, 0)

        resolved, report = _resolve(
            pid,
            [
                DatasetType.CLEANED.value,
                DatasetType.SYNTHETIC.value,
            ],
        )
        self.assertIn(DatasetType.CLEANED.value, resolved)
        self.assertEqual(report["auto_excluded"], [])

    def test_no_op_when_only_cleaned_requested(self):
        """domain_pretrain path passes ``[cleaned]`` alone — must
        respect that without surprise auto-exclusion."""
        pid = self._create_project("auto-exclude-cleaned-only")
        _seed_dataset(pid, DatasetType.SYNTHETIC, 500)

        resolved, report = _resolve(pid, [DatasetType.CLEANED.value])
        self.assertEqual(resolved, [DatasetType.CLEANED.value])
        self.assertEqual(report["auto_excluded"], [])

    def test_no_op_when_only_synthetic_requested(self):
        pid = self._create_project("auto-exclude-synthetic-only")
        _seed_dataset(pid, DatasetType.SYNTHETIC, 500)

        resolved, report = _resolve(pid, [DatasetType.SYNTHETIC.value])
        self.assertEqual(resolved, [DatasetType.SYNTHETIC.value])
        self.assertEqual(report["auto_excluded"], [])

    def test_no_op_when_gold_alone(self):
        pid = self._create_project("auto-exclude-gold-only")
        _seed_dataset(pid, DatasetType.SYNTHETIC, 500)

        resolved, report = _resolve(pid, [DatasetType.GOLD_DEV.value])
        self.assertEqual(resolved, [DatasetType.GOLD_DEV.value])
        self.assertEqual(report["auto_excluded"], [])


if __name__ == "__main__":
    unittest.main()
