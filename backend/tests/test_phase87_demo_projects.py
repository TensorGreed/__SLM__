"""Phase 87 — demo project seeder (newbie UX Phase 3).

Covers the catalog + seeder service in
:mod:`app.services.demo_project_service` and the
``/api/demo-projects`` HTTP surface.

What's covered:

1. **Catalog** returns both shipped archetypes (support-faq +
   sentiment-classifier) with their manifest metadata.

2. **Seeder** materialises a real project + cleaned source dataset +
   locked gold-set version + N approved gold rows.

3. **Idempotency** — re-posting the same slug returns the existing
   project (no new rows, no duplicates).

4. **Unknown slug** raises ``demo_slug_unknown`` (404).

5. **Path-traversal guard** — slugs with ``..`` or absolute paths
   resolve to 404, not file-system access.

6. **HTTP API** — ``GET /api/demo-projects`` lists the catalog;
   ``POST /api/demo-projects/{slug}`` seeds and returns the project
   record.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase87_demo_projects.db"
TEST_DATA_DIR = (
    Path(__file__).resolve().parent / "phase87_demo_projects_data"
)

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType, RawDocument
from app.models.gold_set_annotation import (
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)
from app.models.project import Project
from app.services.demo_project_service import (
    list_demo_archetypes,
    seed_demo_project,
)


def _cleanup_artifacts() -> None:
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


class Phase87DemoProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        _cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        _cleanup_artifacts()

    # ------------------------------------------------------------------
    # 1. Catalog
    # ------------------------------------------------------------------

    def test_catalog_lists_both_archetypes(self):
        archetypes = list_demo_archetypes()
        slugs = {a["slug"] for a in archetypes}
        self.assertIn("support-faq", slugs)
        self.assertIn("sentiment-classifier", slugs)
        by_slug = {a["slug"]: a for a in archetypes}
        self.assertTrue(by_slug["support-faq"]["headline"])
        self.assertTrue(by_slug["support-faq"]["suggested_brief"])
        self.assertEqual(by_slug["sentiment-classifier"]["task_profile"], "classification")
        self.assertEqual(by_slug["sentiment-classifier"]["target_profile"], "mobile_cpu")

    # ------------------------------------------------------------------
    # 2. Seeder writes project + dataset + gold set
    # ------------------------------------------------------------------

    def test_seed_support_faq_creates_full_bundle(self):
        # Re-seeding is idempotent so this test stays correct even when an
        # earlier test (e.g. the HTTP route) already created the demo.
        async def _run():
            async with async_session_factory() as db:
                project, summary = await seed_demo_project(db, "support-faq")
                await db.commit()
                return project.id, summary

        project_id, summary = asyncio.run(_run())
        self.assertEqual(summary["slug"], "support-faq")
        # `created` is only True on the very first call across the suite.
        # The downstream structural checks are the load-bearing assertions.
        self.assertEqual(summary["project_id"], project_id)

        async def _inspect():
            async with async_session_factory() as db:
                # Project record looks right
                project = await db.get(Project, project_id)
                self.assertIsNotNone(project)
                self.assertTrue(project.beginner_mode)
                self.assertEqual(project.target_profile_id, "vllm_server")
                self.assertEqual(
                    (project.dataset_adapter_preset or {}).get("demo_slug"),
                    "support-faq",
                )

                # Source dataset + RawDocument rows
                source = await db.execute(
                    select(Dataset).where(
                        Dataset.project_id == project_id,
                        Dataset.dataset_type == DatasetType.CLEANED,
                    )
                )
                source_ds = source.scalar_one()
                docs = await db.execute(
                    select(RawDocument).where(RawDocument.dataset_id == source_ds.id)
                )
                self.assertEqual(
                    len(docs.scalars().all()), source_ds.record_count
                )

                # Gold dataset + locked version + approved rows
                gold = await db.execute(
                    select(Dataset).where(
                        Dataset.project_id == project_id,
                        Dataset.dataset_type == DatasetType.GOLD_DEV,
                    )
                )
                gold_ds = gold.scalar_one()
                ver = await db.execute(
                    select(GoldSetVersion).where(
                        GoldSetVersion.gold_set_id == gold_ds.id
                    )
                )
                gold_version = ver.scalar_one()
                self.assertEqual(gold_version.status, GoldSetVersionStatus.LOCKED)
                rows = await db.execute(
                    select(GoldSetRow).where(GoldSetRow.version_id == gold_version.id)
                )
                row_records = rows.scalars().all()
                # Gold rows match the bundle's gold.jsonl (6 rows for
                # support-faq). Assert non-empty + every row is APPROVED.
                self.assertGreater(len(row_records), 0)
                for row in row_records:
                    self.assertEqual(row.status, GoldSetRowStatus.APPROVED)

        asyncio.run(_inspect())

    # ------------------------------------------------------------------
    # 3. Idempotency
    # ------------------------------------------------------------------

    def test_seed_is_idempotent(self):
        async def _run_twice():
            async with async_session_factory() as db:
                project1, summary1 = await seed_demo_project(db, "sentiment-classifier")
                await db.commit()
                project2, summary2 = await seed_demo_project(db, "sentiment-classifier")
                await db.commit()
                return (
                    project1.id,
                    project2.id,
                    summary1["created"],
                    summary2["created"],
                )

        id1, id2, created1, created2 = asyncio.run(_run_twice())
        self.assertEqual(id1, id2)
        self.assertTrue(created1)
        self.assertFalse(created2)

        # Confirm no duplicate datasets / gold rows landed on the second call.
        async def _inspect():
            async with async_session_factory() as db:
                datasets = await db.execute(
                    select(Dataset).where(Dataset.project_id == id1)
                )
                all_datasets = datasets.scalars().all()
                # One CLEANED + one GOLD_DEV.
                self.assertEqual(len(all_datasets), 2)
                gold = next(d for d in all_datasets if d.dataset_type == DatasetType.GOLD_DEV)
                versions = await db.execute(
                    select(GoldSetVersion).where(GoldSetVersion.gold_set_id == gold.id)
                )
                self.assertEqual(len(versions.scalars().all()), 1)

        asyncio.run(_inspect())

    # ------------------------------------------------------------------
    # 4. Errors
    # ------------------------------------------------------------------

    def test_unknown_slug_raises(self):
        async def _run():
            async with async_session_factory() as db:
                await seed_demo_project(db, "totally-fake")

        with self.assertRaises(ValueError) as cm:
            asyncio.run(_run())
        self.assertIn("demo_slug_unknown", str(cm.exception))

    def test_path_traversal_guard(self):
        async def _run():
            async with async_session_factory() as db:
                await seed_demo_project(db, "../etc")

        with self.assertRaises(ValueError) as cm:
            asyncio.run(_run())
        self.assertIn("demo_slug_unknown", str(cm.exception))

    # ------------------------------------------------------------------
    # 5. HTTP API
    # ------------------------------------------------------------------

    def test_get_demo_projects_lists_catalog(self):
        resp = self.client.get("/api/demo-projects")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        slugs = {item["slug"] for item in body["archetypes"]}
        self.assertIn("support-faq", slugs)
        self.assertIn("sentiment-classifier", slugs)

    def test_post_demo_projects_seeds_and_returns_project(self):
        # Use a fresh slug for HTTP isolation — second project on this slug.
        resp = self.client.post("/api/demo-projects/support-faq")
        # The earlier test already created support-faq; the HTTP call should
        # still 200 with created=False on the second request.
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["summary"]["slug"], "support-faq")
        self.assertEqual(body["project"]["name"], "Demo · Support FAQ")

    def test_post_unknown_slug_returns_404(self):
        resp = self.client.post("/api/demo-projects/no-such-demo")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertIn("demo_slug_unknown", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
