"""D1 of the data-quality arc — aggregated Data Health Report.

Covers:
- Empty project: no documents → ingestion.no_documents block.
- Recipe-not-selected: shape.no_recipe_selected block.
- Per-group severity bubbling into the top-level `overall`.
- Cleaning signals fire when there are uploads but no cleaning yet.
- Plain-English + why-it-matters carries through every signal.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "data_health.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "data_health_data"

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
from app.models.dataset import (  # noqa: E402
    Dataset, DatasetType, DocumentStatus, RawDocument,
)


def _signal_by_id(body: dict, signal_id: str) -> dict | None:
    for group in body["groups"]:
        for sig in group["signals"]:
            if sig["id"] == signal_id:
                return sig
    return None


def _all_signal_ids(body: dict) -> list[str]:
    return [s["id"] for g in body["groups"] for s in g["signals"]]


class DataHealthTests(unittest.TestCase):
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

    def _create_project(self, recipe_id: str | None = None) -> int:
        body: dict = {"name": f"dh-{uuid.uuid4().hex[:8]}"}
        resp = self.client.post("/api/projects", json=body)
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        if recipe_id:
            # Set the recipe on the project.
            async def _set():
                async with async_session_factory() as db:
                    from app.models.project import Project
                    proj = await db.get(Project, pid)
                    proj.selected_recipe = {"recipe_id": recipe_id}
                    await db.commit()
            asyncio.run(_set())
        return pid

    def _add_docs(
        self,
        project_id: int,
        *,
        statuses: list[DocumentStatus],
        cleaned: bool = False,
        with_pii: bool = False,
        pii_redacted: bool = False,
        with_dups: bool = False,
        quality_scores: list[float] | None = None,
    ) -> None:
        """Seed RawDocuments + their parent Dataset."""
        async def _add():
            async with async_session_factory() as db:
                ds = Dataset(
                    project_id=project_id,
                    name="raw",
                    dataset_type=DatasetType.RAW,
                    file_path="",
                    record_count=len(statuses),
                )
                db.add(ds)
                await db.flush()
                for i, status in enumerate(statuses):
                    meta: dict = {}
                    if cleaned:
                        meta["cleaned_path"] = f"/tmp/doc-{i}.cleaned.txt"
                        meta["text_hash"] = (
                            "duplicate-hash"
                            if (with_dups and i % 2 == 0)
                            else f"unique-{uuid.uuid4().hex[:8]}"
                        )
                        meta["pii_findings"] = (
                            [{"type": "email", "match": "a@b", "position": 0}] * 6
                            if with_pii
                            else []
                        )
                        if pii_redacted:
                            meta["redact_pii"] = True
                    quality = quality_scores[i] if quality_scores and i < len(quality_scores) else 0.7
                    db.add(RawDocument(
                        dataset_id=ds.id,
                        filename=f"doc-{i}.txt",
                        file_type="txt",
                        file_path=f"/tmp/doc-{i}.txt",
                        file_size_bytes=100,
                        source="upload",
                        sensitivity="internal",
                        status=status,
                        quality_score=quality if cleaned else None,
                        chunk_count=5 if cleaned else 0,
                        metadata_=meta,
                    ))
                await db.commit()
        asyncio.run(_add())

    # ── Tests ────────────────────────────────────────────────────

    def test_empty_project_returns_block_on_ingestion_and_shape(self):
        pid = self._create_project()
        resp = self.client.get(f"/api/projects/{pid}/data-health")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Top-level overall = block because no docs + no recipe.
        self.assertEqual(body["overall"], "block")
        # Both block signals fire.
        no_docs = _signal_by_id(body, "ingestion.no_documents")
        self.assertIsNotNone(no_docs)
        assert no_docs is not None
        self.assertEqual(no_docs["severity"], "block")
        # Plain-English text is present (D1 contract).
        self.assertTrue(no_docs["plain_english"])
        self.assertTrue(no_docs["why_it_matters"])
        no_recipe = _signal_by_id(body, "shape.no_recipe_selected")
        self.assertIsNotNone(no_recipe)
        assert no_recipe is not None
        self.assertEqual(no_recipe["severity"], "block")

    def test_uploads_present_but_no_cleaning_warns(self):
        pid = self._create_project()
        self._add_docs(pid, statuses=[DocumentStatus.ACCEPTED] * 5, cleaned=False)
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        not_run = _signal_by_id(body, "cleaning.not_run")
        self.assertIsNotNone(not_run)
        assert not_run is not None
        self.assertEqual(not_run["severity"], "warn")
        # Parse-failure signal reports 0 errors as ok.
        parse = _signal_by_id(body, "ingestion.parse_failure_rate")
        self.assertEqual(parse["severity"], "ok")

    def test_parse_failure_rate_above_block_threshold_surfaces_warn(self):
        pid = self._create_project()
        # 6 errors / 20 docs = 30% > 25% block threshold.
        statuses = [DocumentStatus.ERROR] * 6 + [DocumentStatus.ACCEPTED] * 14
        self._add_docs(pid, statuses=statuses, cleaned=False)
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        parse = _signal_by_id(body, "ingestion.parse_failure_rate")
        self.assertIsNotNone(parse)
        assert parse is not None
        # At/above block threshold the signal severity is at least warn;
        # 30% above the 25% block bucket means it's escalated to block.
        self.assertEqual(parse["severity"], "block")
        self.assertEqual(parse["context"]["errored"], 6)
        # Plain-English mentions "couldn't be read" so a non-technical
        # reader knows what failed.
        self.assertIn("couldn't be read", parse["plain_english"])

    def test_pii_unredacted_fires_when_findings_exist_without_redact_flag(self):
        pid = self._create_project()
        self._add_docs(
            pid,
            statuses=[DocumentStatus.ACCEPTED] * 3,
            cleaned=True,
            with_pii=True,
            pii_redacted=False,
        )
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        pii = _signal_by_id(body, "cleaning.pii_unredacted")
        self.assertIsNotNone(pii)
        assert pii is not None
        # 6 findings/doc * 3 docs = 18 — above warn floor, below 50 block.
        self.assertEqual(pii["context"]["pii_findings"], 18)
        self.assertEqual(pii["severity"], "warn")

    def test_pii_redacted_surfaces_ok_when_flag_set(self):
        pid = self._create_project()
        self._add_docs(
            pid,
            statuses=[DocumentStatus.ACCEPTED] * 3,
            cleaned=True,
            with_pii=True,
            pii_redacted=True,
        )
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        pii = _signal_by_id(body, "cleaning.pii_unredacted")
        self.assertIsNotNone(pii)
        assert pii is not None
        self.assertEqual(pii["severity"], "ok")
        self.assertIn("redacted", pii["headline"].lower())

    def test_duplicate_chunks_detected_via_text_hash(self):
        pid = self._create_project()
        self._add_docs(
            pid,
            statuses=[DocumentStatus.ACCEPTED] * 10,
            cleaned=True,
            with_dups=True,  # every other doc shares "duplicate-hash"
        )
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        dup = _signal_by_id(body, "cleaning.duplicate_chunks")
        self.assertIsNotNone(dup)
        assert dup is not None
        # 5 docs share the same hash → 4 duplicates beyond the first.
        self.assertEqual(dup["context"]["duplicate_count"], 4)
        # 4/10 = 40% > 30% block threshold.
        self.assertEqual(dup["severity"], "block")

    def test_low_quality_docs_above_threshold_warn(self):
        pid = self._create_project()
        # 4 docs below quality threshold (0.3 < 0.4), 6 above. 40% > 30% block.
        scores = [0.3] * 4 + [0.8] * 6
        self._add_docs(
            pid,
            statuses=[DocumentStatus.ACCEPTED] * 10,
            cleaned=True,
            quality_scores=scores,
        )
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        lq = _signal_by_id(body, "cleaning.low_quality_docs")
        self.assertIsNotNone(lq)
        assert lq is not None
        self.assertEqual(lq["context"]["low_quality_count"], 4)
        self.assertEqual(lq["severity"], "block")

    def test_overall_aggregates_to_worst_severity(self):
        """A single block signal anywhere → overall=block. A single warn
        with no blocks → overall=warn. All ok → overall=ok."""
        pid = self._create_project()  # no docs, no recipe — both block
        body = self.client.get(f"/api/projects/{pid}/data-health").json()
        self.assertEqual(body["overall"], "block")
        # Severity summary reflects the actual counts.
        self.assertGreaterEqual(body["severity_summary"]["block"], 2)


if __name__ == "__main__":
    unittest.main()
