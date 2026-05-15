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
from app.models.dataset import Dataset, DatasetType, DatasetVersion, RawDocument
from app.models.gold_set_annotation import (
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)
from app.models.project import PipelineStage, Project
from app.services.demo_project_service import (
    list_demo_archetypes,
    seed_demo_project,
)
from app.services.newbie_autopilot_service import (
    evaluate_newbie_autopilot_dataset_readiness,
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
        self.assertIn("pii-detector", slugs)
        by_slug = {a["slug"]: a for a in archetypes}
        self.assertTrue(by_slug["support-faq"]["headline"])
        self.assertTrue(by_slug["support-faq"]["suggested_brief"])
        self.assertEqual(by_slug["sentiment-classifier"]["task_profile"], "classification")
        self.assertEqual(by_slug["sentiment-classifier"]["target_profile"], "mobile_cpu")
        self.assertEqual(by_slug["pii-detector"]["task_profile"], "structured_extraction")

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
                # Phase 4.1: pipeline_stage advances to TRAINING because the
                # demo materialises prepared splits, so the autopilot can
                # actually launch training without manual prep.
                self.assertEqual(project.pipeline_stage, PipelineStage.TRAINING)
                preset = project.dataset_adapter_preset or {}
                self.assertEqual(preset.get("demo_slug"), "support-faq")
                self.assertEqual(preset.get("adapter_id"), "qa-pair")
                self.assertEqual(preset.get("task_profile"), "instruction_sft")

                # Source dataset is RAW (not CLEANED) so the Pipeline →
                # Data tab + the Cleaning tab can find it via the
                # /ingestion/documents endpoint.
                source = await db.execute(
                    select(Dataset).where(
                        Dataset.project_id == project_id,
                        Dataset.dataset_type == DatasetType.RAW,
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
                # file_path points at the per-project gold dir, NOT the
                # read-only demo-bundle source file. Critical so the
                # legacy GoldSetPanel can read entries from disk.
                self.assertTrue(gold_ds.file_path.endswith("gold/gold_dev.jsonl"))
                gold_jsonl = Path(gold_ds.file_path)
                self.assertTrue(gold_jsonl.exists(), gold_jsonl)
                # Each line has the legacy shape the gold endpoint /
                # GoldSetPanel UI expects.
                import json as _json
                with gold_jsonl.open("r", encoding="utf-8") as handle:
                    entries = [_json.loads(line) for line in handle if line.strip()]
                self.assertGreater(len(entries), 0)
                for entry in entries:
                    self.assertIn("id", entry)
                    self.assertIn("question", entry)
                    self.assertIn("answer", entry)
                    self.assertIsInstance(entry["question"], str)
                    self.assertIsInstance(entry["answer"], str)
                    self.assertTrue(entry["question"])
                    self.assertTrue(entry["answer"])

                # Workbench rows (P10) also exist in parallel.
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
                self.assertGreater(len(row_records), 0)
                for row in row_records:
                    self.assertEqual(row.status, GoldSetRowStatus.APPROVED)

                # Phase 4.1: prepared train/val/test + manifest exist on
                # disk in canonical shape, and matching Dataset rows are
                # present so the Pipeline → Data Prep tab + autopilot
                # readiness check both succeed.
                prepared_dir = (
                    settings.DATA_DIR
                    / "projects"
                    / str(project_id)
                    / "prepared"
                )
                self.assertTrue((prepared_dir / "train.jsonl").exists())
                self.assertTrue((prepared_dir / "val.jsonl").exists())
                self.assertTrue((prepared_dir / "test.jsonl").exists())
                manifest_path = prepared_dir / "manifest.json"
                self.assertTrue(manifest_path.exists())
                import json as _json
                manifest_payload = _json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                self.assertEqual(manifest_payload["adapter_id"], "qa-pair")
                self.assertEqual(
                    manifest_payload["task_profile"], "instruction_sft"
                )
                # Every prepared row must carry the canonical fields the
                # dataset-contract analyzer keys on for qa_pair shape.
                with (prepared_dir / "train.jsonl").open(
                    "r", encoding="utf-8"
                ) as handle:
                    train_entries = [
                        _json.loads(line)
                        for line in handle
                        if line.strip()
                    ]
                self.assertGreater(len(train_entries), 0)
                for entry in train_entries:
                    self.assertTrue(entry.get("question"))
                    self.assertTrue(entry.get("answer"))
                    self.assertTrue(entry.get("text"))
                    self.assertTrue(entry.get("source_text"))
                    self.assertTrue(entry.get("target_text"))

                # TRAIN/VALIDATION/TEST Dataset rows + a DatasetVersion v1
                # per split land in the DB.
                splits_rows = await db.execute(
                    select(Dataset).where(
                        Dataset.project_id == project_id,
                        Dataset.dataset_type.in_(
                            (
                                DatasetType.TRAIN,
                                DatasetType.VALIDATION,
                                DatasetType.TEST,
                            )
                        ),
                    )
                )
                split_datasets = splits_rows.scalars().all()
                self.assertEqual(len(split_datasets), 3)
                for ds in split_datasets:
                    versions = await db.execute(
                        select(DatasetVersion).where(
                            DatasetVersion.dataset_id == ds.id
                        )
                    )
                    self.assertEqual(len(versions.scalars().all()), 1)

        asyncio.run(_inspect())

        # The whole point of Phase 4.1: the autopilot's dataset-readiness
        # gate must clear so the "Apply" button isn't blocked.
        readiness = evaluate_newbie_autopilot_dataset_readiness(
            project_id=project_id, min_rows=1
        )
        self.assertTrue(readiness["ready"], readiness)
        self.assertEqual(readiness["blockers"], [])
        self.assertTrue(readiness["prepared_train_exists"])
        self.assertGreater(readiness["prepared_row_count"], 0)

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
                # 1 RAW + 1 GOLD_DEV + 3 prepared (TRAIN/VALIDATION/TEST).
                types = sorted(d.dataset_type.value for d in all_datasets)
                self.assertEqual(
                    types,
                    sorted(
                        [
                            DatasetType.RAW.value,
                            DatasetType.GOLD_DEV.value,
                            DatasetType.TRAIN.value,
                            DatasetType.VALIDATION.value,
                            DatasetType.TEST.value,
                        ]
                    ),
                )
                gold = next(d for d in all_datasets if d.dataset_type == DatasetType.GOLD_DEV)
                versions = await db.execute(
                    select(GoldSetVersion).where(GoldSetVersion.gold_set_id == gold.id)
                )
                self.assertEqual(len(versions.scalars().all()), 1)

        asyncio.run(_inspect())

        # Phase 5.3.1: sentiment-classifier is a classification demo, so
        # its prepared manifest must carry the candidate `labels` list.
        # The ClassificationHandler reads this to (a) wrap eval prompts
        # with the candidate set and (b) drive per-class P/R/F1.
        prepared_manifest = (
            settings.DATA_DIR
            / "projects"
            / str(id1)
            / "prepared"
            / "manifest.json"
        )
        self.assertTrue(prepared_manifest.exists())
        import json as _json
        payload = _json.loads(prepared_manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["task_profile"], "classification")
        self.assertEqual(
            payload["labels"], ["positive", "neutral", "negative"]
        )

    # ------------------------------------------------------------------
    # 3b. PII / PCI detector demo (structured_extraction)
    # ------------------------------------------------------------------

    def test_seed_pii_detector_full_bundle(self):
        async def _seed():
            async with async_session_factory() as db:
                project, summary = await seed_demo_project(db, "pii-detector")
                await db.commit()
                return project.id, summary

        project_id, summary = asyncio.run(_seed())
        self.assertEqual(summary["slug"], "pii-detector")
        self.assertEqual(summary["adapter_id"], "structured-extraction")
        self.assertEqual(summary["task_profile"], "structured_extraction")
        # The bundle ships 61 training rows and 200 gold rows.
        self.assertEqual(summary["source_row_count"], 61)
        self.assertEqual(summary["gold_row_count"], 200)

        async def _inspect():
            async with async_session_factory() as db:
                project = await db.get(Project, project_id)
                self.assertIsNotNone(project)
                self.assertTrue(project.beginner_mode)
                preset = project.dataset_adapter_preset or {}
                self.assertEqual(preset["demo_slug"], "pii-detector")
                self.assertEqual(preset["adapter_id"], "structured-extraction")
                self.assertEqual(preset["task_profile"], "structured_extraction")

        asyncio.run(_inspect())

        # Prepared manifest carries the structured_extraction shape:
        # task_profile + output_schema + entity_types. The
        # StructuredExtractionHandler reads output_schema to drive its
        # field-level metrics + prompt template.
        import json as _json
        prepared_manifest = (
            settings.DATA_DIR
            / "projects"
            / str(project_id)
            / "prepared"
            / "manifest.json"
        )
        self.assertTrue(prepared_manifest.exists())
        payload = _json.loads(prepared_manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["task_profile"], "structured_extraction")
        self.assertIn("output_schema", payload)
        schema = payload["output_schema"]
        self.assertIn("entities", schema["properties"])
        self.assertIn("entities", schema["required"])
        # Phase 5.3.4b: PII demo declares span_set scoring so the
        # StructuredExtractionHandler routes per-row P/R/F1 + per-class
        # breakdown (load-bearing for compliance-grade PII metrics).
        self.assertEqual(schema.get("scoring_mode"), "span_set")
        # Entity types echoed for the UI / docs.
        self.assertIn("email", payload["entity_types"])
        self.assertIn("credit_card", payload["entity_types"])
        self.assertIn("ssn", payload["entity_types"])

        # Each prepared/train.jsonl row has a JSON-string target_text
        # (the gold entities payload) — the StructuredExtractionHandler
        # parses this at eval time.
        train_path = (
            settings.DATA_DIR
            / "projects"
            / str(project_id)
            / "prepared"
            / "train.jsonl"
        )
        with train_path.open("r", encoding="utf-8") as fh:
            train_rows = [
                _json.loads(line) for line in fh if line.strip()
            ]
        self.assertGreater(len(train_rows), 0)
        for row in train_rows[:5]:
            # input/output shape preserved
            self.assertTrue(row.get("text"))
            self.assertTrue(row.get("target_text"))
            # target_text parses as JSON with an "entities" key
            parsed = _json.loads(row["target_text"])
            self.assertIn("entities", parsed)
            self.assertIsInstance(parsed["entities"], list)

        # Gold JSONL: legacy {id, question, answer} shape with answer
        # holding the JSON-stringified entities list (not Python repr).
        gold_path = (
            settings.DATA_DIR
            / "projects"
            / str(project_id)
            / "gold"
            / "gold_dev.jsonl"
        )
        with gold_path.open("r", encoding="utf-8") as fh:
            first_gold = _json.loads(fh.readline())
        self.assertTrue(first_gold["question"])
        # answer must be valid JSON with single quotes nowhere
        parsed_answer = _json.loads(first_gold["answer"])
        self.assertIn("entities", parsed_answer)

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
