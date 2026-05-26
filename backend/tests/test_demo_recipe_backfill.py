"""Tests for the demo-seeder recipe assignment + backfill.

Covers:
  * ``DEMO_SLUG_TO_RECIPE_ID`` map + ``derive_recipe_id_for_slug`` —
    truth table for the 3 known slugs + unknown slug returns None.
  * ``seed_demo_project`` end-to-end assigns the matching recipe
    (force=True path: always overrides since the project is freshly
    created and has no recipe yet).
  * ``apply_demo_bundle_to_project`` on an existing project that
    already has a recipe — preserves the existing recipe (no
    clobbering of explicit user choice).
  * ``apply_demo_bundle_to_project`` on an existing project that
    has no recipe — assigns the slug's recipe.
  * ``_assign_recipe_from_slug_if_missing`` handles the slug-not-in-
    map case (records skip; doesn't raise).
  * Backfill script's ``_find_demo_projects_needing_backfill``
    matches by project name; skips projects whose recipe is already
    set; skips non-demo projects whose name doesn't match any
    manifest.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "demo_recipe_backfill.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "demo_recipe_backfill_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory, init_db  # noqa: E402
from app.models.dataset import Dataset, DatasetVersion, RawDocument  # noqa: E402
from app.models.gold_set_annotation import GoldSetRow, GoldSetVersion  # noqa: E402
from app.models.project import PipelineStage, Project, ProjectStatus  # noqa: E402
from app.services.demo_project_service import (  # noqa: E402
    DEMO_SLUG_TO_RECIPE_ID,
    _assign_recipe_from_slug_if_missing,
    apply_demo_bundle_to_project,
    derive_recipe_id_for_slug,
    seed_demo_project,
)


def _clear_tree(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()


async def _wipe_demo_state(db) -> None:
    """Wipe every table this test file's seeds could touch. Critical
    for test isolation: ``app.database`` binds the engine at module
    import to whichever test file is loaded first, so different test
    files in the same pytest run share a DB. Without this wipe, the
    rows my tests insert spill into other test files' DBs (caught
    when phase87's seed counts came back as 6 instead of 1)."""
    # Order matters: children before parents to satisfy FK constraints
    # even on engines that enforce them.
    for model in (
        GoldSetRow, GoldSetVersion, RawDocument, DatasetVersion, Dataset, Project,
    ):
        for row in (await db.execute(select(model))).scalars():
            await db.delete(row)
    await db.commit()


# ─────────────────────────────────────────────────────────────────────
# Slug map (pure, no DB)
# ─────────────────────────────────────────────────────────────────────


class SlugMapTests(unittest.TestCase):
    def test_three_known_slugs_have_recipes(self):
        self.assertEqual(DEMO_SLUG_TO_RECIPE_ID["support-faq"], "qa-sft")
        self.assertEqual(DEMO_SLUG_TO_RECIPE_ID["sentiment-classifier"], "classification")
        self.assertEqual(DEMO_SLUG_TO_RECIPE_ID["pii-detector"], "span-extraction")

    def test_unknown_slug_returns_none(self):
        self.assertIsNone(derive_recipe_id_for_slug("imaginary-slug"))

    def test_each_mapped_recipe_resolves_in_catalog(self):
        """Catalog-drift guard: every mapped recipe MUST be in the
        recipe catalog. Without this, a recipe rename / removal
        would silently break demo seeding."""
        from app.services.recipe_service import get_recipe

        for slug, recipe_id in DEMO_SLUG_TO_RECIPE_ID.items():
            with self.subTest(slug=slug, recipe_id=recipe_id):
                self.assertIsNotNone(
                    get_recipe(recipe_id),
                    f"slug {slug!r} maps to {recipe_id!r} but that "
                    f"recipe isn't in the catalog",
                )


# ─────────────────────────────────────────────────────────────────────
# End-to-end seeder + apply behavior
# ─────────────────────────────────────────────────────────────────────


class SeedDemoProjectAssignsRecipeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        asyncio.run(init_db())

    @classmethod
    def tearDownClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)

    async def asyncSetUp(self):
        async with async_session_factory() as db:
            await _wipe_demo_state(db)

    async def asyncTearDown(self):
        async with async_session_factory() as db:
            await _wipe_demo_state(db)

    async def test_seed_support_faq_assigns_qa_sft(self):
        async with async_session_factory() as db:
            project, summary = await seed_demo_project(db, "support-faq")
            await db.commit()
            await db.refresh(project)
        self.assertIsInstance(project.selected_recipe, dict)
        self.assertEqual(project.selected_recipe["recipe_id"], "qa-sft")
        # Recipe assignment is recorded on the summary for observability.
        self.assertTrue(summary["recipe_assignment"]["assigned"])
        self.assertEqual(summary["recipe_assignment"]["recipe_id"], "qa-sft")
        # And the project's base model adopts the recipe's suggestion
        # (Theme 2 contract from recipe_apply_service).
        self.assertEqual(project.base_model_name, "HuggingFaceTB/SmolLM2-135M-Instruct")

    async def test_seed_sentiment_classifier_assigns_classification(self):
        async with async_session_factory() as db:
            project, _ = await seed_demo_project(db, "sentiment-classifier")
            await db.commit()
            await db.refresh(project)
        self.assertEqual(project.selected_recipe["recipe_id"], "classification")

    async def test_seed_pii_detector_assigns_span_extraction(self):
        async with async_session_factory() as db:
            project, _ = await seed_demo_project(db, "pii-detector")
            await db.commit()
            await db.refresh(project)
        self.assertEqual(project.selected_recipe["recipe_id"], "span-extraction")


class ApplyDemoBundlePreservesExistingRecipeTests(unittest.IsolatedAsyncioTestCase):
    """``apply_demo_bundle_to_project`` is the ``force=False`` path —
    used when a user already has a project and clicks the import-
    sample button. If they picked a recipe before clicking, we must
    NOT clobber it."""

    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        asyncio.run(init_db())

    @classmethod
    def tearDownClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)

    async def asyncSetUp(self):
        async with async_session_factory() as db:
            await _wipe_demo_state(db)

    async def asyncTearDown(self):
        async with async_session_factory() as db:
            await _wipe_demo_state(db)

    async def _create_bare_project(self, *, name: str, existing_recipe: dict | None = None) -> int:
        async with async_session_factory() as db:
            project = Project(
                name=name,
                description="",
                status=ProjectStatus.ACTIVE,
                pipeline_stage=PipelineStage.INGESTION,
                beginner_mode=False,
                target_profile_id="vllm_server",
                training_preferred_plan_profile="balanced",
                evaluation_preferred_pack_id=None,
                dataset_adapter_preset={},
                selected_recipe=existing_recipe,
            )
            db.add(project)
            await db.commit()
            await db.refresh(project)
            return project.id

    async def test_apply_bundle_assigns_recipe_when_project_has_none(self):
        pid = await self._create_bare_project(name="Empty Project A")
        async with async_session_factory() as db:
            summary = await apply_demo_bundle_to_project(
                db, pid, slug="support-faq",
            )
            await db.commit()
            project = await db.get(Project, pid)
        self.assertIsNotNone(project.selected_recipe)
        self.assertEqual(project.selected_recipe["recipe_id"], "qa-sft")
        self.assertTrue(summary["recipe_assignment"]["assigned"])

    async def test_apply_bundle_preserves_existing_recipe(self):
        """User picked classification, then clicks 'Import sample CSV'
        with a QA-flavored slug. Classification recipe must stay set."""
        existing = {
            "recipe_id": "classification",
            "name": "Text Classifier",
            "applied_at": "2026-05-25T00:00:00+00:00",
        }
        pid = await self._create_bare_project(
            name="Empty Project B", existing_recipe=existing,
        )
        async with async_session_factory() as db:
            summary = await apply_demo_bundle_to_project(
                db, pid, slug="support-faq",  # would map to qa-sft
            )
            await db.commit()
            project = await db.get(Project, pid)
        # Recipe unchanged — the user's pre-existing classification choice wins.
        self.assertEqual(project.selected_recipe["recipe_id"], "classification")
        self.assertFalse(summary["recipe_assignment"]["assigned"])
        self.assertEqual(
            summary["recipe_assignment"]["reason"],
            "project_already_has_recipe",
        )


# ─────────────────────────────────────────────────────────────────────
# _assign_recipe_from_slug_if_missing edge cases
# ─────────────────────────────────────────────────────────────────────


class AssignRecipeEdgeCasesTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        asyncio.run(init_db())

    @classmethod
    def tearDownClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)

    async def asyncSetUp(self):
        async with async_session_factory() as db:
            await _wipe_demo_state(db)

    async def asyncTearDown(self):
        async with async_session_factory() as db:
            await _wipe_demo_state(db)

    async def test_unknown_slug_records_skip(self):
        async with async_session_factory() as db:
            project = Project(
                name="x", description="", status=ProjectStatus.ACTIVE,
                pipeline_stage=PipelineStage.INGESTION, beginner_mode=False,
                target_profile_id="vllm_server",
                training_preferred_plan_profile="balanced",
                dataset_adapter_preset={},
            )
            db.add(project)
            await db.flush()
            outcome = await _assign_recipe_from_slug_if_missing(
                db, project, "totally-unknown-slug", force=False,
            )
        self.assertFalse(outcome["assigned"])
        self.assertIn("slug_has_no_recipe", outcome["reason"])

    async def test_force_overrides_existing_recipe(self):
        existing = {"recipe_id": "classification", "name": "x", "applied_at": "x"}
        async with async_session_factory() as db:
            project = Project(
                name="forced", description="", status=ProjectStatus.ACTIVE,
                pipeline_stage=PipelineStage.INGESTION, beginner_mode=False,
                target_profile_id="vllm_server",
                training_preferred_plan_profile="balanced",
                dataset_adapter_preset={},
                selected_recipe=existing,
            )
            db.add(project)
            await db.flush()
            outcome = await _assign_recipe_from_slug_if_missing(
                db, project, "support-faq", force=True,
            )
            await db.refresh(project)
        # force=True path overrides; new recipe is qa-sft.
        self.assertTrue(outcome["assigned"])
        self.assertEqual(project.selected_recipe["recipe_id"], "qa-sft")


# ─────────────────────────────────────────────────────────────────────
# Backfill script logic
# ─────────────────────────────────────────────────────────────────────


class BackfillScriptTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        asyncio.run(init_db())

    @classmethod
    def tearDownClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        _clear_tree(TEST_DATA_DIR)

    async def asyncSetUp(self):
        async with async_session_factory() as db:
            await _wipe_demo_state(db)

    async def asyncTearDown(self):
        async with async_session_factory() as db:
            await _wipe_demo_state(db)

    async def _make_project(self, name: str, *, recipe: dict | None = None) -> int:
        async with async_session_factory() as db:
            p = Project(
                name=name, description="", status=ProjectStatus.ACTIVE,
                pipeline_stage=PipelineStage.INGESTION, beginner_mode=False,
                target_profile_id="vllm_server",
                training_preferred_plan_profile="balanced",
                dataset_adapter_preset={},
                selected_recipe=recipe,
            )
            db.add(p)
            await db.commit()
            await db.refresh(p)
            return p.id

    async def test_finds_demo_projects_with_no_recipe(self):
        # Match the real manifest names so name resolution works.
        await self._make_project("Demo · Support FAQ")
        await self._make_project("Demo · Sentiment classifier")
        await self._make_project(
            "Demo · PII / PCI Detector",
            recipe={"recipe_id": "span-extraction", "name": "x", "applied_at": "x"},
        )
        await self._make_project("Some User's Project")  # not a demo

        from scripts.backfill_demo_recipes import (
            _find_demo_projects_needing_backfill,
            _load_manifest_names,
        )
        name_to_slug = _load_manifest_names()
        async with async_session_factory() as db:
            candidates = await _find_demo_projects_needing_backfill(db, name_to_slug)
        # Found the 2 that lack a recipe; skipped the one with a
        # recipe already set + the user's non-demo project.
        names_found = {p.name for p, _ in candidates}
        self.assertEqual(names_found, {"Demo · Support FAQ", "Demo · Sentiment classifier"})

    async def test_dry_run_writes_nothing(self):
        await self._make_project("Demo · Support FAQ")
        from scripts.backfill_demo_recipes import run
        report = await run(dry_run=True)
        self.assertEqual(report["backfilled"], 0)
        # The candidate is reported but no write happened.
        self.assertEqual(len(report["candidates"]), 1)
        async with async_session_factory() as db:
            project = (await db.execute(
                select(Project).where(Project.name == "Demo · Support FAQ")
            )).scalar_one()
            self.assertIsNone(project.selected_recipe)

    async def test_run_assigns_recipe_to_demo_projects(self):
        await self._make_project("Demo · Support FAQ")
        from scripts.backfill_demo_recipes import run
        report = await run(dry_run=False)
        self.assertEqual(report["backfilled"], 1)
        async with async_session_factory() as db:
            project = (await db.execute(
                select(Project).where(Project.name == "Demo · Support FAQ")
            )).scalar_one()
            self.assertEqual(project.selected_recipe["recipe_id"], "qa-sft")

    async def test_run_is_idempotent(self):
        await self._make_project("Demo · Support FAQ")
        from scripts.backfill_demo_recipes import run
        await run(dry_run=False)
        # Second call: project now has the recipe, so the find query
        # filters it out → 0 backfilled the second time.
        report2 = await run(dry_run=False)
        self.assertEqual(report2["backfilled"], 0)


if __name__ == "__main__":
    unittest.main()
