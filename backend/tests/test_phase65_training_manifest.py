"""Phase 65 — immutable training-run manifest (priority.md P14).

Covers:
- ``capture_training_manifest`` writes a new row keyed on ``experiment_id``
  and populates base-model registry, adapter, blueprint, dataset, and
  resolved-config fields from the DB.
- Capture is idempotent on the unique ``experiment_id`` constraint — a
  second call rewrites the same row instead of raising.
- A missing ``experiment_id`` raises the stable reason code
  ``experiment_not_found``.
- ``GET /projects/{id}/training/runs/{id}/manifest`` returns 200 with the
  serialized shape, or 404 ``manifest_not_captured`` when nothing has
  been written yet, or 404 ``experiment_not_found`` for a bad id.
- ``collect_env=False`` skips git/pip subprocesses cleanly in test
  environments (no warnings leak about missing tools).
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase65_training_manifest.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase65_training_manifest_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.base_model_registry import BaseModelRegistryEntry, BaseModelSourceType
from app.models.dataset_adapter_definition import DatasetAdapterDefinition
from app.models.domain_blueprint import DomainBlueprintRevision
from app.models.experiment import Experiment
from app.services.training_manifest_service import (
    capture_training_manifest,
    get_training_manifest,
    serialize_training_manifest,
)


class Phase65TrainingManifestTests(unittest.TestCase):
    @classmethod
    def _cleanup_artifacts(cls):
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

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_artifacts()

    # -- helpers ------------------------------------------------------------

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"phase65-{uuid.uuid4().hex[:8]}",
                "description": "phase65 training manifest",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _seed_base_model_entry(self, source_ref: str) -> int:
        async with async_session_factory() as db:
            entry = BaseModelRegistryEntry(
                model_key=f"catalog:phase65:{uuid.uuid4().hex[:6]}",
                source_type=BaseModelSourceType.HUGGINGFACE,
                source_ref=source_ref,
                display_name=source_ref,
                model_family="test",
                architecture="test",
                cache_fingerprint="fingerprint-abc",
                cache_status="fresh",
                normalization_contract_version="v1",
                normalized_metadata={},
                provenance={},
                modalities=[],
                quantization_support={},
                supported_task_families=[],
                training_mode_support=[],
                estimated_hardware_needs={},
                deployment_target_compatibility=[],
            )
            db.add(entry)
            await db.commit()
            await db.refresh(entry)
            return int(entry.id)

    async def _seed_adapter(self, project_id: int) -> int:
        async with async_session_factory() as db:
            adapter = DatasetAdapterDefinition(
                project_id=project_id,
                adapter_name="phase65-adapter",
                version=4,
                status="active",
                task_profile="chat_sft",
                field_mapping={},
            )
            db.add(adapter)
            await db.commit()
            await db.refresh(adapter)
            return int(adapter.id)

    async def _seed_blueprint(self, project_id: int) -> int:
        async with async_session_factory() as db:
            # DomainBlueprintRevision requires a valid status enum. Import + set lazily.
            from app.models.domain_blueprint import DomainBlueprintStatus

            rev = DomainBlueprintRevision(
                project_id=project_id,
                version=7,
                status=DomainBlueprintStatus.ACTIVE,
                domain_name="phase65",
                task_family="qa",
                confidence_score=0.9,
            )
            db.add(rev)
            await db.commit()
            await db.refresh(rev)
            return int(rev.id)

    async def _create_experiment(self, project_id: int, *, config: dict) -> int:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=project_id,
                name=f"exp-{uuid.uuid4().hex[:6]}",
                description="phase65",
                status="completed",
                base_model="microsoft/phi-2",
                config=config,
            )
            db.add(exp)
            await db.commit()
            await db.refresh(exp)
            return int(exp.id)

    # -- tests --------------------------------------------------------------

    def test_capture_populates_all_reference_fields_from_db(self):
        async def run():
            project_id = self._create_project()
            reg_id = await self._seed_base_model_entry("microsoft/phi-2")
            await self._seed_adapter(project_id)
            await self._seed_blueprint(project_id)

            exp_id = await self._create_experiment(
                project_id,
                config={
                    "recipe": "recipe.pipeline.sft_default",
                    "training_mode": "sft",
                    "training_runtime_id": "runtime.hf_trainer",
                    "seed": 42,
                    "chat_template": "llama3",
                    "max_seq_length": 2048,
                    "tokenizer": "microsoft/phi-2",
                    "_runtime": {"task_id": "celery-abc", "backend": "hf_trainer"},
                },
            )

            async with async_session_factory() as db:
                row = await capture_training_manifest(
                    db,
                    project_id=project_id,
                    experiment_id=exp_id,
                    collect_env=False,
                )
            return project_id, exp_id, reg_id, row

        project_id, exp_id, reg_id, row = asyncio.run(run())

        self.assertEqual(row.experiment_id, exp_id)
        self.assertEqual(row.project_id, project_id)
        # Base model registry resolved via source_ref lookup.
        self.assertEqual(row.base_model_registry_id, reg_id)
        self.assertEqual(row.base_model_cache_fingerprint, "fingerprint-abc")
        self.assertEqual(row.base_model_source_ref, "microsoft/phi-2")
        # Adapter + blueprint versions are faithful to the seeded state.
        self.assertIsNotNone(row.dataset_adapter_id)
        self.assertEqual(row.dataset_adapter_version, 4)
        self.assertIsNotNone(row.blueprint_revision_id)
        self.assertEqual(row.blueprint_version, 7)
        # Authoritative config fields pulled out to columns.
        self.assertEqual(row.recipe_id, "recipe.pipeline.sft_default")
        self.assertEqual(row.runtime_id, "runtime.hf_trainer")
        self.assertEqual(row.training_mode, "sft")
        self.assertEqual(row.seed, 42)
        self.assertEqual(row.tokenizer_name, "microsoft/phi-2")
        self.assertTrue(row.tokenizer_config_hash)
        # The `_runtime` sub-block is stripped from the stored config so
        # rerun replays the authored config, not the transient task state.
        self.assertNotIn("_runtime", row.resolved_config)
        self.assertEqual(row.resolved_config.get("recipe"), "recipe.pipeline.sft_default")
        # collect_env=False → no env subprocess warnings leak.
        self.assertFalse(any(w.startswith("git_") or w.startswith("pip_") for w in (row.capture_warnings or [])))
        # Env digest is still deterministic even when git/pip are skipped.
        self.assertTrue(row.env_digest)
        self.assertEqual(len(row.env_digest), 64)
        self.assertEqual(row.schema_version, 1)

    def test_capture_is_idempotent_and_rewrites_on_second_call(self):
        async def run():
            project_id = self._create_project()
            exp_id = await self._create_experiment(
                project_id,
                config={"seed": 1, "training_runtime_id": "runtime.vllm_serve"},
            )

            async with async_session_factory() as db:
                first = await capture_training_manifest(
                    db, project_id=project_id, experiment_id=exp_id, collect_env=False
                )

            # Mutate the experiment config and recapture — same row, new payload.
            async with async_session_factory() as db:
                from sqlalchemy import select as _sel

                exp = (
                    await db.execute(_sel(Experiment).where(Experiment.id == exp_id))
                ).scalar_one()
                exp.config = {"seed": 7, "training_runtime_id": "runtime.hf_trainer"}
                await db.commit()

            async with async_session_factory() as db:
                second = await capture_training_manifest(
                    db, project_id=project_id, experiment_id=exp_id, collect_env=False
                )
            return first, second

        first, second = asyncio.run(run())
        self.assertEqual(first.id, second.id)  # same row
        self.assertEqual(second.seed, 7)
        self.assertEqual(second.runtime_id, "runtime.hf_trainer")

    def test_capture_raises_for_unknown_experiment(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                with self.assertRaises(ValueError) as ctx:
                    await capture_training_manifest(
                        db,
                        project_id=project_id,
                        experiment_id=999_999,
                        collect_env=False,
                    )
                return str(ctx.exception)

        message = asyncio.run(run())
        self.assertEqual(message, "experiment_not_found")

    def test_get_manifest_returns_404_when_no_capture_exists_yet(self):
        project_id = self._create_project()

        async def run():
            return await self._create_experiment(
                project_id, config={"seed": 1}
            )

        exp_id = asyncio.run(run())
        resp = self.client.get(
            f"/api/projects/{project_id}/training/runs/{exp_id}/manifest"
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "manifest_not_captured")

    def test_get_manifest_endpoint_returns_serialized_row_after_capture(self):
        async def run():
            project_id = self._create_project()
            exp_id = await self._create_experiment(
                project_id,
                config={
                    "seed": 99,
                    "training_mode": "sft",
                    "training_runtime_id": "runtime.hf_trainer",
                    "recipe": "recipe.pipeline.dpo",
                    "chat_template": "llama3",
                },
            )
            async with async_session_factory() as db:
                await capture_training_manifest(
                    db, project_id=project_id, experiment_id=exp_id, collect_env=False
                )
            return project_id, exp_id

        project_id, exp_id = asyncio.run(run())
        resp = self.client.get(
            f"/api/projects/{project_id}/training/runs/{exp_id}/manifest"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["experiment_id"], exp_id)
        self.assertEqual(body["project_id"], project_id)
        self.assertEqual(body["seed"], 99)
        self.assertEqual(body["recipe_id"], "recipe.pipeline.dpo")
        self.assertEqual(body["runtime_id"], "runtime.hf_trainer")
        self.assertEqual(body["training_mode"], "sft")
        self.assertIn("env", body)
        self.assertIn("tokenizer", body)
        self.assertIn("datasets", body)
        self.assertIn("resolved_config", body)
        # The env block must always carry a digest, even without git/pip.
        self.assertEqual(len(body["env"]["env_digest"]), 64)

    def test_get_manifest_returns_404_for_unknown_experiment(self):
        project_id = self._create_project()
        resp = self.client.get(
            f"/api/projects/{project_id}/training/runs/999999/manifest"
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "experiment_not_found")

    def test_serialize_shape_is_stable(self):
        # Fast regression-guard on the serialized top-level keys so consumers
        # (CLI, frontend) can rely on the field names without poking the DB.
        async def run():
            project_id = self._create_project()
            exp_id = await self._create_experiment(project_id, config={"seed": 2})
            async with async_session_factory() as db:
                row = await capture_training_manifest(
                    db, project_id=project_id, experiment_id=exp_id, collect_env=False
                )
            return row

        row = asyncio.run(run())
        payload = serialize_training_manifest(row)
        expected_keys = {
            "id", "experiment_id", "project_id", "schema_version", "captured_at",
            "base_model", "dataset_adapter", "blueprint", "datasets",
            "recipe_id", "runtime_id", "training_mode", "tokenizer", "seed",
            "resolved_config", "env", "artifact_ids", "capture_warnings",
        }
        self.assertEqual(set(payload.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
