"""Phase 71 — brewslm.yaml schema + serializer (priority.md P21).

End-to-end coverage of the pipeline-as-code manifest contract:

- :func:`serialize_project_to_manifest` reads project state (project row,
  active blueprint, datasets, adapters) and produces a :class:`BrewslmManifest`
  with the expected sections.
- :func:`manifest_to_yaml` / :func:`manifest_from_yaml` round-trip a
  manifest through YAML without information loss.
- :func:`deserialize_manifest_to_apply_plan` produces:
  - ``create`` actions for every section when ``project_id`` is None,
  - ``noop`` actions when the manifest matches current state, and
  - ``update`` actions with ``fields_changed`` when fields drift.
- ``GET /api/manifest/schema`` returns a JSON Schema document.
- ``GET /api/projects/{id}/manifest/export`` returns YAML by default and
  JSON when ``format=json``.
- ``POST /api/projects/{id}/manifest/apply-plan`` validates the manifest
  and returns the same shape the service produces.
- Strict-mode (``extra='forbid'``) rejects unknown top-level fields and
  the API version is enforced.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase71_brewslm_manifest.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase71_brewslm_manifest_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType, DatasetVersion
from app.models.dataset_adapter_definition import DatasetAdapterDefinition
from app.models.domain_blueprint import DomainBlueprintRevision, DomainBlueprintStatus
from app.models.project import Project
from app.schemas.brewslm_manifest import (
    BrewslmManifest,
    MANIFEST_API_VERSION,
    MANIFEST_KIND,
)
from app.services.brewslm_manifest_service import (
    deserialize_manifest_to_apply_plan,
    manifest_from_yaml,
    manifest_to_yaml,
    serialize_project_to_manifest,
)


class Phase71BrewslmManifestTests(unittest.TestCase):
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

    def _create_project(self, **overrides) -> int:
        body = {
            "name": f"phase71-{uuid.uuid4().hex[:8]}",
            "description": "phase71 brewslm manifest",
        }
        body.update(overrides)
        resp = self.client.post("/api/projects", json=body)
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _seed_dataset(
        self,
        project_id: int,
        *,
        name: str,
        dataset_type: DatasetType = DatasetType.CLEANED,
        record_count: int = 100,
    ) -> int:
        async with async_session_factory() as db:
            ds = Dataset(
                project_id=project_id,
                name=name,
                dataset_type=dataset_type,
                description=f"{name} description",
                record_count=record_count,
                file_path=f"data/{name}.jsonl",
                metadata_={"source": "phase71"},
            )
            db.add(ds)
            await db.commit()
            await db.refresh(ds)
            version = DatasetVersion(
                dataset_id=ds.id,
                version=1,
                file_path=f"data/{name}.v1.jsonl",
                record_count=record_count,
                manifest={"snapshot": "phase71"},
            )
            db.add(version)
            await db.commit()
            return int(ds.id)

    async def _seed_adapter(
        self,
        project_id: int,
        *,
        adapter_name: str = "phase71-adapter",
        version: int = 1,
        task_profile: str = "chat_sft",
    ) -> int:
        async with async_session_factory() as db:
            adapter = DatasetAdapterDefinition(
                project_id=project_id,
                adapter_name=adapter_name,
                version=version,
                status="active",
                task_profile=task_profile,
                source_type="raw",
                base_adapter_id="default-canonical",
                field_mapping={"input": "prompt", "output": "completion"},
                adapter_config={"chat_template": "llama3"},
                output_contract={"format": "chatml"},
            )
            db.add(adapter)
            await db.commit()
            await db.refresh(adapter)
            return int(adapter.id)

    async def _seed_blueprint(
        self,
        project_id: int,
        *,
        domain_name: str = "Phase 71 Domain",
        version: int = 1,
        active: bool = True,
    ) -> int:
        async with async_session_factory() as db:
            rev = DomainBlueprintRevision(
                project_id=project_id,
                version=version,
                status=DomainBlueprintStatus.ACTIVE,
                source="manual",
                brief_text="phase71 brief",
                domain_name=domain_name,
                problem_statement="Solve phase 71 problems.",
                target_user_persona="Domain expert",
                task_family="instruction_sft",
                input_modality="text",
                expected_output_schema={"answer": "string"},
                expected_output_examples=[{"answer": "yes"}],
                safety_compliance_notes=["no_pii"],
                deployment_target_constraints={"max_latency_ms": 200},
                success_metrics=[{
                    "metric_id": "exact_match",
                    "label": "Exact match",
                    "target": ">=0.85",
                    "why_it_matters": "Correctness",
                }],
                glossary=[{
                    "term": "Phase71",
                    "plain_language": "Test domain",
                    "category": "general",
                }],
                confidence_score=0.9,
                unresolved_assumptions=["seed_data_quality"],
            )
            db.add(rev)
            await db.commit()
            await db.refresh(rev)
            if active:
                project = (
                    await db.execute(
                        # late import to avoid circular wrapping
                        __import__("sqlalchemy").select(Project).where(Project.id == project_id)
                    )
                ).scalar_one()
                project.active_domain_blueprint_version = rev.version
                await db.commit()
            return int(rev.id)

    # -- schema-level tests -------------------------------------------------

    def test_manifest_rejects_unknown_top_level_field(self):
        with self.assertRaises(ValidationError):
            BrewslmManifest.model_validate({
                "api_version": MANIFEST_API_VERSION,
                "kind": MANIFEST_KIND,
                "metadata": {"name": "x"},
                "spec": {},
                "extras": "this should be rejected",
            })

    def test_manifest_rejects_unsupported_api_version(self):
        with self.assertRaises(ValidationError):
            BrewslmManifest.model_validate({
                "api_version": "brewslm/v999",
                "kind": MANIFEST_KIND,
                "metadata": {"name": "x"},
                "spec": {},
            })

    def test_manifest_rejects_unsupported_kind(self):
        with self.assertRaises(ValidationError):
            BrewslmManifest.model_validate({
                "api_version": MANIFEST_API_VERSION,
                "kind": "Recipe",
                "metadata": {"name": "x"},
                "spec": {},
            })

    def test_yaml_round_trip_preserves_data(self):
        async def run():
            project_id = self._create_project()
            await self._seed_blueprint(project_id)
            await self._seed_dataset(project_id, name="train_set")
            await self._seed_adapter(project_id)
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            return manifest

        original = asyncio.run(run())
        body = manifest_to_yaml(original)
        # Sanity-check the YAML is human-readable + has the section headers.
        self.assertIn("api_version: brewslm/v1", body)
        self.assertIn("metadata:", body)
        self.assertIn("spec:", body)
        # Round-trip through PyYAML and reparse.
        reparsed = manifest_from_yaml(body)
        self.assertEqual(reparsed.model_dump(mode="json"), original.model_dump(mode="json"))

    # -- serializer ---------------------------------------------------------

    def test_serializer_captures_project_blueprint_datasets_adapters(self):
        async def run():
            project_id = self._create_project(
                description="serializer test",
                target_profile_id="ollama_local",
                beginner_mode=True,
                base_model_name="microsoft/phi-2",
            )
            await self._seed_blueprint(project_id, domain_name="Phase71 SOTA")
            await self._seed_dataset(project_id, name="train", dataset_type=DatasetType.TRAIN)
            await self._seed_dataset(project_id, name="dev", dataset_type=DatasetType.GOLD_DEV, record_count=50)
            # Two versions of an adapter — only the latest should appear.
            await self._seed_adapter(project_id, adapter_name="my-adapter", version=1)
            await self._seed_adapter(project_id, adapter_name="my-adapter", version=2, task_profile="chat_sft")
            async with async_session_factory() as db:
                return await serialize_project_to_manifest(db, project_id=project_id), project_id

        manifest, project_id = asyncio.run(run())

        # Header.
        self.assertEqual(manifest.api_version, MANIFEST_API_VERSION)
        self.assertEqual(manifest.kind, MANIFEST_KIND)
        self.assertTrue(manifest.metadata.name.startswith("phase71-"))
        self.assertEqual(manifest.metadata.description, "serializer test")

        # Workflow.
        self.assertTrue(manifest.spec.workflow.beginner_mode)
        self.assertEqual(manifest.spec.workflow.target_profile_id, "ollama_local")

        # Blueprint.
        self.assertIsNotNone(manifest.spec.blueprint)
        self.assertEqual(manifest.spec.blueprint.domain_name, "Phase71 SOTA")
        self.assertEqual(len(manifest.spec.blueprint.success_metrics), 1)
        self.assertEqual(manifest.spec.blueprint.success_metrics[0].metric_id, "exact_match")
        self.assertEqual(len(manifest.spec.blueprint.glossary), 1)

        # Model.
        self.assertEqual(manifest.spec.model.base_model, "microsoft/phi-2")

        # Data sources — both datasets, sorted by id.
        names = [ds.name for ds in manifest.spec.data_sources]
        self.assertEqual(sorted(names), ["dev", "train"])
        train = next(ds for ds in manifest.spec.data_sources if ds.name == "train")
        self.assertEqual(train.type, "train")
        self.assertEqual(len(train.versions), 1)
        self.assertEqual(train.versions[0].version, 1)

        # Adapters — only the latest version per name.
        self.assertEqual(len(manifest.spec.adapters), 1)
        self.assertEqual(manifest.spec.adapters[0].name, "my-adapter")
        self.assertEqual(manifest.spec.adapters[0].version, 2)

    def test_serializer_uses_active_blueprint_when_set(self):
        async def run():
            project_id = self._create_project()
            # Two revisions; only v1 is active. Latest version > 1 is v2 but
            # we mark v1 active to verify the active pointer wins over latest.
            await self._seed_blueprint(project_id, domain_name="Active V1", version=1, active=True)
            await self._seed_blueprint(project_id, domain_name="Latest V2", version=2, active=False)
            async with async_session_factory() as db:
                return await serialize_project_to_manifest(db, project_id=project_id)

        manifest = asyncio.run(run())
        self.assertIsNotNone(manifest.spec.blueprint)
        self.assertEqual(manifest.spec.blueprint.domain_name, "Active V1")
        self.assertEqual(manifest.spec.blueprint.version, 1)

    def test_serializer_falls_back_to_latest_when_no_active(self):
        async def run():
            project_id = self._create_project()
            # No active version set; serializer should pick latest revision.
            await self._seed_blueprint(project_id, domain_name="Draft V1", version=1, active=False)
            await self._seed_blueprint(project_id, domain_name="Draft V2", version=2, active=False)
            async with async_session_factory() as db:
                return await serialize_project_to_manifest(db, project_id=project_id)

        manifest = asyncio.run(run())
        self.assertIsNotNone(manifest.spec.blueprint)
        self.assertEqual(manifest.spec.blueprint.version, 2)

    def test_serializer_omits_blueprint_when_none_exists(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                return await serialize_project_to_manifest(db, project_id=project_id)

        manifest = asyncio.run(run())
        self.assertIsNone(manifest.spec.blueprint)
        self.assertEqual(manifest.spec.data_sources, [])
        self.assertEqual(manifest.spec.adapters, [])

    def test_serializer_raises_for_missing_project(self):
        async def run():
            async with async_session_factory() as db:
                await serialize_project_to_manifest(db, project_id=999_999)

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(run())
        self.assertEqual(str(ctx.exception), "project_not_found")

    # -- deserializer / apply-plan ------------------------------------------

    def test_apply_plan_for_new_project_emits_creates(self):
        async def run():
            # Build a manifest from a serialized project, then ask the
            # deserializer to plan it as if no project existed yet.
            project_id = self._create_project()
            await self._seed_blueprint(project_id)
            await self._seed_dataset(project_id, name="train")
            await self._seed_adapter(project_id)
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
                plan = await deserialize_manifest_to_apply_plan(
                    db, manifest=manifest, project_id=None
                )
            return plan

        plan = asyncio.run(run())
        self.assertIsNone(plan.project_id)
        targets = {action.target for action in plan.actions}
        self.assertIn("project", targets)
        self.assertIn("blueprint", targets)
        self.assertIn("data_source", targets)
        self.assertIn("adapter", targets)
        self.assertIn("training_plan", targets)
        self.assertIn("eval_pack", targets)
        # Every action should be a create.
        self.assertTrue(all(a.operation == "create" for a in plan.actions))
        self.assertEqual(plan.summary.get("create"), len(plan.actions))

    def test_apply_plan_against_self_is_all_noop(self):
        async def run():
            project_id = self._create_project(base_model_name="microsoft/phi-2")
            await self._seed_blueprint(project_id)
            await self._seed_dataset(project_id, name="train")
            await self._seed_adapter(project_id, adapter_name="primary")
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
                plan = await deserialize_manifest_to_apply_plan(
                    db, manifest=manifest, project_id=project_id
                )
            return plan

        plan = asyncio.run(run())
        non_noop = [a for a in plan.actions if a.operation != "noop"]
        self.assertEqual(non_noop, [], f"expected all-noop, got: {non_noop}")
        self.assertEqual(plan.warnings, [])

    def test_apply_plan_detects_blueprint_field_changes(self):
        async def run():
            project_id = self._create_project()
            await self._seed_blueprint(project_id, domain_name="Original")
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            # Edit the manifest to change one blueprint field.
            assert manifest.spec.blueprint is not None
            manifest.spec.blueprint.problem_statement = "Updated statement."
            async with async_session_factory() as db:
                plan = await deserialize_manifest_to_apply_plan(
                    db, manifest=manifest, project_id=project_id
                )
            return plan

        plan = asyncio.run(run())
        blueprint_actions = [a for a in plan.actions if a.target == "blueprint"]
        self.assertEqual(len(blueprint_actions), 1)
        self.assertEqual(blueprint_actions[0].operation, "update")
        self.assertIn("problem_statement", blueprint_actions[0].fields_changed)

    def test_apply_plan_warns_when_dataset_dropped_from_manifest(self):
        async def run():
            project_id = self._create_project()
            await self._seed_dataset(project_id, name="train")
            await self._seed_dataset(project_id, name="legacy")
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            # Remove `legacy` from the manifest so apply would lose track of it.
            manifest.spec.data_sources = [
                ds for ds in manifest.spec.data_sources if ds.name != "legacy"
            ]
            async with async_session_factory() as db:
                plan = await deserialize_manifest_to_apply_plan(
                    db, manifest=manifest, project_id=project_id
                )
            return plan

        plan = asyncio.run(run())
        self.assertIn("data_source_not_in_manifest:legacy", plan.warnings)
        # Existing `train` should be a noop, no destructive deletes implied.
        train_actions = [a for a in plan.actions if a.target == "data_source" and a.name == "train"]
        self.assertEqual(len(train_actions), 1)
        self.assertEqual(train_actions[0].operation, "noop")

    # -- HTTP endpoints -----------------------------------------------------

    def test_schema_endpoint_returns_json_schema(self):
        resp = self.client.get("/api/manifest/schema")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # JSON Schema documents always carry a top-level `properties` map.
        self.assertIn("properties", body)
        self.assertIn("metadata", body["properties"])
        self.assertIn("spec", body["properties"])

    def test_export_endpoint_default_returns_yaml(self):
        async def run():
            project_id = self._create_project(base_model_name="microsoft/phi-2")
            await self._seed_blueprint(project_id)
            await self._seed_dataset(project_id, name="train")
            return project_id

        project_id = asyncio.run(run())
        resp = self.client.get(f"/api/projects/{project_id}/manifest/export")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn("application/x-yaml", resp.headers.get("content-type", ""))
        parsed = yaml.safe_load(resp.text)
        self.assertEqual(parsed["api_version"], MANIFEST_API_VERSION)
        self.assertEqual(parsed["kind"], MANIFEST_KIND)
        self.assertEqual(parsed["spec"]["model"]["base_model"], "microsoft/phi-2")

    def test_export_endpoint_json_format(self):
        project_id = self._create_project()
        resp = self.client.get(
            f"/api/projects/{project_id}/manifest/export?format=json"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["api_version"], MANIFEST_API_VERSION)

    def test_export_endpoint_404_for_missing_project(self):
        resp = self.client.get("/api/projects/999999/manifest/export")
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "project_not_found")

    def test_apply_plan_endpoint_round_trip(self):
        async def run():
            project_id = self._create_project()
            await self._seed_blueprint(project_id)
            await self._seed_dataset(project_id, name="train")
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            return project_id, manifest_to_yaml(manifest)

        project_id, yaml_body = asyncio.run(run())
        resp = self.client.post(
            f"/api/projects/{project_id}/manifest/apply-plan",
            json={"manifest_yaml": yaml_body},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], project_id)
        non_noop = [a for a in body["actions"] if a["operation"] != "noop"]
        self.assertEqual(non_noop, [])

    def test_apply_plan_endpoint_rejects_invalid_manifest(self):
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/manifest/apply-plan",
            json={"manifest": {"api_version": "bogus", "kind": "Project", "metadata": {"name": "x"}, "spec": {}}},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertTrue(resp.json()["detail"].startswith("manifest_invalid"))

    def test_apply_plan_endpoint_for_new_project(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            return manifest

        manifest = asyncio.run(run())
        resp = self.client.post(
            "/api/manifest/apply-plan",
            json={"manifest": manifest.model_dump(mode="json")},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIsNone(body["project_id"])
        self.assertTrue(all(a["operation"] == "create" for a in body["actions"]))


if __name__ == "__main__":
    unittest.main()
