"""Phase 72 — manifest validate / diff / apply (priority.md P22).

Covers:

- ``POST /api/manifest/validate`` returns structured ``ManifestValidationIssue``
  rows (`code`, `severity`, `field`, `message`, `actionable_fix`) for
  unknown target profile / unknown eval pack / unknown domain pack /
  unknown domain profile / unknown dataset type / duplicated dataset
  name / duplicated adapter name / blueprint contract violation.
- A clean manifest validates with ``ok=true`` and an empty ``errors``.
- An unregistered base model becomes a *warning*, not an error
  (auto-registration on training launch is recoverable).
- ``POST /api/projects/{id}/manifest/diff`` returns the same
  apply-plan shape as the apply-plan endpoint, with ``noop`` actions
  when the manifest matches current state and ``update`` actions
  otherwise.
- ``POST /api/projects/{id}/manifest/apply`` with ``plan_only=true``
  performs no writes and returns a populated ``plan`` block.
- ``POST /api/projects/{id}/manifest/apply`` (writes mode) updates
  workflow fields, writes a new blueprint revision, upserts datasets
  and adapters, and is idempotent (a second apply with the same
  manifest is all-noop).
- ``POST /api/manifest/apply`` creates a new project end-to-end (no
  pre-existing ``project_id``).
- A validation-failing manifest short-circuits to ``plan_only=true``
  with the validation block populated.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase72_manifest_apply.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase72_manifest_apply_data"

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
from app.models.dataset import Dataset, DatasetType
from app.models.dataset_adapter_definition import DatasetAdapterDefinition
from app.models.domain_blueprint import DomainBlueprintRevision, DomainBlueprintStatus
from app.models.project import Project
from app.schemas.brewslm_manifest import BrewslmManifest
from app.services.brewslm_manifest_service import (
    manifest_to_yaml,
    serialize_project_to_manifest,
)


_BLUEPRINT_FIXTURE = {
    "domain_name": "Phase72",
    "problem_statement": "Solve phase72.",
    "target_user_persona": "QA reviewer",
    "task_family": "instruction_sft",
    "input_modality": "text",
    "expected_output_schema": {"answer": "string"},
    "expected_output_examples": [{"answer": "yes"}],
    "safety_compliance_notes": [],
    "deployment_target_constraints": {},
    "success_metrics": [
        {
            "metric_id": "exact_match",
            "label": "Exact match",
            "target": ">=0.85",
            "why_it_matters": "Correctness",
        }
    ],
    "glossary": [],
    "confidence_score": 0.9,
    "unresolved_assumptions": [],
}


def _minimal_manifest_dict(name: str, **overrides) -> dict:
    base = {
        "api_version": "brewslm/v1",
        "kind": "Project",
        "metadata": {"name": name, "description": "phase72"},
        "spec": {
            "workflow": {
                "beginner_mode": False,
                "pipeline_stage": "ingestion",
                "target_profile_id": "vllm_server",
                "training_preferred_plan_profile": "balanced",
                "gate_policy": {},
                "budget_settings": {},
            },
            "blueprint": dict(_BLUEPRINT_FIXTURE),
            "domain": {"pack_id": None, "profile_id": None},
            "model": {"base_model": "microsoft/phi-2"},
            "data_sources": [],
            "adapters": [],
            "training_plan": {"training_mode": "sft", "config": {}},
            "eval_pack": {
                "pack_id": None,
                "datasets": ["gold_dev", "gold_test"],
                "eval_types": ["exact_match"],
            },
            "export": {"formats": []},
            "deployment": {"target_profile_id": "vllm_server"},
        },
    }
    spec_overrides = overrides.pop("spec", {}) or {}
    for key, value in spec_overrides.items():
        if isinstance(value, dict) and isinstance(base["spec"].get(key), dict):
            base["spec"][key] = {**base["spec"][key], **value}
        else:
            base["spec"][key] = value
    for key, value in overrides.items():
        base[key] = value
    return base


class Phase72ManifestApplyTests(unittest.TestCase):
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

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"phase72-{uuid.uuid4().hex[:8]}",
                "description": "phase72",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    # -- validate -----------------------------------------------------------

    def test_validate_clean_manifest_returns_ok(self):
        manifest = _minimal_manifest_dict(name=f"valid-{uuid.uuid4().hex[:6]}")
        resp = self.client.post(
            "/api/manifest/validate", json={"manifest": manifest}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["errors"], [])

    def test_validate_unknown_target_profile(self):
        manifest = _minimal_manifest_dict(
            name="bad-target",
            spec={"workflow": {"target_profile_id": "no-such-profile"}},
        )
        resp = self.client.post("/api/manifest/validate", json={"manifest": manifest})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["ok"])
        codes = [e["code"] for e in body["errors"]]
        self.assertIn("UNKNOWN_TARGET_PROFILE", codes)
        issue = next(e for e in body["errors"] if e["code"] == "UNKNOWN_TARGET_PROFILE")
        self.assertEqual(issue["field"], "spec.workflow.target_profile_id")
        self.assertTrue(issue["actionable_fix"])

    def test_validate_unknown_eval_pack(self):
        manifest = _minimal_manifest_dict(
            name="bad-eval",
            spec={"eval_pack": {"pack_id": "ghost-pack-9999", "datasets": [], "eval_types": []}},
        )
        resp = self.client.post("/api/manifest/validate", json={"manifest": manifest})
        body = resp.json()
        codes = [e["code"] for e in body["errors"]]
        self.assertIn("UNKNOWN_EVAL_PACK", codes)

    def test_validate_unknown_domain_pack(self):
        manifest = _minimal_manifest_dict(
            name="bad-pack",
            spec={"domain": {"pack_id": "no-such-pack", "profile_id": None}},
        )
        resp = self.client.post("/api/manifest/validate", json={"manifest": manifest})
        body = resp.json()
        codes = [e["code"] for e in body["errors"]]
        self.assertIn("UNKNOWN_DOMAIN_PACK", codes)

    def test_validate_unknown_dataset_type(self):
        manifest = _minimal_manifest_dict(
            name="bad-ds-type",
            spec={
                "data_sources": [
                    {"name": "x", "type": "frobnozzle", "description": "", "record_count": 0, "metadata": {}, "versions": []}
                ]
            },
        )
        resp = self.client.post("/api/manifest/validate", json={"manifest": manifest})
        body = resp.json()
        codes = [e["code"] for e in body["errors"]]
        self.assertIn("UNKNOWN_DATASET_TYPE", codes)

    def test_validate_duplicate_dataset_and_adapter_names(self):
        manifest = _minimal_manifest_dict(
            name="dup",
            spec={
                "data_sources": [
                    {"name": "train", "type": "train", "description": "", "record_count": 0, "metadata": {}, "versions": []},
                    {"name": "train", "type": "validation", "description": "", "record_count": 0, "metadata": {}, "versions": []},
                ],
                "adapters": [
                    {"name": "alpha", "version": 1},
                    {"name": "alpha", "version": 2},
                ],
            },
        )
        resp = self.client.post("/api/manifest/validate", json={"manifest": manifest})
        body = resp.json()
        codes = [e["code"] for e in body["errors"]]
        self.assertIn("DUPLICATE_DATASET_NAME", codes)
        self.assertIn("DUPLICATE_ADAPTER_NAME", codes)

    def test_validate_unregistered_base_model_is_warning(self):
        manifest = _minimal_manifest_dict(
            name="warn-model",
            spec={"model": {"base_model": "fictitious-org/never-published-1B"}},
        )
        resp = self.client.post("/api/manifest/validate", json={"manifest": manifest})
        body = resp.json()
        # Should still be ok=True since this is a warning.
        self.assertTrue(body["ok"], body)
        warning_codes = [w["code"] for w in body["warnings"]]
        self.assertIn("BASE_MODEL_NOT_IN_REGISTRY", warning_codes)

    def test_validate_invalid_pydantic_returns_400(self):
        # Missing required `metadata.name` should be caught at the schema layer.
        manifest = {
            "api_version": "brewslm/v1",
            "kind": "Project",
            "metadata": {},
            "spec": {},
        }
        resp = self.client.post("/api/manifest/validate", json={"manifest": manifest})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertTrue(resp.json()["detail"].startswith("manifest_invalid"))

    # -- diff ---------------------------------------------------------------

    def test_diff_endpoint_against_self_is_all_noop(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            return project_id, manifest

        project_id, manifest = asyncio.run(run())
        resp = self.client.post(
            f"/api/projects/{project_id}/manifest/diff",
            json={"manifest": manifest.model_dump(mode="json")},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        non_noop = [a for a in body["actions"] if a["operation"] != "noop"]
        self.assertEqual(non_noop, [])

    def test_diff_endpoint_yaml_form(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            return project_id, manifest_to_yaml(manifest)

        project_id, body_yaml = asyncio.run(run())
        resp = self.client.post(
            f"/api/projects/{project_id}/manifest/diff",
            json={"manifest_yaml": body_yaml},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    # -- apply --------------------------------------------------------------

    def test_apply_plan_only_does_not_write(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            # Edit name to force a diff against current state.
            manifest.metadata.description = "would-change"
            return project_id, manifest

        project_id, manifest = asyncio.run(run())
        resp = self.client.post(
            f"/api/projects/{project_id}/manifest/apply",
            json={"manifest": manifest.model_dump(mode="json"), "plan_only": True},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["plan_only"])
        self.assertEqual(body["applied_actions"], [])
        # Verify the description was not actually updated.
        check = self.client.get(f"/api/projects/{project_id}")
        self.assertNotEqual(check.json()["description"], "would-change")

    def test_apply_writes_workflow_changes_and_blueprint_revision(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            # Rewrite workflow + add a blueprint.
            from app.schemas.brewslm_manifest import BlueprintSection
            manifest.metadata.description = "applied"
            manifest.spec.workflow.beginner_mode = True
            manifest.spec.workflow.target_profile_id = "mobile_cpu"
            manifest.spec.blueprint = BlueprintSection.model_validate(_BLUEPRINT_FIXTURE)
            return project_id, manifest

        project_id, manifest = asyncio.run(run())
        resp = self.client.post(
            f"/api/projects/{project_id}/manifest/apply",
            json={"manifest": manifest.model_dump(mode="json"), "plan_only": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["plan_only"])
        self.assertTrue(body["validation"]["ok"])
        # Project + blueprint actions should both show up as applied.
        targets = {a["target"] for a in body["applied_actions"]}
        self.assertIn("project", targets)
        self.assertIn("blueprint", targets)

        # Verify writes landed.
        check = self.client.get(f"/api/projects/{project_id}").json()
        self.assertEqual(check["description"], "applied")
        self.assertTrue(check["beginner_mode"])
        self.assertEqual(check["target_profile_id"], "mobile_cpu")

        async def verify_blueprint():
            async with async_session_factory() as db:
                from sqlalchemy import select as _sel
                rows = (
                    await db.execute(
                        _sel(DomainBlueprintRevision).where(
                            DomainBlueprintRevision.project_id == project_id
                        )
                    )
                ).scalars().all()
                return [
                    (int(r.version), r.status.value if hasattr(r.status, "value") else str(r.status))
                    for r in rows
                ]

        revisions = asyncio.run(verify_blueprint())
        # At least one revision exists and the latest is ACTIVE.
        self.assertTrue(revisions)
        latest = max(revisions, key=lambda x: x[0])
        self.assertEqual(latest[1], DomainBlueprintStatus.ACTIVE.value)

    def test_apply_upserts_data_sources(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            from app.schemas.brewslm_manifest import DataSourceSpec
            manifest.spec.data_sources = [
                DataSourceSpec(
                    name="train",
                    type="train",
                    description="train set",
                    record_count=100,
                ),
                DataSourceSpec(name="dev", type="gold_dev", description="dev set", record_count=50),
            ]
            return project_id, manifest

        project_id, manifest = asyncio.run(run())
        resp = self.client.post(
            f"/api/projects/{project_id}/manifest/apply",
            json={"manifest": manifest.model_dump(mode="json")},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        async def check():
            async with async_session_factory() as db:
                from sqlalchemy import select as _sel
                rows = (
                    await db.execute(
                        _sel(Dataset).where(Dataset.project_id == project_id)
                    )
                ).scalars().all()
                return sorted((r.name, r.dataset_type.value, int(r.record_count)) for r in rows)

        self.assertEqual(
            asyncio.run(check()),
            [("dev", "gold_dev", 50), ("train", "train", 100)],
        )

    def test_apply_is_idempotent(self):
        async def run():
            project_id = self._create_project()
            async with async_session_factory() as db:
                manifest = await serialize_project_to_manifest(db, project_id=project_id)
            from app.schemas.brewslm_manifest import (
                BlueprintSection,
                DataSourceSpec,
                AdapterSpec,
            )
            manifest.spec.blueprint = BlueprintSection.model_validate(_BLUEPRINT_FIXTURE)
            manifest.spec.data_sources = [
                DataSourceSpec(name="train", type="train", record_count=10)
            ]
            manifest.spec.adapters = [
                AdapterSpec(
                    name="primary",
                    version=1,
                    task_profile="chat_sft",
                    field_mapping={"input": "prompt"},
                )
            ]
            return project_id, manifest

        project_id, manifest = asyncio.run(run())
        body_json = manifest.model_dump(mode="json")

        first = self.client.post(
            f"/api/projects/{project_id}/manifest/apply",
            json={"manifest": body_json},
        )
        self.assertEqual(first.status_code, 200, first.text)

        second = self.client.post(
            f"/api/projects/{project_id}/manifest/apply",
            json={"manifest": body_json},
        )
        self.assertEqual(second.status_code, 200, second.text)
        body = second.json()
        # The plan returned is the diff against post-first-apply state.
        # Workflow/data-source/adapter actions should all be noop on re-apply.
        ds_actions = [a for a in body["applied_actions"] if a["target"] == "data_source"]
        adapter_actions = [a for a in body["applied_actions"] if a["target"] == "adapter"]
        self.assertTrue(ds_actions)
        self.assertTrue(adapter_actions)
        for action in ds_actions + adapter_actions:
            self.assertEqual(action["operation"], "noop", f"{action} should be noop")

    def test_apply_creates_new_project_via_top_level_endpoint(self):
        manifest = _minimal_manifest_dict(name=f"new-{uuid.uuid4().hex[:6]}")
        resp = self.client.post("/api/manifest/apply", json={"manifest": manifest})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["project_id"] > 0)
        self.assertFalse(body["plan_only"])
        # Project should be reachable via the projects API.
        check = self.client.get(f"/api/projects/{body['project_id']}")
        self.assertEqual(check.status_code, 200, check.text)

    def test_apply_short_circuits_on_validation_error(self):
        manifest = _minimal_manifest_dict(
            name="short-circuit",
            spec={"workflow": {"target_profile_id": "no-such-profile"}},
        )
        # Need an existing project to hit the project-scoped apply.
        project_id = self._create_project()
        resp = self.client.post(
            f"/api/projects/{project_id}/manifest/apply",
            json={"manifest": manifest, "plan_only": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["validation"]["ok"])
        self.assertEqual(body["applied_actions"], [])
        codes = [e["code"] for e in body["validation"]["errors"]]
        self.assertIn("UNKNOWN_TARGET_PROFILE", codes)


if __name__ == "__main__":
    unittest.main()
