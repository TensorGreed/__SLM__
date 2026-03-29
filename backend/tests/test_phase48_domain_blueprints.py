"""Phase 48 tests: domain blueprint analysis, versioning, and beginner-mode bootstrap."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase48_domain_blueprints.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase48_domain_blueprints_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["DOMAIN_BLUEPRINT_ENABLE_LLM_ENRICHMENT"] = "false"

from fastapi.testclient import TestClient

from app.main import app


class Phase48DomainBlueprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def test_global_analyze_returns_machine_readable_guidance(self):
        resp = self.client.post(
            "/api/domain-blueprints/analyze",
            json={
                "brief_text": "Build a support assistant that answers FAQ questions from customer tickets.",
                "sample_inputs": ["How do I reset my password?"],
                "sample_outputs": ['{"answer":"Use the reset link on the sign-in page."}'],
                "deployment_target": "edge_gpu",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertIn("blueprint", payload)
        self.assertIn("guidance", payload)
        self.assertIn("validation", payload)
        self.assertEqual(payload["blueprint"]["task_family"], "qa")
        self.assertGreaterEqual(float(payload["blueprint"]["confidence_score"]), 0.5)

    def test_project_create_from_brief_bootstraps_versioned_blueprint(self):
        create = self.client.post(
            "/api/projects",
            json={
                "name": "phase48-brief-bootstrap",
                "description": "phase48",
                "beginner_mode": True,
                "brief_text": "Extract liability clauses from contracts into JSON fields.",
                "sample_inputs": ["Section 12: Liability limitations apply only to direct damages."],
                "sample_outputs": ['{"liability_clause":"direct damages only","confidence":0.91}'],
            },
        )
        self.assertEqual(create.status_code, 201, create.text)
        project = create.json()
        project_id = int(project["id"])
        self.assertTrue(bool(project.get("beginner_mode")))
        self.assertEqual(project.get("active_domain_blueprint_version"), 1)

        revisions = self.client.get(f"/api/projects/{project_id}/domain-blueprints")
        self.assertEqual(revisions.status_code, 200, revisions.text)
        payload = revisions.json()
        self.assertEqual(payload.get("count"), 1)
        self.assertEqual(payload.get("active_version"), 1)
        self.assertEqual(payload["revisions"][0]["version"], 1)
        self.assertEqual(payload["revisions"][0]["status"], "active")

    def test_save_diff_apply_and_glossary_help(self):
        create = self.client.post(
            "/api/projects",
            json={"name": "phase48-diff-apply", "description": "phase48"},
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = int(create.json()["id"])

        analysis = self.client.post(
            f"/api/projects/{project_id}/domain-blueprints/analyze",
            json={
                "brief_text": "Classify support tickets by urgency labels.",
                "sample_inputs": ["Customer cannot access account in production."],
                "sample_outputs": ['{"label":"urgent","confidence":0.93}'],
                "deployment_target": "server",
            },
        )
        self.assertEqual(analysis.status_code, 200, analysis.text)
        blueprint = analysis.json()["blueprint"]

        save_v1 = self.client.post(
            f"/api/projects/{project_id}/domain-blueprints",
            json={
                "blueprint": blueprint,
                "source": "test",
                "brief_text": "v1",
            },
        )
        self.assertEqual(save_v1.status_code, 201, save_v1.text)
        self.assertEqual(save_v1.json()["revision"]["version"], 1)

        blueprint_v2 = dict(blueprint)
        blueprint_v2["domain_name"] = "Support Ops"
        blueprint_v2["problem_statement"] = "Classify tickets for faster routing."
        save_v2 = self.client.post(
            f"/api/projects/{project_id}/domain-blueprints",
            json={
                "blueprint": blueprint_v2,
                "source": "test",
                "brief_text": "v2",
            },
        )
        self.assertEqual(save_v2.status_code, 201, save_v2.text)
        self.assertEqual(save_v2.json()["revision"]["version"], 2)

        diff_resp = self.client.get(
            f"/api/projects/{project_id}/domain-blueprints/diff",
            params={"from_version": 1, "to_version": 2},
        )
        self.assertEqual(diff_resp.status_code, 200, diff_resp.text)
        changed_fields = {item["field"] for item in diff_resp.json().get("changed_fields", [])}
        self.assertIn("domain_name", changed_fields)
        self.assertIn("problem_statement", changed_fields)

        apply_resp = self.client.post(
            f"/api/projects/{project_id}/domain-blueprints/2/apply",
            json={"adopt_project_description": True, "adopt_target_profile": True, "set_beginner_mode": True},
        )
        self.assertEqual(apply_resp.status_code, 200, apply_resp.text)
        project = apply_resp.json().get("project") or {}
        self.assertEqual(project.get("active_domain_blueprint_version"), 2)
        self.assertTrue(bool(project.get("beginner_mode")))

        glossary = self.client.get(
            f"/api/projects/{project_id}/domain-blueprints/glossary/help",
            params={"term": "task"},
        )
        self.assertEqual(glossary.status_code, 200, glossary.text)
        self.assertGreaterEqual(int(glossary.json().get("count", 0)), 1)

    def test_validation_rejects_contradictory_constraints(self):
        create = self.client.post(
            "/api/projects",
            json={"name": "phase48-invalid-blueprint", "description": "phase48"},
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = int(create.json()["id"])

        invalid_blueprint = {
            "domain_name": "Operations",
            "problem_statement": "Route incidents by priority.",
            "target_user_persona": "Ops analysts",
            "task_family": "classification",
            "input_modality": "text",
            "expected_output_schema": {
                "type": "object",
                "properties": {"label": "string"},
                "required": ["label"],
            },
            "expected_output_examples": [{"label": "high"}],
            "safety_compliance_notes": [],
            "deployment_target_constraints": {
                "target_profile_id": "vllm_server",
                "offline_required": True,
                "requires_cloud_inference": True,
            },
            "success_metrics": [{"metric_id": "macro_f1", "label": "Macro F1", "target": ">=0.80", "why_it_matters": "quality"}],
            "glossary": [],
            "confidence_score": 0.7,
            "unresolved_assumptions": [],
        }
        save_resp = self.client.post(
            f"/api/projects/{project_id}/domain-blueprints",
            json={"blueprint": invalid_blueprint, "source": "test"},
        )
        self.assertEqual(save_resp.status_code, 400, save_resp.text)
        payload = save_resp.json()
        self.assertEqual(payload.get("detail", {}).get("error_code"), "DOMAIN_BLUEPRINT_VALIDATION_FAILED")
        errors = payload.get("detail", {}).get("validation", {}).get("errors", [])
        codes = {str(item.get("code")) for item in errors if isinstance(item, dict)}
        self.assertIn("DEPLOYMENT_CONSTRAINT_CONFLICT", codes)


if __name__ == "__main__":
    unittest.main()
