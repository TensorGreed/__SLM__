"""Tests for the project_template service + endpoints."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "project_template_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "project_template_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.project_template_service import (
    get_project_template,
    list_project_templates,
)


class ProjectTemplateCatalogTests(unittest.TestCase):
    """Catalog reads (no DB needed; pure file-system inspection)."""

    def test_list_includes_both_shipped_templates(self):
        catalog = list_project_templates()
        slugs = {t["slug"] for t in catalog}
        self.assertIn("security-alert-summarizer", slugs)
        self.assertIn("ticket-router", slugs)

    def test_each_template_carries_the_required_metadata(self):
        for template in list_project_templates():
            with self.subTest(slug=template["slug"]):
                self.assertTrue(template["name"], "name is required")
                self.assertTrue(template["headline"], "headline is required")
                self.assertTrue(template["description"], "description is required")
                self.assertTrue(template["task_profile"], "task_profile is required")
                self.assertGreater(
                    template["minimum_dataset_size"], 0,
                    "minimum_dataset_size must be a positive int",
                )
                self.assertGreaterEqual(
                    len(template["recommended_base_models"]), 1,
                    "at least one recommended base model required",
                )

    def test_get_known_template_returns_summary(self):
        template = get_project_template("security-alert-summarizer")
        self.assertIsNotNone(template)
        self.assertEqual(template["recipe_id"], "summarization")
        self.assertEqual(template["task_profile"], "summarization")
        self.assertGreaterEqual(template["minimum_dataset_size"], 50)

    def test_get_unknown_template_returns_none(self):
        self.assertIsNone(get_project_template("nope"))

    def test_get_handles_path_traversal_attempt(self):
        self.assertIsNone(get_project_template("../etc"))
        self.assertIsNone(get_project_template("/etc/passwd"))


class ProjectTemplateInstantiateApiTests(unittest.TestCase):
    """End-to-end via the FastAPI test client. Confirms templates
    are cloneable (multiple projects per template) and the
    recipe-apply hook fires."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

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
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def test_list_endpoint_returns_catalog(self):
        resp = self.client.get("/api/project-templates")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertGreaterEqual(payload["count"], 2)
        slugs = {t["slug"] for t in payload["templates"]}
        self.assertIn("ticket-router", slugs)

    def test_get_endpoint_returns_404_for_unknown_slug(self):
        resp = self.client.get("/api/project-templates/does-not-exist")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_get_endpoint_returns_template_detail(self):
        resp = self.client.get("/api/project-templates/ticket-router")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["slug"], "ticket-router")
        self.assertEqual(payload["task_profile"], "classification")
        self.assertEqual(payload["target_profile"], "mobile_cpu")
        self.assertIn("billing", payload["labels"])

    def test_instantiate_creates_a_new_project_with_template_data(self):
        resp = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": "Acme Ticket Router"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        project = resp.json()
        self.assertEqual(project["name"], "Acme Ticket Router")
        # selected_recipe should be populated by the recipe-apply hook.
        self.assertIsInstance(project.get("selected_recipe"), dict)
        self.assertEqual(project["selected_recipe"]["recipe_id"], "classification")
        # base_model_name carries the template's first recommended pick.
        self.assertIn("SmolLM2", (project.get("base_model_name") or ""))

    def test_instantiate_same_template_twice_yields_two_distinct_projects(self):
        first = self.client.post(
            "/api/project-templates/security-alert-summarizer/instantiate",
            json={"project_name": "Acme Security Summarizer"},
        )
        second = self.client.post(
            "/api/project-templates/security-alert-summarizer/instantiate",
            json={"project_name": "Acme Security Summarizer"},
        )
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 201, second.text)
        first_payload = first.json()
        second_payload = second.json()
        # Two distinct projects.
        self.assertNotEqual(first_payload["id"], second_payload["id"])
        # Name collision is resolved with a suffix.
        self.assertEqual(first_payload["name"], "Acme Security Summarizer")
        self.assertEqual(second_payload["name"], "Acme Security Summarizer (2)")

    def test_instantiate_defaults_name_to_template_name(self):
        # Need a fresh slug to avoid collision with prior tests.
        resp = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={},  # no project_name
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        # Template's default name is "Ticket Router SLM"; previous
        # test grabbed "Acme Ticket Router" so this should land
        # at the default with no suffix.
        self.assertEqual(resp.json()["name"], "Ticket Router SLM")

    def test_instantiate_unknown_template_returns_404(self):
        resp = self.client.post(
            "/api/project-templates/not-a-real-template/instantiate",
            json={"project_name": "x"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)


if __name__ == "__main__":
    unittest.main()
