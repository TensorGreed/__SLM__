"""Phase 1 integration tests: auth, RBAC, membership, and audit logging."""

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "phase1_test.db"

os.environ["AUTH_ENABLED"] = "true"
os.environ["AUTH_BOOTSTRAP_API_KEY"] = "phase1-admin-key"
os.environ["AUTH_BOOTSTRAP_USERNAME"] = "phase1-admin"
os.environ["AUTH_BOOTSTRAP_ROLE"] = "admin"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DEBUG"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Phase1AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth = (
            settings.AUTH_ENABLED,
            settings.AUTH_BOOTSTRAP_API_KEY,
            settings.AUTH_BOOTSTRAP_USERNAME,
            settings.AUTH_BOOTSTRAP_ROLE,
        )
        settings.AUTH_ENABLED = True
        settings.AUTH_BOOTSTRAP_API_KEY = "phase1-admin-key"
        settings.AUTH_BOOTSTRAP_USERNAME = "phase1-admin"
        settings.AUTH_BOOTSTRAP_ROLE = "admin"
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()
        cls.admin_headers = {"x-api-key": "phase1-admin-key"}

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        (
            settings.AUTH_ENABLED,
            settings.AUTH_BOOTSTRAP_API_KEY,
            settings.AUTH_BOOTSTRAP_USERNAME,
            settings.AUTH_BOOTSTRAP_ROLE,
        ) = cls._prev_auth
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    def test_health_open_and_projects_protected(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json().get("auth_enabled"))

        protected = self.client.get("/api/projects")
        self.assertEqual(protected.status_code, 401)

    def test_project_creation_membership_and_rbac(self):
        # Create project as bootstrap admin.
        create = self.client.post(
            "/api/projects",
            headers=self.admin_headers,
            json={"name": "phase1-project", "description": "phase1", "base_model_name": "microsoft/phi-2"},
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = create.json()["id"]

        # Create engineer user + key.
        create_user = self.client.post(
            "/api/auth/users",
            headers=self.admin_headers,
            json={"username": "phase1-engineer", "role": "engineer"},
        )
        self.assertEqual(create_user.status_code, 201, create_user.text)
        engineer_key = create_user.json()["api_key"]
        engineer_headers = {"x-api-key": engineer_key}

        # Engineer has no membership yet -> project access denied.
        no_access = self.client.get(f"/api/projects/{project_id}", headers=engineer_headers)
        self.assertEqual(no_access.status_code, 403, no_access.text)

        # Grant viewer membership to engineer.
        add_member = self.client.post(
            f"/api/auth/projects/{project_id}/members",
            headers=self.admin_headers,
            json={"user_id": create_user.json()["user_id"], "role": "viewer"},
        )
        self.assertEqual(add_member.status_code, 201, add_member.text)

        # Engineer can now read project.
        can_read = self.client.get(f"/api/projects/{project_id}", headers=engineer_headers)
        self.assertEqual(can_read.status_code, 200, can_read.text)

        # Engineer cannot delete project (owner/admin required).
        cannot_delete = self.client.delete(f"/api/projects/{project_id}", headers=engineer_headers)
        self.assertEqual(cannot_delete.status_code, 403, cannot_delete.text)

    def test_audit_logs_and_cleaning_validation(self):
        # Create one project for audit trail.
        create = self.client.post(
            "/api/projects",
            headers=self.admin_headers,
            json={"name": "phase1-audit-project", "description": "audit"},
        )
        self.assertEqual(create.status_code, 201, create.text)
        project_id = create.json()["id"]

        # Trigger validation error: chunk_overlap >= chunk_size.
        invalid_clean = self.client.post(
            f"/api/projects/{project_id}/cleaning/clean",
            headers=self.admin_headers,
            json={
                "document_id": 1,
                "chunk_size": 200,
                "chunk_overlap": 200,
                "redact_pii": True,
            },
        )
        self.assertEqual(invalid_clean.status_code, 422, invalid_clean.text)

        # Audit logs should include mutating requests.
        logs = self.client.get(
            f"/api/projects/{project_id}/audit/logs",
            headers=self.admin_headers,
        )
        self.assertEqual(logs.status_code, 200, logs.text)
        self.assertGreaterEqual(logs.json()["count"], 1)


if __name__ == "__main__":
    unittest.main()
