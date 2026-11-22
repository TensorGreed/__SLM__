"""A duplicate project name returns 400 with ``error_code: PROJECT_NAME_TAKEN``.

The create dialog (UI simplification phase 6) retries a name it derived from the
brief with the next suffix only on this code; ``detail`` stays the message string.
"""

from __future__ import annotations

import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


class ProjectNameTakenTests(unittest.TestCase):
    def test_duplicate_name_envelope(self):
        name = f"Name taken {uuid4().hex[:8]}"
        first = client.post("/api/projects", json={"name": name})
        self.assertEqual(first.status_code, 201, first.text)
        second = client.post("/api/projects", json={"name": name})
        self.assertEqual(second.status_code, 400, second.text)
        body = second.json()
        self.assertEqual(body["error_code"], "PROJECT_NAME_TAKEN")
        self.assertEqual(body["detail"], f"Project '{name}' already exists")
        self.assertEqual(body["message"], body["detail"])
        self.assertEqual(body["metadata"], {"name": name})


if __name__ == "__main__":
    unittest.main()
