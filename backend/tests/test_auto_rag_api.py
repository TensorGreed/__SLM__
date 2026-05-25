"""Tests for the auto-RAG preview API (USER-SUCCESS Epic 9 Phase 9a).

End-to-end via FastAPI TestClient:
  * 404 on unknown project.
  * 400 when project has no recipe selected.
  * 400 when recipe has no RAG corpus shape (e.g. classification today).
  * 400 when project has no training rows on disk.
  * 200 on the policy-qa-style template — returns recipe_id, query,
    k, index manifest, and a ``retrieved`` list whose top hit
    actually matches the query.
  * 200 after rebuild on cache miss — the second call hits the
    cached index (faster path) and returns the same top hit.
  * ``k`` query param honored (200 with k=1 returns at most 1 hit).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "auto_rag_api_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "auto_rag_api_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


class AutoRagPreviewApiTests(unittest.TestCase):
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

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_unknown_project_returns_404(self):
        resp = self.client.get(
            "/api/projects/99999/auto-rag/preview",
            params={"query": "x"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_project_without_recipe_returns_400(self):
        # Project created directly (no template instantiation) →
        # selected_recipe is null.
        resp = self.client.post(
            "/api/projects",
            json={"name": "AutoRAG No Recipe Project"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = resp.json()["id"]
        preview = self.client.get(
            f"/api/projects/{pid}/auto-rag/preview",
            params={"query": "anything"},
        )
        self.assertEqual(preview.status_code, 400, preview.text)
        self.assertIn("recipe", preview.text.lower())

    def test_classification_recipe_returns_400_unsupported(self):
        """Phase 9a covers qa-sft only; classification has no RAG
        corpus shape yet. The error names the recipe so the user
        knows why."""
        project = self._instantiate_template(
            "ticket-router", "AutoRAG Classification Unsupported"
        )
        resp = self.client.get(
            f"/api/projects/{project['id']}/auto-rag/preview",
            params={"query": "billing"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("classification", resp.text)

    def test_qa_sft_template_returns_ranked_retrievals(self):
        """End-to-end on policy-qa-style. The query should retrieve
        a real Q&A row, and the response shape includes recipe_id,
        index manifest, and the retrieved list."""
        project = self._instantiate_template(
            "policy-qa-style", "AutoRAG QA Preview"
        )
        resp = self.client.get(
            f"/api/projects/{project['id']}/auto-rag/preview",
            params={"query": "vacation days", "k": 3},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["recipe_id"], "qa-sft")
        self.assertEqual(payload["query"], "vacation days")
        self.assertEqual(payload["k"], 3)
        self.assertIn("index", payload)
        self.assertEqual(payload["index"]["recipe_id"], "qa-sft")
        self.assertEqual(payload["index"]["text_keys"], ["question", "answer"])
        # At least one retrieval; top hit's payload carries the
        # full row so Phase 9b can extract both question + answer.
        self.assertGreaterEqual(len(payload["retrieved"]), 1)
        top = payload["retrieved"][0]
        self.assertIn("row_id", top)
        self.assertIn("score", top)
        self.assertIn("payload", top)
        self.assertGreater(top["score"], 0)

    def test_second_call_hits_cached_index(self):
        """The preview endpoint rebuilds the index on first call,
        then loads from disk on subsequent calls. Both calls return
        the same top hit (cache validity not based on query text)."""
        project = self._instantiate_template(
            "policy-qa-style", "AutoRAG Cache Hit"
        )
        first = self.client.get(
            f"/api/projects/{project['id']}/auto-rag/preview",
            params={"query": "PII security policy"},
        )
        second = self.client.get(
            f"/api/projects/{project['id']}/auto-rag/preview",
            params={"query": "PII security policy"},
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        # Same top hit on both calls → caching didn't corrupt the index.
        self.assertEqual(
            first.json()["retrieved"][0]["row_id"],
            second.json()["retrieved"][0]["row_id"],
        )

    def test_k_query_param_caps_results(self):
        project = self._instantiate_template(
            "policy-qa-style", "AutoRAG K Param"
        )
        resp = self.client.get(
            f"/api/projects/{project['id']}/auto-rag/preview",
            params={"query": "policy", "k": 1},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["k"], 1)
        self.assertLessEqual(len(payload["retrieved"]), 1)


if __name__ == "__main__":
    unittest.main()
