"""Endpoint tests for ``POST /api/projects/{id}/gold/add`` after the
per-recipe generalization. Verifies:
  * legacy qa-sft wire shape (``{question, answer}``) still works
  * classification rows (``{text, label}``) round-trip into the JSONL
  * span-extraction rows (``{text, entities}``) round-trip
  * summarization rows (``{document, summary}``) round-trip
  * empty rows (no recipe content) are rejected with EMPTY_GOLD_ROW
  * system-owned fields (``id``, ``created_at``) are NOT overridable
    even when the caller tries to send them
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "gold_add_per_recipe.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "gold_add_per_recipe_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["DOMAIN_BLUEPRINT_ENABLE_LLM_ENRICHMENT"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


class GoldAddPerRecipeTests(unittest.TestCase):

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

    def _make_project(self, name: str) -> int:
        resp = self.client.post("/api/projects", json={"name": name})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    # ── Legacy qa-sft wire shape ──────────────────────────────────

    def test_qa_sft_legacy_shape_still_works(self):
        pid = self._make_project("Add legacy QA")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "question": "What is the capital of France?",
                "answer": "Paris",
                "difficulty": "easy",
                "is_hallucination_trap": False,
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        entry = resp.json()
        self.assertEqual(entry["question"], "What is the capital of France?")
        self.assertEqual(entry["answer"], "Paris")
        self.assertEqual(entry["difficulty"], "easy")
        self.assertEqual(entry["is_hallucination_trap"], False)
        self.assertIn("id", entry)
        self.assertIn("created_at", entry)

    def test_qa_sft_hallucination_trap_flag_round_trips(self):
        pid = self._make_project("Add QA trap")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "question": "What is the meaning of life?",
                "answer": "I don't know.",
                "difficulty": "hard",
                "is_hallucination_trap": True,
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        entry = resp.json()
        self.assertEqual(entry["is_hallucination_trap"], True)
        self.assertEqual(entry["difficulty"], "hard")

    # ── Classification ────────────────────────────────────────────

    def test_classification_row_round_trips_into_jsonl(self):
        pid = self._make_project("Add classification")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "text": "I love this product!",
                "label": "positive",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        entry = resp.json()
        self.assertEqual(entry["text"], "I love this product!")
        self.assertEqual(entry["label"], "positive")
        # qa-sft keys should NOT appear when the caller didn't send them.
        self.assertNotIn("question", entry)
        self.assertNotIn("answer", entry)
        # Defaults still applied so eval code that reads them survives.
        self.assertEqual(entry["difficulty"], "medium")
        self.assertEqual(entry["is_hallucination_trap"], False)

        # Read back via /gold/entries to confirm JSONL round-trip.
        entries = self.client.get(
            f"/api/projects/{pid}/gold/entries",
            params={"dataset_type": "gold_dev"},
        )
        self.assertEqual(entries.status_code, 200, entries.text)
        rows = entries.json()["entries"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "I love this product!")
        self.assertEqual(rows[0]["label"], "positive")

    # ── Span-extraction ────────────────────────────────────────────

    def test_span_extraction_row_round_trips(self):
        pid = self._make_project("Add span")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "text": "Contact jane@example.com today",
                "entities": [
                    {
                        "type": "email",
                        "start": 8,
                        "end": 24,
                        "text": "jane@example.com",
                    },
                ],
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        entry = resp.json()
        self.assertEqual(entry["text"], "Contact jane@example.com today")
        self.assertEqual(len(entry["entities"]), 1)
        self.assertEqual(entry["entities"][0]["type"], "email")
        self.assertEqual(entry["entities"][0]["start"], 8)
        self.assertEqual(entry["entities"][0]["end"], 24)

    def test_span_extraction_row_with_empty_entities_accepted(self):
        # Negative-example rows (empty entities) are legitimate gold
        # data — they teach the model what NOT to extract.
        pid = self._make_project("Add span negative")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "text": "no PII in this row",
                "entities": [],
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        entry = resp.json()
        self.assertEqual(entry["entities"], [])

    # ── Summarization ──────────────────────────────────────────────

    def test_summarization_row_round_trips(self):
        pid = self._make_project("Add summary")
        long_doc = "The board meeting on March 14 covered three topics: " * 5
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "document": long_doc,
                "summary": "Board reviewed three topics on March 14.",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        entry = resp.json()
        self.assertEqual(entry["document"], long_doc)
        self.assertEqual(entry["summary"], "Board reviewed three topics on March 14.")

    # ── Empty-row guard ────────────────────────────────────────────

    def test_completely_empty_row_returns_400(self):
        pid = self._make_project("Add empty")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json()  # structured-error envelope: error_code is top-level
        self.assertEqual(detail.get("error_code"), "EMPTY_GOLD_ROW")

    def test_only_system_fields_returns_400(self):
        # Caller sent dataset_type + difficulty but no recipe content
        # — caught as empty-row.
        pid = self._make_project("Add system-only")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "dataset_type": "gold_test",
                "difficulty": "hard",
                "is_hallucination_trap": True,
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json()  # structured-error envelope: error_code is top-level
        self.assertEqual(detail.get("error_code"), "EMPTY_GOLD_ROW")

    def test_whitespace_only_strings_treated_as_empty(self):
        # A user pasting "   " in the answer field shouldn't sneak a
        # row in. The guard strips strings before counting.
        pid = self._make_project("Add whitespace")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "question": "   ",
                "answer": "\n\t",
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json()  # structured-error envelope: error_code is top-level
        self.assertEqual(detail.get("error_code"), "EMPTY_GOLD_ROW")

    # ── System-field protection ────────────────────────────────────

    def test_caller_cannot_override_id_or_created_at(self):
        # Even if the caller tries to send id=999 + created_at=2020-01,
        # the service overlays its own values.
        pid = self._make_project("Add system protect")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "question": "Q",
                "answer": "A",
                "id": 999,
                "created_at": "2020-01-01T00:00:00",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        entry = resp.json()
        # System fields won.
        self.assertNotEqual(entry["id"], 999)
        self.assertNotIn("2020", entry["created_at"])

    # ── Dataset routing ────────────────────────────────────────────

    def test_dataset_type_routes_to_gold_test(self):
        pid = self._make_project("Add to test set")
        resp = self.client.post(
            f"/api/projects/{pid}/gold/add",
            json={
                "question": "Test set Q",
                "answer": "Test set A",
                "dataset_type": "gold_test",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        # The new row lands in gold_test, NOT gold_dev.
        dev = self.client.get(
            f"/api/projects/{pid}/gold/entries",
            params={"dataset_type": "gold_dev"},
        )
        test = self.client.get(
            f"/api/projects/{pid}/gold/entries",
            params={"dataset_type": "gold_test"},
        )
        self.assertEqual(dev.json()["entries"], [])
        self.assertEqual(len(test.json()["entries"]), 1)
        self.assertEqual(test.json()["entries"][0]["question"], "Test set Q")


if __name__ == "__main__":
    unittest.main()
