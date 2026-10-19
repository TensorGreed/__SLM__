"""Duplicate a project (project_duplicate_service, POST /projects/{id}/duplicate).

Pins: the data tree (raw / cleaned / gold / prepared incl. versions /
synthetic) is copied byte-for-byte — the test examples stay identical —
Dataset rows point at the copy's files, the base model can be overridden,
and nothing measured (runs, auto_rag comparisons) or sibling flags come
along.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def setUpModule():
    global _cm, client, _prev
    _prev = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _cm = TestClient(app)
    client = _cm.__enter__()


def tearDownModule():
    _cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev


class DuplicateProjectTests(unittest.TestCase):
    def test_copies_the_data_and_split_but_not_the_measurements(self):
        resp = client.post("/api/projects", json={"name": f"dup-{uuid4().hex[:6]}", "description": "d"})
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = resp.json()["id"]
        client.put(f"/api/projects/{pid}", json={"base_model_name": "org/big-model"})
        root = Path(settings.DATA_DIR) / "projects" / str(pid)
        for rel, body in {
            "prepared/test.jsonl": '{"question": "q", "answer": "a"}\n',
            "prepared/versions/1/test.jsonl": '{"question": "q", "answer": "a"}\n',
            "cleaned/cleaned.jsonl": '{"text": "passage", "source_document_id": 1, "chunk_id": 0}\n',
            "synthetic/synthetic.jsonl": '{"id": 1, "question": "q2", "answer": "a2"}\n',
            "auto_rag/comparison_base_documents.json": json.dumps({"base_model": "org/big-model"}),
        }.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(body, encoding="utf-8")

        out = client.post(f"/api/projects/{pid}/duplicate", json={"base_model_name": "org/small-model"})
        self.assertEqual(out.status_code, 201, out.text)
        copy = out.json()
        self.assertNotEqual(copy["id"], pid)
        self.assertEqual(copy["base_model_name"], "org/small-model")
        self.assertIn("(copy)", copy["name"])
        dst = Path(settings.DATA_DIR) / "projects" / str(copy["id"])
        for rel in ("prepared/test.jsonl", "prepared/versions/1/test.jsonl", "cleaned/cleaned.jsonl", "synthetic/synthetic.jsonl"):
            self.assertEqual((dst / rel).read_bytes(), (root / rel).read_bytes(), rel)
        # Comparisons are measured on a base model: never copied.
        self.assertFalse((dst / "auto_rag").exists())
        cfg = copy.get("runtime_config") or {}
        self.assertEqual(cfg["duplicated_from"]["project_id"], pid)
        self.assertNotIn("rag_first", cfg)
        # The source keeps its base model.
        self.assertEqual(client.get(f"/api/projects/{pid}").json()["base_model_name"], "org/big-model")

    def test_unknown_project_is_404(self):
        self.assertEqual(client.post("/api/projects/99999999/duplicate", json={}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
