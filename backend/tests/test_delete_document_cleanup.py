"""Deleting a document also drops its rows from cleaned.jsonl, its
``.cleaned.txt`` sidecar, and makes the passages index stale."""

from __future__ import annotations

import asyncio
import json
import os
import time
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import auto_rag_service
from app.services.span_relabel_service import load_document_passages

_PARA = "The harbour authority publishes tide tables every spring. "


def setUpModule():
    global _cm, client, _prev
    _prev = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _cm = TestClient(app)
    client = _cm.__enter__()


def tearDownModule():
    _cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev


class DeleteDocumentCleanupTests(unittest.TestCase):
    def setUp(self):
        r = client.post("/api/projects", json={"name": f"del-{uuid4().hex[:6]}", "description": ""})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = int(r.json()["id"])
        self.base = f"/api/projects/{self.pid}"
        self.cleaned = settings.DATA_DIR / "projects" / str(self.pid) / "cleaned" / "cleaned.jsonl"
        self.raw = settings.DATA_DIR / "projects" / str(self.pid) / "raw"

    def _doc(self, name: str, word: str, clean: bool = True) -> dict:
        body = (f"{word} " + _PARA * 30).encode()
        r = client.post(f"{self.base}/ingestion/upload", files={"file": (name, body, "text/plain")})
        self.assertIn(r.status_code, (200, 201), r.text)
        doc = r.json()
        r = client.post(f"{self.base}/ingestion/documents/{doc['id']}/process")
        self.assertEqual(r.status_code, 200, r.text)
        if clean:
            r = client.post(f"{self.base}/cleaning/clean", json={"document_id": doc["id"]})
            self.assertEqual(r.status_code, 200, r.text)
        return doc

    def _rows(self) -> list[dict]:
        return [json.loads(x) for x in self.cleaned.read_text().splitlines() if x.strip()]

    def test_delete_removes_only_that_docs_rows_and_sidecars(self):
        a = self._doc("a.txt", "alpha")
        b = self._doc("b.txt", "bravo")
        c = self._doc("c.txt", "charlie")
        before = self._rows()
        self.assertEqual({a["id"], b["id"], c["id"]}, {r["source_document_id"] for r in before})
        self.assertTrue((self.raw / "b.cleaned.txt").exists())
        (self.raw / "b.chunks.jsonl").write_text("{}\n")

        self.assertEqual(client.delete(f"{self.base}/ingestion/documents/{b['id']}").status_code, 204)

        after = self._rows()
        self.assertNotIn(b["id"], [r["source_document_id"] for r in after])
        self.assertEqual(
            [r["text"] for r in after], [r["text"] for r in before if r["source_document_id"] != b["id"]]
        )
        self.assertEqual([r["id"] for r in after], list(range(1, len(after) + 1)))
        self.assertFalse((self.raw / "b.cleaned.txt").exists())
        self.assertFalse((self.raw / "b.extracted.txt").exists())
        self.assertFalse((self.raw / "b.chunks.jsonl").exists())
        self.assertTrue((self.raw / "a.cleaned.txt").exists())
        self.assertEqual(load_document_passages(self.pid, b["id"]), [])
        self.assertNotIn(
            b["id"], [p["source_document_id"] for p in auto_rag_service.load_document_passages(self.pid)]
        )

    def test_deleting_a_never_cleaned_document_works(self):
        a = self._doc("a.txt", "alpha")
        n = self._doc("n.txt", "november", clean=False)
        before = self._rows()
        self.assertEqual(client.delete(f"{self.base}/ingestion/documents/{n['id']}").status_code, 204)
        self.assertEqual(self._rows(), before)
        self.assertEqual({r["source_document_id"] for r in self._rows()}, {a["id"]})

    def test_deleting_with_no_cleaned_file_works(self):
        n = self._doc("n.txt", "november", clean=False)
        self.assertFalse(self.cleaned.exists())
        self.assertEqual(client.delete(f"{self.base}/ingestion/documents/{n['id']}").status_code, 204)

    def test_index_goes_stale_after_delete(self):
        self._doc("a.txt", "alpha")
        b = self._doc("b.txt", "bravo")
        st = auto_rag_service.ensure_document_index(self.pid)
        self.assertTrue(st["available"], st)
        # Index built "in the past" so the rewrite's mtime is clearly newer.
        idx = auto_rag_service.document_index_dir(self.pid) / "bm25_index.json"
        old = time.time() - 60
        os.utime(idx, (old, old))
        os.utime(self.cleaned, (old - 10, old - 10))
        self.assertTrue(auto_rag_service.document_index_status(self.pid)["available"])
        client.delete(f"{self.base}/ingestion/documents/{b['id']}")
        self.assertTrue(auto_rag_service.document_index_status(self.pid)["stale"])
        st = auto_rag_service.ensure_document_index(self.pid)
        self.assertTrue(st["built"], st)
        payload = json.loads(idx.read_text())
        self.assertEqual(payload["doc_count"], len(auto_rag_service.load_document_passages(self.pid)))

    def test_batch_remover_rewrites_once(self):
        from app.database import async_session_factory
        from app.services import cleaning_service

        a = self._doc("a.txt", "alpha")
        b = self._doc("b.txt", "bravo")
        c = self._doc("c.txt", "charlie")

        async def go():
            async with async_session_factory() as db:
                n = await cleaning_service.remove_documents_from_cleaned(db, self.pid, [a["id"], c["id"]])
                await db.commit()
                return n

        self.assertGreater(asyncio.run(go()), 0)
        self.assertEqual({r["source_document_id"] for r in self._rows()}, {b["id"]})


if __name__ == "__main__":
    unittest.main()
