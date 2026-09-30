"""Document-level retrieval (Wave 2c-2).

Pins:
  * cleaned document passages (not structured Q&A rows) are indexed with
    BM25 under ``auto_rag/documents/``, lazily, and rebuilt when cleaning
    has run since;
  * the playground grounds answers in those passages for projects without
    a Q&A recipe, with a cite-or-say-you-don't-know preamble and per-hit
    ``source_doc`` / passage provenance; Q&A projects keep the Q&A index;
  * end to end from a real .docx upload → process → clean → index → chat;
  * RAG-compare's retriever uses the passage index / extracted text — it
    used to read PDF/DOCX files as raw bytes.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services import auto_rag_service as rag
from app.services.rag_sandbox_service import retrieve_project_rag_snippets


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


def _docx_bytes(paragraphs: list[str]) -> bytes:
    from docx import Document

    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


MANUAL = [
    "Zorbex K-7 blender warranty: the motor housing is covered for fourteen months from purchase.",
    "To descale the Quillon M2 kettle, fill it with CitraClean solution every six weeks and boil twice.",
    "The Varnet X9 toaster ships with a crumb tray that must be emptied weekly to avoid smoke.",
] * 4


class DocumentRetrievalTests(unittest.TestCase):
    def setUp(self):
        resp = client.post("/api/projects", json={"name": f"docs-{uuid4().hex[:6]}", "description": ""})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.pid = int(resp.json()["id"])

    def _ingest_docx(self, name: str, paragraphs: list[str]) -> int:
        doc = client.post(
            f"/api/projects/{self.pid}/ingestion/upload",
            files={"file": (name, _docx_bytes(paragraphs), "application/octet-stream")},
        ).json()
        self.assertEqual(
            client.post(f"/api/projects/{self.pid}/ingestion/documents/{doc['id']}/process").json()["status"],
            "accepted",
        )
        resp = client.post(
            f"/api/projects/{self.pid}/cleaning/clean",
            json={"document_id": doc["id"], "chunk_size": 200, "chunk_overlap": 0},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return doc["id"]

    def test_docx_to_grounded_chat_end_to_end(self):
        self._ingest_docx("manual.docx", MANUAL)
        status = client.get(f"/api/projects/{self.pid}/auto-rag/documents").json()
        self.assertTrue(status["available"], status)
        self.assertGreater(status["passages"], 1)

        resp = client.post(
            f"/api/projects/{self.pid}/training/playground/chat",
            json={
                "provider": "mock",
                "auto_rag": True,
                "messages": [{"role": "user", "content": "How long is the K-7 blender warranty?"}],
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        block = resp.json()["auto_rag"]
        self.assertTrue(block["applied"])
        self.assertEqual(block["corpus"], "documents")
        top = block["retrieved"][0]["payload"]
        self.assertIn("fourteen months", top["text"])
        self.assertEqual(top["source_doc"], "manual.docx")

    def test_preamble_cites_passages_and_admits_ignorance(self):
        self._ingest_docx("manual.docx", MANUAL)

        async def _go():
            async with async_session_factory() as db:
                return await rag.build_preamble_from_query(db, self.pid, "descale kettle CitraClean", k=2)

        out = asyncio.run(_go())
        self.assertEqual(out["corpus"], "documents")
        self.assertIn("[1] (manual.docx · passage", out["preamble_text"])
        self.assertIn("say you don't know", out["preamble_text"])
        self.assertIn("CitraClean", out["retrieved"][0]["payload"]["text"])

    def test_index_skips_structured_rows_and_rebuilds_when_stale(self):
        csv = b"question,answer\nWhat is the K-7 warranty?,Fourteen months\n"
        doc = client.post(
            f"/api/projects/{self.pid}/ingestion/upload",
            files={"file": ("faq.csv", csv, "text/csv")},
        ).json()
        client.post(f"/api/projects/{self.pid}/ingestion/documents/{doc['id']}/process")
        client.post(f"/api/projects/{self.pid}/cleaning/clean", json={"document_id": doc["id"]})
        self.assertEqual(rag.load_document_passages(self.pid), [])
        self.assertFalse(rag.ensure_document_index(self.pid)["available"])

        self._ingest_docx("manual.docx", MANUAL[:3])
        first = rag.ensure_document_index(self.pid)
        self.assertTrue(first["built"])
        time.sleep(0.01)
        self._ingest_docx("second.docx", ["The Pellix T4 grill needs a ceramic plate replaced yearly."])
        self.assertTrue(rag.document_index_status(self.pid)["stale"])
        second = rag.ensure_document_index(self.pid)
        self.assertTrue(second["built"])
        self.assertGreater(second["passages"], first["passages"])

    def test_qa_projects_keep_the_qa_index(self):
        self._ingest_docx("manual.docx", MANUAL)
        client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "qa-sft"})
        qa_dir = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "auto_rag"
        rag.build_bm25_index(
            [{"id": 1, "question": "K-7 warranty?", "answer": "Fourteen months"}],
            recipe_id="qa-sft",
            output_dir=qa_dir,
            corpus_source=rag.QA_CORPUS_SOURCE,
        )

        async def _go(corpus):
            async with async_session_factory() as db:
                return await rag.build_preamble_from_query(db, self.pid, "K-7 warranty", corpus=corpus)

        self.assertEqual(asyncio.run(_go("auto"))["corpus"], "qa")
        self.assertEqual(asyncio.run(_go("documents"))["corpus"], "documents")

    def test_rag_compare_retriever_reads_passages_not_bytes(self):
        self._ingest_docx("manual.docx", MANUAL)

        async def _go():
            async with async_session_factory() as db:
                return await retrieve_project_rag_snippets(db, project_id=self.pid, query="toaster crumb tray")

        snippets = asyncio.run(_go())
        self.assertTrue(snippets)
        self.assertIn("crumb tray", snippets[0]["text"])
        self.assertEqual(snippets[0]["source_doc"], "manual.docx")
        json.dumps(snippets)  # plain text, no binary garbage


if __name__ == "__main__":
    unittest.main()
