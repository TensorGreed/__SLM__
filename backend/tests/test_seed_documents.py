"""Seed documents: a generation model writes sample documents for a domain
when the user has no corpus; they are ingested as one text document and
cleaned into passages, so the rest of the pipeline is unchanged.

Covered:
  * the prompt names the genre, the required content and the locale;
  * documents are normalised to one paragraph each; empty / short replies
    and backend failures are counted, not fatal; nothing usable → raises;
  * the file is ingested (source="seed_documents"), parsed and cleaned —
    passages exist afterwards;
  * the endpoint runs it as a Job (202) and refuses a second one (409).
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services import seed_documents_service as svc
from app.services.synth_backends.base import SynthBackendError

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


class FakeWriter:
    name = "fake"
    calls: list[str] = []
    fail_on: set[int] = set()
    short_on: set[int] = set()

    def describe(self) -> str:
        return "fake:writer"

    async def complete(self, prompt, *, system_prompt=None, max_tokens=1024, temperature=0.7, response_schema=None):
        FakeWriter.calls.append(prompt)
        n = len(FakeWriter.calls)
        if n in FakeWriter.fail_on:
            raise SynthBackendError("boom")
        if n in FakeWriter.short_on:
            return "No."
        genre = prompt.split("realistic ")[1].split(" (")[0]
        body = (
            f"Hello, this is a {genre} number {n}. The customer Ada Lovelace (ada{n}@example.test, +1-555-01{n:02d}) "
            f"paid with card 4111 1111 1111 1{n:03d} expiring 09/27 and asked about order ORD-{1000 + n}. "
            "Please process the refund to account GB29 NWBK 6016 1331 9268 19 within five business days. "
            "Thanks for your patience; the team will follow up by email."
        )
        return "```\n" + body + "\n```"


class SeedDocumentsTests(unittest.TestCase):
    def setUp(self):
        FakeWriter.calls = []
        FakeWriter.fail_on = set()
        FakeWriter.short_on = set()
        self.pid = client.post("/api/projects", json={"name": f"seed-{time.time_ns()}", "description": "sensitive data"}).json()["id"]

    def _run(self, **kwargs):
        """Runs the service and reads the ingested document back IN THE SAME
        LOOP (a later TestClient read can lose the write on the shared
        StaticPool connection)."""
        from app.models.dataset import RawDocument

        async def _go():
            async with async_session_factory() as db:
                summary = await svc.run_seed_documents(db, self.pid, backend_override=FakeWriter(), **kwargs)
                doc = await db.get(RawDocument, summary.document_id) if summary.document_id else None
                self.last_doc = {"filename": doc.filename, "source": doc.source} if doc else None
                return summary

        return asyncio.run(_go())

    def test_prompt_names_genre_content_and_locale(self):
        prompt = svc.build_document_prompt("clinical visit note", "patient name, MRN, diagnosis", words=150, locale="UK", style="terse", domain_hint="a hospital")
        self.assertIn("clinical visit note (UK conventions)", prompt)
        self.assertIn("It must naturally contain: patient name, MRN, diagnosis.", prompt)
        self.assertIn("The organisation: a hospital.", prompt)
        self.assertIn("about 150 words", prompt)
        self.assertEqual(svc.normalise_document("```text\nline one\n\nline   two\n```"), "line one line two")

    def test_writes_ingests_and_cleans_counting_failures(self):
        FakeWriter.fail_on = {2}
        FakeWriter.short_on = {3}
        summary = self._run(kinds="card numbers, emails", genres=["support ticket", "billing email"], count=6, words=120)
        self.assertEqual((summary.requested, summary.written, summary.failed), (6, 4, 2))
        self.assertEqual(summary.backend, "fake:writer")
        self.assertTrue(summary.filename.startswith("seed-documents-"))
        self.assertGreaterEqual(summary.passages, 1)
        self.assertIn("2 of 6 documents were empty or failed", summary.warnings[0])
        # Genres alternate; the content requirement is in every prompt.
        self.assertIn("support ticket", FakeWriter.calls[0])
        self.assertIn("billing email", FakeWriter.calls[1])
        self.assertTrue(all("card numbers, emails" in c for c in FakeWriter.calls))
        # The document is in the project, marked by source, with passages on disk.
        self.assertEqual(self.last_doc["filename"], summary.filename)
        self.assertEqual(self.last_doc["source"], "seed_documents")
        self.assertTrue((settings.DATA_DIR / "projects" / str(self.pid) / "raw" / summary.filename).exists())
        cleaned = settings.DATA_DIR / "projects" / str(self.pid) / "cleaned" / "cleaned.jsonl"
        rows = [json.loads(l) for l in cleaned.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(rows), summary.passages)
        self.assertTrue(all("Ada Lovelace" in r["text"] or "refund" in r["text"] for r in rows))

    def test_nothing_usable_raises(self):
        FakeWriter.short_on = {1, 2, 3}
        with self.assertRaises(RuntimeError) as ctx:
            self._run(count=3)
        self.assertIn("wrote no usable documents", str(ctx.exception))

    def test_endpoint_runs_as_a_job_and_refuses_a_second(self):
        from unittest import mock

        from app.services.synth_backends import base as backends_base

        started = {}

        async def fake_run(db, project_id, **kwargs):
            started.update(kwargs, project_id=project_id)
            await asyncio.sleep(0.5)
            return svc.SeedDocumentsSummary(project_id=project_id, requested=kwargs["count"], written=2, failed=0, backend="fake", filename="seed-documents-x.txt", document_id=1, passages=3)

        with mock.patch("app.api.flows.run_seed_documents", fake_run):
            res = client.post(f"/api/projects/{self.pid}/flows/seed-documents", json={"kinds": "card numbers", "count": 2, "genres": ["ticket"]})
            self.assertEqual(res.status_code, 202, res.text)
            job_id = res.json()["id"]
            dup = client.post(f"/api/projects/{self.pid}/flows/seed-documents", json={"count": 2})
            self.assertEqual(dup.status_code, 409)
            for _ in range(40):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.25)
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["result"]["written"], 2)
        self.assertEqual((started["kinds"], started["count"], started["genres"]), ("card numbers", 2, ["ticket"]))
        self.assertEqual(client.get(f"/api/projects/{self.pid}/flows/seed-documents/defaults").json()["genres"][0], "customer support ticket")
        del backends_base


    def test_cloud_pin_needs_a_saved_key_and_reaches_the_runner(self):
        from unittest import mock

        pin = "cloud:anthropic:claude-test"
        missing = client.post(f"/api/projects/{self.pid}/flows/seed-documents", json={"count": 2, "backend": pin})
        self.assertEqual(missing.status_code, 402, missing.text)
        self.assertEqual(missing.json()["error_code"], "CLOUD_KEY_MISSING")
        self.assertEqual(client.post(f"/api/projects/{self.pid}/flows/seed-documents", json={"count": 2, "backend": "cloud::"}).status_code, 400)
        saved = client.put(f"/api/projects/{self.pid}/secrets", json={"provider": "cloud_llm_anthropic", "key_name": "api_key", "value": "sk-test-key"})
        self.assertEqual(saved.status_code, 201, saved.text)

        started = {}

        async def fake_run(db, project_id, **kwargs):
            started.update(kwargs)
            await asyncio.sleep(0.5)
            return svc.SeedDocumentsSummary(project_id=project_id, requested=2, written=2, failed=0, backend="cloud", filename="seed-documents-x.txt", document_id=1, passages=3)

        with mock.patch("app.api.flows.run_seed_documents", fake_run):
            res = client.post(f"/api/projects/{self.pid}/flows/seed-documents", json={"count": 2, "backend": pin})
            self.assertEqual(res.status_code, 202, res.text)
            job_id = res.json()["id"]
            for _ in range(40):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.25)
        self.assertEqual(job["status"], "succeeded", job)
        override = started["backend_override"]
        self.assertEqual(override.describe(), "cloud:anthropic:claude-test")


if __name__ == "__main__":
    unittest.main()


class PagedLayoutTests(unittest.TestCase):
    """layout="paged": pages with header / footer lines, varied, invented,
    never the held-out bracketed "[i/n]" style."""

    def test_paginate_inserts_furniture_lines(self):
        import random
        import re

        from app.services.seed_documents_service import paginate_document

        text = " ".join(f"Sentence number {i} is here." for i in range(120))
        styles = set()
        for seed in range(40):
            paged = paginate_document(text, random.Random(seed), genre="clinical visit note or discharge summary", words_per_page=60)
            lines = paged.split("\n")
            body = [l for l in lines if l.startswith("Sentence number")]
            furniture = [l for l in lines if not l.startswith("Sentence number")]
            self.assertGreaterEqual(len(body), 7, "about 480 words → 8 pages")
            self.assertGreaterEqual(len(furniture), len(body), "a footer per page, a header on most")
            self.assertEqual(" ".join(body).split(), text.split(), "the prose is unchanged, only split")
            self.assertFalse(any(re.fullmatch(r"\[\d+/\d+\]", l) for l in furniture), "the held-out style never appears")
            styles.add(lines[-1].split()[0] if lines[-1].split() else "")
        self.assertGreater(len(styles), 3, "footer styles vary across documents")

    def test_plain_layout_is_unchanged(self):
        from app.services.seed_documents_service import LAYOUTS

        self.assertEqual(LAYOUTS, ("plain", "paged", "scanned"))


class ScannedLayoutTests(unittest.TestCase):
    def test_scanned_layout_is_accepted_and_renders_an_ocr_readable_pdf(self):
        import random

        from app.services.seed_documents_service import LAYOUTS, paginate_document, render_scanned_pdf
        from app.utils.file_parsers import ocr_available, parse_pdf

        self.assertIn("scanned", LAYOUTS)
        text = " ".join(f"Sentence {i} mentions Ada Lovelace." for i in range(40))
        pdf = render_scanned_pdf([paginate_document(text, random.Random(1), genre="note")], random.Random(2))
        self.assertTrue(pdf.startswith(b"%PDF"))
        if not ocr_available():
            self.skipTest("tesseract / pdftoppm not installed")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.pdf"
            path.write_bytes(pdf)
            read = parse_pdf(path)
        self.assertGreater(read.count("Lovelace"), 10, "OCR reads most of it back")
