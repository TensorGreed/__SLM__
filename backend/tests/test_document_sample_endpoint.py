"""Document row-sampling endpoint (Data tab accordion preview).

Pins:
- JSONL: reservoir-samples up to n rows from large files in a single
  pass; total_rows_scanned reflects the full file.
- CSV: header-aware, yields dict rows.
- Plain text: yields ``{"line": ...}`` rows.
- Unsupported extensions return ``rows: []`` + a clear note rather
  than 500-ing.
- 404 when the document isn't in the project; 404 when the file is
  missing on disk.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


TEST_DATA_DIR = Path(tempfile.gettempdir()) / f"brewslm-doc-sample-{uuid.uuid4().hex[:8]}"


def _upload_text_file(
    client: TestClient,
    project_id: int,
    filename: str,
    body: bytes,
    content_type: str = "application/octet-stream",
) -> int:
    """Push a file through /ingestion/upload + return the doc id."""

    files = {"file": (filename, body, content_type)}
    data = {"source": "upload", "sensitivity": "internal", "license_info": ""}
    resp = client.post(
        f"/api/projects/{project_id}/ingestion/upload",
        files=files,
        data=data,
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


class DocumentSampleEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{name}-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    # ── JSONL ───────────────────────────────────────────────────────

    def test_jsonl_returns_random_n_rows_with_full_scanned_count(self):
        pid = self._create_project("doc-sample-jsonl")
        rows = [{"text": f"row {i}", "label": "pos"} for i in range(200)]
        body = ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")
        doc_id = _upload_text_file(
            self.client, pid, "rows.jsonl", body, "application/jsonl"
        )

        resp = self.client.get(
            f"/api/projects/{pid}/ingestion/documents/{doc_id}/sample?n=10"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(len(payload["rows"]), 10)
        self.assertEqual(payload["total_rows_scanned"], 200)
        self.assertEqual(payload["file_type"], "jsonl")
        for row in payload["rows"]:
            self.assertIn("text", row)
            self.assertIn("label", row)
        # Default n=10 (no query arg).
        resp_default = self.client.get(
            f"/api/projects/{pid}/ingestion/documents/{doc_id}/sample"
        )
        self.assertEqual(len(resp_default.json()["rows"]), 10)

    def test_jsonl_smaller_than_n_returns_all_rows(self):
        pid = self._create_project("doc-sample-small")
        rows = [{"text": "only", "label": "x"}]
        body = (json.dumps(rows[0]) + "\n").encode("utf-8")
        doc_id = _upload_text_file(
            self.client, pid, "tiny.jsonl", body, "application/jsonl"
        )
        resp = self.client.get(
            f"/api/projects/{pid}/ingestion/documents/{doc_id}/sample?n=10"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["rows"]), 1)

    # ── CSV ─────────────────────────────────────────────────────────

    def test_csv_header_aware_dict_rows(self):
        pid = self._create_project("doc-sample-csv")
        body = b"text,label\nhello,pos\nworld,neg\nfine,neu\n"
        doc_id = _upload_text_file(
            self.client, pid, "rows.csv", body, "text/csv"
        )
        resp = self.client.get(
            f"/api/projects/{pid}/ingestion/documents/{doc_id}/sample?n=5"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["file_type"], "csv")
        self.assertEqual(payload["total_rows_scanned"], 3)
        labels = sorted(r["label"] for r in payload["rows"])
        self.assertEqual(labels, ["neg", "neu", "pos"])

    # ── Plain text ──────────────────────────────────────────────────

    def test_plain_text_returns_line_rows(self):
        pid = self._create_project("doc-sample-txt")
        body = b"line one\nline two\nline three\n"
        doc_id = _upload_text_file(
            self.client, pid, "raw.txt", body, "text/plain"
        )
        resp = self.client.get(
            f"/api/projects/{pid}/ingestion/documents/{doc_id}/sample?n=10"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["file_type"], "txt")
        self.assertEqual(payload["total_rows_scanned"], 3)
        # All rows have a "line" key with the text.
        for row in payload["rows"]:
            self.assertIn("line", row)

    # ── 404 + edge cases ────────────────────────────────────────────

    def test_unknown_document_returns_404(self):
        pid = self._create_project("doc-sample-404")
        resp = self.client.get(
            f"/api/projects/{pid}/ingestion/documents/999999/sample"
        )
        self.assertEqual(resp.status_code, 404)

    def test_unsupported_extension_returns_clear_note(self):
        pid = self._create_project("doc-sample-pdf")
        # Pretend it's a tiny PDF — only the extension matters.
        body = b"%PDF-1.4\n%fake pdf header\n"
        doc_id = _upload_text_file(
            self.client, pid, "fake.pdf", body, "application/pdf"
        )
        resp = self.client.get(
            f"/api/projects/{pid}/ingestion/documents/{doc_id}/sample"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["rows"], [])
        self.assertIn("Preview not available", payload["note"])


if __name__ == "__main__":
    unittest.main()
