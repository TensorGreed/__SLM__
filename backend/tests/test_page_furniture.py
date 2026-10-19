"""Page furniture: repeated page headers / footers / page numbers are dropped
before chunking (``services/page_furniture``), per page — the PDF parser
keeps page breaks as a form feed on its own line.

Covers:
  * the detector on synthetic multi-page text: a header on every page, a
    "Page i of n" footer, a print-date line, a footer fused onto the last
    body line, a running head with a page counter; body lines that repeat
    on fewer pages (or in the middle of a page) stay;
  * no page breaks / too few pages → unchanged; off → only the page breaks
    become blank lines (the old join);
  * parse_pdf joins pages with PAGE_SEPARATOR;
  * through the API: a generated multi-page PDF cleaned with the default
    drops the furniture and reports it; ``remove_page_furniture: false``
    keeps it; a re-clean without the field keeps the document's last
    setting; a CSV is untouched.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import page_furniture as pf
from app.utils.file_parsers import PAGE_SEPARATOR, parse_pdf


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


N_PAGES = 9
_WORDS = ("alpha", "bravo", "cedar", "delta", "ember", "fjord", "grove", "harbor", "island", "juniper")
_CHAPTERS = ("Basics", "Appeals", "Review")


def _page(i: int, n: int = N_PAGES) -> list[str]:
    """A page: a header, ten body lines, a print stamp, a footer and a
    running head ("Chapter c <title> <page>"). Page 3's footer is fused onto
    its last body line, as PDF extraction does."""
    w = _WORDS[i]
    body = [
        f"The {w} section opens the topic for this page.",
        f"Readers of the {w} part should note the exceptions.",
        f"Each {w} example is worked through in full.",
        "Every claim must be filed within thirty days of the notice.",  # every page, mid-page
        f"Later {w} cases are handled by the regional board.",
        f"Further {w} details follow on the next lines.",
        f"Some {w} claims need a second signature.",
        f"Other {w} claims can be filed online.",
        f"Notes on {w} close the section.",
        f"The {w} page ends with this sentence.",
    ]
    if i in (2, 5):
        # At the page edge, but only on 2 of 9 pages: below the share → kept.
        body.append("See the appendix for the full schedule.")
    if i in (4, 6):
        body.insert(5, "Keep a copy of every form you send.")
    lines = ["Northwind Claims Manual — Internal", *body]
    stamp = f"Printed 2026-01-{10 + i:02d} 13:{40 + i}"
    footer = f"Page {i} of {n}"
    if i == 3:
        lines[-1] = lines[-1] + footer
        lines.append(stamp)
    else:
        lines += [stamp, footer]
    chapter = (i - 1) // 3
    lines.append(f"Chapter {chapter + 1} {_CHAPTERS[chapter]} {i + 10}")
    return lines


def _document() -> str:
    return PAGE_SEPARATOR.join("\n".join(_page(i)) for i in range(1, N_PAGES + 1))


class DetectorTests(unittest.TestCase):
    def test_removes_header_footer_stamp_and_running_head(self):
        text, report = pf.remove_page_furniture(_document())
        self.assertNotIn("Northwind Claims Manual", text)
        self.assertNotIn("Page 1 of 9", text)
        self.assertNotIn("Page 3 of 9", text)
        self.assertNotIn("Printed 2026", text)
        self.assertNotIn("Chapter 1 Basics", text)
        self.assertNotIn("Chapter 3 Review", text)
        self.assertNotIn("\f", text)
        # The fused footer is cut off; the body line it was glued to stays.
        self.assertIn("The cedar page ends with this sentence.\n", text + "\n")
        self.assertIn("The delta page ends with this sentence.", text)
        # Body lines stay, including ones that repeat (mid-page, or at the
        # edge of fewer pages than the share).
        self.assertEqual(text.count("Every claim must be filed within thirty days of the notice."), N_PAGES)
        self.assertEqual(text.count("See the appendix for the full schedule."), 2)
        self.assertEqual(text.count("Keep a copy of every form you send."), 2)
        for w in _WORDS[1:]:
            self.assertIn(f"The {w} section opens the topic for this page.", text)

        d = report.as_dict()
        self.assertEqual(d["pages"], N_PAGES)
        # header + stamp + running head on 9 pages, footer on its own line on 8.
        self.assertEqual(d["furniture_lines_removed"], 9 * 3 + 8)
        self.assertEqual(d["furniture_segments_trimmed"], 1)
        self.assertEqual(d["by_rule"]["running_head"], 9)
        self.assertEqual(d["by_rule"]["page_number"], 8)
        self.assertEqual(d["by_rule"]["repeated"], 9 * 2 + 1)  # header, stamp, the fused footer
        self.assertTrue(d["examples"])
        self.assertLessEqual(len(d["examples"]), pf.MAX_EXAMPLES)

    def test_page_number_patterns(self):
        for line in ("Page 9", "page 9 of 142", "- 9 -", "9 / 142", "[9/142]", "p. 9", "Pg. 12"):
            self.assertTrue(pf.is_page_number_line(line), line)
        self.assertTrue(pf.is_page_number_line("12"))
        self.assertFalse(pf.is_page_number_line("12", outermost=False))
        self.assertFalse(pf.is_page_number_line("Page nine explains it"))
        self.assertFalse(pf.is_page_number_line("12,500"))

    def test_page_numbers_go_even_when_they_never_repeat_as_text(self):
        pages = [f"Body text number {i} is here.\nMore body text {i} follows.\n- {i} -" for i in range(1, 4)]
        text, report = pf.remove_page_furniture(PAGE_SEPARATOR.join(pages))
        self.assertNotIn("- 2 -", text)
        self.assertEqual(report.by_rule.get("page_number"), 3)

    def test_a_table_row_of_changing_numbers_is_not_a_running_head(self):
        # The row sits at the page edge on 3 of 8 pages (under the share),
        # its figures change page to page like a counter — but it has no
        # words, so it is a table, not a running head.
        pages = []
        for i in range(8):
            w = _WORDS[i]
            lines = [f"The {w} heading", f"Body about {w} goes here.", f"More on {w} below."]
            if i in (1, 4, 6):
                lines.append(f"{i},100 {i},200 {i},300")
            pages.append("\n".join(lines))
        text, _ = pf.remove_page_furniture(PAGE_SEPARATOR.join(pages))
        for i in (1, 4, 6):
            self.assertIn(f"{i},100 {i},200 {i},300", text)

    def test_no_page_breaks_or_too_few_pages_is_unchanged(self):
        plain = "Page 1 of 2\nBody.\nPage 2 of 2"
        self.assertFalse(pf.has_page_breaks(plain))
        two = "Header\nBody one.\nPage 1 of 2" + PAGE_SEPARATOR + "Header\nBody two.\nPage 2 of 2"
        text, report = pf.remove_page_furniture(two)
        self.assertEqual(text, "Header\nBody one.\nPage 1 of 2\n\nHeader\nBody two.\nPage 2 of 2")
        self.assertEqual(report.lines_removed, 0)

    def test_join_pages_is_the_old_parser_join(self):
        pages = ["one\ntwo", "three", "four"]
        self.assertEqual(pf.join_pages(pf.split_pages(PAGE_SEPARATOR.join(pages))), "\n\n".join(pages))


# ── a tiny text PDF, no extra dependency ────────────────────────────────


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_text_pdf(pages: list[list[str]]) -> bytes:
    """A PDF with one Helvetica text line per entry, pages in order."""
    objects: list[bytes] = []
    n = len(pages)
    page_ids = [3 + 2 * k for k in range(n)]
    font_id = 3 + 2 * n
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for k, lines in enumerate(pages):
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 800 Td"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {page_ids[k] + 1} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _pdf_pages() -> list[list[str]]:
    pages = []
    for i in range(1, 6):
        w = _WORDS[i]
        pages.append([
            "Northwind Claims Manual - Internal",
            f"The {w} section opens the topic.",
            f"Readers of the {w} part note the exceptions.",
            f"Each {w} example is worked in full.",
            "Every claim must be filed within thirty days of the notice.",
            f"Later {w} cases go to the regional board.",
            f"Some {w} claims need a second signature.",
            f"Notes on {w} close the section.",
            f"Printed 2026-01-{10 + i:02d}",
            f"Page {i} of 5",
        ])
    return pages


class ParsePdfTests(unittest.TestCase):
    def test_pages_are_joined_with_the_page_separator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual.pdf"
            path.write_bytes(make_text_pdf(_pdf_pages()))
            text = parse_pdf(path)
        self.assertEqual(text.count("\f"), 4)
        self.assertIn("Each delta example", text)


class CleanDocumentTests(unittest.TestCase):
    def setUp(self):
        resp = client.post("/api/projects", json={"name": f"furniture-{uuid4().hex[:6]}", "description": ""})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.pid = int(resp.json()["id"])

    def _ingest(self, name: str, data: bytes) -> int:
        resp = client.post(
            f"/api/projects/{self.pid}/ingestion/upload",
            files={"file": (name, data, "application/octet-stream")},
        )
        self.assertIn(resp.status_code, (200, 201), resp.text)
        doc_id = int(resp.json()["id"])
        resp = client.post(f"/api/projects/{self.pid}/ingestion/documents/{doc_id}/process")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["status"], "accepted", resp.text)
        return doc_id

    def _clean(self, doc_id: int, **extra) -> dict:
        resp = client.post(
            f"/api/projects/{self.pid}/cleaning/clean",
            json={"document_id": doc_id, "redact_pii": False, **extra},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _passages(self) -> str:
        path = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "cleaned" / "cleaned.jsonl"
        return "\n".join(json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines())

    def test_pdf_furniture_is_dropped_by_default_and_kept_when_off(self):
        doc_id = self._ingest("manual.pdf", make_text_pdf(_pdf_pages()))

        body = self._clean(doc_id)
        self.assertEqual(body["furniture_lines_removed"], 15)  # header, stamp, footer × 5
        self.assertTrue(body["page_furniture"]["enabled"])
        self.assertEqual(body["page_furniture"]["pages"], 5)
        passages = self._passages()
        self.assertNotIn("Page 2 of 5", passages)
        self.assertNotIn("Northwind Claims Manual", passages)
        self.assertNotIn("Printed 2026", passages)
        self.assertNotIn("\f", passages)
        self.assertIn("Later ember cases go to the regional board.", passages)

        off = self._clean(doc_id, remove_page_furniture=False)
        self.assertEqual(off["furniture_lines_removed"], 0)
        self.assertFalse(off["page_furniture"]["enabled"])
        passages = self._passages()
        self.assertIn("Page 2 of 5", passages)
        self.assertNotIn("\f", passages)

        # Re-cleaning without the field keeps the document's last setting.
        again = self._clean(doc_id)
        self.assertFalse(again["page_furniture"]["enabled"])
        self.assertIn("Page 2 of 5", self._passages())

    def test_row_files_are_untouched(self):
        rows = "".join(f"Page {i} of 5,Every claim must be filed.\n" for i in range(1, 6))
        doc_id = self._ingest("rows.csv", ("question,answer\n" + rows).encode())
        body = self._clean(doc_id)
        self.assertTrue(body.get("structured"))
        self.assertEqual(body["rows_kept"], 5)
        self.assertNotIn("page_furniture", body)


if __name__ == "__main__":
    unittest.main()
