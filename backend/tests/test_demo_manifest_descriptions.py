"""Demo / template card text describes what the project does; it names no third-party tool."""
import csv
import json
import re
import unittest
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"

_PII_DESCRIPTION = (
    "Pre-loaded with 61 synthetic chat / log / form snippets and a 200-row gold set. Each row's gold answer is a "
    "JSON object listing every PII / PCI span found in the text — type, start offset, end offset, and the raw "
    "value — so the trained model can plug into any redaction or guardrail pipeline that needs structured entity "
    "output. Covers email, phone, SSN, credit card, person_name, street_address, date_of_birth, ip_address, "
    "api_key, bank_account."
)


class DemoManifestDescriptionTests(unittest.TestCase):
    def test_pii_card_description_is_generic(self):
        m = json.loads((_DATA / "demo_samples" / "pii-detector" / "manifest.json").read_text())
        self.assertEqual(m["description"], _PII_DESCRIPTION)
        self.assertIn("guardrail scanner or any redaction pipeline", m["suggested_brief"])

    def test_log_triage_template_names_no_tool(self):
        m = json.loads((_DATA / "project_templates" / "log-triage" / "manifest.json").read_text())
        self.assertIn("a streaming filter without a GPU round-trip", m["description"])

    def test_demo_row_counts_match_the_files(self):
        # 7h follow-up: the gold files grew to 200 rows (aa04beb) while the cards still said 10 / 25.
        for slug in ("sentiment-classifier", "pii-detector"):
            root = _DATA / "demo_samples" / slug
            m = json.loads((root / "manifest.json").read_text())
            gold = sum(1 for line in (root / m["gold_filename"]).read_text().splitlines() if line.strip())
            with (root / m["dataset_filename"]).open(newline="") as fh:
                rows = sum(1 for _ in csv.DictReader(fh))
            self.assertIn(f"{gold}-row gold set", m["description"], slug)
            numbers = [int(n) for n in re.findall(r"\b(\d+) (?:short|synthetic)", m["description"])]
            self.assertEqual(numbers, [rows], slug)
