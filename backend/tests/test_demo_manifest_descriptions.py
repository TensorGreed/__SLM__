"""Demo / template card text describes what the project does; it names no third-party tool."""
import json
import unittest
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"

_PII_DESCRIPTION = (
    "Pre-loaded with 60 synthetic chat / log / form snippets and a 25-row gold set. Each row's gold answer is a "
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
