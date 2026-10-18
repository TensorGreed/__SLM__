"""Benchmark gold-VALUE contamination scan: do the answers a benchmark
expects appear verbatim in the project's training text? (The 2026-10-17
Gretel incident: 32 values × 591 occurrences, nothing flagged it.)"""

from __future__ import annotations

import json
import time
import unittest

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import benchmark_contamination_service as svc

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


class PureTests(unittest.TestCase):
    def test_gold_values_per_shape(self):
        spans = [{"text": "Call EMP-48213 or jo@x.io", "entities": [
            {"type": "employee_id", "start": 5, "end": 14, "text": "EMP-48213"},
            {"type": "email", "start": 18, "end": 25},  # no text → sliced from the row
        ]}]
        self.assertEqual(svc.gold_values(spans, "structured_extraction"),
                         [("EMP-48213", "employee_id"), ("jo@x.io", "email")])
        self.assertEqual(svc.gold_values([{"question": "q", "answer": "Paris is the capital"}], "qa"),
                         [("Paris is the capital", "answer")])
        self.assertEqual(svc.gold_values([{"text": "t", "label": "positive"}], "classification"), [])

    def test_scan_counts_rows_occurrences_sources_and_severity(self):
        training = [
            {"text": "Employee EMP-48213 called twice; EMP-48213 again.", "synth_source": "seed_documents:doc=7"},
            {"text": "Nothing to see", "synth_source": "documents_qa_flow"},
            {"text": "emp-48213 lower-cased and  spaced", "source": "cleaned"},
        ]
        values = [("EMP-48213", "employee_id"), ("ACCT-1", "account"), ("jo@x.io", "email"), ("ab", "tiny")]
        block = svc.scan_values(values, training)
        self.assertEqual(block["values_checked"], 3, "'ab' is below MIN_VALUE_CHARS")
        self.assertEqual(block["values_found"], 1)
        self.assertEqual(block["occurrences"], 3)
        self.assertEqual(block["rows_hit"], 2)
        self.assertEqual(block["by_type"]["employee_id"], {"checked": 1, "found": 1, "occurrences": 3})
        self.assertEqual(block["by_type"]["email"]["found"], 0)
        self.assertEqual(block["by_source"], {"seed_documents:doc=7": 1, "cleaned": 1})
        self.assertEqual(block["examples"][0]["value"], "EMP-48213")
        self.assertEqual(block["severity"], "severe", "1 of 3 values = 33 % of the benchmark's values")
        self.assertEqual(svc.severity_of(0.0, 0), "clean")
        self.assertEqual(svc.severity_of(0.01, 2), "low")
        self.assertEqual(svc.severity_of(0.05, 2), "notable")

    def test_block_and_line(self):
        self.assertEqual(svc.benchmark_contamination([{"text": "t", "label": "x"}], "classification", [])["applicable"], False)
        block = svc.benchmark_contamination(
            [{"text": "see PT-1234", "entities": [{"type": "patient_id", "start": 4, "end": 11}]}],
            "structured_extraction", [{"text": "record PT-1234"}],
        )
        self.assertTrue(block["applicable"])
        self.assertIn("1 of 1 gold values appear verbatim", svc.contamination_line(block))
        self.assertIsNone(svc.contamination_line({"applicable": True, "values_found": 0}))
        self.assertIsNone(svc.contamination_line(None))


def _project(name: str, train_rows):
    pid = client.post("/api/projects", json={"name": f"{name}-{time.time_ns()}", "description": "x"}).json()["id"]
    prepared = settings.DATA_DIR / "projects" / str(pid) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows))
    (prepared / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows))
    return pid


class ImportTests(unittest.TestCase):
    def test_import_scans_gold_values_and_rescan_follows_the_training_rows(self):
        train = [{"text": "Hi Ada Lovelace, your id is EMP-48213.", "entities": [{"type": "person_name", "start": 3, "end": 15}]}]
        pid = _project("bench-contam", train)
        imports = settings.DATA_DIR / "projects" / str(pid) / "imports"
        imports.mkdir(parents=True, exist_ok=True)
        path = imports / "pii.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in [
            {"text": "Badge EMP-48213 for Grace Hopper", "spans": [
                {"start": 6, "end": 15, "label": "ID"}, {"start": 20, "end": 32, "label": "PER"}]},
            {"text": "Badge EMP-99999 for Alan Turing", "spans": [
                {"start": 6, "end": 15, "label": "ID"}, {"start": 20, "end": 31, "label": "PER"}]},
        ]))
        created = client.post(f"/api/projects/{pid}/benchmarks", json={
            "locator": f"file:{path}", "name": "PII", "mapping": {"ID": "person_name", "PER": "person_name"}}).json()
        contamination = created["contamination"]
        self.assertTrue(contamination["applicable"])
        self.assertEqual((contamination["values_checked"], contamination["values_found"]), (4, 1))
        self.assertEqual(contamination["examples"][0]["value"], "EMP-48213")
        self.assertEqual(contamination["severity"], "severe")
        manifest = json.loads((settings.DATA_DIR / "projects" / str(pid) / "benchmarks" / created["slug"] / "manifest.json").read_text())
        self.assertEqual(manifest["contamination"]["values_found"], 1)

        # The training rows change (the contaminated value is scrubbed) → rescan.
        prepared = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        (prepared / "train.jsonl").write_text(json.dumps({"text": "Hi Ada Lovelace, your id is EMP-00001.", "entities": []}) + "\n")
        again = client.post(f"/api/projects/{pid}/benchmarks/{created['id']}/contamination").json()
        self.assertEqual(again["contamination"]["values_found"], 0)
        self.assertEqual(again["contamination"]["severity"], "clean")
        listed = client.get(f"/api/projects/{pid}/benchmarks").json()["benchmarks"][0]
        self.assertEqual(listed["contamination"]["values_found"], 0, "the DB manifest was updated too")
        self.assertEqual(client.post(f"/api/projects/{pid}/benchmarks/999999/contamination").status_code, 404)


if __name__ == "__main__":
    unittest.main()
