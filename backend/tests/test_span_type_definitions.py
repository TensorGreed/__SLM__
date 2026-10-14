"""Entity-type definitions: one labelling convention for every model that
labels the project's spans — stored per project, served with suggestions,
and put into BOTH the training-label prompt and the answer-key prompt."""

from __future__ import annotations

import asyncio
import json
import time
import unittest

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services import span_relabel_service as relabel
from app.services import span_type_definitions_service as defs
from app.services.synthetic_service import generate_span_extraction_rows

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


DEFS = {
    "Date of Birth": "a person's birth date only",
    "visit_date": "an appointment or visit date",
    "iban": "",  # empty → dropped
}


class PureTests(unittest.TestCase):
    def test_normalise_clause_and_suggestions(self):
        norm = defs.normalise_definitions(DEFS)
        self.assertEqual(norm, {"date_of_birth": "a person's birth date only", "visit_date": "an appointment or visit date"})
        clause = defs.definitions_clause(["visit_date", "email", "date_of_birth"], norm)
        self.assertTrue(clause.startswith("Labelling conventions"))
        self.assertLess(clause.index("visit_date"), clause.index("date_of_birth"), "in the order of the type list")
        self.assertNotIn("email", clause)
        self.assertEqual(defs.definitions_clause(["email"], norm), "")
        self.assertEqual(defs.definitions_clause(["email"], {}), "")
        sugg = defs.suggest_definitions(["bank_account", "made_up_type", "provider_name"])
        self.assertEqual(sorted(sugg), ["bank_account", "provider_name"])
        self.assertIn("iban", sugg["bank_account"])


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"defs-{time.time_ns()}", "description": "x"}).json()["id"]
        syn = settings.DATA_DIR / "projects" / str(self.pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        (syn / "synthetic.jsonl").write_text(json.dumps({"text": "x", "entities": [{"type": "bank_account"}, {"type": "date_of_birth"}]}) + "\n")

    def test_endpoints_store_and_suggest(self):
        got = client.get(f"/api/projects/{self.pid}/synthetic/type-definitions").json()
        self.assertEqual(got["definitions"], {})
        self.assertEqual(got["types"], ["bank_account", "date_of_birth"])
        self.assertEqual(sorted(got["suggestions"]), ["bank_account", "date_of_birth"])
        put = client.put(f"/api/projects/{self.pid}/synthetic/type-definitions", json={"definitions": DEFS}).json()
        self.assertEqual(sorted(put["definitions"]), ["date_of_birth", "visit_date"])
        again = client.get(f"/api/projects/{self.pid}/synthetic/type-definitions").json()
        self.assertEqual(sorted(again["suggestions"]), ["bank_account"], "a defined type is no longer suggested")

    def test_both_prompts_carry_the_definitions(self):
        client.put(f"/api/projects/{self.pid}/synthetic/type-definitions", json={"definitions": DEFS})
        prompts: list[str] = []

        class Teacher:
            name = "fake"

            def describe(self):
                return "fake:teacher"

            async def complete(self, prompt, **kw):
                prompts.append(prompt)
                if "Label every entity" in prompt:
                    return json.dumps({"rows": [{"index": 0, "entities": [{"type": "date_of_birth", "text": "01/02/1990"}]}]})
                return '{"rows":[{"text":"Born 01/02/1990, seen on 03/04/2024 for a routine check-up at the clinic downtown today.","entities":[{"type":"date_of_birth","text":"01/02/1990"}]}]}'

        async def _go():
            async with async_session_factory() as db:
                await generate_span_extraction_rows(db, self.pid, "src", num_rows=1, entity_types=["date_of_birth", "visit_date"], backend_override=Teacher())
            prepared = settings.DATA_DIR / "projects" / str(self.pid) / "prepared"
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "test.jsonl").write_text(json.dumps({"text": "Born 01/02/1990.", "entities": [{"type": "date_of_birth", "start": 5, "end": 15, "text": "01/02/1990"}]}) + "\n")
            async with async_session_factory() as db:
                return await relabel.relabel_split_as_answer_key(db, self.pid, split="test", backend_override=Teacher(), entity_types=["date_of_birth", "visit_date"])

        summary = asyncio.run(_go())
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("Labelling conventions", prompt)
            self.assertIn("- date_of_birth: a person's birth date only", prompt)
            self.assertIn("- visit_date: an appointment or visit date", prompt)
        self.assertEqual(summary.conventions_types, ["date_of_birth", "visit_date"])


class TrainingRelabelTests(unittest.TestCase):
    def test_training_rows_are_relabelled_in_place_under_the_conventions(self):
        pid = client.post("/api/projects", json={"name": f"defs-train-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"date_of_birth": "birth date only"}})
        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows = [
            {"text": "Seen 03/04/2024, call 555-0100.", "entities": [{"type": "date_of_birth", "start": 5, "end": 15, "text": "03/04/2024"}], "review_status": "accepted"},
            {"text": "Born 01/02/1990.", "entities": [{"type": "date_of_birth", "start": 5, "end": 15, "text": "01/02/1990"}]},
            {"not": "a span row"},
        ]
        (syn / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        prompts: list[str] = []

        class Labeller:
            name = "fake"

            def describe(self):
                return "fake:labeller"

            async def complete(self, prompt, **kw):
                prompts.append(prompt)
                return json.dumps({"rows": [
                    {"index": 0, "entities": [{"type": "visit_date", "text": "03/04/2024"}]},
                    {"index": 1, "entities": [{"type": "date_of_birth", "text": "01/02/1990"}]},
                ]})

        async def _go():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(db, pid, backend_override=Labeller(), entity_types=["date_of_birth", "visit_date", "phone"])

        summary = asyncio.run(_go())
        self.assertIn("- date_of_birth: birth date only", prompts[0])
        self.assertEqual((summary.rows, summary.labelled_rows), (2, 2))
        saved = [json.loads(l) for l in (syn / "synthetic.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(len(saved), 3, "non-span rows are kept")
        self.assertEqual(sorted(e["type"] for e in saved[0]["entities"]), ["phone", "visit_date"], "relabelled + regex pass")
        self.assertEqual(saved[0]["review_status"], "accepted", "other fields kept")
        self.assertEqual(saved[0]["relabelled_by"], "fake:labeller")
        # Agreement old vs new: the DOB→visit_date change is the one disagreement.
        self.assertEqual((summary.agreement["tp"], summary.agreement["fn"]), (1, 2))


if __name__ == "__main__":
    unittest.main()


class PassageLabelTests(unittest.TestCase):
    def test_passages_are_labelled_as_they_are_and_appended_once(self):
        pid = client.post("/api/projects", json={"name": f"passages-{time.time_ns()}", "description": "x"}).json()["id"]
        base = settings.DATA_DIR / "projects" / str(pid)
        (base / "cleaned").mkdir(parents=True, exist_ok=True)
        (base / "synthetic").mkdir(parents=True, exist_ok=True)
        passages = [
            {"source_document_id": 5, "chunk_id": 0, "text": "NORTHWIND HEALTH PARTNERS\nPatient Ada Lovelace, call 555-0100.\nPage 1 of 2"},
            {"source_document_id": 5, "chunk_id": 1, "text": "Northwind\nNo identifiers on this page at all.\nPage 2 of 2"},
            {"source_document_id": 9, "chunk_id": 0, "text": "Another document's passage, call 555-0199."},
            {"row_index": 3, "text": "a structured row, not a passage"},
        ]
        (base / "cleaned" / "cleaned.jsonl").write_text("\n".join(json.dumps(p) for p in passages) + "\n")
        (base / "synthetic" / "synthetic.jsonl").write_text(json.dumps({"text": "old row", "entities": [{"type": "person_name", "start": 0, "end": 3, "text": "old"}]}) + "\n")
        prompts: list[str] = []

        class Labeller:
            name = "fake"

            def describe(self):
                return "fake:labeller"

            async def complete(self, prompt, **kw):
                prompts.append(prompt)
                return json.dumps({"rows": [{"index": 0, "entities": [{"type": "person_name", "text": "Ada Lovelace"}]}, {"index": 1, "entities": []}]})

        async def _go():
            async with async_session_factory() as db:
                return await relabel.label_passages_as_training_rows(
                    db, pid, document_id=5, backend_override=Labeller(), entity_types=["person_name", "phone"],
                )

        summary = asyncio.run(_go())
        self.assertEqual((summary.rows, summary.labelled_rows), (2, 2))
        self.assertIn("Page 1 of 2", prompts[0], "the passage goes to the labeller as it is, layout lines included")
        rows = [json.loads(l) for l in (base / "synthetic" / "synthetic.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 3, "appended to the existing rows")
        first = rows[1]
        self.assertEqual(first["text"], passages[0]["text"])
        self.assertEqual(sorted(e["type"] for e in first["entities"]), ["person_name", "phone"], "labeller + regex pass")
        self.assertEqual(first["synth_source"], "passage_labels:doc=5")
        self.assertEqual([r.get("id") for r in rows[1:]], [1, 2], "ids continue after the highest existing one (none here)")
        accepted = client.post(f"/api/projects/{pid}/synthetic/review-queue/bulk-update-by-source", json={"source": "passage_labels:doc=5", "action": "accept"}).json()
        self.assertEqual(accepted["accepted"], 2, "the review queue can accept them")
        self.assertEqual(rows[2]["entities"], [], "a page with nothing sensitive is a useful negative")
        with self.assertRaisesRegex(RuntimeError, "already labelled"):
            async def _again():
                async with async_session_factory() as db:
                    return await relabel.label_passages_as_training_rows(db, pid, document_id=5, backend_override=Labeller(), entity_types=["person_name"])
            asyncio.run(_again())
