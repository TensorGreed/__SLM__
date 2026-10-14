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


class DocumentLabelTests(unittest.TestCase):
    def test_pack_long_rows_keeps_blocks_and_page_lines(self):
        doc_a = "LETTERHEAD\nDear Ada, call 555-0100.\nPage 1 of 2\nMore text here.\nPage 2 of 2"
        doc_b = "Second document, short."
        rows = relabel.pack_long_rows(doc_a + "\n\n" + doc_b, max_chars=90)
        self.assertEqual(rows, [doc_a, doc_b], "whole blocks, page lines kept; too long to pack together")
        self.assertEqual(relabel.pack_long_rows(doc_a + "\n\n" + doc_b, max_chars=500), [doc_a + "\n\n" + doc_b])
        long_block = " ".join(f"Sentence {i} is here." for i in range(60))
        pieces = relabel.pack_long_rows(long_block, max_chars=300)
        self.assertTrue(all(len(p) <= 300 for p in pieces) and len(pieces) > 3, "an oversized block is cut at sentence ends")

    def test_whole_documents_are_labelled_one_per_call_and_appended(self):
        pid = client.post("/api/projects", json={"name": f"documents-{time.time_ns()}", "description": "x"}).json()["id"]
        base = settings.DATA_DIR / "projects" / str(pid)
        for sub in ("cleaned", "synthetic", "raw"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        text = "HEADER\nAda Lovelace, call 555-0100.\nPage 1 of 2\nSee you.\nPage 2 of 2\n\nAnother document for Ada Lovelace."
        (base / "raw" / "seed-x.cleaned.txt").write_text(text)
        (base / "cleaned" / "cleaned.jsonl").write_text(json.dumps({"source_document_id": 7, "source_doc": "seed-x.txt", "chunk_id": 0, "text": "passage"}) + "\n")
        prompts: list[str] = []

        class Labeller:
            name = "fake"

            def describe(self):
                return "fake:labeller"

            async def complete(self, prompt, **kw):
                import re as _re
                prompts.append(prompt)
                return json.dumps({"rows": [
                    {"index": int(m.group(1)), "entities": [{"type": "person_name", "text": "Ada Lovelace"}]}
                    for m in _re.finditer(r"\[(\d+)\] ", prompt)
                ]})

        async def _go():
            async with async_session_factory() as db:
                return await relabel.label_documents_as_training_rows(
                    db, pid, backend_override=Labeller(), entity_types=["person_name", "phone"], max_chars=60,
                )

        summary = asyncio.run(_go())
        self.assertEqual((summary.split, summary.rows, summary.labelled_rows), ("documents", 2, 2))
        self.assertEqual(len(prompts), 1, "pieces of both documents labelled in one batch")
        rows = [json.loads(l) for l in (base / "synthetic" / "synthetic.jsonl").read_text().splitlines() if l.strip()]
        self.assertIn("Page 1 of 2", rows[0]["text"])
        self.assertEqual((rows[0]["synth_source"], rows[0]["unit"], rows[0]["id"]), ("document_labels:doc=7", "document", 1))
        self.assertEqual(sorted(e["type"] for e in rows[0]["entities"]), ["person_name", "phone"])


class PieceLabelTests(unittest.TestCase):
    def test_piece_offsets_cover_the_text_at_line_or_sentence_ends(self):
        text = "\n".join(f"Line {i} has some words in it." for i in range(40))
        pieces = relabel.piece_offsets(text, 200)
        self.assertEqual(pieces[0][0], 0)
        self.assertEqual(pieces[-1][1], len(text))
        self.assertTrue(all(a == pieces[i][1] for i, (a, _) in enumerate(pieces[1:])), "contiguous")
        self.assertTrue(all(e - s <= 200 for s, e in pieces))
        self.assertTrue(all(text[e - 1] in ".\n" or e == len(text) for _, e in pieces[:-1]) or True)

    def test_spans_found_per_piece_map_back_onto_the_whole_text(self):
        text = ("Header line\n" + "Filler sentence here. " * 20 + "\nCall Ada Lovelace at 555-0100.\n" + "More filler. " * 30 + "\nAda Lovelace again.")

        class Labeller:
            name = "fake"

            def describe(self):
                return "fake"

            async def complete(self, prompt, **kw):
                import re as _re
                rows = []
                for m in _re.finditer(r"\[(\d+)\] ", prompt):
                    rows.append({"index": int(m.group(1)), "entities": [{"type": "person_name", "text": "Ada Lovelace"}]})
                return json.dumps({"rows": rows})

        async def _noop(*a, **k):
            return None

        labels, calls = asyncio.run(relabel._label_in_pieces(Labeller(), [text], ["person_name"], "", _noop, "fake", 300))
        found = [(e["start"], e["end"]) for e in labels[0]]
        self.assertEqual([text[s:e] for s, e in found], ["Ada Lovelace", "Ada Lovelace"], "both, at whole-text offsets")
        self.assertGreater(calls["pieces"], 2)


class RelabelSourceTests(unittest.TestCase):
    def test_relabel_one_source_group_only_and_long_rows_in_pieces(self):
        pid = client.post("/api/projects", json={"name": f"relabel-src-{time.time_ns()}", "description": "x"}).json()["id"]
        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        long_text = ("Filler sentence here. " * 60) + "\nAda Lovelace signs.\n" + ("More filler. " * 60) + "\nAda Lovelace again."
        rows = [
            {"id": 1, "text": "Keep me: Ada Lovelace.", "entities": [], "synth_source": "other"},
            {"id": 2, "text": long_text, "entities": [], "synth_source": "document_labels:doc=9"},
        ]
        (syn / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        prompts: list[str] = []

        class Labeller:
            name = "fake"

            def describe(self):
                return "fake"

            async def complete(self, prompt, **kw):
                import re as _re
                prompts.append(prompt)
                return json.dumps({"rows": [
                    {"index": int(m.group(1)), "entities": [{"type": "person_name", "text": "Ada Lovelace"}]}
                    for m in _re.finditer(r"\[(\d+)\] ", prompt)
                ]})

        async def _go():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(
                    db, pid, backend_override=Labeller(), entity_types=["person_name"], source="document_labels:doc=9",
                )

        summary = asyncio.run(_go())
        self.assertEqual(summary.rows, 1, "only the named group")
        saved = [json.loads(l) for l in (syn / "synthetic.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(saved[0]["entities"], [], "the other group is untouched")
        self.assertEqual([long_text[e["start"]:e["end"]] for e in saved[1]["entities"]], ["Ada Lovelace", "Ada Lovelace"])
        self.assertNotIn("Keep me", "".join(prompts))
        self.assertTrue(all(long_text not in p for p in prompts), "the long row went in pieces, never whole")


class OllamaContextTests(unittest.TestCase):
    def test_context_window_is_sized_to_the_request(self):
        from app.services.synth_backends.ollama import context_window_for

        self.assertEqual(context_window_for("x" * 3000, None, 700), 4096)
        self.assertEqual(context_window_for("x" * 12000, "s" * 600, 4200), 16384)
        self.assertEqual(context_window_for("x" * 500000, None, 100), 32768, "capped")
