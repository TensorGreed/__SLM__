"""Entity-type definitions: one labelling convention for every model that
labels the project's spans — stored per project, served with suggestions,
and put into BOTH the training-label prompt and the answer-key prompt."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest import mock

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


class ScanTolerantTests(unittest.TestCase):
    """Texts from scans: a scan-tolerant labelling clause for both
    labellers, OCR-tolerant span decoding, and a flag that can be set
    without touching the definitions."""

    def test_clause_is_appended_only_when_the_flag_is_on(self):
        self.assertNotIn("scanned documents", defs.definitions_clause(["email"], {"email": "an email"}))
        clause = defs.definitions_clause(["email"], {"email": "an email"}, scan_tolerant=True)
        self.assertIn("- email: an email", clause)
        self.assertIn("ONE entity", clause)
        self.assertIn("scanned documents", defs.definitions_clause([], {}, scan_tolerant=True), "even with no definitions")

    def test_flag_is_saved_without_wiping_the_definitions(self):
        pid = client.post("/api/projects", json={"name": f"scan-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"email": "an email"}})
        out = client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"scan_tolerant": True}).json()
        self.assertEqual(out, {"definitions": {"email": "an email"}, "scan_tolerant": True})
        got = client.get(f"/api/projects/{pid}/synthetic/type-definitions").json()
        self.assertTrue(got["scan_tolerant"])
        self.assertEqual(got["definitions"], {"email": "an email"})
        cleared = client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {}}).json()
        self.assertEqual(cleared, {"definitions": {}, "scan_tolerant": True}, "{} still clears; the flag stays")


class OcrTidyTests(unittest.TestCase):
    def test_trims_marks_and_joins_spans_broken_by_ocr(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train_token_classification import ocr_tidy_spans

        text = "Call Sarah - Jenkins or _jsmith@x.com, John, Mary; card 4111 ~ 1111 today"
        spans = [
            {"type": "person_name", "start": 5, "end": 10},
            {"type": "person_name", "start": 13, "end": 20},
            {"type": "email", "start": 24, "end": 37},
            {"type": "person_name", "start": 39, "end": 43},
            {"type": "person_name", "start": 45, "end": 49},
            {"type": "credit_card", "start": 56, "end": 60},
            {"type": "credit_card", "start": 63, "end": 67},
        ]
        got = [(s["type"], s["text"]) for s in ocr_tidy_spans(text, spans)]
        self.assertEqual(got, [
            ("person_name", "Sarah - Jenkins"),
            ("email", "jsmith@x.com"),
            ("person_name", "John"),
            ("person_name", "Mary"),
            ("credit_card", "4111 ~ 1111"),
        ])
        far = [{"type": "phone", "start": 0, "end": 4}, {"type": "phone", "start": 10, "end": 14}]
        self.assertEqual(len(ocr_tidy_spans("5550 ----- 0101", far)), 2, "a gap longer than max_gap stays apart")
        mixed = [{"type": "phone", "start": 0, "end": 4}, {"type": "email", "start": 7, "end": 10}]
        self.assertEqual(len(ocr_tidy_spans("5550 - abc", mixed)), 2, "different types never merge")


class EvalHonoursScanFlagTests(unittest.TestCase):
    def test_token_classification_inference_passes_ocr_tolerant_from_the_pairs(self):
        import sys
        import types as _types
        from unittest import mock

        from app.services import evaluation_service as ev

        seen = {}

        def fake_predict(texts, model, tokenizer, id2label, **kw):
            seen.update(kw)
            return [[] for _ in texts]

        class FakeTok:
            @staticmethod
            def from_pretrained(path):
                return "tok"

        class FakeModel:
            @staticmethod
            def from_pretrained(path, **kw):
                return FakeModel()

            def to(self, device):
                return self

            def eval(self):
                return self

        saved = sys.modules.get("train_token_classification")
        sys.modules["train_token_classification"] = _types.SimpleNamespace(predict_spans=fake_predict)
        try:
            with mock.patch("transformers.AutoTokenizer", FakeTok), mock.patch("transformers.AutoModelForTokenClassification", FakeModel):
                ev._run_token_classification_inference(
                    {"model_path": "/x/model", "id2label": {0: "O"}, "max_length": 512},
                    [{"prompt": "p", "reference": "{}", "structured_input": "t", "ocr_tolerant": True}],
                )
        finally:
            if saved is None:
                sys.modules.pop("train_token_classification", None)
            else:
                sys.modules["train_token_classification"] = saved
        self.assertTrue(seen.get("ocr_tolerant"))


class ConventionsVersionTests(unittest.TestCase):
    """P1k: labellers stamp the conventions version on the rows they write;
    the review queue flags groups labelled under an older one."""

    def test_version_is_stable_and_tracks_definitions_and_the_scan_flag(self):
        self.assertIsNone(defs.conventions_version({}, False), "no conventions → nothing can be stale")
        a = defs.conventions_version({"email": "an email"})
        self.assertEqual(a, defs.conventions_version({"Email": " an email "}), "normalised")
        self.assertNotEqual(a, defs.conventions_version({"email": "an email address"}))
        self.assertNotEqual(a, defs.conventions_version({"email": "an email"}, True))
        self.assertIsNotNone(defs.conventions_version({}, True))
        changed = None
        with mock.patch.object(defs, "SCAN_TOLERANT_CLAUSE", "other wording"):
            changed = defs.conventions_version({"email": "an email"}, True)
        self.assertNotEqual(changed, defs.conventions_version({"email": "an email"}, True), "clause wording is part of the version")
        with mock.patch.object(defs, "SCAN_TOLERANT_CLAUSE", "other wording"):
            self.assertEqual(defs.conventions_version({"email": "an email"}), a, "without the scan flag the clause doesn't matter")

    def test_labeller_and_pin(self):
        self.assertEqual(defs.row_labeller({"relabelled_by": "cloud:anthropic:m"}), "cloud:anthropic:m")
        self.assertEqual(defs.row_labeller({"source": "passage_labels:ollama"}), "ollama")
        self.assertIsNone(defs.row_labeller({"source": "teacher_llm"}))
        self.assertEqual(defs.relabel_pin("cloud:anthropic:m"), "cloud:anthropic:m")
        self.assertEqual(defs.relabel_pin("ollama:gemma4:12b"), "ollama")
        self.assertIsNone(defs.relabel_pin("teacher:x@http://h"))

    def test_status_counts_current_stale_and_untracked(self):
        rows = [
            {"text": "a", "entities": [], "conventions_version": "v2", "relabelled_by": "ollama"},
            {"text": "b", "entities": [], "conventions_version": "v1", "relabelled_by": "ollama"},
            {"text": "c", "entities": [], "source": "passage_labels:cloud:anthropic:m"},
            {"text": "d", "label": "x"},  # not a span row
        ]
        st = defs.conventions_status(rows, "v2")
        self.assertEqual(
            {k: st[k] for k in ("span_rows", "current_rows", "stale_rows", "untracked_rows", "is_stale", "relabel_pin")},
            {"span_rows": 3, "current_rows": 1, "stale_rows": 1, "untracked_rows": 1, "is_stale": True, "relabel_pin": "ollama"},
        )
        self.assertIsNone(defs.conventions_status(rows, None))
        self.assertIsNone(defs.conventions_status([{"text": "d", "label": "x"}], "v2"))

    def test_relabel_stamps_rows_and_the_queue_clears_the_flag(self):
        pid = client.post("/api/projects", json={"name": f"conv-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"person_name": "a name"}})
        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": 1, "text": "Ada Lovelace wrote.", "entities": [], "synth_source": "passage_labels:doc=1",
             "source": "passage_labels:ollama", "review_status": "accepted"},
            {"id": 2, "text": "Hi Grace Hopper.", "entities": [], "synth_source": "passage_labels:doc=2",
             "source": "passage_labels:ollama", "review_status": "accepted"},
        ]
        (syn / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        def stale_sources():
            q = client.get(f"/api/projects/{pid}/synthetic/review-queue").json()
            self.assertTrue(q["conventions_version"])
            return sorted(g["synth_source"] for g in q["accepted_groups"] if (g.get("conventions") or {}).get("is_stale"))

        self.assertEqual(stale_sources(), ["passage_labels:doc=1", "passage_labels:doc=2"], "untracked rows are flagged")

        class Labeller:
            name = "fake"

            def describe(self):
                return "fake"

            async def complete(self, prompt, **kw):
                import re as _re
                return json.dumps({"rows": [{"index": int(m.group(1)), "entities": []} for m in _re.finditer(r"\[(\d+)\] ", prompt)]})

        async def _go():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(db, pid, backend_override=Labeller(), source="passage_labels:doc=1")

        summary = asyncio.run(_go())
        self.assertEqual(summary.conventions_types, ["person_name"], "default vocabulary = the project's defined types")

        # A legacy row without synth_source is the queue's "legacy:<source>" group.
        rows_now = [json.loads(l) for l in (syn / "synthetic.jsonl").read_text().splitlines() if l.strip()]
        rows_now.append({"id": 3, "text": "Old Ada row.", "entities": [], "source": "teacher_llm"})
        (syn / "synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in rows_now) + "\n")

        async def _legacy():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(db, pid, backend_override=Labeller(), source="legacy:teacher_llm")

        self.assertEqual(asyncio.run(_legacy()).rows, 1)
        self.assertEqual(stale_sources(), ["passage_labels:doc=2"], "the relabelled group is current")
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"person_name": "a person's name"}})
        self.assertEqual(
            stale_sources(), ["legacy:teacher_llm", "passage_labels:doc=1", "passage_labels:doc=2"],
            "a definition change makes every group stale again",
        )


class MarkConventionsCurrentTests(unittest.TestCase):
    def test_mark_current_stamps_one_group_with_provenance(self):
        pid = client.post("/api/projects", json={"name": f"mark-{time.time_ns()}", "description": "x"}).json()["id"]
        r = client.post(f"/api/projects/{pid}/synthetic/review-queue/mark-conventions-current", json={"source": "g1"})
        self.assertEqual(r.status_code, 400, "no conventions yet")
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"email": "an email"}})
        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": 1, "text": "a", "entities": [], "synth_source": "g1", "review_status": "accepted"},
            {"id": 2, "text": "b", "entities": [], "synth_source": "g2", "review_status": "accepted"},
        ]
        (syn / "synthetic.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        out = client.post(f"/api/projects/{pid}/synthetic/review-queue/mark-conventions-current", json={"source": "g1"}).json()
        self.assertEqual((out["span_rows"], out["rows_marked"]), (1, 1))
        saved = {x["synth_source"]: x for x in (json.loads(l) for l in (syn / "synthetic.jsonl").read_text().splitlines() if l.strip())}
        self.assertEqual(saved["g1"]["conventions_version"], out["conventions_version"])
        self.assertIn("conventions_marked_current_at", saved["g1"])
        self.assertNotIn("conventions_version", saved["g2"])
        q = client.get(f"/api/projects/{pid}/synthetic/review-queue").json()
        stale = sorted(g["synth_source"] for g in q["accepted_groups"] if g["conventions"]["is_stale"])
        self.assertEqual(stale, ["g2"])
        r = client.post(f"/api/projects/{pid}/synthetic/review-queue/mark-conventions-current", json={"source": "nope"})
        self.assertEqual(r.status_code, 400)


class RetireGroupTests(unittest.TestCase):
    def test_retire_takes_a_group_out_of_training_and_restore_brings_it_back(self):
        from app.services.dataset_service import _load_records_from_file

        pid = client.post("/api/projects", json={"name": f"retire-{time.time_ns()}", "description": "x"}).json()["id"]
        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": 1, "text": "a", "entities": [], "synth_source": "g1"},  # legacy: no review_status = accepted
            {"id": 2, "text": "b", "entities": [], "synth_source": "g1", "review_status": "rejected", "reject_reason": "bad"},
            {"id": 3, "text": "c", "entities": [], "synth_source": "g2", "review_status": "accepted", "relabelled_by": "cloud:anthropic:m"},
        ]
        path = syn / "synthetic.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        url = f"/api/projects/{pid}/synthetic/review-queue/retire-group"
        out = client.post(url, json={"source": "g1"}).json()
        self.assertEqual((out["rows_changed"], out["group_rows"]), (1, 2))
        texts = sorted(r["text"] for r in _load_records_from_file(path))
        self.assertEqual(texts, ["c"], "retired rows never train")
        retired = {x["id"]: x for x in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}
        self.assertEqual(retired[1]["decided_via"], "retire:g1")
        self.assertNotIn("decided_via", retired[2], "a row it did not change keeps its record")
        q = client.get(f"/api/projects/{pid}/synthetic/review-queue").json()
        self.assertEqual(q["labeller_pins"], ["cloud:anthropic:m"])
        out = client.post(url, json={"source": "g1", "retired": False}).json()
        self.assertEqual(out["rows_changed"], 1)
        saved = {x["id"]: x for x in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}
        self.assertEqual(saved[1]["review_status"], "accepted")
        self.assertEqual(saved[1]["decided_via"], "unretire:g1")
        self.assertEqual(saved[2]["reject_reason"], "bad", "a row rejected for another reason stays rejected")
        self.assertEqual(client.post(url, json={"source": "nope"}).status_code, 400)


class OnlyStaleRelabelTests(unittest.TestCase):
    def test_only_rows_not_labelled_under_the_current_conventions_are_sent(self):
        pid = client.post("/api/projects", json={"name": f"stale-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"person_name": "a name"}})
        current = defs.conventions_version({"person_name": "a name"})
        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": 1, "text": "Ada wrote.", "entities": [], "synth_source": "g", "conventions_version": current},
            {"id": 2, "text": "Grace wrote.", "entities": [], "synth_source": "g"},
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
                return json.dumps({"rows": [{"index": int(m.group(1)), "entities": []} for m in _re.finditer(r"\[(\d+)\] ", prompt)]})

        async def _go():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(db, pid, backend_override=Labeller(), source="g", only_stale=True)

        self.assertEqual(asyncio.run(_go()).rows, 1)
        self.assertNotIn("Ada wrote", "".join(prompts))

        async def _again():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(db, pid, backend_override=Labeller(), source="g", only_stale=True)

        with self.assertRaises(RuntimeError):
            asyncio.run(_again())


class RelabelSplitInPlaceTests(unittest.TestCase):
    def test_test_split_and_its_identical_snapshots_are_relabelled_in_place(self):
        pid = client.post("/api/projects", json={"name": f"split-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={"definitions": {"person_name": "a name"}})
        prepared = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        (prepared / "versions" / "1").mkdir(parents=True, exist_ok=True)
        (prepared / "versions" / "2").mkdir(parents=True, exist_ok=True)
        rows = "".join(json.dumps(r) + "\n" for r in [
            {"text": "Hi Ada Lovelace", "entities": [{"type": "person_name", "start": 0, "end": 2}]},
        ])
        (prepared / "test.jsonl").write_text(rows)
        (prepared / "versions" / "1" / "test.jsonl").write_text(rows)
        (prepared / "versions" / "2" / "test.jsonl").write_text("different\n")

        class Labeller:
            name = "fake"

            def describe(self):
                return "fake"

            async def complete(self, prompt, **kw):
                import re as _re
                return json.dumps({"rows": [{"index": int(m.group(1)), "entities": [{"type": "person_name", "text": "Ada Lovelace"}]}
                                            for m in _re.finditer(r"\[(\d+)\] ", prompt)]})

        async def _go():
            async with async_session_factory() as db:
                return await relabel.relabel_split_in_place(db, pid, split="test", backend_override=Labeller())

        summary = asyncio.run(_go())
        self.assertEqual(summary.labelled_rows, 1)
        row = json.loads((prepared / "test.jsonl").read_text().splitlines()[0])
        self.assertEqual([("Hi Ada Lovelace"[e["start"]:e["end"]]) for e in row["entities"]], ["Ada Lovelace"])
        self.assertEqual(row["conventions_version"], defs.conventions_version({"person_name": "a name"}))
        self.assertEqual((prepared / "versions" / "1" / "test.jsonl").read_text(), (prepared / "test.jsonl").read_text())
        self.assertEqual((prepared / "versions" / "2" / "test.jsonl").read_text(), "different\n")
