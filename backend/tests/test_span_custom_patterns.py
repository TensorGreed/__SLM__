"""Custom infotypes: a span project's own identifier formats as regexes —
validated on write, used by every regex pass beside the built-ins, and
testable against the project's own text before saving."""

from __future__ import annotations

import asyncio
import json
import time
import unittest

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services import span_custom_patterns_service as cps
from app.services import span_relabel_service as relabel
from app.services.synthetic_service import (
    _extract_entities_via_regex,
    _merge_regex_entities,
    augment_saved_rows_with_regex,
    generate_span_extraction_rows,
)

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


EMP = {"type": "Employee ID", "pattern": r"\bEMP-\d{5}\b", "description": "staff id"}
TCK = {"type": "ticket_id", "pattern": r"\btck-\d{4}\b", "flags": "i"}


class ValidationTests(unittest.TestCase):
    def test_good_patterns_pass(self):
        self.assertIsNone(cps.validate_pattern("employee_id", r"\bEMP-\d{5}\b"))
        self.assertIsNone(cps.validate_pattern("ref", r"(?:-[A-Z]{2})+", check_speed=False))

    def test_rejections_name_the_reason(self):
        cases = {
            ("", r"x"): "type is required",
            ("t", ""): "pattern is required",
            ("t", "x" * (cps.MAX_PATTERN_CHARS + 1)): "longer than",
            ("t", r"(EMP"): "does not compile",
            ("t", r"\d*"): "empty string",
            ("t", r"(?=EMP)"): "empty string",
            ("t", r"x?"): "empty string",
            ("t", r"(a+)+$"): "nests one unbounded repeat",
            ("t", r"(\w+\s?)*!"): "nests one unbounded repeat",
        }
        for (name, pattern), reason in cases.items():
            with self.subTest(pattern=pattern):
                self.assertIn(reason, cps.validate_pattern(name, pattern, check_speed=False) or "")

    def test_exponential_alternation_is_caught_by_the_timed_child_process(self):
        start = time.perf_counter()
        error = cps.validate_pattern("t", r"(a|a)+b")
        self.assertIn("too slow", error or "")
        self.assertLess(time.perf_counter() - start, cps.SPEED_CHECK_TIMEOUT + 3, "the child is killed, the server isn't hung")

    def test_normalise_types_flags_and_drop_bad_entries(self):
        valid, errors = cps.normalise_patterns([EMP, TCK, EMP, {"type": "bad", "pattern": "("}, "junk"])
        self.assertEqual([p["type"] for p in valid], ["employee_id", "ticket_id"], "type normalised, duplicate dropped")
        self.assertEqual(valid[1]["flags"], "i")
        self.assertEqual(valid[0]["description"], "staff id")
        self.assertEqual([e["index"] for e in errors], [3, 4])


class MergeTests(unittest.TestCase):
    TEXT = "Ticket TCK-0042 for EMP-48213, reply to ana@example.com."

    def test_custom_patterns_add_beside_the_builtins(self):
        patterns, _ = cps.normalise_patterns([EMP, TCK])
        ents = _extract_entities_via_regex(self.TEXT, None, patterns)
        self.assertEqual([e["type"] for e in ents], ["ticket_id", "employee_id", "email"])
        emp = ents[1]
        self.assertEqual(self.TEXT[emp["start"]:emp["end"]], "EMP-48213")
        self.assertEqual([e["type"] for e in _extract_entities_via_regex(self.TEXT)], ["email"], "no patterns → built-ins only")

    def test_vocabulary_filter_applies_to_custom_types(self):
        patterns, _ = cps.normalise_patterns([EMP, TCK])
        ents = _extract_entities_via_regex(self.TEXT, ["employee_id", "email"], patterns)
        self.assertEqual(sorted(e["type"] for e in ents), ["email", "employee_id"])

    def test_merge_keeps_teacher_entities_and_skips_overlaps(self):
        patterns, _ = cps.normalise_patterns([EMP, TCK])
        teacher = [{"type": "case_ref", "start": 7, "end": 15, "text": "TCK-0042"}]
        merged, augmented = _merge_regex_entities(self.TEXT, teacher, None, patterns)
        self.assertTrue(augmented)
        self.assertEqual([e["type"] for e in merged], ["case_ref", "employee_id", "email"])

    def test_a_custom_pattern_for_a_builtin_type_adds_matches(self):
        patterns, _ = cps.normalise_patterns([{"type": "phone", "pattern": r"\b0\d{4} \d{6}\b"}])
        ents = _extract_entities_via_regex("Call 01632 960001 or 555-0100.", ["phone"], patterns)
        self.assertEqual([e["text"] for e in ents], ["01632 960001", "555-0100"])

    def test_preview_reports_count_and_snippets(self):
        texts = ["EMP-00001 and EMP-00002", "nothing", "x EMP-12345 y"]
        out = cps.preview_pattern_matches(r"EMP-\d{5}", "Employee ID", texts, sources=["a", "b", "c"], max_snippets=2)
        self.assertTrue(out["valid"])
        self.assertEqual((out["count"], out["texts_matched"], out["texts_scanned"]), (3, 2, 3))
        self.assertEqual(len(out["matches"]), 2)
        self.assertEqual(out["matches"][1]["before"], "EMP-00001 and ")
        self.assertEqual(out["matches"][0]["source"], "a")
        bad = cps.preview_pattern_matches(r"\d*", "t", texts)
        self.assertFalse(bad["valid"])
        self.assertIn("empty string", bad["error"])


def _new_project(prefix: str) -> int:
    return client.post("/api/projects", json={"name": f"{prefix}-{time.time_ns()}", "description": "x"}).json()["id"]


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.pid = _new_project("patterns")
        base = settings.DATA_DIR / "projects" / str(self.pid)
        (base / "synthetic").mkdir(parents=True, exist_ok=True)
        (base / "cleaned").mkdir(parents=True, exist_ok=True)
        (base / "synthetic" / "synthetic.jsonl").write_text(
            json.dumps({"text": "Badge EMP-48213 lost.", "entities": []}) + "\n"
        )
        (base / "cleaned" / "cleaned.jsonl").write_text(
            json.dumps({"source_document_id": 1, "text": "Staff EMP-00007 and EMP-00008."}) + "\n"
            + json.dumps({"row_index": 0, "text": "structured EMP-99999"}) + "\n"
        )

    def test_get_put_round_trip(self):
        url = f"/api/projects/{self.pid}/synthetic/type-patterns"
        got = client.get(url).json()
        self.assertEqual(got["patterns"], [])
        self.assertIn("email", got["builtin_types"])
        self.assertEqual(got["limits"]["max_pattern_chars"], cps.MAX_PATTERN_CHARS)
        put = client.put(url, json={"patterns": [EMP, TCK]})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual([p["type"] for p in put.json()["patterns"]], ["employee_id", "ticket_id"])
        self.assertEqual([p["type"] for p in client.get(url).json()["patterns"]], ["employee_id", "ticket_id"])

    def test_put_rejects_and_saves_nothing(self):
        url = f"/api/projects/{self.pid}/synthetic/type-patterns"
        client.put(url, json={"patterns": [EMP]})
        bad = client.put(url, json={"patterns": [TCK, {"type": "x", "pattern": "(a+)+"}]})
        self.assertEqual(bad.status_code, 422)
        body = bad.json()
        self.assertEqual(body["error_code"], "INVALID_TYPE_PATTERN")
        self.assertEqual(body["metadata"]["errors"][0]["index"], 1)
        self.assertEqual([p["type"] for p in client.get(url).json()["patterns"]], ["employee_id"], "old list kept")

    def test_test_endpoint_scans_saved_rows_and_passages(self):
        out = client.post(
            f"/api/projects/{self.pid}/synthetic/type-patterns/test",
            json={"pattern": r"EMP-\d{5}", "type": "employee_id"},
        ).json()
        self.assertTrue(out["valid"])
        self.assertEqual(out["count"], 3, "structured cleaned rows are not passages")
        self.assertEqual([m["source"] for m in out["matches"]], ["saved_row", "passage", "passage"])
        self.assertEqual(out["matches"][0]["match"], "EMP-48213")
        bad = client.post(f"/api/projects/{self.pid}/synthetic/type-patterns/test", json={"pattern": "(", "type": "t"}).json()
        self.assertFalse(bad["valid"])

    def test_regex_augment_uses_the_project_patterns(self):
        client.put(f"/api/projects/{self.pid}/synthetic/type-patterns", json={"patterns": [EMP]})
        out = client.post(f"/api/projects/{self.pid}/synthetic/spans/regex-augment", json={}).json()
        self.assertEqual(out["added_by_type"], {"employee_id": 1})
        again = augment_saved_rows_with_regex(self.pid, None, cps.normalise_patterns([EMP])[0])
        self.assertEqual(again["entities_added"], 0, "idempotent")

    def test_missing_project_is_404(self):
        self.assertEqual(client.get("/api/projects/999999/synthetic/type-patterns").status_code, 404)


class LabellerThreadingTests(unittest.TestCase):
    def test_generation_and_training_relabel_apply_custom_patterns(self):
        pid = _new_project("patterns-labellers")
        client.put(f"/api/projects/{pid}/synthetic/type-patterns", json={"patterns": [EMP]})
        prompts: list[str] = []
        snippet = "Employee EMP-48213 asked about the parking permit renewal for the north building lot this week, thanks."

        class Teacher:
            name = "fake"

            def describe(self):
                return "fake:teacher"

            async def complete(self, prompt, **kw):
                prompts.append(prompt)
                if "Label every entity" in prompt:
                    return json.dumps({"rows": [{"index": 0, "entities": []}]})
                return json.dumps({"rows": [{"text": snippet, "entities": [{"type": "person_name", "text": "parking"}]}]})

        async def _gen():
            async with async_session_factory() as db:
                return await generate_span_extraction_rows(
                    db, pid, "src", num_rows=1, entity_types=["employee_id", "person_name"], backend_override=Teacher(),
                )

        rows = asyncio.run(_gen())
        self.assertEqual(sorted(e["type"] for e in rows[0]["entities"]), ["employee_id", "person_name"])
        self.assertIn("employee_id", prompts[0], "the prompt names the custom regex type")

        syn = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        syn.mkdir(parents=True, exist_ok=True)
        (syn / "synthetic.jsonl").write_text(json.dumps({"text": snippet, "entities": []}) + "\n")

        async def _relabel():
            async with async_session_factory() as db:
                return await relabel.relabel_training_rows(db, pid, backend_override=Teacher(), entity_types=["employee_id"])

        asyncio.run(_relabel())
        saved = [json.loads(l) for l in (syn / "synthetic.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual([e["text"] for e in saved[0]["entities"]], ["EMP-48213"])


if __name__ == "__main__":
    unittest.main()
