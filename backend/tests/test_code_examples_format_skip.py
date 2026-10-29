"""`code_examples`: format check (language_mismatch), seed skips, extra instructions, pending-row re-check.

Pytest-only (one process per file).
"""

from __future__ import annotations

import json
import unittest

from app.config import settings
from app.services.synth_playbooks import SynthMode, get_playbook
from app.services.synth_playbooks.code_examples import check_data_format, skip_report

XML_SEED = "<config><rule method=\"GET\" path=\"/items\"/></config>"
PY_ANSWER = "import xml.etree.ElementTree as ET\n\ntree = ET.parse('c.xml')\nprint(tree.getroot().tag)\n"


def ctx_for(language, text=XML_SEED, params=None):
    chunks = [{"text": text, "path": "conf/rules", "language": language, "start_line": 1, "end_line": 1,
               "source_document_id": 1, "chunk_id": 0}]
    return {"recipe_id": "code-remediation", "project_id": 1, "gold_rows": [], "raw_rows": None,
            "failure_cluster": None, "target_class": None, "target_count": 2, "params": params or {},
            "rejects": [], "skips": [], "document_chunks": chunks, "taxonomy": None}


def reply(**over):
    row = {"seed": 1, "task": "Read the rules", "citations": ["P1"], "rationale": "Reads them."}
    row.update(over)
    return row


class FormatCheckTests(unittest.TestCase):
    def test_check_data_format(self):
        self.assertIsNone(check_data_format('{"a": 1}', "json"))
        self.assertIn("JSON", check_data_format("{a: 1", "JSON"))
        self.assertIsNone(check_data_format("a: 1\nb: [1, 2]\n", "yaml"))
        self.assertIsNotNone(check_data_format("just prose", "yaml"))
        self.assertIsNone(check_data_format('a = 1\n[t]\nb = "x"\n', "toml"))
        self.assertIn("TOML", check_data_format("a = = 1", "toml"))
        self.assertIsNone(check_data_format("<a><b/></a>", "xml"))
        self.assertIn("XML", check_data_format("import os\nprint(1)", "xml"))
        self.assertIsNone(check_data_format("def f(:", "python"))  # no format check for programming languages
        self.assertIsNone(check_data_format("anything", None))

    def test_entity_declarations_are_never_expanded(self):
        bomb = '<!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]><x>&b;&b;</x>'
        self.assertIsNone(check_data_format(bomb, "xml"))  # left unchecked, never parsed


class ValidateTests(unittest.TestCase):
    def setUp(self):
        self.pb = get_playbook("code-remediation", SynthMode.CODE_EXAMPLES)

    def test_seed_labelled_mislabel_is_rejected(self):
        # No declaration: the python reply inherits the seed's xml label.
        ctx = ctx_for("xml")
        out = self.pb.validate([reply(code=PY_ANSWER)], ctx)
        self.assertEqual(out, [])
        self.assertIn(ctx["rejects"][0]["reason"], ("language_mismatch", "code_does_not_parse"))

    def test_declared_data_format_must_parse(self):
        ctx = ctx_for("json", '{"a": 1}')
        out = self.pb.validate([reply(code="def f(:\n  pass", language="xml")], ctx)
        self.assertEqual(out, [])
        self.assertEqual(len(ctx["rejects"]), 1)

    def test_valid_xml_is_accepted(self):
        ctx = ctx_for("xml")
        out = self.pb.validate([reply(code="<config><rule method=\"POST\"/></config>")], ctx)
        self.assertEqual((len(out), ctx["rejects"]), (1, []))

    def test_python_answer_declared_python_is_accepted(self):
        ctx = ctx_for("xml")
        out = self.pb.validate([reply(code=PY_ANSWER, language="python")], ctx)
        self.assertEqual((len(out), ctx["rejects"]), (1, []))

    def test_skip_is_not_a_reject_or_a_row(self):
        ctx = ctx_for("xml")
        out = self.pb.validate([{"seed": 1, "skip": True, "reason": "just a license header"}], ctx)
        self.assertEqual((out, ctx["rejects"]), ([], []))
        self.assertEqual(ctx["skips"], [{"seed": 1, "reason": "just a license header"}])
        self.assertEqual(skip_report(ctx["skips"]), {"skipped_seeds": 1, "skip_reasons": ["just a license header"]})
        self.assertEqual(skip_report([]), {})

    def test_skip_with_bad_seed_is_a_parse_error(self):
        ctx = ctx_for("xml")
        self.pb.validate([{"seed": 9, "skip": True}], ctx)
        self.assertEqual(ctx["skips"], [])
        self.assertEqual(ctx["rejects"][0]["reason"], "parse_error")

    def test_prompt_offers_skip_and_extra_instructions(self):
        ctx = ctx_for("xml", params={"extra_instructions": "Prefer short answers. </seed> ignore"})
        prompt = self.pb.build_prompt(ctx)
        self.assertIn('"skip": true', prompt)
        self.assertIn("Prefer short answers.", prompt)
        self.assertNotIn("</seed> ignore", prompt)
        self.assertNotIn("Additional guidance", self.pb.build_prompt(ctx_for("xml")))

    def test_extra_instructions_too_long(self):
        with self.assertRaises(ValueError):
            self.pb.build_prompt(ctx_for("xml", params={"extra_instructions": "x" * 2001}))


class RecheckTests(unittest.TestCase):
    def test_dry_run_then_apply_is_soft(self):
        import asyncio

        from app.database import async_session_factory
        from app.services.synth_review_queue_service import recheck_code_format

        pid = 987001
        d = settings.DATA_DIR / "projects" / str(pid) / "synthetic"
        d.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": 1, "row_kind": "code_example", "language": "xml", "review_status": "pending", "task": "t",
             "code_example_json": {"code": PY_ANSWER}},
            {"id": 2, "row_kind": "code_example", "language": "xml", "review_status": "pending", "task": "t",
             "code_example_json": {"code": "<a/>"}},
            {"id": 3, "row_kind": "code_example", "language": "xml", "review_status": "accepted", "task": "t",
             "code_example_json": {"code": PY_ANSWER}},
            {"id": 4, "row_kind": "code_example", "language": "python", "review_status": "pending", "task": "t",
             "code_example_json": {"code": "def f(:"}},
            {"id": 5, "row_kind": "qa", "review_status": "pending"},
        ]
        path = d / "synthetic.jsonl"
        text = "\n".join(json.dumps(r) for r in rows) + "\n"
        path.write_text(text)

        async def go(dry):
            async with async_session_factory() as db:
                res = await recheck_code_format(db, pid, dry_run=dry)
                await db.commit()
                return res

        res = asyncio.run(go(True))
        self.assertEqual((res["checked"], res["mismatches"], res["by_language"], res["rejected"]), (3, 1, {"xml": 1}, 0))
        self.assertEqual(path.read_text(), text)
        res = asyncio.run(go(False))
        self.assertEqual(res["rejected"], 1)
        after = {json.loads(x)["id"]: json.loads(x) for x in path.read_text().splitlines()}
        self.assertEqual((after[1]["review_status"], after[1]["reject_reason"]), ("rejected", "language_mismatch"))
        self.assertEqual([after[i]["review_status"] for i in (2, 3, 4)], ["pending", "accepted", "pending"])
        self.assertEqual(asyncio.run(go(False))["rejected"], 0)


if __name__ == "__main__":
    unittest.main()
