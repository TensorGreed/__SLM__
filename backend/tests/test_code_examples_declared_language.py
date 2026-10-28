"""`code_examples` playbook: the reply declares its language; validation checks the code under it.

A data/config seed (JSON, YAML, ...) often gets a task answered in a programming language. The
declared language is checked when it differs from the seed's and has a runnable check; otherwise the
seed's language applies (so a declaration never hides a real parse failure). Pytest-only.
"""

from __future__ import annotations

import unittest

from app.services.synth_playbooks import SynthMode, get_playbook
from app.services.synth_playbooks.code_examples import canonical_language, known_languages

JSON_SEED = '{"rules": [{"method": "GET", "path": "/items", "allow": true}]}'
PY_ANSWER = (
    "import json\n\nwith open('rules.json') as f:\n    cfg = json.load(f)\n\n"
    "for rule in cfg.get('rules', []):\n    print(rule['method'], rule['path'])\n"
)
YAML_PY_ANSWER = (
    "import yaml\n\nwith open('config.yaml') as f:\n    doc = yaml.safe_load(f)\n\n"
    "print(doc['kind'], doc['metadata']['name'])\n"
)
JSON_ANSWER = (
    "{\"version\": 2, \"defaults\": {\"deny\": true}, \"rules\": [{\"method\": \"POST\", \"path\": \"/orders\", "
    "\"allow\": false, \"audit\": [\"ops\", \"sec\"]}]}"
)


def ctx_for(language, text=JSON_SEED):
    chunks = [{"text": text, "path": "conf/rules", "language": language, "start_line": 1, "end_line": 1,
               "source_document_id": 1, "chunk_id": 0}]
    return {"recipe_id": "code-remediation", "project_id": 1, "gold_rows": [], "raw_rows": None,
            "failure_cluster": None, "target_class": None, "target_count": 2, "params": {}, "rejects": [],
            "document_chunks": chunks, "taxonomy": None}


def reply(**over):
    row = {"seed": 1, "task": "Check the rules for a GET", "citations": ["P1"], "rationale": "Reads the rules."}
    row.update(over)
    return row


class DeclaredLanguageTests(unittest.TestCase):
    def setUp(self):
        self.pb = get_playbook("code-remediation", SynthMode.CODE_EXAMPLES)

    def run_one(self, language, row, seed_text=JSON_SEED):
        ctx = ctx_for(language, seed_text)
        out = self.pb.validate([row], ctx)
        return out, ctx

    def test_python_answer_to_json_seed_is_accepted_as_python(self):
        out, ctx = self.run_one("json", reply(code=PY_ANSWER, language="python"))
        self.assertEqual(ctx["rejects"], [])
        p = out[0]["payload"]
        self.assertEqual((p["language"], p["seed_language"]), ("python", "json"))

    def test_alias_is_normalised(self):
        out, _ = self.run_one("json", reply(code=PY_ANSWER, language="Python3"))
        self.assertEqual(out[0]["payload"]["language"], "python")

    def test_yaml_seed_python_answer(self):
        out, ctx = self.run_one("yaml", reply(code=YAML_PY_ANSWER, language="py"), "kind: Thing\nmetadata:\n  name: x\n")
        self.assertEqual((len(out), ctx["rejects"]), (1, []))
        self.assertEqual(out[0]["payload"]["language"], "python")

    def test_json_answer_to_json_seed_keeps_seed_language(self):
        out, ctx = self.run_one("json", reply(code=JSON_ANSWER, language="json"))
        self.assertEqual(ctx["rejects"], [])
        p = out[0]["payload"]
        self.assertEqual(p["language"], "json")
        self.assertNotIn("seed_language", p)

    def test_invalid_code_in_declared_language_is_rejected_with_that_detail(self):
        out, ctx = self.run_one("json", reply(code="def f(:\n    pass", language="python"))
        self.assertEqual(out, [])
        r = ctx["rejects"][0]
        self.assertEqual(r["reason"], "code_does_not_parse")
        self.assertTrue(r["detail"].startswith("python"), r["detail"])
        self.assertEqual(r["payload"]["language"], "python")

    def test_missing_language_is_the_old_behaviour(self):
        out, ctx = self.run_one("json", reply(code=PY_ANSWER))
        self.assertEqual(out, [])
        self.assertEqual(ctx["rejects"][0]["reason"], "code_does_not_parse")
        self.assertTrue(ctx["rejects"][0]["detail"].startswith("json_parse"))
        out, ctx = self.run_one("json", reply(code=JSON_ANSWER))
        self.assertEqual(out[0]["payload"]["language"], "json")

    def test_unknown_language_falls_back_to_the_seed(self):
        for bad in ("klingon", "", 7, None):
            out, ctx = self.run_one("json", reply(code=PY_ANSWER, language=bad))
            self.assertEqual(out, [], bad)
            self.assertTrue(ctx["rejects"][0]["detail"].startswith("json_parse"), bad)

    def test_declared_language_without_a_check_cannot_dodge_the_seed_check(self):
        # "cobol" is not known; a known language with no runnable check behaves the same way.
        out, ctx = self.run_one("json", reply(code=PY_ANSWER, language="cobol"))
        self.assertEqual(out, [])
        self.assertEqual(ctx["rejects"][0]["reason"], "code_does_not_parse")

    def test_unchecked_seed_language_stays_unchecked(self):
        out, ctx = self.run_one("cobol", reply(code="def f(:\n    pass"))
        self.assertEqual((len(out), ctx["rejects"]), (1, []))
        self.assertEqual(out[0]["payload"]["language"], "cobol")

    def test_yaml_declaration_cannot_dodge_a_seed_check(self):
        bad = ["def f(:\n    return 1", "import yaml\nprint(yaml.safe_load(open('a.yml'))",
               "This is not code, just an apology."]
        for seed, seed_text in (("python", "def g(x):\n    return x + 1\n"), ("json", JSON_SEED)):
            for code in bad:
                out, ctx = self.run_one(seed, reply(code=code, language="yaml"), seed_text)
                self.assertEqual(out, [], (seed, code))
                self.assertEqual(ctx["rejects"][0]["reason"], "code_does_not_parse", (seed, code))
                self.assertTrue(ctx["rejects"][0]["detail"].startswith(("json_parse", "python")), ctx["rejects"][0])

    def test_data_seed_may_switch_to_a_yaml_mapping(self):
        code = "version: 2\ndefaults:\n  deny: true\nrules:\n  - method: POST\n    path: /orders\n"
        out, ctx = self.run_one("json", reply(code=code, language="yaml"))
        self.assertEqual(ctx["rejects"], [])
        self.assertEqual((out[0]["payload"]["language"], out[0]["payload"]["seed_language"]), ("yaml", "json"))

    def test_data_seed_yaml_scalar_falls_back_to_the_seed(self):
        out, ctx = self.run_one("json", reply(code="just some prose here", language="yaml"))
        self.assertEqual(out, [])
        self.assertTrue(ctx["rejects"][0]["detail"].startswith("json_parse"))

    def test_unchecked_seed_with_unchecked_declared_language_is_labelled(self):
        out, ctx = self.run_one("cobol", reply(code="x", language="xml"))
        p = out[0]["payload"]
        self.assertEqual((p["language"], p["seed_language"], ctx["rejects"]), ("xml", "cobol", []))

    def test_no_seed_language_adds_no_seed_language_key(self):
        out, _ = self.run_one(None, reply(code="plain text answer here", language="cobol"))
        self.assertNotIn("seed_language", out[0]["payload"])

    def test_prompt_asks_for_language(self):
        prompt = self.pb.build_prompt(ctx_for("json"))
        self.assertIn('"language": "<language of `code`>"', prompt)
        self.assertIn("python", prompt)
        self.assertIn("data or config format", prompt)

    def test_helpers(self):
        self.assertEqual(canonical_language("Py"), "python")
        self.assertEqual(canonical_language("C#"), "csharp")
        self.assertEqual(canonical_language(3), "")
        langs = known_languages()
        self.assertTrue({"python", "json", "yaml"} <= set(langs))


if __name__ == "__main__":
    unittest.main()
