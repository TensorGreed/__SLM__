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

    # A code file that documents config usage (a module whose docs show the YAML task that calls it) gets
    # a YAML answer; it used to be checked as the seed's language and rejected ("illegal target for annotation").
    # The switch to YAML needs config shape AND a YAML block embedded in the seed (a docs / examples string)
    # that shares two keys with the answer, or one non-generic key.
    MODULE_SEED = (
        'DOCUMENTATION = r"""\nmodule: widget_save\noptions:\n  api_node:\n    type: dict\n  op_type:\n    type: str\n"""\n\n'
        'EXAMPLES = r"""\n- name: Save a widget\n  example.widgets.widget_save:\n    api_node: "{{ node }}"\n'
        '    op_type: create\n"""\n\n'
        'def main():\n    module = setup_module()\n    module.exit_json(changed=False)\n'
    )
    PLAIN_SEED = (
        'def fetch_user(user_id, timeout=30, retries=3):\n    """Fetch a user record.\n\n    Args:\n'
        '        user_id: the id\n        timeout: seconds\n        retries: attempts\n    """\n'
        '    resp = session.get(f"{BASE}/users/{user_id}", timeout=timeout)\n    logger.info("fetched %s", user_id)\n'
        '    return resp.json()\n'
    )
    TASK_LIST_YAML = (
        '- name: Create a widget\n  example.widgets.widget_save:\n    api_node: "{{ api_node }}"\n'
        '    op_type: create\n    size: 3\n  register: created\n'
    )
    NESTED_MAP_YAML = "api_node:\n  server: api.example.test\n  op_type: update\n"

    def run_task(self, seed_lang, seed_text, code, task="Check the rules for a GET"):
        return self.run_one(seed_lang, reply(code=code, language="yaml", task=task), seed_text)

    def assert_not_yaml(self, seed_lang, seed_text, code, task="Check the rules for a GET"):
        out, ctx = self.run_task(seed_lang, seed_text, code, task)
        self.assertFalse(out and out[0]["payload"]["language"] == "yaml", (code, out))
        return out, ctx

    def test_programming_seed_with_embedded_yaml_examples_switches(self):
        for code in (self.TASK_LIST_YAML, self.NESTED_MAP_YAML,
                     "- name: Loop\n  example.widgets.widget_save:\n    op_type: {{ item }}\n  loop: [a, b]\n"):
            out, ctx = self.run_task("python", self.MODULE_SEED, code)
            self.assertEqual(ctx["rejects"], [], code)
            p = out[0]["payload"]
            self.assertEqual((p["language"], p["seed_language"]), ("yaml", "python"), code)

    def test_programming_seed_yaml_switch_still_runs_the_yaml_check(self):
        broken = '- name: Create a widget\n  example.widgets.widget_save:\n    op_type: [unclosed\n'
        out, ctx = self.run_task("python", self.MODULE_SEED, broken)
        self.assertEqual(out, [])
        self.assertEqual(ctx["rejects"][0]["reason"], "code_does_not_parse")

    def test_seed_without_embedded_yaml_stays_rejected(self):
        out, ctx = self.run_task("python", self.PLAIN_SEED, self.TASK_LIST_YAML)
        self.assertEqual(out, [])
        self.assertTrue(ctx["rejects"][0]["detail"].startswith("python"), ctx["rejects"][0])

    def test_generic_shared_keys_are_never_evidence(self):
        seed = 'CONFIG = """\nclient:\n  timeout: 30\n  retries: 3\n  pool_size: 4\n"""\n'
        self.assert_not_yaml("python", seed, "worker:\n  timeout: 10\n")  # one shared key, generic
        self.assert_not_yaml("python", seed, "worker:\n  timeout: 10\n  retries: 1\n")  # two, both generic
        self.assert_not_yaml("python", "x='''\nthing:\n  name: a\n  type: b\n'''", "server:\n  name: x\n  type: y\n")
        self.assert_not_yaml("python", "x='''\nsrv:\n  host: a\n  port: 2\n'''", "a:\n  host: x\n  port: 1\n")
        out, ctx = self.run_task("python", seed, "worker:\n  pool_size: 2\n")  # one non-generic shared key
        self.assertEqual(out[0]["payload"]["language"], "yaml")

    def test_docstring_sections_are_not_an_embedded_config_block(self):
        q = '"' * 3
        args_only = (f"def fetch_user(user_id, timeout=30, retries=3):\n    {q}\n    Args:\n        user_id: the id\n"
                     f"        timeout: seconds\n        retries: how many attempts\n    {q}\n    return get(user_id)\n")
        typed = (f"def connect(host, port, use_tls=False):\n    {q}\n    Args:\n        host (str): hostname\n"
                 f"        use_tls (bool): wrap in TLS\n    Raises:\n        ValueError: bad port\n    {q}\n")
        for seed, code in ((args_only, "fetch_user:\n  user_id: the id of the user\n  timeout: seconds\n"),
                           (args_only, "steps:\n  - fetch_user:\n      user_id: 42\n      retries: 3\n"),
                           (typed, "Args:\n  use_tls: true\n  host: x\n"),
                           (typed, "errors:\n  ValueError: bad port\n")):
            self.assert_not_yaml("python", seed, code)

    def test_code_declared_as_yaml_is_still_checked_as_the_seed(self):
        for code in ("if ready:\n    total = (1\n", "class Widget:\n    size: int = (\n",
                     "def f(x):\n    return (x\n", "for i in items:\n    print(i\n", "total: (1\n"):
            out, ctx = self.run_task("python", self.MODULE_SEED, code)
            self.assertEqual(out, [], code)
            self.assertTrue(ctx["rejects"][0]["detail"].startswith("python"), (code, ctx["rejects"][0]))

    def test_wrong_switch_probes_stay_off_yaml(self):
        # Review probes: each parses as nested YAML; none may be labelled yaml against a code seed.
        js_seed = ("export async function createOrder(items, options = {}) {\n  const response = await fetch('/api/orders',"
                   " { method: 'POST' });\n  return response.json();\n}\n")
        probes = [
            ("python", self.PLAIN_SEED, '{"a": {"b": 1}}', "Fetch a user"),
            ("python", self.PLAIN_SEED, "Addresses:\n  inet: 10.0.0.1 netmask 255.0.0.0\n  inet6: fe80::1\n", "Show it"),
            ("python", self.PLAIN_SEED, "step2:\n  call: api.fetch(x)\n", "Fetch"),
            ("python", self.PLAIN_SEED, "cfg:\n  user_id: List[str]\n", "Type it"),
            ("python", self.PLAIN_SEED, "Steps:\n  - Call fetch_user(x) with the id\n  - Check the result\n", "How"),
            ("python", self.PLAIN_SEED, "steps:\n  fetch_user:\n    call: session.get\n    then: return resp\n", "Fetch a user"),
            ("python", self.PLAIN_SEED, "Results:\n  fetch_user:\n    timeout: 30\n    retries: 3\n", "Call fetch_user"),
            ("python", self.PLAIN_SEED, "Explanation:\n  timeout: the number of seconds to wait\n  retries: attempts\n", "Explain"),
            ("python", self.PLAIN_SEED, "a:\n  b: 1\n", "Write a function for the checkout workflow"),
            ("javascript", js_seed, "a:\n  b: 1\n", "Render a bar chart of orders"),
            ("javascript", js_seed, "a:\n  b: 1\n", "Read the package manifest and create an order"),
            ("python", self.PLAIN_SEED, "a:\n  b: 1\n", "Load the values file and parse the yaml"),
            ("javascript", js_seed, "createOrder:\n  items:\n    - sku: A1\n  options:\n    currency: USD\n", "Order"),
            ("python", self.PLAIN_SEED, "INFO:\n  fetched: 42\nERROR:\n  timeout: after 30s\n", "Show log output"),
            ("python", self.PLAIN_SEED, "install:\n  - pip install requests\n  - python -c 'import x'\n", "Install"),
        ]
        import shutil
        from types import SimpleNamespace
        from unittest import mock

        from app.config import settings
        from app.services import sandbox_exec

        failing = SimpleNamespace(returncode=1, stdout="", stderr="SyntaxError: Unexpected token ':'", timed_out=False)
        # node_check stubbed (and failing) so the JS seed is a checked seed, like the Python one.
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "node_check"), \
                mock.patch.object(shutil, "which", return_value="/usr/bin/node"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", return_value=failing):
            for seed_lang, seed, code, task in probes:
                self.assert_not_yaml(seed_lang, seed, code, task)

    def test_embedded_workflow_and_openapi_examples(self):
        workflow_seed = ('"""Run the suite in CI.\n\nExample workflow:\n\n```yaml\non:\n  push:\n    branches: [main]\n'
                         'jobs:\n  test:\n    runs-on: ubuntu-latest\n```\n"""\nimport subprocess\n\n'
                         'def main():\n    subprocess.run(["pytest", "-q"], check=True)\n')
        workflow = ("on:\n  pull_request:\n    branches: [main]\njobs:\n  lint:\n    runs-on: ubuntu-latest\n"
                    "    steps:\n      - run: ruff check .\n")
        out, ctx = self.run_task("python", workflow_seed, workflow)
        self.assertEqual((ctx["rejects"], out[0]["payload"]["language"]), ([], "yaml"))

        import shutil
        from types import SimpleNamespace
        from unittest import mock

        from app.config import settings
        from app.services import sandbox_exec

        js_seed = ("const spec = `\nopenapi: 3.0.0\npaths:\n  /users/{id}:\n    get:\n      operationId: getUser\n`;\n"
                   "router.get('/users/:id', getUser);\n")
        spec = ("paths:\n  /users/{id}:\n    delete:\n      operationId: deleteUser\n      responses:\n"
                "        204:\n          description: deleted\n")
        failing = SimpleNamespace(returncode=1, stdout="", stderr="SyntaxError: Unexpected token ':'", timed_out=False)
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "node_check"), \
                mock.patch.object(shutil, "which", return_value="/usr/bin/node"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", return_value=failing):
            out, ctx = self.run_task("javascript", js_seed, spec)
            self.assertEqual(ctx["rejects"], [])
            self.assertEqual((out[0]["payload"]["language"], out[0]["payload"]["seed_language"]), ("yaml", "javascript"))
            out, ctx = self.run_task("javascript", "const total = rows => rows.length;\n", spec)
            self.assertEqual(out, [])
            self.assertTrue(ctx["rejects"][0]["detail"].startswith("node_check"), ctx["rejects"][0])

    def test_yaml_seed_manifest_is_the_unchanged_path(self):
        manifest = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\nspec:\n  replicas: 2\n"
                    "  template:\n    spec:\n      containers:\n        - name: web\n          image: web:1\n")
        out, ctx = self.run_one("yaml", reply(code=manifest, language="yaml"), "kind: Service\nmetadata:\n  name: web\n")
        self.assertEqual(ctx["rejects"], [])
        self.assertEqual(out[0]["payload"]["language"], "yaml")
        self.assertNotIn("seed_language", out[0]["payload"])

    def test_brace_keys_only_without_spaces(self):
        from app.services.synth_playbooks.code_examples import check_data_format

        self.assertIsNone(check_data_format("paths:\n  /users/{id}:\n    get: {}\n", "yaml"))
        self.assertIsNotNone(check_data_format("msg:\n  hello {user}: greeting\n", "yaml"))
        self.assertIsNotNone(check_data_format("out:\n  f{x}: 1\n", "yaml"))
        self.assertIsNone(check_data_format("env:\n  ${HOME}/bin: path\n", "yaml"))

    def test_data_seed_yaml_scalar_falls_back_to_the_seed(self):
        out, ctx = self.run_one("json", reply(code="just some prose here", language="yaml"))
        self.assertEqual(out, [])
        self.assertTrue(ctx["rejects"][0]["detail"].startswith("json_parse"))

    def test_unchecked_seed_with_unchecked_declared_language_is_labelled(self):
        out, ctx = self.run_one("cobol", reply(code="<a><b/></a>", language="xml"))
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
