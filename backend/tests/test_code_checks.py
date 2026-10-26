"""Code-check registry, built-ins, external checks (sandbox stubbed), plugins, API."""

import shutil
import sys
import types
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.config import settings
from app.services import code_check_service as svc
from app.services import sandbox_exec
from app.services.code_checks import external
from app.services.code_checks.base import BaseCheck, make_result, normalize_language


def sbx(returncode=0, stdout="", stderr="", timed_out=False):
    return sandbox_exec.SandboxResult(returncode, stdout, stderr, timed_out, False, False, 0.01)


class Allow:
    def __init__(self, ids):
        self.p = mock.patch.object(settings, "CODE_CHECK_ALLOWED", ids)

    def __enter__(self):
        self.p.start()

    def __exit__(self, *a):
        self.p.stop()


class LanguageTests(unittest.TestCase):
    def test_aliases(self):
        for raw, want in {"py": "python", "JS": "javascript", "sh": "shell", "bash": "shell", "tf": "terraform",
                          "hcl": "terraform", "cs": "csharp", "golang": "go", "yml": "yaml", " Python ": "python", None: ""}.items():
            self.assertEqual(normalize_language(raw), want)


class BuiltinTests(unittest.TestCase):
    def test_python_ast(self):
        ok = svc.run_code_checks("x = 1\n", language="py")
        self.assertIs(ok["ok"], True)
        bad = svc.run_code_checks("def f(:\n  pass", language="python")
        self.assertIs(bad["ok"], False)
        self.assertIn("line 1", bad["results"][0]["messages"][0])

    def test_python_indented_fragment_dedents(self):
        self.assertIs(svc.run_code_checks("    x = 1\n    y = 2\n", language="python")["ok"], True)
        self.assertIs(svc.run_code_checks("    x = (1\n    y = 2\n", language="python")["ok"], False)

    def test_json_and_yaml(self):
        self.assertIs(svc.run_code_checks('{"a": 1}', language="json")["ok"], True)
        r = svc.run_code_checks('{"a": 1,}', language="json")
        self.assertIs(r["ok"], False)
        self.assertEqual([x["check_id"] for x in r["results"]], ["json_parse"])  # json_schema needs a schema
        self.assertIs(svc.run_code_checks("a: 1\nb: [1, 2]\n", language="yml")["ok"], True)
        self.assertIs(svc.run_code_checks("a: [1, 2\n", language="yaml")["ok"], False)
        self.assertIs(svc.run_code_checks("a: &x 1\nb: *x\n", language="yaml")["ok"], False)  # aliases refused

    def test_json_schema(self):
        schema = {"type": "object", "required": ["patch"], "properties": {"patch": {"type": "string"}}}
        ok = svc.run_code_checks('{"patch": "x"}', language="json", context={"json_schema": schema})
        self.assertIs(ok["ok"], True)
        self.assertEqual({r["check_id"] for r in ok["results"]}, {"json_parse", "json_schema"})
        bad = svc.run_code_checks('{"patch": 3}', language="json", check_ids=["json_schema"], context={"json_schema": schema})
        self.assertIs(bad["ok"], False)
        self.assertIn("patch", bad["results"][0]["messages"][0])
        none = svc.run_code_checks("{}", language="json", check_ids=["json_schema"])
        self.assertIsNone(none["ok"])
        self.assertEqual(none["results"][0]["skipped"], "no_schema")


class RegistryTests(unittest.TestCase):
    def test_list_shape_and_gating(self):
        rows = {r["id"]: r for r in svc.list_code_checks()}
        self.assertTrue({"python_ast", "json_parse", "yaml_parse", "json_schema", "java_syntax", "go_syntax", "go_vet",
                         "terraform_fmt", "terraform_validate", "shellcheck", "node_check", "csharp_syntax"} <= set(rows))
        self.assertTrue(rows["python_ast"]["available"] and rows["python_ast"]["allowed"])
        self.assertEqual(rows["java_syntax"]["binary"], "javac")
        self.assertFalse(rows["java_syntax"]["allowed"])  # default: nothing external is allowed
        self.assertEqual(rows["python_ast"]["languages"], ["python"])

    def test_default_selection_excludes_unallowed_and_unavailable(self):
        self.assertEqual([c.id for c in svc.checks_for_language("java")], [])
        with Allow("java_syntax,shellcheck"):
            ids = [c.id for c in svc.checks_for_language("java")]
            self.assertEqual(ids, ["java_syntax"] if shutil.which("javac") else [])
        self.assertEqual([c.id for c in svc.checks_for_language("sh", ["shellcheck"])], ["shellcheck"])

    def test_skip_reasons(self):
        r = svc.run_code_checks("class A {}", language="java", check_ids=["java_syntax"])
        self.assertEqual(r["results"][0]["skipped"], "not_allowed")
        self.assertIsNone(r["ok"])
        with Allow("terraform_fmt"), mock.patch.object(shutil, "which", return_value=None):
            r = svc.run_code_checks("x", language="terraform", check_ids=["terraform_fmt"])
        self.assertEqual(r["results"][0]["skipped"], "unavailable")
        r = svc.run_code_checks("x = 1", language="go", check_ids=["python_ast", "nope"])
        self.assertEqual([x["skipped"] for x in r["results"]], ["language_mismatch", "unknown_check"])

    def test_ok_semantics_and_message_limits(self):
        class Noisy(BaseCheck):
            id = "zz_noisy"
            languages = frozenset({"python"})

            def run(self, code, *, language, context):
                return make_result(self.id, False, ["m" * 500] * 40)

        svc.register_code_check(Noisy())
        try:
          with Allow("zz_noisy"):
            r = svc.run_code_checks("x = 1", language="python")
            self.assertIs(r["ok"], False)  # python_ast ok, noisy failed
            noisy = [x for x in r["results"] if x["check_id"] == "zz_noisy"][0]
            self.assertEqual(len(noisy["messages"]), 20)
            self.assertTrue(all(len(m) <= 300 for m in noisy["messages"]))
        finally:
            svc._REGISTRY.pop("zz_noisy", None)

    def test_crashing_check_does_not_raise(self):
        class Boom(BaseCheck):
            id = "zz_boom"
            languages = frozenset({"python"})

            def run(self, code, *, language, context):
                raise RuntimeError("boom")

        svc.register_code_check(Boom())
        try:
            with Allow("zz_boom"):
                r = svc.run_code_checks("x = 1", language="python", check_ids=["zz_boom"])
            self.assertEqual(r["results"][0]["skipped"], "unavailable")
            self.assertIsNone(r["ok"])
        finally:
            svc._REGISTRY.pop("zz_boom", None)

    def test_detector_kind_is_gated_and_not_default(self):
        class Det(BaseCheck):
            id = "zz_det"
            languages = frozenset({"python"})
            kind = "detector"
            binary = None

            def run(self, code, *, language, context):
                return make_result(self.id, True)

        svc.register_code_check(Det())
        try:
            self.assertNotIn("zz_det", [c.id for c in svc.checks_for_language("python")])
            self.assertEqual(svc.run_code_checks("x", language="python", check_ids=["zz_det"])["results"][0]["skipped"], "not_allowed")
            with Allow("zz_det"):
                self.assertIs(svc.run_code_checks("x", language="python", check_ids=["zz_det"])["ok"], True)
        finally:
            svc._REGISTRY.pop("zz_det", None)


class JavaTests(unittest.TestCase):
    SYNTAX_ERR = "Snippet.java:3: error: ';' expected\n    int x = 1\n             ^\n1 error\n"
    RESOLUTION_ERR = ("Snippet.java:2: error: cannot find symbol\n  Foo f;\n  ^\n"
                      "Snippet.java:1: error: package com.acme does not exist\n2 errors\n")

    def run_java(self, code, results):
        calls = []

        def fake(argv, *, files, **kw):
            calls.append((argv, files))
            return results.pop(0) if isinstance(results, list) else results

        with Allow("java_syntax"), mock.patch.object(shutil, "which", return_value="/usr/bin/javac"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=fake):
            out = svc.run_code_checks(code, language="java", check_ids=["java_syntax"])
        return out, calls

    def test_filter_classification(self):
        syn, ign = external.java_syntax_messages(self.SYNTAX_ERR + self.RESOLUTION_ERR)
        self.assertEqual(len(syn), 1)
        self.assertEqual(len(ign), 2)
        self.assertIn("';' expected", syn[0])
        for text in ("illegal start of expression", "not a statement", "reached end of file while parsing",
                     "unclosed string literal", "class, interface, enum, or record expected", "orphaned case"):
            s, _ = external.java_syntax_messages(f"Snippet.java:1: error: {text}")
            self.assertEqual(len(s), 1, text)
        for text in ("incompatible types: String cannot be converted to int", "missing return statement"):
            s, i = external.java_syntax_messages(f"Snippet.java:1: error: {text}")
            self.assertEqual((len(s), len(i)), (0, 1), text)

    def test_resolution_errors_pass(self):
        out, _ = self.run_java("class A { Foo f; }", sbx(1, stderr=self.RESOLUTION_ERR))
        self.assertIs(out["ok"], True)

    def test_syntax_error_fails_with_line(self):
        out, calls = self.run_java("class A { int x = 1 }", sbx(1, stderr=self.SYNTAX_ERR))
        self.assertIs(out["ok"], False)
        self.assertIn("line 3", out["results"][0]["messages"][0])
        self.assertEqual(len(calls), 1)  # a whole class is not re-wrapped

    def test_fragment_wrapping(self):
        out, calls = self.run_java("int x = 1;", [sbx(1, stderr=self.SYNTAX_ERR), sbx(0)])
        self.assertIs(out["ok"], True)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1]["Snippet.java"].startswith("class Snippet { int x = 1;"))
        self.assertIn("void __snippet()", calls[1][1]["Snippet.java"])
        self.assertIn("-proc:none", calls[0][0])

    def test_member_fragment_passes_first_wrap(self):
        out, calls = self.run_java("void f() { int a; }", [sbx(0)])
        self.assertIs(out["ok"], True)
        self.assertEqual(len(calls), 1)

    def test_timeout_and_refusal(self):
        out, _ = self.run_java("class A {}", sbx(-9, timed_out=True))
        self.assertEqual(out["results"][0]["skipped"], "timeout")
        with Allow("java_syntax"), mock.patch.object(shutil, "which", return_value="/usr/bin/javac"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=sandbox_exec.SandboxError("no isolation")):
            out = svc.run_code_checks("class A {}", language="java", check_ids=["java_syntax"])
        self.assertEqual(out["results"][0]["skipped"], "unavailable")

    @unittest.skipUnless(shutil.which("javac"), "javac not installed")
    def test_real_javac(self):
        with Allow("java_syntax"):
            good = svc.run_code_checks("int add(int a, int b) { return a + b; }", language="java", check_ids=["java_syntax"])
            unresolved = svc.run_code_checks("class A { Missing m; }", language="java", check_ids=["java_syntax"])
            bad = svc.run_code_checks("class A { void f() { int x = ; } }", language="java", check_ids=["java_syntax"])
        self.assertIs(good["ok"], True, good)
        self.assertIs(unresolved["ok"], True, unresolved)
        self.assertIs(bad["ok"], False, bad)


class GoShellNodeTests(unittest.TestCase):
    def test_go_wrapping_and_messages(self):
        seen = []

        def fake(argv, *, files, stdin=None, **kw):
            seen.append(stdin)
            return sbx(0) if "func _snippet" in stdin else sbx(2, stderr="<standard input>:1:20: expected declaration, found x := 1\n")

        with Allow("go_syntax"), mock.patch.object(shutil, "which", return_value="/usr/bin/gofmt"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=fake):
            out = svc.run_code_checks("x := 1", language="go", check_ids=["go_syntax"])
        self.assertIs(out["ok"], True)
        self.assertTrue(seen[0].startswith("package snippet; x := 1"))
        with Allow("go_syntax"), mock.patch.object(shutil, "which", return_value="/usr/bin/gofmt"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", return_value=sbx(2, stderr="<standard input>:3:1: bad\n")):
            out = svc.run_code_checks("package p\nfunc f( {", language="golang", check_ids=["go_syntax"])
        self.assertIs(out["ok"], False)
        self.assertEqual(out["results"][0]["messages"], ["line 3:1: bad"])

    def test_go_fragment_that_closes_the_wrapper_fails(self):
        with Allow("go_syntax"), mock.patch.object(shutil, "which", return_value="/usr/bin/gofmt"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", return_value=sbx(0)) as m:
            out = svc.run_code_checks("}\nfunc g() {", language="go", check_ids=["go_syntax"])
            raw = svc.run_code_checks("s := `}`\nuse(s)", language="go", check_ids=["go_syntax"])
        self.assertIs(out["ok"], False)
        self.assertIn("unbalanced", out["results"][0]["messages"][0])
        self.assertIs(raw["ok"], True)  # a brace inside a raw string doesn't count
        self.assertEqual(m.call_count, 1)

    def test_go_vet_requires_package_clause(self):
        with Allow("go_vet"), mock.patch.object(shutil, "which", return_value="/usr/bin/go"), \
                mock.patch.object(sandbox_exec, "run_sandboxed") as m:
            out = svc.run_code_checks("x := 1", language="go", check_ids=["go_vet"])
        m.assert_not_called()
        self.assertIsNone(out["ok"])

    def test_terraform_validate_needs_module_dir(self):
        with Allow("terraform_validate"), mock.patch.object(shutil, "which", return_value="/usr/bin/terraform"):
            out = svc.run_code_checks("x", language="tf", check_ids=["terraform_validate"])
        self.assertEqual(out["results"][0]["skipped"], "no_module_dir")

    def test_shellcheck_and_node(self):
        with Allow("shellcheck,node_check"), mock.patch.object(shutil, "which", return_value="/x"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", return_value=sbx(1, stdout="-:2:5: error: bad\n")):
            sh = svc.run_code_checks("echo $(", language="bash", check_ids=["shellcheck"])
        self.assertIs(sh["ok"], False)
        self.assertEqual(sh["results"][0]["messages"], ["line 2:5: error: bad"])
        with Allow("node_check"), mock.patch.object(shutil, "which", return_value="/x"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", return_value=sbx(0)) as m:
            js = svc.run_code_checks("let a = 1", language="js", check_ids=["node_check"])
        self.assertIs(js["ok"], True)
        self.assertEqual(m.call_args.args[0][:2], ["node", "--check"])

    @unittest.skipUnless(shutil.which("gofmt"), "gofmt not installed")
    def test_real_gofmt(self):
        with Allow("go_syntax"):
            good = svc.run_code_checks("x := compute(1)", language="go", check_ids=["go_syntax"])
            bad = svc.run_code_checks("package p\nfunc f( {", language="go", check_ids=["go_syntax"])
        self.assertIs(good["ok"], True, good)
        self.assertIs(bad["ok"], False, bad)


class PluginTests(unittest.TestCase):
    def tearDown(self):
        for k in ("zz_plug_a", "zz_plug_b"):
            svc._REGISTRY.pop(k, None)
        for m in ("zz_plugin_ok", "zz_plugin_dict", "zz_plugin_bad"):
            sys.modules.pop(m, None)
            svc._LOADED_PLUGIN_MODULES.discard(m)
            svc._PLUGIN_ERRORS.pop(m, None)

    def _module(self, name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    def test_register_hook_and_list(self):
        class A(BaseCheck):
            id = "zz_plug_a"
            languages = frozenset({"Python"})

            def run(self, code, *, language, context):
                return make_result(self.id, True)

        self._module("zz_plugin_ok", register_code_checks=lambda register: register(A()))
        self._module("zz_plugin_dict", CODE_CHECKS=[type("B", (A,), {"id": "zz_plug_b"})])
        self._module("zz_plugin_bad", nothing=1)
        diag = svc.load_code_check_plugins(["zz_plugin_ok", "zz_plugin_dict", "zz_plugin_bad", "zz_does_not_exist"])
        self.assertEqual(diag["registered_checks"], 2)
        self.assertEqual(set(diag["loaded_modules"]), {"zz_plugin_ok", "zz_plugin_dict"})
        self.assertEqual(set(diag["failed_modules"]), {"zz_plugin_bad", "zz_does_not_exist"})
        self.assertIn("zz_plug_a", {r["id"] for r in svc.list_code_checks()})
        self.assertEqual(svc.list_code_checks()[0].keys(), {"id", "languages", "kind", "binary", "available", "allowed"})
        # a plugin is never trusted by what it declares: it runs only when allow-listed
        self.assertEqual(svc.run_code_checks("x", language="python", check_ids=["zz_plug_a"])["results"][0]["skipped"], "not_allowed")
        with Allow("zz_plug_a"):
            self.assertIs(svc.run_code_checks("x", language="python", check_ids=["zz_plug_a"])["ok"], True)

    def test_self_declared_builtin_is_not_trusted(self):
        class Liar(BaseCheck):
            id = "zz_plug_a"
            kind = "builtin"
            languages = frozenset({"python"})

            def run(self, code, *, language, context):
                return make_result(self.id, True)

        svc.register_code_check(Liar())
        try:
            self.assertFalse(svc.is_allowed(Liar()))
            r = svc.run_code_checks("x", language="python", check_ids=["zz_plug_a"])
            self.assertEqual(r["results"][0]["skipped"], "not_allowed")
            self.assertTrue(svc.is_allowed(svc.get_code_check("python_ast")))
        finally:
            svc._REGISTRY.pop("zz_plug_a", None)

    def test_from_settings(self):
        self._module("zz_plugin_ok", CODE_CHECKS=[])
        with mock.patch.object(settings, "CODE_CHECK_PLUGIN_MODULES", " zz_plugin_ok , "):
            diag = svc.load_code_check_plugins_from_settings()
        self.assertEqual(diag["loaded_modules"].count("zz_plugin_ok"), 1)
        with mock.patch.object(settings, "CODE_CHECK_PLUGIN_MODULES", ""):
            self.assertEqual(svc.load_code_check_plugins_from_settings()["status"], "no_plugin_modules_configured")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.main import app

        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def test_list(self):
        body = self.client.get("/api/code-checks").json()
        self.assertIn("python_ast", {c["id"] for c in body["checks"]})
        self.assertEqual(set(body["network_isolation"]), {"mode", "available"})
        self.assertEqual(body["network_isolation"]["mode"], "auto")

    def test_run(self):
        r = self.client.post("/api/code-checks/run", json={"language": "py", "code": "def f(:\n"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIs(r.json()["ok"], False)
        r = self.client.post("/api/code-checks/run", json={"language": "json", "code": '{"a":1}', "check_ids": ["json_parse"]})
        self.assertIs(r.json()["ok"], True)

    def test_too_big_and_bad_body(self):
        self.assertEqual(self.client.post("/api/code-checks/run", json={"language": "py", "code": "x" * 200_001}).status_code, 422)
        self.assertEqual(self.client.post("/api/code-checks/run", json={"code": "x"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
