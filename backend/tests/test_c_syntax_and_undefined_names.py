"""c_syntax (clang / gcc -fsyntax-only, sandbox stubbed + real-binary) and python_undefined_names (pyflakes)."""

import importlib.util
import shutil
import unittest
from unittest import mock

from app.config import settings
from app.services import code_check_service as svc
from app.services import sandbox_exec
from app.services.code_checks import external


def sbx(returncode=0, stdout="", stderr="", timed_out=False):
    return sandbox_exec.SandboxResult(returncode, stdout, stderr, timed_out, False, False, 0.01)


class Allow:
    def __init__(self, ids):
        self.p = mock.patch.object(settings, "CODE_CHECK_ALLOWED", ids)

    def __enter__(self):
        self.p.start()

    def __exit__(self, *a):
        self.p.stop()


HAS_CC = bool(shutil.which("clang") or shutil.which("gcc"))
HAS_CXX = bool(shutil.which("clang++") or shutil.which("g++"))
HAS_PYFLAKES = importlib.util.find_spec("pyflakes") is not None


class CSyntaxStubbedTests(unittest.TestCase):
    def run_c(self, code, results, language="c"):
        calls = []

        def fake(argv, *, files, **kw):
            calls.append((argv, files))
            return results.pop(0)

        with Allow("c_syntax"), mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=fake), \
                mock.patch.object(external.CSyntax, "compiler_for", return_value="clang"), \
                mock.patch.object(external.CSyntax, "available", return_value=True):
            out = svc.run_code_checks(code, language=language, check_ids=["c_syntax"])
        return out["results"][0], calls

    def test_gated_by_allow_list(self):
        r = svc.run_code_checks("int x;", language="c", check_ids=["c_syntax"])
        self.assertEqual(r["results"][0]["skipped"], "not_allowed")
        self.assertNotIn("c_syntax", [c.id for c in svc.checks_for_language("c")])

    def test_clean_and_resolution_only_pass(self):
        out, calls = self.run_c("int main(void) { return 0; }", [sbx(0)])
        self.assertIs(out["ok"], True)
        self.assertEqual(calls[0][0][:3], ["clang", "-fsyntax-only", "-w"])
        self.assertIn("snippet.c", calls[0][1])
        err = "snippet.c:2:5: error: use of undeclared identifier 'acme_widget_t'\n"
        out, _ = self.run_c("int main(void) {\n    acme_widget_t *w;\n}", [sbx(1, stderr=err)])
        self.assertIs(out["ok"], True)

    def test_syntax_error_fails_and_cpp_uses_cpp_file(self):
        err = "snippet.cpp:3:12: error: expected ';' at end of declaration\n"
        wrapped_err = "snippet.cpp:3:12: error: expected ';' at end of declaration\n"
        out, calls = self.run_c("int main() {\n  int x = 1\n  int y = 2;\n}\n", [sbx(1, stderr=err), sbx(1, stderr=wrapped_err)],
                                language="c++")
        self.assertIs(out["ok"], False)
        self.assertEqual(out["messages"], ["line 3: expected ';' at end of declaration"])
        self.assertIn("snippet.cpp", calls[0][1])
        self.assertIn("-std=gnu++17", calls[0][0])

    def test_syntax_error_on_a_line_with_an_unknown_name_is_ignored(self):
        err = ("snippet.c:1:1: error: unknown type name 'ACME_EXPORT'\n"
               "snippet.c:1:12: error: expected ';' after top level declarator\n")
        out, _ = self.run_c("ACME_EXPORT int f(int x) { return x; }\n", [sbx(1, stderr=err)])
        self.assertIs(out["ok"], True)

    def test_missing_headers_are_stripped_and_rechecked(self):
        code = '#include <stdio.h>\n#include "acme/widget.h"\n#include <acme/extra.h>\nint main(void) { return 0; }\n'
        clang_missing = "snippet.c:2:10: fatal error: 'acme/widget.h' file not found\n"
        gcc_missing = "snippet.c:3:10: fatal error: acme/extra.h: No such file or directory\n"
        out, calls = self.run_c(code, [sbx(1, stderr=clang_missing), sbx(1, stderr=gcc_missing), sbx(0)])
        self.assertIs(out["ok"], True)
        self.assertEqual(len(calls), 3)
        final = calls[2][1]["snippet.c"]
        self.assertNotIn("acme/widget.h", final)
        self.assertNotIn("acme/extra.h", final)
        self.assertIn("#include <stdio.h>", final)
        self.assertEqual(final.count("\n"), code.count("\n"))  # line numbers kept

    def test_syntax_error_after_stripping_still_fails_with_a_note(self):
        code = '#include "acme.h"\nint main(void) {\n  int x = 1\n  return x;\n}\n'
        missing = "snippet.c:1:10: fatal error: 'acme.h' file not found\n"
        err = "snippet.c:3:12: error: expected ';' at end of declaration\n"
        out, _ = self.run_c(code, [sbx(1, stderr=missing), sbx(1, stderr=err), sbx(1, stderr=err)])
        self.assertIs(out["ok"], False)
        self.assertIn("acme.h", out["messages"][0])
        self.assertIn("line 3: expected ';'", out["messages"][1])

    def test_unstrippable_header_is_skipped(self):
        # the compiler names a header the code does not include literally (e.g. pulled in by a macro)
        missing = "snippet.c:1:10: fatal error: 'acme_generated.h' file not found\n"
        out, _ = self.run_c("#include ACME_HEADER\nint x;\n", [sbx(1, stderr=missing)])
        self.assertIsNone(out["ok"])
        self.assertEqual(out["skipped"], "unresolvable_includes")

    def test_statement_fragment_is_retried_wrapped(self):
        code = "int x = acme_count();\nif (x > 2) { acme_log(x); }\n"
        top = "snippet.c:2:1: error: expected identifier or '('\n"
        out, calls = self.run_c(code, [sbx(1, stderr=top), sbx(0)])
        self.assertIs(out["ok"], True)
        wrapped = calls[1][1]["snippet.c"]
        self.assertIn("void __brewslm_snippet(void) {", wrapped)
        self.assertIn("#line 1", wrapped)

    def test_unbalanced_fragment_is_not_wrapped(self):
        top = "snippet.c:1:20: error: extraneous closing brace ('}')\n"
        out, calls = self.run_c("int x = 1; }\n", [sbx(1, stderr=top)])
        self.assertIs(out["ok"], False)
        self.assertEqual(len(calls), 1)

    def test_timeout(self):
        out, _ = self.run_c("int x;", [sbx(-9, timed_out=True)])
        self.assertEqual(out["skipped"], "timeout")

    def test_unavailable_without_a_compiler(self):
        with Allow("c_syntax"), mock.patch.object(external.shutil, "which", return_value=None):
            r = svc.run_code_checks("int x;", language="c", check_ids=["c_syntax"])
        self.assertEqual(r["results"][0]["skipped"], "unavailable")


@unittest.skipUnless(HAS_CC, "no C compiler on PATH")
class CSyntaxRealTests(unittest.TestCase):
    def check(self, code, language="c"):
        with Allow("c_syntax"):
            return svc.run_code_checks(code, language=language, check_ids=["c_syntax"])["results"][0]

    def test_real_compiler(self):
        missing_header = ('#include <stdio.h>\n#include <acme/widget.h>\nint main(void) {\n'
                          '    acme_widget_t *w = acme_widget_new(3);\n    printf("%d\\n", w->size);\n    return 0;\n}\n')
        self.assertIs(self.check(missing_header)["ok"], True)
        self.assertIs(self.check("int main(void) {\n  int x = 1\n  return x;\n}\n")["ok"], False)
        self.assertIs(self.check("int x = acme_count();\nif (x > 2) { acme_log(x); }\n")["ok"], True)

    @unittest.skipUnless(HAS_CXX, "no C++ compiler on PATH")
    def test_real_cpp(self):
        ok = '#include <vector>\n#include "acme/client.hpp"\nint main() {\n  std::vector<int> v;\n  v.push_back(1);\n  return 0;\n}\n'
        self.assertIs(self.check(ok, "cpp")["ok"], True)
        self.assertIs(self.check("#include <vector>\nint main() {\n  std::vector<int> v\n  return 0;\n}\n", "cpp")["ok"], False)


class PythonUndefinedNamesTests(unittest.TestCase):
    def test_gated_and_not_default_unless_allowed(self):
        self.assertEqual([c.id for c in svc.checks_for_language("python")], ["python_ast"])
        r = svc.run_code_checks("x = y\n", language="python", check_ids=["python_undefined_names"])
        self.assertEqual(r["results"][0]["skipped"], "not_allowed")
        rows = {c["id"]: c for c in svc.list_code_checks()}
        self.assertFalse(rows["python_undefined_names"]["allowed"])
        self.assertEqual(rows["python_undefined_names"]["kind"], "builtin")

    def test_unavailable_without_pyflakes(self):
        with Allow("python_undefined_names"), mock.patch("importlib.util.find_spec", return_value=None):
            r = svc.run_code_checks("x = y\n", language="python", check_ids=["python_undefined_names"])
            self.assertEqual(r["results"][0]["skipped"], "unavailable")
            self.assertEqual([c.id for c in svc.checks_for_language("python")], ["python_ast"])

    @unittest.skipUnless(HAS_PYFLAKES, "pyflakes is not installed")
    def test_real_pyflakes(self):
        def check(code):
            with Allow("python_undefined_names"):
                return svc.run_code_checks(code, language="python", check_ids=["python_undefined_names"])["results"][0]

        self.assertIs(check("import os\nprint(os.getcwd())\n")["ok"], True)
        self.assertIs(check("import os\nimport sys\nx = 1\n")["ok"], True)  # unused imports are not failures
        bad = check("def f():\n    return widget_count + 1\n")
        self.assertIs(bad["ok"], False)
        self.assertEqual(bad["messages"], ["line 2: undefined name 'widget_count'"])
        self.assertIs(check("    return self.size\n")["ok"], True)  # a method body: self is fine
        self.assertIs(check("def f(:\n")["ok"], False)
        self.assertIs(check("__all__ = ['missing']\n")["ok"], False)
        with Allow("python_undefined_names"):
            self.assertIn("python_undefined_names", [c.id for c in svc.checks_for_language("python")])


if __name__ == "__main__":
    unittest.main()
