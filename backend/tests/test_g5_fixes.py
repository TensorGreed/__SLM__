"""Regression tests for the G5 review fixes (diff applier, scoring denominators, fingerprint,
Java / Go checks, own-chunk exclusion, detector identity, playbook extras). pytest-only."""

from __future__ import annotations

import asyncio
import json
import shutil
import unittest
from unittest import mock

from app.config import settings
from app.services import code_remediation_prompt as crp
from app.services import remediation_detector as rd
from app.services import remediation_scoring as rs
from app.services import sandbox_exec
from app.services import sft_lift_summary_service as lift
from app.services.code_checks import external
from app.services.unified_diff import apply_patch_to_chunk as ap
from app.services.unified_diff import make_unified_diff as mk


def sbx(returncode=0, stdout="", stderr="", timed_out=False):
    return sandbox_exec.SandboxResult(returncode, stdout, stderr, timed_out, False, False, 0.01)


class DiffFixTests(unittest.TestCase):
    CODE = "def f(q):\n    sql = 'a' + q\n\n    log(password)\n    return sql\n"

    def test_blank_context_after_header_counts_are_satisfied_keeps_the_body(self):
        p = ("@@ -1,2 +1,2 @@\n def f(q):\n-    sql = 'a' + q\n+    sql = bind(q)\n\n"
             "-    log(password)\n+    log('redacted')\n     return sql\n")
        r = ap(self.CODE, 1, p)
        self.assertTrue(r.ok, r.reason)
        self.assertFalse(r.header_counts_ok)  # the body disagrees with the header: the body wins
        self.assertEqual(r.patched_code, "def f(q):\n    sql = bind(q)\n\n    log('redacted')\n    return sql\n")

    def test_trailing_blank_lines_still_end_the_hunk(self):
        r = ap("a\nb\n", 1, "@@ -1,1 +1,1 @@\n-a\n+A\n\n\n")
        self.assertTrue(r.ok)
        self.assertTrue(r.header_counts_ok)
        self.assertEqual(r.patched_code, "A\nb\n")

    def test_overlapping_hunks_fail_instead_of_matching_later(self):
        code = "x = 1\ny = 2\nx = 1\ny = 2\n"
        p = "@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 3\n@@ -2,1 +2,1 @@\n-y = 2\n+y = 4\n"
        r = ap(code, 1, p)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "overlapping_hunks")

    def test_indented_fence_in_context_round_trips(self):
        code = 'doc = """\n ```python\nprint(1)\n ```\n"""\nval = 1\n'
        new = code.replace("val = 1", "val = 2")
        d = mk(code, new, path="m.py", start_line=1)
        self.assertEqual(ap(code, 1, d, path="m.py").patched_code, new)
        self.assertEqual(ap(code, 1, "```diff\n" + d + "```\n", path="m.py").patched_code, new)

    def test_column_zero_fence_is_still_stripped(self):
        r = ap("a\nb\n", 1, "Here you go:\n```diff\n@@ -1,1 +1,1 @@\n-a\n+A\n```\nthanks\n")
        self.assertTrue(r.ok)
        self.assertEqual(r.patched_code, "A\nb\n")

    def test_pure_insert_out_of_range_fails(self):
        r = ap("a = 1\nb = 2\n", 1, "@@ -12,0 +13,1 @@\n+z = 9\n")
        self.assertEqual((r.ok, r.reason), (False, "out_of_range"))
        self.assertTrue(ap("a = 1\nb = 2\n", 1, "@@ -2,0 +3,1 @@\n+z = 9\n").ok)  # after the last line

    def test_ambiguous_context_at_the_same_distance_fails(self):
        r = ap("a\nb\na\nb\na\n", 1, "@@ -2,1 +2,1 @@\n-a\n+A\n")
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "context_mismatch")
        self.assertIn("ambiguous", r.detail)

    def test_nearest_unique_match_still_applies_with_offset(self):
        r = ap("a\nb\nc\nd\n", 1, "@@ -1,1 +1,1 @@\n-c\n+C\n")
        self.assertTrue(r.ok)
        self.assertEqual(r.offsets, [2])


class ScoringFixTests(unittest.TestCase):
    def setUp(self):
        from test_code_remediation_recipe import CODE, FIXED, GOOD, patch_of, prediction_for, remediation_row, score

        self.mod = dict(CODE=CODE, FIXED=FIXED, GOOD=GOOD, patch_of=patch_of, prediction_for=prediction_for,
                        remediation_row=remediation_row, score=score)

    def test_checkable_depends_on_the_original_only(self):
        m = self.mod
        good = m["prediction_for"](m["remediation_row"](), m["GOOD"])
        malformed = m["prediction_for"](m["remediation_row"](), "garbage")
        no_apply = m["prediction_for"](m["remediation_row"](), {**m["GOOD"], "patch": "just prose"})
        metrics = m["score"]([good, malformed, no_apply])
        self.assertEqual(metrics["code_checkable_count"], 3)
        self.assertAlmostEqual(metrics["code_parses_rate"], round(1 / 3, 4))
        self.assertIs(malformed["row_code_checkable"], True)
        self.assertIs(malformed["row_code_parses"], False)

    def test_blind_detector_rows_and_the_overall_rate(self):
        from test_code_remediation_recipe import DetectorTests

        m = self.mod
        t = DetectorTests()
        rows = [
            m["prediction_for"](m["remediation_row"](), m["GOOD"]),   # reported on the original, gone: resolved
            m["prediction_for"](m["remediation_row"](), m["GOOD"]),   # not reported on the original: blind
            m["prediction_for"](m["remediation_row"](), "garbage"),   # no patch: not resolved
        ]
        # both applied rows share one code, so make the second one blind through its own chunk text
        rows[1]["remediation_row"] = {**rows[1]["remediation_row"], "code": m["CODE"] + "# other\n"}
        rows[1]["prediction"] = json.dumps({**m["GOOD"], "patch": m["patch_of"](m["CODE"] + "# other\n", m["FIXED"] + "# other\n")})
        metrics, _ = t.run_detector(rows, lambda code: ["injection.query"] if code == m["CODE"] else [])
        self.assertTrue(rows[0]["row_issue_resolved"])
        self.assertIsNone(rows[1]["row_issue_resolved"])
        self.assertEqual(rows[1]["row_detector_note"], "detector_blind")
        self.assertEqual(metrics["detector_blind_count"], 1)
        self.assertEqual(metrics["detector_evaluated_count"], 1)
        self.assertEqual(metrics["issue_resolved_rate"], 1.0)
        # resolved / (scored - blind): the unapplied row counts as not resolved
        self.assertAlmostEqual(metrics["issue_resolved_overall_rate"], 0.5)
        # a blind row's pass ignores issue_resolved
        self.assertTrue(rows[1]["row_remediation_pass"])

    def test_lift_metrics(self):
        base = {"remediation_pass_rate": 0.25, "issue_resolved_rate": 1.0, "new_findings_rate": 0.0,
                "issue_resolved_overall_rate": 0.1, "malformed_rate": 0.5, "clean_chunk_fp_rate": 0.4}
        tuned = {"remediation_pass_rate": 0.75, "issue_resolved_rate": 0.5, "new_findings_rate": 0.2,
                 "issue_resolved_overall_rate": 0.6, "malformed_rate": 0.1, "clean_chunk_fp_rate": 0.1}
        lifts = {r["metric_id"]: r for r in lift._compute_metric_lifts(lift._normalize_metrics(base), lift._normalize_metrics(tuned))}
        self.assertNotIn("issue_resolved_rate", lifts)
        self.assertNotIn("new_findings_rate", lifts)
        self.assertIn("issue_resolved_overall_rate", lifts)
        self.assertEqual(lifts["malformed_rate"]["direction"], "improved")  # a drop is the improvement
        self.assertEqual(lifts["clean_chunk_fp_rate"]["direction"], "improved")
        self.assertEqual(lifts["issue_resolved_overall_rate"]["direction"], "improved")


class FingerprintTests(unittest.TestCase):
    def test_baseline_reuse_requires_a_matching_fingerprint(self):
        from app.services.post_training_eval_service import _baseline_scoring_matches

        res = mock.Mock(metrics={"remediation_fingerprint": "abc"})
        self.assertTrue(_baseline_scoring_matches(res, "abc"))
        self.assertFalse(_baseline_scoring_matches(res, "def"))
        self.assertFalse(_baseline_scoring_matches(mock.Mock(metrics={}), "abc"))  # older result: re-run
        self.assertTrue(_baseline_scoring_matches(mock.Mock(metrics={}), None))  # not a remediation project

    def fp(self, runtime=None, schema=None, detector_identity="none"):
        async def ident(db, cfg):
            return detector_identity

        with mock.patch.object(rd, "detector_identity", ident):
            return asyncio.run(rd.remediation_scoring_fingerprint(None, 5, runtime, schema))

    def test_fingerprint_follows_every_scoring_input(self):
        base = self.fp({"remediation": {"retrieve_passages": False}})
        self.assertEqual(base, self.fp({"remediation": {"retrieve_passages": False}}))
        self.assertNotEqual(base, self.fp({"remediation": {"retrieve_passages": True}}))
        self.assertNotEqual(base, self.fp({"remediation": {"retrieve_passages": False, "top_k": 5}}))
        self.assertNotEqual(base, self.fp({"remediation": {"retrieve_passages": False, "checks": ["json_parse"]}}))
        self.assertNotEqual(base, self.fp({"remediation": {"retrieve_passages": False}}, detector_identity="project:1:run:9"))
        self.assertNotEqual(base, self.fp({"remediation": {"retrieve_passages": False}},
                                          schema={"properties": {"severity": {"type": "string"}}}))
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "java_syntax"):
            self.assertNotEqual(base, self.fp({"remediation": {"retrieve_passages": False}}))

    def test_cached_baseline_is_not_reused_after_the_config_changed(self):
        """End to end through ``_reusable_baseline_result`` with a fake DB."""
        from datetime import datetime, timezone

        from app.services import post_training_eval_service as pte

        cand = mock.Mock(created_at=datetime.now(timezone.utc), details={"row_scores": {}},
                         metrics={"remediation_fingerprint": "old", "inference": {}})
        db = mock.Mock()
        scalars = mock.Mock()
        scalars.scalars.return_value.all.return_value = [cand]

        async def execute(*a, **k):
            return scalars

        db.execute = execute

        async def mtime(_pid):
            return None

        with mock.patch.object(pte, "_test_split_mtime", mtime), \
                mock.patch.object(pte, "_baseline_judge_matches", return_value=True), \
                mock.patch.object(pte, "_baseline_decoding_matches", return_value=True):
            exp = mock.Mock(id=1)
            self.assertIs(asyncio.run(pte._reusable_baseline_result(db, exp, 1, scoring_fingerprint="old")), cand)
            self.assertIsNone(asyncio.run(pte._reusable_baseline_result(db, exp, 1, scoring_fingerprint="new")))


class DetectorIdentityTests(unittest.TestCase):
    def test_recipe_fingerprint_changes_with_the_prompt_builder(self):
        a = rd.detector_recipe_fingerprint(["injection"])
        self.assertEqual(a, rd.detector_recipe_fingerprint(["injection"]))
        self.assertNotEqual(a, rd.detector_recipe_fingerprint(["injection", "secret.hardcoded"]))
        with mock.patch("app.services.code_findings_prompt.build_code_findings_prompt_from_row", return_value="changed prompt"):
            self.assertNotEqual(a, rd.detector_recipe_fingerprint(["injection"]))

    def test_binary_fingerprint_follows_the_file(self):
        import os
        import tempfile

        self.assertEqual(rd.binary_fingerprint("definitely-not-a-binary"), "none")
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "mydet")
            with open(exe, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(exe, 0o755)
            os.utime(exe, (1_000_000, 1_000_000))
            with mock.patch.dict(os.environ, {"PATH": d}):
                one = rd.binary_fingerprint("mydet")
                os.utime(exe, (2_000_000, 2_000_000))
                two = rd.binary_fingerprint("mydet")
        self.assertNotEqual(one, "none")
        self.assertNotEqual(one, two)


class OwnChunkTests(unittest.TestCase):
    ROW = {"code": "a = 1\nb = 2\nc = 3\n", "path": "src/app/db.py", "start_line": 10, "end_line": 12}

    def test_suffix_path_match_with_overlapping_lines(self):
        p = {"path": "/repo/src/app/db.py", "start_line": 11, "end_line": 20, "text": "x"}
        self.assertTrue(crp._overlaps_own_chunk(self.ROW, p))
        self.assertFalse(crp._overlaps_own_chunk(self.ROW, {**p, "start_line": 30, "end_line": 40}))
        self.assertFalse(crp._overlaps_own_chunk(self.ROW, {**p, "path": "/repo/src/app/xdb.py"}))  # not at a "/" boundary

    def test_document_id_prefix(self):
        row = {**self.ROW, "_source_document_id": 7, "path": None}
        self.assertTrue(crp._overlaps_own_chunk(row, {"passage_id": "7:3", "path": "other.py", "start_line": 10, "end_line": 12, "text": "x"}))
        self.assertFalse(crp._overlaps_own_chunk(row, {"passage_id": "8:3", "path": "other.py", "start_line": 10, "end_line": 12, "text": "x"}))

    def test_pathless_passage_that_contains_the_code_is_excluded(self):
        self.assertTrue(crp._overlaps_own_chunk(self.ROW, {"text": "# header\na = 1\nb = 2\nc = 3\n"}))
        self.assertFalse(crp._overlaps_own_chunk(self.ROW, {"text": "a = 1\nunrelated"}))

    def test_retrieval_drops_the_row_s_own_chunk(self):
        hits = [
            {"payload": {"text": "a = 1\nb = 2\nc = 3\n", "path": "/r/src/app/db.py", "start_line": 10, "end_line": 12}},
            {"payload": {"text": "other guidance", "source_doc": "guide"}},
        ]
        out = crp.retrieve_passages_for_row(1, self.ROW, {}, 2, retrieve_fn=lambda q, c: hits)
        self.assertEqual([p["text"] for p in out], ["other guidance"])


class JavaTests(unittest.TestCase):
    def test_brace_lexer_ignores_literals_and_comments(self):
        bal = external.java_brace_balance
        self.assertEqual(bal('String s = "}{"; char c = \'}\'; // }\n/* { */ f();'), (None, 0))
        self.assertEqual(bal("void f() { a(); }"), (None, 0))
        self.assertEqual(bal("void f() { a(); } }  class Z {"), (1, 0))
        self.assertEqual(bal("if (x) {\n a();\n"), (None, 1))
        self.assertEqual(bal('String t = """\n }\n""";'), (None, 0))

    def test_toplevel_detection(self):
        self.assertTrue(external.java_is_toplevel("package a;\nclass A {}"))
        self.assertTrue(external.java_is_toplevel("// c\npublic final class A { }"))
        self.assertFalse(external.java_is_toplevel("int a = 1;\nclass Local { }\nuse(a);"))
        self.assertFalse(external.java_is_toplevel("    class Local { }\n    use(1);"))  # indented: a local class
        self.assertEqual(len(external.JavaSyntax.candidates("class A {}")), 1)
        wrapped = external.JavaSyntax.candidates("void f() { }")
        self.assertEqual(len(wrapped), 2)
        self.assertTrue(all("__brewslm_end_sentinel" in w for w in wrapped))

    def run_check(self, code, results):
        """``results``: stub sandbox results, one per javac call."""
        calls = []

        def fake(argv, **kw):
            calls.append(kw["files"]["Snippet.java"])
            return results[min(len(calls) - 1, len(results) - 1)]

        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "java_syntax"), \
                mock.patch.object(shutil, "which", return_value="/usr/bin/javac"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", fake):
            from app.services import code_check_service as svc

            return svc.run_code_checks(code, language="java", check_ids=["java_syntax"])["results"][0], calls

    def test_unbalanced_fragments_fail_without_javac(self):
        res, calls = self.run_check("void f() { a(); } }  class Z {", [sbx()])
        self.assertIs(res["ok"], False)
        self.assertIn("unbalanced", res["messages"][0])
        self.assertEqual(calls, [])
        res, _ = self.run_check("if (x) {\n a();\n", [sbx()])
        self.assertIs(res["ok"], False)

    def test_syntax_error_at_or_after_the_sentinel_fails_even_when_javac_exits_ok_elsewhere(self):
        code = "void f() { a(); }"  # sentinel is on line 2
        res, _ = self.run_check(code, [sbx(1, stderr="Snippet.java:2: error: reached end of file while parsing\n")] * 2)
        self.assertIs(res["ok"], False)

    def test_resolution_errors_are_ignored(self):
        res, _ = self.run_check("a.b();", [sbx(1, stderr="Snippet.java:1: error: cannot find symbol\n")])
        self.assertIs(res["ok"], True)

    def test_local_class_inside_statements_passes(self):
        # first wrapper (member) fails with a syntax error, the statement wrapper passes
        res, calls = self.run_check("int a = 1;\nclass Local { }\nuse(a);",
                                    [sbx(1, stderr="Snippet.java:1: error: illegal start of type\n"), sbx(0)])
        self.assertIs(res["ok"], True)
        self.assertEqual(len(calls), 2)

    @unittest.skipUnless(shutil.which("javac"), "javac not installed")
    def test_real_javac(self):
        from app.services import code_check_service as svc

        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "java_syntax"):
            def ok(code):
                return svc.run_code_checks(code, language="java", check_ids=["java_syntax"])["ok"]

            self.assertTrue(ok('String q = "a" + x;\nrun(q);'))
            self.assertFalse(ok('String q = "a" + x\nrun(q);'))
            self.assertTrue(ok("void f() { a(); }"))
            self.assertFalse(ok("void f() { a(); } }  class Z {"))
            self.assertTrue(ok("int a = 1;\nclass Local { }\nuse(a);"))


class GoTests(unittest.TestCase):
    def run_check(self, code, results):
        calls = []

        def fake(argv, **kw):
            calls.append(kw.get("stdin"))
            return results[min(len(calls) - 1, len(results) - 1)]

        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "go_syntax"), \
                mock.patch.object(shutil, "which", return_value="/usr/bin/gofmt"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", fake):
            from app.services import code_check_service as svc

            return svc.run_code_checks(code, language="go", check_ids=["go_syntax"])["results"][0], calls

    def test_fragment_failures_report_the_func_wrapped_candidate_against_original_lines(self):
        first = sbx(2, stderr="<standard input>:1:20: expected 'IDENT', found 'x'\n")
        second = sbx(2, stderr="<standard input>:1:45: expected ')', found newline\n<standard input>:3:2: expected '}', found 'EOF'\n")
        res, calls = self.run_check("x := (1\nuse(x)", [first, second])
        self.assertIs(res["ok"], False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(res["messages"][0], f"line 1:{45 - len(external._GO_FUNC_PREFIX)}: expected ')', found newline")
        self.assertEqual(res["messages"][1], "line 3:2: expected '}', found 'EOF'")

    def test_whole_file_is_checked_as_is(self):
        res, calls = self.run_check("package a\nfunc f() {}\n", [sbx(0)])
        self.assertIs(res["ok"], True)
        self.assertEqual(len(calls), 1)

    def test_go_vet_cache_is_per_call_inside_the_sandbox_temp_dir(self):
        seen = {}

        def fake(argv, **kw):
            seen.update(kw)
            return sbx(0)

        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "go_vet"), \
                mock.patch.object(shutil, "which", return_value="/usr/bin/go"), \
                mock.patch.object(sandbox_exec, "run_sandboxed", fake):
            from app.services import code_check_service as svc

            svc.run_code_checks("package a\n", language="go", check_ids=["go_vet"])
        env = seen["env"]
        self.assertTrue(env["GOCACHE"].startswith("{tmp}/"))
        self.assertTrue(env["GOPATH"].startswith("{tmp}/"))
        self.assertNotIn(str(settings.DATA_DIR), env["GOCACHE"])

    def test_sandbox_expands_the_temp_dir_placeholder(self):
        env = sandbox_exec._sandbox_env("/tmp/x", {"GOCACHE": "{tmp}/.gocache"})
        self.assertEqual(env["GOCACHE"], "/tmp/x/.gocache")


class PlaybookFixTests(unittest.TestCase):
    def test_extra_fields_reach_the_prompt_are_required_and_carried(self):
        from test_code_remediation_playbooks import diff, make_ctx, reasons
        from app.services.synth_playbooks import SynthMode, get_playbook

        pb = get_playbook("code-remediation", SynthMode.REMEDIATION_FROM_FINDINGS)
        extra = {"severity": {"type": ["string", "null"], "description": "How serious the finding is."}}
        ctx = make_ctx(remediation_extra_fields=extra)
        prompt = pb.build_prompt(ctx)
        self.assertIn("severity", prompt)
        self.assertIn("How serious the finding is.", prompt)
        self.assertIn("severity", pb.response_schema(ctx)["required"])
        good = {"task": 1, "patch": diff(), "rationale": "Bind it.", "citations": [], "severity": "high"}
        missing = {k: v for k, v in good.items() if k != "severity"}
        out = pb.validate([good, missing], ctx)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["payload"]["remediation_json"]["severity"], "high")
        self.assertEqual(reasons(ctx), ["missing_extra_field"])

    def test_fix_pairs_without_a_label_or_changed_lines_are_reported(self):
        from test_code_remediation_playbooks import BAD, fix_pair_row
        from app.services.synth_playbooks._code_remediation_common import build_seeds_from_rows

        rejected: list = []
        seeds = build_seeds_from_rows(
            [fix_pair_row(labels=[]), fix_pair_row(fixed_code=BAD), fix_pair_row()], "findings_dataset", rejected
        )
        self.assertEqual(len(seeds), 1)
        self.assertEqual([r for r, _d in rejected], ["no_label", "no_changed_lines"])


class CodeCheckTrustTests(unittest.TestCase):
    def test_core_builtins_are_trusted_by_identity(self):
        from app.services import code_check_service as svc

        self.assertTrue(svc.is_allowed(svc.get_code_check("python_ast")))
        self.assertFalse(svc.is_allowed(svc.get_code_check("java_syntax")))


if __name__ == "__main__":
    unittest.main()
