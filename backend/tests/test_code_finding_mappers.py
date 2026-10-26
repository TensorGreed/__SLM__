"""Findings-row mappers (code_label_to_findings, fix_pair, contrast_functions)
on a neutral toy taxonomy, the introspector's code hypotheses, and the
end-to-end import through the API (labels resolved to taxonomy ids)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import label_taxonomy_service as lts
from app.services.dataset_import import resolve_mapper
from app.services.dataset_import.introspector import detect_shape, sniff_columns
from app.services.dataset_import.mappers.fix_pair import changed_before_lines
from app.services.dataset_import.protocols import ImportContext, RejectedRow, TransformedRow

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


TOY_TAXONOMY = {
    "name": "toy",
    "entries": [
        {"id": "LEAK-1", "title": "Data leak", "aliases": ["leakage", "data exposure"], "crosswalk": {"stdA": ["A-12"]}},
        {"id": "LEAK-1.1", "title": "Leak in logs", "parent": "LEAK-1"},
        {"id": "KEY-2", "title": "Hard-coded key", "aliases": ["secret in code"]},
    ],
}

CODE = "def f(k):\n    log(k)\n    return k\n"


def index():
    return lts.TaxonomyIndex(lts.parse_taxonomy(TOY_TAXONOMY))


def run(mapper_id, rows, field_map=None, taxonomy="toy", source_id="jsonl"):
    extra = {"taxonomy": index()} if taxonomy == "toy" else {}
    ctx = ImportContext(project_id=0, project_task_profile=None, source_id=source_id, mapper_id=mapper_id,
                        locator=f"{source_id}:/x", extra=extra)
    out = list(resolve_mapper(mapper_id).transform(rows, field_map or {}, ctx=ctx))
    return ([o for o in out if isinstance(o, TransformedRow)], [o for o in out if isinstance(o, RejectedRow)])


def one(mapper_id, row, field_map=None, taxonomy="toy"):
    ok, bad = run(mapper_id, [row], field_map, taxonomy)
    return ok, bad


class CodeLabelTests(unittest.TestCase):
    M = "code_label_to_findings"

    def test_accepted_row_shape(self):
        ok, bad = one(self.M, {"id": 7, "code": CODE, "label": "Leakage", "path": "a/b.py"})
        self.assertEqual(bad, [])
        row = ok[0]
        self.assertEqual(row.row_key, "jsonl-7")
        p = row.payload
        self.assertEqual(p["row_kind"], "findings")
        self.assertEqual((p["language"], p["path"], p["start_line"], p["end_line"]), ("python", "a/b.py", 1, 3))
        self.assertEqual(p["labels_raw"], ["Leakage"])
        self.assertEqual(p["taxonomy_version"], index().version)
        self.assertEqual(p["findings_json"]["findings"], [
            {"label": "LEAK-1", "start_line": 1, "end_line": 3, "evidence": "def f(k):\nlog(k)\nreturn k", "attrs": {}},
        ])
        self.assertEqual(row.warnings, [fc_warn("lines_unknown")])
        for forbidden in ("id", "source", "status", "row_key"):
            self.assertNotIn(forbidden, p)

    def test_columns_fallback_and_language_column(self):
        ok, _ = one(self.M, {"func": CODE, "category": "KEY-2", "lang": "py"})
        self.assertEqual((ok[0].payload["language"], ok[0].payload["findings_json"]["findings"][0]["label"]), ("python", "KEY-2"))

    def test_rejections(self):
        cases = [
            ({"code": "", "label": "KEY-2"}, "empty_code"),
            ({"code": "   \n", "label": "KEY-2"}, "empty_code"),
            ({"code": CODE}, "missing_label"),
            ({"code": CODE, "label": "nope"}, "unknown_label"),
            ({"code": CODE, "label": "KEY-2", "line": 9}, "line_out_of_range"),
            ({"code": CODE, "label": "KEY-2", "line": "3-2"}, "line_out_of_range"),
            ({"code": CODE, "label": "KEY-2", "line": "abc"}, "line_out_of_range"),
            ({"code": CODE, "label": "KEY-2", "line": 0}, "line_out_of_range"),
            ({"__parse_error__": "boom"}, "parse_error"),
        ]
        for row, reason in cases:
            ok, bad = one(self.M, row)
            self.assertEqual((ok, [b.reason for b in bad]), ([], [reason]), row)
        self.assertIn("'nope'", one(self.M, {"code": CODE, "label": "nope"})[1][0].detail)
        _, bad = one(self.M, {"code": "x" * 30, "label": "KEY-2"}, {"max_code_chars": 10})
        self.assertEqual(bad[0].reason, "code_too_long")
        _, bad = one(self.M, {"code": CODE, "label": "KEY-2", "path": "a.go"}, {"languages": ["python", "java"]})
        self.assertEqual(bad[0].reason, "language_out_of_scope")
        ok, _ = one(self.M, {"code": CODE, "label": "KEY-2", "path": "a.py"}, {"languages": ["Python"]})
        self.assertEqual(len(ok), 1)
        _, bad = run(self.M, [{"code": CODE, "label": "KEY-2"}], {"max_code_chars": "zero"})
        self.assertEqual(bad[0].reason, "parse_error")

    def test_label_prefix_retry_and_partial_labels(self):
        ok, bad = one(self.M, {"code": CODE, "label": "1"}, {"label_prefix": "LEAK-"})
        self.assertEqual(ok[0].payload["findings_json"]["findings"][0]["label"], "LEAK-1")
        ok, bad = one(self.M, {"code": CODE, "label": "1"})
        self.assertEqual(bad[0].reason, "unknown_label")
        row = {"code": CODE, "label": "KEY-2, mystery"}
        self.assertEqual(one(self.M, row)[1][0].reason, "unknown_label")
        ok, bad = one(self.M, row, {"allow_partial_labels": True})
        self.assertEqual(bad, [])
        self.assertEqual([f["label"] for f in ok[0].payload["findings_json"]["findings"]], ["KEY-2"])
        self.assertTrue(any(w.startswith("unknown_labels_dropped") and "mystery" in w for w in ok[0].warnings))
        _, bad = one(self.M, {"code": CODE, "label": "mystery"}, {"allow_partial_labels": True})
        self.assertEqual(bad[0].reason, "unknown_label")

    def test_clean_labels(self):
        for raw in ("", "None", "clean", "SAFE", "benign", "0", "False", "no"):
            ok, bad = one(self.M, {"code": CODE, "label": raw})
            self.assertEqual((bad, ok[0].payload["findings_json"]["findings"]), ([], []), raw)
        ok, _ = one(self.M, {"code": CODE, "label": "KEY-2;none"})
        self.assertEqual([f["label"] for f in ok[0].payload["findings_json"]["findings"]], ["KEY-2"])
        ok, bad = one(self.M, {"code": CODE, "label": "ok"}, {"clean_labels": ["ok"]})
        self.assertEqual(ok[0].payload["findings_json"]["findings"], [])
        _, bad = one(self.M, {"code": CODE, "label": ""}, {"clean_labels": ["ok"]})
        self.assertEqual(bad[0].reason, "missing_label")

    def test_vulnerable_column(self):
        fm = {"vulnerable": "vuln"}
        ok, _ = one(self.M, {"code": CODE, "label": "KEY-2", "vuln": 0}, fm)
        self.assertEqual(ok[0].payload["findings_json"]["findings"], [], "falsy flag = clean regardless of label")
        ok, _ = one(self.M, {"code": CODE, "label": "KEY-2", "vuln": "true"}, fm)
        self.assertEqual(len(ok[0].payload["findings_json"]["findings"]), 1)
        _, bad = one(self.M, {"code": CODE, "label": "", "vuln": True}, fm)
        self.assertEqual(bad[0].reason, "missing_label")
        ok, _ = one(self.M, {"code": CODE, "vuln": "no"}, fm)
        self.assertEqual(ok[0].payload["findings_json"]["findings"], [])

    def test_lines_absolute_relative_ranges_lists(self):
        code = "a\nb\nc\nd\ne"
        fm = {"start_line": 100}
        ok, bad = one(self.M, {"code": code, "label": "KEY-2", "line": 102}, fm)
        f = ok[0].payload["findings_json"]["findings"][0]
        self.assertEqual((f["start_line"], f["end_line"], f["evidence"], ok[0].warnings), (102, 102, "c", []))
        self.assertEqual((ok[0].payload["start_line"], ok[0].payload["end_line"]), (100, 104))
        ok, _ = one(self.M, {"code": code, "label": "KEY-2", "line": "101-103"}, fm)
        f = ok[0].payload["findings_json"]["findings"][0]
        self.assertEqual((f["start_line"], f["end_line"], f["evidence"]), (101, 103, "b\nc\nd"))
        ok, _ = one(self.M, {"code": code, "label": "KEY-2", "line": 2, "end_line": 4}, {**fm, "lines_relative": True})
        f = ok[0].payload["findings_json"]["findings"][0]
        self.assertEqual((f["start_line"], f["end_line"]), (101, 103))
        ok, bad = one(self.M, {"code": code, "label": "KEY-2", "line": 1}, fm)
        self.assertEqual(bad[0].reason, "line_out_of_range", "absolute line 1 is before the chunk starts at 100")
        ok, bad = one(self.M, {"code": code, "label": "KEY-2|LEAK-1", "line": [100, "103-104"]}, fm)
        fs = ok[0].payload["findings_json"]["findings"]
        self.assertEqual([(f["label"], f["start_line"], f["end_line"]) for f in fs], [("KEY-2", 100, 100), ("LEAK-1", 103, 104)])
        ok, bad = one(self.M, {"code": code, "label": "KEY-2|LEAK-1", "line": "[100, 101]"}, fm)
        self.assertEqual(len(ok[0].payload["findings_json"]["findings"]), 2, "JSON-list string")
        _, bad = one(self.M, {"code": code, "label": "KEY-2|LEAK-1", "line": [100, 101, 102]}, fm)
        self.assertEqual(bad[0].reason, "line_count_mismatch")
        ok, _ = one(self.M, {"code": code, "label": "KEY-2|LEAK-1", "line": 101}, fm)
        self.assertEqual([f["start_line"] for f in ok[0].payload["findings_json"]["findings"]], [101, 101], "one span applies to every label")

    def test_start_line_column_and_attr_columns_and_evidence(self):
        ok, _ = one(self.M, {"code": "a\nb", "label": "KEY-2", "line": 11, "start": 10, "sev": "high", "why": "because"},
                    {"start_line": "start", "attr_columns": {"severity": "sev"}, "evidence": "why"})
        f = ok[0].payload["findings_json"]["findings"][0]
        self.assertEqual((f["start_line"], f["attrs"], f["evidence"]), (11, {"severity": "high"}, "because"))
        ok, _ = one(self.M, {"code": "a\n" + "x" * 500, "label": "KEY-2", "line": "1-2"})
        self.assertEqual(len(ok[0].payload["findings_json"]["findings"][0]["evidence"]), 400)

    def test_multi_label_list_and_separator(self):
        ok, _ = one(self.M, {"code": CODE, "labels": ["KEY-2", "Leakage", "key 2"]})
        self.assertEqual([f["label"] for f in ok[0].payload["findings_json"]["findings"]], ["KEY-2", "LEAK-1", "KEY-2"][:0] or ["KEY-2", "LEAK-1"])
        ok, _ = one(self.M, {"code": CODE, "label": '["KEY-2", "leakage"]'})
        self.assertEqual(len(ok[0].payload["findings_json"]["findings"]), 2)
        ok, _ = one(self.M, {"code": CODE, "label": "KEY-2 / leakage"}, {"label_separator": r"\s*/\s*"})
        self.assertEqual(len(ok[0].payload["findings_json"]["findings"]), 2)

    def test_no_taxonomy_keeps_raw_labels_with_warning(self):
        ok, bad = one(self.M, {"code": CODE, "label": " Whatever "}, taxonomy=None)
        self.assertEqual(bad, [])
        self.assertEqual(ok[0].payload["findings_json"]["findings"][0]["label"], "Whatever")
        self.assertIsNone(ok[0].payload["taxonomy_version"])
        self.assertIn("no_taxonomy", ok[0].warnings)
        ok, _ = one(self.M, {"code": CODE, "label": "none"}, taxonomy=None)
        self.assertIn("no_taxonomy", ok[0].warnings)


def fc_warn(name):
    return name


class FixPairTests(unittest.TestCase):
    M = "fix_pair"
    BEFORE = "def f(k):\n    log(k)\n    save(k)\n    return k"
    AFTER = "def f(k):\n    log(mask(k))\n    save(k)\n    return k"

    def test_default_findings_pos_neg(self):
        ok, bad = one(self.M, {"id": 3, "before": self.BEFORE, "after": self.AFTER, "label": "leakage", "path": "x/y.py"})
        self.assertEqual(bad, [])
        self.assertEqual([r.row_key for r in ok], ["jsonl-3-pos", "jsonl-3-neg"])
        pos, neg = ok[0].payload, ok[1].payload
        self.assertEqual(pos["code"], self.BEFORE)
        self.assertEqual(neg["code"], self.AFTER)
        self.assertEqual(neg["findings_json"], {"findings": []})
        f = pos["findings_json"]["findings"][0]
        self.assertEqual((f["label"], f["start_line"], f["end_line"], f["evidence"]), ("LEAK-1", 2, 2, "log(k)"))
        self.assertEqual(pos["taxonomy_version"], index().version)
        self.assertEqual(pos["labels_raw"], ["leakage"])
        self.assertEqual(neg["labels_raw"], [])

    def test_diff_lines_change_deletion_insertion(self):
        self.assertEqual(changed_before_lines("a\nb\nc", "a\nB\nc"), (2, 2))
        self.assertEqual(changed_before_lines("a\nb\nc\nd", "a\nd"), (2, 3), "deleted lines")
        self.assertEqual(changed_before_lines("a\nb\nc", "a\nb\nNEW\nc"), (2, 2), "pure insertion anchors on the line before")
        self.assertEqual(changed_before_lines("a\nb", "NEW\na\nb"), (1, 1), "insertion at the top clamps to line 1")
        self.assertEqual(changed_before_lines("a\nb", "a\nb\nNEW"), (2, 2), "insertion at the end")
        self.assertEqual(changed_before_lines("a\nb\nc\nd\ne", "A\nb\nc\nd\nE"), (1, 5), "several hunks: first to last")
        self.assertIsNone(changed_before_lines("a", "a"))

    def test_start_line_offsets_finding_lines_and_whole_mode(self):
        ok, _ = one(self.M, {"before": self.BEFORE, "after": self.AFTER, "label": "KEY-2", "start_line": 40})
        f = ok[0].payload["findings_json"]["findings"][0]
        self.assertEqual((f["start_line"], f["end_line"]), (41, 41))
        ok, _ = one(self.M, {"before": self.BEFORE, "after": self.AFTER, "label": "KEY-2", "start_line": 40}, {"finding_lines": "whole"})
        f = ok[0].payload["findings_json"]["findings"][0]
        self.assertEqual((f["start_line"], f["end_line"]), (40, 43))
        _, bad = one(self.M, {"before": "a", "after": "b", "label": "KEY-2"}, {"finding_lines": "some"})
        self.assertEqual(bad[0].reason, "parse_error")

    def test_multi_label_one_finding_each(self):
        ok, _ = one(self.M, {"before": self.BEFORE, "after": self.AFTER, "label": "KEY-2,leakage"})
        self.assertEqual([f["label"] for f in ok[0].payload["findings_json"]["findings"]], ["KEY-2", "LEAK-1"])

    def test_patch_and_preference_rows(self):
        ok, bad = one(self.M, {"id": 1, "before": self.BEFORE, "after": self.AFTER, "label": "leakage", "path": "src/m.py"},
                      {"emit": ["patch", "preference"]})
        self.assertEqual(bad, [])
        self.assertEqual([r.row_key for r in ok], ["jsonl-1-patch", "jsonl-1-pref"])
        patch, pref = ok[0].payload, ok[1].payload
        self.assertEqual((patch["row_kind"], patch["code"], patch["fixed_code"], patch["labels"], patch["path"]),
                         ("patch", self.BEFORE, self.AFTER, ["LEAK-1"], "src/m.py"))
        self.assertTrue(patch["patch"].startswith("--- a/src/m.py\n+++ b/src/m.py\n@@"))
        self.assertIn("-    log(k)\n+    log(mask(k))", patch["patch"])
        self.assertEqual(pref["row_kind"], "preference")
        self.assertEqual(pref["prompt"], f"Fix the Data leak issue in this python code:\n\n{self.BEFORE}")
        self.assertEqual((pref["chosen"], pref["rejected"], pref["reference"], pref["labels"]), (self.AFTER, self.BEFORE, self.AFTER, ["LEAK-1"]))
        ok, _ = one(self.M, {"before": self.BEFORE, "after": self.AFTER, "label": "leakage"}, {"emit": ["findings", "patch", "preference"]})
        self.assertEqual([r.row_key for r in ok], ["jsonl-0-pos", "jsonl-0-neg", "jsonl-0-patch", "jsonl-0-pref"], "row index when there is no id")
        self.assertIn("--- a/code", ok[2].payload["patch"])
        self.assertTrue(ok[3].payload["prompt"].startswith("Fix the Data leak issue in this code:"))

    def test_row_keys_unique_across_rows(self):
        rows = [{"before": self.BEFORE, "after": self.AFTER, "label": "KEY-2"} for _ in range(3)]
        ok, _ = run(self.M, rows, {"emit": ["findings", "patch", "preference"]})
        keys = [r.row_key for r in ok]
        self.assertEqual(len(keys), len(set(keys)), keys)

    def test_rejections(self):
        base = {"before": self.BEFORE, "after": self.AFTER, "label": "KEY-2"}
        cases = [
            ({**base, "after": self.BEFORE.replace("\n", "  \n")}, "identical_pair"),
            ({**base, "after": ""}, "empty_code"),
            ({**base, "before": ""}, "empty_code"),
            ({**base, "label": "nope"}, "unknown_label"),
            ({k: v for k, v in base.items() if k != "label"}, "missing_label"),
            ({**base, "label": "none"}, "missing_label"),
            ({**base, "before": "x" * 50}, "code_too_long"),
            ({**base, "language": "go"}, "language_out_of_scope"),
        ]
        for row, reason in cases:
            ok, bad = one(self.M, row, {"max_code_chars": 40} if reason == "code_too_long" else {"languages": ["python"]} if reason == "language_out_of_scope" else None)
            self.assertEqual([b.reason for b in bad], [reason], (row, reason))
        _, bad = one(self.M, base, {"emit": ["bogus"]})
        self.assertEqual(bad[0].reason, "parse_error")

    def test_fallback_columns(self):
        ok, _ = one(self.M, {"func_before": self.BEFORE, "func_after": self.AFTER, "label": "KEY-2"})
        self.assertEqual(len(ok), 2)
        ok, _ = one(self.M, {"vulnerable_code": self.BEFORE, "fixed_code": self.AFTER, "label": "KEY-2"})
        self.assertEqual(len(ok), 2)

    def test_no_taxonomy(self):
        ok, _ = one(self.M, {"before": self.BEFORE, "after": self.AFTER, "label": "RawLabel"}, {"emit": ["findings", "preference"]}, taxonomy=None)
        self.assertEqual(ok[0].payload["findings_json"]["findings"][0]["label"], "RawLabel")
        self.assertIn("no_taxonomy", ok[0].warnings)
        self.assertIn("RawLabel issue", ok[2].payload["prompt"])


JAVA_SUITE = """package toy;

public class CaseA {
    private int shared;

    public void badLeak(String key) {
        System.out.println(key);
    }

    public void goodLeak(String key) {
        System.out.println(mask(key));
    }

    private void helper() {
        shared++;
    }
}
"""

PY_SUITE = """import os

def bad_key():
    key = "abc123"
    return key

def good_key():
    return os.environ["KEY"]

def helper():
    return 1
"""


class ContrastTests(unittest.TestCase):
    M = "contrast_functions"

    def test_java_bad_good_and_helper_ignored(self):
        ok, bad = one(self.M, {"id": 1, "path": "suite/LEAK-1/CaseA.java", "code": JAVA_SUITE, "label": "leakage"})
        self.assertEqual(bad, [])
        self.assertEqual([r.row_key for r in ok], ["jsonl-1-CaseA.badLeak", "jsonl-1-CaseA.goodLeak"])
        bad_row, good_row = ok
        p = bad_row.payload
        self.assertEqual((p["language"], p["start_line"], p["end_line"], p["path"]), ("java", 6, 8, "suite/LEAK-1/CaseA.java"))
        self.assertEqual(p["code"].splitlines()[0].strip(), "public void fn_1(String key) {")
        self.assertEqual([(f["label"], f["start_line"], f["end_line"]) for f in p["findings_json"]["findings"]], [("LEAK-1", 6, 8)])
        self.assertEqual(p["labels_raw"], ["leakage"])
        self.assertEqual((good_row.payload["findings_json"]["findings"], good_row.payload["start_line"]), ([], 10))
        self.assertTrue(any("ignored 1 other function" in w for w in bad_row.warnings))

    def test_python_and_label_from_path(self):
        ok, bad = one(self.M, {"path": "suite/KEY-2/case.py", "code": PY_SUITE}, {"label_from_path": r"/([A-Z]+-\d+)/"})
        self.assertEqual(bad, [])
        self.assertEqual([r.row_key for r in ok], ["jsonl-suite/KEY-2/case.py-bad_key", "jsonl-suite/KEY-2/case.py-good_key"])
        f = ok[0].payload["findings_json"]["findings"][0]
        self.assertEqual((f["label"], f["start_line"], f["end_line"]), ("KEY-2", 3, 5))
        self.assertEqual(ok[1].payload["start_line"], 7)

    def test_custom_patterns_and_no_variants(self):
        suite = "def vuln_a():\n    return 1\n\ndef fixed_a():\n    return 2\n"
        ok, _ = one(self.M, {"path": "a.py", "code": suite, "label": "KEY-2"}, {"bad_pattern": "^vuln", "good_pattern": "^fixed"})
        self.assertEqual(len(ok), 2)
        ok, bad = one(self.M, {"path": "a.py", "code": suite, "label": "KEY-2"})
        self.assertEqual(bad[0].reason, "no_variants")

    def test_rejections(self):
        base = {"path": "s/a.java", "code": JAVA_SUITE, "label": "KEY-2"}
        for row, reason, fm in [
            ({**base, "code": ""}, "empty_code", None),
            ({**base, "label": "nope"}, "unknown_label", None),
            ({k: v for k, v in base.items() if k != "label"}, "missing_label", None),
            ({**base}, "language_out_of_scope", {"languages": ["python"]}),
            ({**base}, "parse_error", {"bad_pattern": "("}),
            ({**base}, "parse_error", {"label_from_path": "nogroup"}),
            ({"__parse_error__": "x"}, "parse_error", None),
        ]:
            ok, bad = one(self.M, row, fm)
            self.assertEqual((ok, [b.reason for b in bad]), ([], [reason]), (reason, fm))
        _, bad = one(self.M, {**base, "path": "s/none.java"}, {"label": "label", "label_from_path": r"/(ZZ-\d)/"})
        self.assertEqual(bad, [])
        _, bad = one(self.M, {k: v for k, v in base.items() if k != "label"}, {"label_from_path": r"/(ZZ-\d)/"})
        self.assertEqual(bad[0].reason, "missing_label")
        ok, bad = one(self.M, base, {"max_code_chars": 60})
        self.assertEqual({b.reason for b in bad}, {"code_too_long"})

    def test_no_taxonomy_and_repeated_symbol_keys(self):
        dup = "class A {\n  void badX() {\n    a();\n  }\n  void badX(int n) {\n    b();\n  }\n}\n"
        ok, _ = one(self.M, {"path": "d.java", "code": dup, "label": "raw-label"}, taxonomy=None)
        self.assertEqual([r.row_key for r in ok], ["jsonl-d.java-A.badX", "jsonl-d.java-A.badX#2"])
        self.assertEqual(ok[0].payload["findings_json"]["findings"][0]["label"], "raw-label")
        self.assertIn("no_taxonomy", ok[0].warnings)


def run_with(mapper_id, rows, field_map=None, entries=None):
    """Like run() but over a custom list of taxonomy entries."""
    tax = lts.TaxonomyIndex(lts.parse_taxonomy({"entries": entries})) if entries else None
    ctx = ImportContext(project_id=0, project_task_profile=None, source_id="jsonl", mapper_id=mapper_id,
                        locator="jsonl:/x", extra={"taxonomy": tax} if tax else {})
    out = list(resolve_mapper(mapper_id).transform(rows, field_map or {}, ctx=ctx))
    return ([o for o in out if isinstance(o, TransformedRow)], [o for o in out if isinstance(o, RejectedRow)])


def spans(row):
    return [(f["label"], f["start_line"], f["end_line"]) for f in row.payload["findings_json"]["findings"]]


class RobustnessTests(unittest.TestCase):
    """NaN cells, odd rows, ambiguity, line endings, row keys."""

    NAN = float("nan")

    def test_nan_and_inf_cells(self):
        ok, bad = one("code_label_to_findings", {"code": self.NAN, "label": "LEAK-1"})
        self.assertEqual([b.reason for b in bad], ["empty_code"])
        ok, bad = one("code_label_to_findings", {"code": "a=1\nb=2", "label": self.NAN})
        self.assertEqual((len(ok), spans(ok[0])), (1, []), "a NaN label is blank, i.e. clean")
        for value in (self.NAN, float("inf"), float("-inf")):
            ok, bad = one("code_label_to_findings", {"code": "a=1\nb=2", "label": "LEAK-1", "line": value})
            if ok:  # NaN = no line info; inf is not a line number
                self.assertEqual((spans(ok[0]), ok[0].warnings), ([("LEAK-1", 1, 2)], ["lines_unknown"]))
            else:
                self.assertEqual(bad[0].reason, "line_out_of_range")
        ok, bad = one("code_label_to_findings", {"code": "a=1\nb=2", "label": "LEAK-1", "start_line": float("inf")})
        self.assertEqual([b.reason for b in bad], ["line_out_of_range"])
        ok, _ = one("code_label_to_findings", {"code": "a=1\nb=2", "label": "LEAK-1", "flag": self.NAN}, {"vulnerable": "flag"})
        self.assertEqual(len(spans(ok[0])), 1, "a NaN flag is blank: the label decides")
        ok, bad = one("fix_pair", {"before": self.NAN, "after": "x", "label": "KEY-2"})
        self.assertEqual([b.reason for b in bad], ["empty_code"])
        ok, bad = one("contrast_functions", {"path": "a.py", "code": self.NAN, "label": "KEY-2"})
        self.assertEqual([b.reason for b in bad], ["empty_code"])

    def test_unexpected_exception_rejects_the_row_not_the_import(self):
        from unittest import mock

        from app.services.dataset_import.mappers import _findings_common as fc
        rows = {
            "code_label_to_findings": {"code": CODE, "label": "KEY-2"},
            "fix_pair": {"before": "a\nb", "after": "a\nc", "label": "KEY-2"},
            "contrast_functions": {"path": "a.py", "code": PY_SUITE, "label": "KEY-2"},
        }
        for mapper_id, row in rows.items():
            with mock.patch.object(fc, "scrub", side_effect=RuntimeError("boom")):
                ok, bad = run(mapper_id, [row, {**row}])
            self.assertEqual((ok, [b.reason for b in bad]), ([], ["parse_error", "parse_error"]), mapper_id)
            self.assertIn("RuntimeError: boom", bad[0].detail)

    def test_ambiguous_label_is_rejected_with_candidates(self):
        entries = [{"id": "LEAK-11"}, {"id": "LEAK-1.1"}, {"id": "KEY-2"}]
        for mapper_id, row in (
            ("code_label_to_findings", {"code": CODE, "label": "LEAK 1.1"}),
            ("fix_pair", {"before": "a\nb", "after": "a\nc", "label": "LEAK 1.1"}),
            ("contrast_functions", {"path": "a.py", "code": PY_SUITE, "label": "LEAK 1.1"}),
        ):
            ok, bad = run_with(mapper_id, [row], entries=entries)
            self.assertEqual([b.reason for b in bad], ["ambiguous_label"], mapper_id)
            self.assertIn("LEAK-11", bad[0].detail)
            self.assertIn("LEAK-1.1", bad[0].detail)
        ok, bad = run_with("code_label_to_findings", [{"code": CODE, "label": "LEAK-1.1"}], entries=entries)
        self.assertEqual(spans(ok[0])[0][0], "LEAK-1.1", "an exact id is never ambiguous")

    def test_lone_cr_is_not_a_line_break_but_crlf_is(self):
        code = "x = 'a\rb'\r\ny\r\nz"
        ok, bad = one("code_label_to_findings", {"code": code, "label": "KEY-2", "line": 3})
        self.assertEqual(spans(ok[0]), [("KEY-2", 3, 3)])
        self.assertEqual(ok[0].payload["code"], "x = 'a\rb'\ny\nz")
        self.assertEqual(ok[0].payload["findings_json"]["findings"][0]["evidence"], "z")
        suite = 'class C {\n  void bad() { s = "a\rb"; }\n  void good() { s = "c"; }\n}\n'
        ok, bad = one("contrast_functions", {"path": "C.java", "code": suite, "label": "KEY-2"}, {"anonymize_symbols": False})
        self.assertEqual(bad, [])
        self.assertEqual([r.payload["start_line"] for r in ok], [2, 3], "lines are counted on \\n only")
        self.assertEqual(ok[0].payload["code"].count("\r"), 1, "a lone CR stays inside the code")
        crlf = suite.replace("\r", "").replace("\n", "\r\n")
        ok, _ = one("contrast_functions", {"path": "C.java", "code": crlf, "label": "KEY-2"}, {"anonymize_symbols": False})
        self.assertEqual([r.payload["start_line"] for r in ok], [2, 3])

    def test_row_keys_never_none_never_duplicated(self):
        rows = [{"id": 5, "code": CODE, "label": "KEY-2"}, {"id": 5, "code": CODE, "label": "KEY-2"},
                {"code": CODE, "label": "KEY-2"}, {"id": "", "code": CODE, "label": "KEY-2"}, {"id": None, "code": CODE, "label": "KEY-2"}]
        ok, _ = run("code_label_to_findings", rows)
        keys = [r.row_key for r in ok]
        self.assertEqual(keys, ["jsonl-5", "jsonl-5#2", "jsonl-2", "jsonl-3", "jsonl-4"])
        ok, _ = run("fix_pair", [{"id": 1, "before": "a\nb", "after": "a\nc", "label": "KEY-2"}] * 2, {"emit": ["findings", "patch"]})
        keys = [r.row_key for r in ok]
        self.assertEqual(len(keys), len(set(keys)), keys)
        self.assertTrue(all(keys))
        files = [{"path": "a.py", "code": PY_SUITE, "label": "KEY-2"}] * 2 + [{"code": PY_SUITE, "label": "KEY-2"}]
        ok, _ = run("contrast_functions", files)
        keys = [r.row_key for r in ok]
        self.assertEqual(len(keys), len(set(keys)), keys)
        self.assertTrue(all(keys))


class LineModeTests(unittest.TestCase):
    BEFORE = "\n".join(f"l{i}" for i in range(1, 13))

    def after(self, *changed):
        return "\n".join(f"L{i}" if i in changed else f"l{i}" for i in range(1, 13))

    def test_hunk_default_one_finding_per_hunk_and_close_hunks_merge(self):
        ok, _ = one("fix_pair", {"before": self.BEFORE, "after": self.after(2, 9), "label": "KEY-2", "start_line": 100})
        self.assertEqual(spans(ok[0]), [("KEY-2", 101, 101), ("KEY-2", 108, 108)])
        ok, _ = one("fix_pair", {"before": self.BEFORE, "after": self.after(2, 4), "label": "KEY-2"})
        self.assertEqual(spans(ok[0]), [("KEY-2", 2, 4)], "one unchanged line between: merged")
        ok, _ = one("fix_pair", {"before": self.BEFORE, "after": self.after(2, 5), "label": "KEY-2"})
        self.assertEqual(spans(ok[0]), [("KEY-2", 2, 5)], "two unchanged lines between: still merged")
        ok, _ = one("fix_pair", {"before": self.BEFORE, "after": self.after(2, 6), "label": "KEY-2"})
        self.assertEqual(spans(ok[0]), [("KEY-2", 2, 2), ("KEY-2", 6, 6)], "three unchanged lines between: apart")

    def test_hunk_two_labels_each_get_every_hunk(self):
        ok, _ = one("fix_pair", {"before": self.BEFORE, "after": self.after(2, 9), "label": "KEY-2, leakage"})
        self.assertEqual(spans(ok[0]), [("KEY-2", 2, 2), ("KEY-2", 9, 9), ("LEAK-1", 2, 2), ("LEAK-1", 9, 9)])

    def test_span_and_whole_modes(self):
        ok, _ = one("fix_pair", {"before": self.BEFORE, "after": self.after(2, 9), "label": "KEY-2"}, {"finding_lines": "span"})
        self.assertEqual(spans(ok[0]), [("KEY-2", 2, 9)])
        ok, _ = one("fix_pair", {"before": self.BEFORE, "after": self.after(2, 9), "label": "KEY-2"}, {"finding_lines": "whole"})
        self.assertEqual(spans(ok[0]), [("KEY-2", 1, 12)])
        _, bad = one("fix_pair", {"before": "a", "after": "b", "label": "KEY-2"}, {"finding_lines": "diff"})
        self.assertEqual(bad[0].reason, "parse_error")

    def test_indentation_only_change_is_not_identical(self):
        ok, bad = one("fix_pair", {"before": "if x:\n  y()\nz()", "after": "if x:\n  y()\n  z()", "label": "KEY-2"})
        self.assertEqual((bad, spans(ok[0])), ([], [("KEY-2", 3, 3)]))
        _, bad = one("fix_pair", {"before": "a\nb", "after": "a  \nb\t", "label": "KEY-2"})
        self.assertEqual([b.reason for b in bad], ["identical_pair"], "trailing whitespace is ignored")

    def test_one_label_several_spans_gives_one_finding_per_span(self):
        code = "a\nb\nc\nd\ne"
        ok, _ = one("code_label_to_findings", {"code": code, "label": "KEY-2", "line": [1, "3-4"]})
        self.assertEqual(spans(ok[0]), [("KEY-2", 1, 1), ("KEY-2", 3, 4)])
        ok, _ = one("code_label_to_findings", {"code": code, "label": "KEY-2", "line": [1, 3], "end_line": [2, 4]})
        self.assertEqual(spans(ok[0]), [("KEY-2", 1, 2), ("KEY-2", 3, 4)])
        ok, _ = one("code_label_to_findings", {"code": code, "label": "KEY-2|leakage", "line": 2})
        self.assertEqual(spans(ok[0]), [("KEY-2", 2, 2), ("LEAK-1", 2, 2)], "one span applies to every label")
        for label, line in (("KEY-2|leakage", [1, 2, 3]), ("KEY-2", [1, 2]) and ("KEY-2|leakage|LEAK-1.1", [1, 2])):
            _, bad = one("code_label_to_findings", {"code": code, "label": label, "line": line})
            self.assertEqual([b.reason for b in bad], ["line_count_mismatch"], (label, line))


class LeakScrubTests(unittest.TestCase):
    LEAKY = """public class Case {
    /* FLAW: the key is logged */
    public void badSink(String k) {
        log(k); // flaw here, incidental call below
        goodG2B(k);
    }

    private void goodG2B(String k) { /* good: masked */
        log(mask(k)); // the fix
    }
}
"""

    def test_contrast_anonymises_symbols_and_blanks_comments_by_default(self):
        ok, bad = one("contrast_functions", {"path": "Case.java", "code": self.LEAKY, "label": "KEY-2"})
        self.assertEqual(bad, [])
        text = "\n".join(r.payload["code"] for r in ok)
        for leak in ("badSink", "goodG2B", "FLAW", "flaw", "fix", "good:"):
            self.assertNotIn(leak, text)
        bad_row = ok[0]
        self.assertIn("fn_1(String k)", bad_row.payload["code"])
        self.assertIn("fn_2(k);", bad_row.payload["code"], "callees are renamed too, consistently within the row")
        self.assertEqual(bad_row.payload["code"].count("\n") + 1, bad_row.payload["end_line"] - bad_row.payload["start_line"] + 1, "line count unchanged")
        self.assertEqual(bad_row.payload["start_line"], 2, "the leading comment belongs to the unit")
        self.assertEqual(bad_row.payload["code"].split("\n")[0], "", "its text is blanked, the line stays")
        self.assertNotIn("badSink", bad_row.payload["findings_json"]["findings"][0]["evidence"], "evidence comes from the rewritten code")
        self.assertIn("log(k);", bad_row.payload["findings_json"]["findings"][0]["evidence"])
        self.assertIn("fn_1(String k)", ok[1].payload["code"], "numbering restarts per row")

    def test_contrast_options_can_switch_it_off_or_blank_all_comments(self):
        ok, _ = one("contrast_functions", {"path": "Case.java", "code": self.LEAKY, "label": "KEY-2"},
                    {"anonymize_symbols": False, "blank_comments": "none"})
        self.assertIn("badSink", ok[0].payload["code"])
        self.assertIn("flaw here", ok[0].payload["code"])
        ok, _ = one("contrast_functions", {"path": "Case.java", "code": self.LEAKY, "label": "KEY-2"},
                    {"anonymize_symbols": False, "blank_comments": "all", "comment_pattern": "zzz"})
        self.assertNotIn("flaw here", ok[0].payload["code"])
        ok, _ = one("contrast_functions", {"path": "Case.java", "code": "class C {\n  void bad1() {\n    s(); // note\n  }\n}\n", "label": "KEY-2"},
                    {"anonymize_symbols": False, "blank_comments": "matching"})
        self.assertIn("// note", ok[0].payload["code"], "a comment that does not match comment_pattern stays")

    def test_multi_line_block_comments_and_hash_comments(self):
        ok, _ = one("code_label_to_findings", {"code": "x = 1\n/* the flaw\n   continues */\ny = 2", "label": "KEY-2", "language": "c"},
                    {"blank_comments": "matching"})
        code = ok[0].payload["code"]
        self.assertEqual(code.split("\n"), ["x = 1", "", "", "y = 2"])
        ok, _ = one("code_label_to_findings", {"code": "# insecure on purpose\nurl = 'http://x/#frag'\nz = 1", "label": "KEY-2", "language": "python"},
                    {"blank_comments": "matching"})
        self.assertEqual(ok[0].payload["code"].split("\n"), ["", "url = 'http://x/#frag'", "z = 1"])
        ok, _ = one("code_label_to_findings", {"code": "#include <bad.h>\nint a;", "label": "KEY-2", "language": "c"}, {"blank_comments": "matching"})
        self.assertIn("#include <bad.h>", ok[0].payload["code"], "a preprocessor line is not a comment in C")

    def test_code_label_and_fix_pair_scrub_only_on_request(self):
        row = {"code": "def bad_fn(x):\n    # vulnerable\n    return good_fn(x)", "label": "KEY-2"}
        ok, _ = one("code_label_to_findings", row)
        self.assertIn("bad_fn", ok[0].payload["code"])
        ok, _ = one("code_label_to_findings", row, {"anonymize_symbols": True, "blank_comments": "matching"})
        self.assertEqual(ok[0].payload["code"].split("\n"), ["def fn_1(x):", "", "    return fn_2(x)"])
        self.assertEqual(ok[0].payload["findings_json"]["findings"][0]["evidence"], "def fn_1(x):\nreturn fn_2(x)")
        pair = {"before": "def bad_fn(x):\n    # flaw\n    return x", "after": "def bad_fn(x):\n    # fixed\n    return str(x)", "label": "KEY-2"}
        ok, _ = one("fix_pair", pair)
        self.assertIn("bad_fn", ok[0].payload["code"])
        ok, _ = one("fix_pair", pair, {"anonymize_symbols": True, "blank_comments": "matching"})
        pos, neg = ok[0].payload, ok[1].payload
        self.assertEqual(pos["code"].split("\n"), ["def fn_1(x):", "", "    return x"])
        self.assertEqual(neg["code"].split("\n"), ["def fn_1(x):", "", "    return str(x)"])
        self.assertEqual(spans(ok[0]), [("KEY-2", 3, 3)])
        _, bad = one("fix_pair", pair, {"blank_comments": "sometimes"})
        self.assertEqual(bad[0].reason, "parse_error")


class CliAndServiceTests(unittest.TestCase):
    def test_cli_preview_survives_a_taxonomy_load_failure(self):
        import contextlib
        import io
        from unittest import mock

        from app.cli.dataset_import import main

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps({"code": CODE, "label": "KEY-2"}) + "\n", encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with mock.patch("app.services.label_taxonomy_service.load_taxonomy_index", side_effect=RuntimeError("db down")), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(["preview", "--locator", f"jsonl:{path}", "--mapper", "code_label_to_findings", "--project", "1"])
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("accepted=1", out.getvalue())
        self.assertIn("could not load the label taxonomy", err.getvalue())
        self.assertIn("db down", err.getvalue())

    def test_warnings_are_listed_once_even_on_many_rows(self):
        from app.services.dataset_import.protocols import TransformedRow as TR
        from app.services.dataset_import.service import _drain_pipeline

        class FakeMapper:
            def transform(self, rows, field_map, *, ctx):
                for i in range(2000):
                    yield TR(payload={"i": i}, row_key=f"k{i}", warnings=["no_taxonomy", f"w{i % 3}"])

        class FakeSource:
            def load(self, rest, limit=None):
                return iter([{}])

        from unittest import mock
        ctx = ImportContext(project_id=0, project_task_profile=None, source_id="jsonl", mapper_id="x", locator="jsonl:/x")
        with mock.patch("app.services.dataset_import.service.resolve_source", return_value=FakeSource()), \
                mock.patch("app.services.dataset_import.service.resolve_mapper", return_value=FakeMapper()):
            accepted, _, _, warnings = _drain_pipeline(ctx=ctx, drop_reasons=None, sample_cap=5)
        self.assertEqual((len(accepted), warnings), (5, ["no_taxonomy", "w0", "w1", "w2"]))


class IntrospectorTests(unittest.TestCase):
    def ranked(self, rows):
        return detect_shape(sniff_columns(rows), rows)

    def test_fix_pair(self):
        rows = [{"func_before": "int f() {\n  return 1;\n}", "func_after": "int f() {\n  return 2;\n}", "label": "LEAK-1" if i % 2 else "KEY-2",
                 "message": "patched the parser to stop leaking the buffer"} for i in range(6)]
        hyps = self.ranked(rows)
        self.assertEqual(hyps[0].mapper_id, "fix_pair")
        self.assertAlmostEqual(hyps[0].confidence, 0.93)
        self.assertEqual(hyps[0].field_map, {"before": "func_before", "after": "func_after", "label": "label"})
        self.assertIn("func_before", hyps[0].rationale)
        lcl = [h for h in hyps if h.mapper_id == "label_to_classification"]
        self.assertTrue(all(h.confidence < hyps[0].confidence for h in lcl))

    def test_code_label_outranks_classification(self):
        rows = [{"code": "def f(x):\n    return eval(x)\n", "label": "LEAK-1" if i % 2 else "KEY-2", "note": "calls eval on the input argument directly"} for i in range(6)]
        hyps = self.ranked(rows)
        self.assertEqual(hyps[0].mapper_id, "code_label_to_findings")
        self.assertEqual(hyps[0].field_map["code"], "code")
        self.assertAlmostEqual(hyps[0].confidence, 0.92)
        ids = [h.mapper_id for h in hyps]
        self.assertLess(ids.index("code_label_to_findings"), ids.index("label_to_classification"))

    def test_code_detected_by_content_without_a_hint_name(self):
        rows = [{"body": "if (x) {\n  run();\n}\nreturn;\n", "kind": "a" if i % 2 else "b"} for i in range(6)]
        sigs = sniff_columns(rows)
        self.assertEqual(sigs["body"].column_type, "code_like")
        prose = [{"body": "A long sentence of plain words.\nAnother line of prose here.\n"} for _ in range(4)]
        self.assertNotEqual(sniff_columns(prose)["body"].column_type, "code_like")

    def test_contrast_functions(self):
        rows = [{"path": f"suite/case{i}.java", "code": JAVA_SUITE, "label": "KEY-2" if i % 2 else "LEAK-1"} for i in range(4)]
        hyps = self.ranked(rows)
        top = next(h for h in hyps if h.mapper_id == "contrast_functions")
        self.assertAlmostEqual(top.confidence, 0.9)
        self.assertEqual(top.field_map["code"], "code")
        self.assertNotIn("code_label_to_findings", [h.mapper_id for h in hyps], "contrast data is file-level, not labelled snippets")
        self.assertIn("'code'", top.rationale)

    def test_prose_is_not_code(self):
        """Prose with the odd ';', '{...}' or 'let me know' stays a text column."""
        clauses = [{"text": "(a) the Licensee shall keep records;\n(b) the Licensee shall not sublicense;\n(c) fees are payable monthly.",
                    "label": ["obligation", "payment"][i % 2]} for i in range(10)]
        tickets = [{"text": f"Hi team,\nlet me know when the refund for order {i} lands.\nlog says {{see log}};\nThanks", "label": ["billing", "bug"][i % 2]}
                   for i in range(10)]
        poems = [{"text": "Roses are red;\nviolets are blue;\nsugar is sweet;", "label": ["poem", "verse"][i % 2]} for i in range(10)]
        steps = [{"text": "Steps:\n  1. open the app\n  2. click save;\nresult: crash {see log}", "label": ["bug", "ux"][i % 2]} for i in range(10)]
        grammar = [{"before": "She go to school (every day).\nHe do too.", "after": "She goes to school (every day).\nHe does too.",
                    "type": ["verb", "agreement"][i % 2]} for i in range(10)]
        for name, rows in (("clauses", clauses), ("tickets", tickets), ("poems", poems), ("steps", steps), ("grammar", grammar)):
            sigs = sniff_columns(rows)
            self.assertNotIn("code_like", [s.column_type for s in sigs.values()], name)
            hyps = detect_shape(sigs, rows)
            self.assertEqual(hyps[0].mapper_id, "label_to_classification", name)
            self.assertNotIn("fix_pair", [h.mapper_id for h in hyps], name)

    def test_before_after_names_need_code_content(self):
        rows = [{"before": "if (x) {\n  run();\n}\nreturn 1;", "after": "if (x) {\n  run2();\n}\nreturn 1;", "type": "a" if i % 2 else "b"} for i in range(6)]
        self.assertEqual(self.ranked(rows)[0].mapper_id, "fix_pair")

    def test_code_column_still_offers_classification_below_the_code_readings(self):
        rows = [{"code": "def f(x):\n    return eval(x)\n", "label": "LEAK-1" if i % 2 else "KEY-2"} for i in range(6)]
        hyps = self.ranked(rows)
        lcl = next(h for h in hyps if h.mapper_id == "label_to_classification")
        self.assertEqual(lcl.field_map["text_field"], "code")
        self.assertLess(lcl.confidence, hyps[0].confidence)
        self.assertEqual(hyps[0].mapper_id, "code_label_to_findings")

    def test_label_fallback_skips_language_and_path_columns_and_finds_the_flag(self):
        rows = [{"func": "int f(char *s) {\n  strcpy(buf, s);\n  return 0;\n}", "language": "c" if i % 5 else "cpp",
                 "project": "alpha" if i % 3 else "beta", "target": i % 2, "kind": ["LEAK-1", "KEY-2"][i % 2]} for i in range(10)]
        top = self.ranked(rows)[0]
        self.assertEqual(top.mapper_id, "code_label_to_findings")
        self.assertEqual(top.field_map["label"], "kind")
        self.assertEqual(top.field_map["vulnerable"], "target")
        only_lang = [{"func": r["func"], "language": r["language"], "target": r["target"]} for r in rows]
        self.assertNotIn("code_label_to_findings", [h.mapper_id for h in self.ranked(only_lang)])

    def test_proposed_mappers_are_registered(self):
        from app.services.dataset_import import list_registered_mappers
        rows = [{"code": "def f(x):\n    return x\n", "label": "A" if i % 2 else "B"} for i in range(6)]
        for h in self.ranked(rows):
            self.assertIn(h.mapper_id, list_registered_mappers())


class AcceptanceTests(unittest.TestCase):
    """Taxonomy-resolved import through the API: contrast suite + fix commits."""

    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"findings-{time.time_ns()}", "description": "x"}).json()["id"]
        put = client.put(f"/api/projects/{self.pid}/label-taxonomy", json={"entries": TOY_TAXONOMY["entries"]})
        self.assertEqual(put.status_code, 200, put.text)
        self.version = put.json()["version"]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _jsonl(self, rows):
        path = Path(self.tmp.name) / f"rows-{time.time_ns()}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return f"jsonl:{path}"

    def _written(self, result):
        return [json.loads(line) for line in Path(result["written_path"]).read_text().splitlines() if line.strip()]

    def test_contrast_suite_import(self):
        locator = self._jsonl([
            {"id": 1, "path": "suite/LEAK-1/CaseA.java", "code": JAVA_SUITE},
            {"id": 2, "path": "suite/KEY-2/case_b.py", "code": PY_SUITE},
            {"id": 3, "path": "suite/ZZZ-9/CaseC.java", "code": JAVA_SUITE},
            {"id": 4, "path": "suite/KEY-2/helpers.py", "code": "def helper():\n    return 1\n"},
        ])
        body = {"locator": locator, "mapper_id": "contrast_functions",
                "field_map": {"label_from_path": r"/([A-Z]+-\d+)/"}}
        preview = client.post(f"/api/projects/{self.pid}/dataset-import/preview", json=body).json()
        self.assertEqual(preview["taxonomy"], {"version": self.version, "entry_count": 3})
        self.assertEqual(preview["rejection_counts"], {"unknown_label": 1, "no_variants": 1})
        self.assertEqual(preview["accepted_count"], 4)

        resp = client.post(f"/api/projects/{self.pid}/dataset-import/run", json={**body, "drop_reasons": ["unknown_label"]})
        self.assertEqual(resp.status_code, 200, resp.text)
        result = resp.json()
        self.assertEqual(result["taxonomy"], {"version": self.version, "entry_count": 3})
        self.assertEqual(result["accepted_count"], 4)
        self.assertEqual(result["rejection_counts"], {"unknown_label": 1, "no_variants": 1})
        self.assertEqual([r["reason"] for r in result["rejected_sample"]], ["no_variants"], "unknown_label bulk-dropped")
        rows = self._written(result)
        self.assertEqual(len(rows), 4)
        bads = [r for r in rows if r["findings_json"]["findings"]]
        self.assertEqual(sorted(f["label"] for r in bads for f in r["findings_json"]["findings"]), ["KEY-2", "LEAK-1"])
        for r in rows:
            self.assertEqual(r["row_kind"], "findings")
            self.assertEqual(r["taxonomy_version"], self.version)
            self.assertEqual(r["source"], "dataset_import")
            self.assertTrue(r["row_key"])
        self.assertEqual(len({r["row_key"] for r in rows}), 4)

    def test_fix_commit_import_with_alias_labels(self):
        before = "def f(k):\n    log(k)\n    return k"
        after = "def f(k):\n    log(mask(k))\n    return k"
        locator = self._jsonl([
            {"id": 10, "func_before": before, "func_after": after, "label": "Data Exposure", "path": "m.py"},
            {"id": 11, "func_before": before, "func_after": after, "label": "secret in code", "path": "n.py"},
            {"id": 12, "func_before": before, "func_after": after, "label": "mystery", "path": "o.py"},
            {"id": 13, "func_before": before, "func_after": before, "label": "KEY-2", "path": "p.py"},
        ])
        body = {"locator": locator, "mapper_id": "fix_pair", "field_map": {}}
        first = client.post(f"/api/projects/{self.pid}/dataset-import/run", json=body).json()
        self.assertEqual(first["accepted_count"], 4)
        self.assertEqual(first["rejection_counts"], {"unknown_label": 1, "identical_pair": 1})
        rows = self._written(first)
        pos = [r for r in rows if r["findings_json"]["findings"]]
        self.assertEqual(sorted(f["label"] for r in pos for f in r["findings_json"]["findings"]), ["KEY-2", "LEAK-1"])
        self.assertEqual({r["taxonomy_version"] for r in rows}, {self.version})
        self.assertEqual(sorted(r["row_key"] for r in rows), ["jsonl-10-neg", "jsonl-10-pos", "jsonl-11-neg", "jsonl-11-pos"])
        reasons = {r["reason"] for r in first["rejected_sample"]}
        self.assertEqual(reasons, {"unknown_label", "identical_pair"})
        dropped = client.post(f"/api/projects/{self.pid}/dataset-import/run",
                              json={**body, "drop_reasons": ["unknown_label", "identical_pair"]}).json()
        self.assertEqual(dropped["rejected_sample"], [])
        self.assertEqual(dropped["rejection_counts"], {"unknown_label": 1, "identical_pair": 1})

    def test_project_without_taxonomy(self):
        pid = client.post("/api/projects", json={"name": f"findings-none-{time.time_ns()}", "description": "x"}).json()["id"]
        locator = self._jsonl([{"code": CODE, "label": "anything"}])
        preview = client.post(f"/api/projects/{pid}/dataset-import/preview",
                              json={"locator": locator, "mapper_id": "code_label_to_findings"}).json()
        self.assertIsNone(preview["taxonomy"])
        self.assertEqual(preview["accepted_count"], 1)
        self.assertIn("no_taxonomy", preview["warnings"])

    def test_saved_config_run_uses_taxonomy(self):
        locator = self._jsonl([{"code": CODE, "label": "leakage"}])
        cfg = client.post(f"/api/projects/{self.pid}/dataset-import/configs", json={
            "name": "findings", "locator": locator, "mapper_id": "code_label_to_findings", "field_map": {}}).json()
        result = client.post(f"/api/projects/{self.pid}/dataset-import/configs/{cfg['id']}/run").json()
        self.assertEqual(result["taxonomy"]["version"], self.version)
        rows = self._written(result)
        self.assertEqual(rows[-1]["findings_json"]["findings"][0]["label"], "LEAK-1")


if __name__ == "__main__":
    unittest.main()
