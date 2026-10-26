"""Unified-diff parser / applier (pure)."""

import unittest

from app.services.unified_diff import DiffError, apply_patch_to_chunk, make_unified_diff, parse_unified_diff

CODE = "\n".join([
    "def lookup(db, name):",
    "    query = \"SELECT * FROM t WHERE n = '\" + name + \"'\"",
    "    rows = db.execute(query)",
    "    return rows",
    "",
    "def other():",
    "    return 1",
])
START = 10
PATCH = """--- a/svc/db.py
+++ b/svc/db.py
@@ -10,4 +10,4 @@
 def lookup(db, name):
-    query = "SELECT * FROM t WHERE n = '" + name + "'"
-    rows = db.execute(query)
+    query = "SELECT * FROM t WHERE n = ?"
+    rows = db.execute(query, (name,))
     return rows
"""


class ApplyTests(unittest.TestCase):
    def test_exact(self):
        r = apply_patch_to_chunk(CODE, START, PATCH, path="svc/db.py")
        self.assertTrue(r.ok, r.reason)
        self.assertIn("db.execute(query, (name,))", r.patched_code)
        self.assertEqual(r.offsets, [0])
        self.assertTrue(r.header_counts_ok)
        self.assertEqual(r.hunks, 1)

    def test_offset(self):
        r = apply_patch_to_chunk(CODE, START + 3, PATCH)  # patch states line 10, code really starts at 13
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.offsets, [3])  # applied 3 lines after the stated position
        # beyond max_offset fails
        r2 = apply_patch_to_chunk(CODE, START + 30, PATCH)
        self.assertFalse(r2.ok)
        self.assertEqual(r2.reason, "out_of_range")
        r3 = apply_patch_to_chunk(CODE, START + 3, PATCH, max_offset=1)
        self.assertEqual(r3.reason, "out_of_range")

    def test_fence_prose_and_no_headers(self):
        body = PATCH.split("+++ b/svc/db.py\n", 1)[1]
        text = "Here is the fix:\n```diff\n" + body + "```\nHope it helps"
        r = apply_patch_to_chunk(CODE, START, text, path="svc/db.py")
        self.assertTrue(r.ok, r.reason)

    def test_trailing_whitespace_ignored_but_nothing_else(self):
        code = CODE.replace("return rows", "return rows   ")
        self.assertTrue(apply_patch_to_chunk(code, START, PATCH).ok)
        r = apply_patch_to_chunk(CODE.replace("return rows", "return  rows"), START, PATCH)
        self.assertEqual(r.reason, "context_mismatch")

    def test_crlf(self):
        r = apply_patch_to_chunk(CODE.replace("\n", "\r\n"), START, PATCH.replace("\n", "\r\n"))
        self.assertTrue(r.ok, r.reason)
        self.assertIn("\r\n", r.patched_code)
        self.assertNotIn("\n\n", r.patched_code.replace("\r\n", ""))

    def test_no_newline_marker(self):
        patch = "@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+c\n\\ No newline at end of file\n"
        r = apply_patch_to_chunk("a\nb", 1, patch)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.patched_code, "a\nc")

    def test_overlapping_hunks(self):
        patch = "@@ -1,2 +1,2 @@\n a\n-b\n+B\n@@ -2,2 +2,2 @@\n b\n-c\n+C\n"
        r = apply_patch_to_chunk("a\nb\nc\nd", 1, patch)
        self.assertEqual(r.reason, "overlapping_hunks")

    def test_two_hunks(self):
        patch = "@@ -1,2 +1,2 @@\n a\n-b\n+B\n@@ -6,2 +6,2 @@\n f\n-g\n+G\n"
        code = "\n".join("abcdefgh")
        r = apply_patch_to_chunk(code, 1, patch)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.patched_code, "\n".join("aBcdefGh"))

    def test_other_file_and_multiple_files(self):
        self.assertEqual(apply_patch_to_chunk(CODE, START, PATCH, path="svc/other.py").reason, "other_file")
        self.assertTrue(apply_patch_to_chunk(CODE, START, PATCH, path="x/y/db.py").ok)  # basenames
        two = PATCH + PATCH.replace("db.py", "x.py")
        self.assertEqual(apply_patch_to_chunk(CODE, START, two).reason, "multiple_files")

    def test_empty_not_a_diff_noop(self):
        self.assertEqual(apply_patch_to_chunk(CODE, START, "  \n").reason, "empty_patch")
        self.assertEqual(apply_patch_to_chunk(CODE, START, "just some words").reason, "not_a_diff")
        r = apply_patch_to_chunk("a\nb", 1, "@@ -1,2 +1,2 @@\n a\n-b\n+b\n")
        self.assertEqual(r.reason, "no_op")

    def test_header_counts_wrong_tolerated(self):
        patch = "@@ -11,9 +11,9 @@\n     query = \"SELECT * FROM t WHERE n = '\" + name + \"'\"\n-    rows = db.execute(query)\n+    rows = db.run(query)\n"
        r = apply_patch_to_chunk(CODE, START, patch)
        self.assertTrue(r.ok, r.reason)
        self.assertFalse(r.header_counts_ok)
        # more body than the header says
        patch2 = "@@ -10,1 +10,1 @@\n def lookup(db, name):\n+    # note\n+    # more\n"
        r2 = apply_patch_to_chunk(CODE, START, patch2)
        self.assertTrue(r2.ok, r2.reason)
        self.assertFalse(r2.header_counts_ok)

    def test_out_of_range(self):
        patch = "@@ -90,2 +90,2 @@\n x\n-y\n+z\n"
        self.assertEqual(apply_patch_to_chunk(CODE, START, patch).reason, "out_of_range")

    def test_pure_insertion(self):
        patch = "@@ -11,0 +12,1 @@\n+    # inserted\n"
        r = apply_patch_to_chunk(CODE, START, patch)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.patched_code.split("\n")[2], "    # inserted")

    def test_parse_structure(self):
        fp = parse_unified_diff(PATCH)
        self.assertEqual(len(fp), 1)
        self.assertEqual((fp[0].old_path, fp[0].new_path), ("svc/db.py", "svc/db.py"))
        h = fp[0].hunks[0]
        self.assertEqual((h.old_start, h.old_len, h.new_start, h.new_len), (10, 4, 10, 4))
        self.assertEqual(h.lines[0], (" ", "def lookup(db, name):"))
        with self.assertRaises(DiffError) as cm:
            parse_unified_diff("")
        self.assertEqual(cm.exception.code, "empty_patch")


class RoundTripTests(unittest.TestCase):
    CASES = [
        ("a\nb\nc", "a\nB\nc"),
        ("a\nb\nc\n", "a\nb\nc\nd\n"),
        ("a\nb\nc", "b\nc"),
        ("a\nb\nc", "x\na\nb\nc\ny"),
        ("a\nb", "a\nb\n"),
        ("a\nb\n", "a\nb"),
        ("\n".join(str(i) for i in range(40)), "\n".join(str(i) if i not in (3, 30) else "X" for i in range(40))),
        ("", "a\nb"),
        ("a\nb\nc\nd\ne\nf\ng\nh\ni\nj", "a\nb\nc\nd\ne\nf\ng\nh\ni"),
    ]

    def test_round_trip(self):
        for old, new in self.CASES:
            for start in (1, 57):
                d = make_unified_diff(old, new, path="f.py", start_line=start)
                if old == new:
                    continue
                r = apply_patch_to_chunk(old, start, d, path="f.py")
                self.assertTrue(r.ok, (old, new, start, r.reason, d))
                self.assertEqual(r.patched_code, new, (old, new, d))

    def test_identical_is_empty_and_headers_absolute(self):
        self.assertEqual(make_unified_diff("a\nb", "a\nb", path="f"), "")
        d = make_unified_diff("a\nb\nc", "a\nB\nc", path="p/f.py", start_line=100, context=1)
        self.assertIn("--- a/p/f.py", d)
        self.assertIn("@@ -100,3 +100,3 @@", d)


if __name__ == "__main__":
    unittest.main()
