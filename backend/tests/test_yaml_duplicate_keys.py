"""The ``yaml_parse`` code check refuses duplicate mapping keys at any nesting level (PyYAML keeps the
last value silently, so a doubled key used to pass). Multi-document streams, anchors, ``<<`` merge keys
and template-expression keys still parse. Pytest-only; built-in check, no subprocess."""

from __future__ import annotations

import unittest

from app.services import code_check_service as svc


def check(code: str) -> dict:
    return svc.run_code_checks(code, language="yaml", check_ids=["yaml_parse"])


class YamlDuplicateKeyTests(unittest.TestCase):
    def assert_duplicate(self, code: str, key: str, line: int, first: int) -> None:
        res = check(code)
        self.assertIs(res["ok"], False, res)
        msg = res["results"][0]["messages"][0]
        self.assertIn(f"found duplicate key '{key}'", msg)
        self.assertTrue(msg.startswith(f"line {line}:"), msg)
        self.assertIn(f"first at line {first}", msg)

    def test_top_level_duplicate(self):
        self.assert_duplicate("name: a\nreplicas: 2\nname: b\n", "name", 3, 1)

    def test_nested_duplicate(self):
        code = (
            "kind: Deployment\nspec:\n  template:\n    spec:\n      containers:\n        - name: app\n"
            "          image: app:1\n          envFrom:\n            - configMapRef: {name: base}\n"
            "          envFrom:\n            - secretRef: {name: extra}\n"
        )
        self.assert_duplicate(code, "envFrom", 10, 8)

    def test_duplicate_in_a_flow_mapping_and_in_a_later_document(self):
        self.assert_duplicate("a: {x: 1, x: 2}\n", "x", 1, 1)
        self.assert_duplicate("a: 1\n---\nb: 1\nb: 2\n", "b", 4, 3)

    def test_same_key_in_sibling_mappings_and_documents_is_fine(self):
        self.assertIs(check("- name: a\n  value: 1\n- name: b\n  value: 2\n")["ok"], True)
        self.assertIs(check("a: {name: x}\nb: {name: y}\n")["ok"], True)

    def test_multi_document(self):
        self.assertIs(check("---\nkind: A\nname: one\n---\nkind: B\nname: two\n...\n")["ok"], True)

    def test_anchor_and_merge_key(self):
        merged = "base: &base {x: 1}\nchild:\n  <<: {x: 1, y: 2}\n  x: 3\n"
        self.assertIs(check(merged)["ok"], True)  # a key overriding a merged one is not a duplicate
        many = "child:\n  <<: [{a: 1}, {b: 2}]\n  c: 3\n"
        self.assertIs(check(many)["ok"], True)

    def test_aliases_stay_refused(self):
        # Unchanged guard (expansion bombs): an alias, so a ``<<: *base`` merge, is still refused.
        res = check("base: &base {x: 1}\nchild:\n  <<: *base\n")
        self.assertIs(res["ok"], False)
        self.assertIn("aliases are not allowed", res["results"][0]["messages"][0])

    def test_keys_of_different_types_are_different_keys(self):
        self.assertIs(check("1: a\n'1': b\n")["ok"], True)

    def test_template_branches_are_not_duplicates(self):
        helm = ("spec:\n{{- if .Values.ha }}\n  replicas: 3\n{{- else }}\n  replicas: 1\n{{- end }}\n")
        jinja = "server:\n{% if prod %}\n  port: 443\n{% else %}\n  port: 80\n{% endif %}\n"
        helm_range = "env:\n{{- range .Values.env }}\n  - name: {{ .name }}\n{{- end }}\n"
        for code in (helm, jinja, helm_range):
            self.assertIs(check(code)["ok"], True, code)
        # Templated text without control lines (inline expressions only) is still checked; so is text whose
        # only template-only lines are comments (no branch).
        self.assert_duplicate("image: {{ .Values.image }}\nimage: other\n", "image", 2, 1)
        self.assert_duplicate("{{- /* a comment */}}\nspec:\n  a: 1\n  a: 2\n", "a", 4, 3)
        self.assert_duplicate("{{- /* note */ -}}\nspec:\n  a: 1\n  a: 2\n", "a", 4, 3)
        self.assert_duplicate("{# jinja note #}\nspec:\n  a: 1\n  a: 2\n", "a", 4, 3)
        self.assert_duplicate("spec:\n  a: {{ .x }} {{/* c */}}\n  a: 2\n", "a", 3, 2)

    def test_keys_compare_by_source_text_not_yaml11_value(self):
        self.assertIs(check("1.0: a\n1: b\n")["ok"], True)
        self.assertIs(check("on:\n  push: {}\ntrue: x\n")["ok"], True)
        self.assert_duplicate("a: 1\n'a': 2\n", "a", 2, 1)  # quoting does not make a different string key
        self.assert_duplicate("1: a\n1: b\n", "1", 2, 1)

    def test_template_expression_keys_are_not_compared(self):
        self.assertIs(check("labels:\n  {{ .a }}: one\n  {{ .b }}: two\n")["ok"], True)


class CheckRevisionFingerprintTests(unittest.TestCase):
    """The rule changed what passes yaml_parse, so the scoring fingerprint must change for check selections
    that include it (an older cached baseline is not reused or paired) and stay put for the others."""

    def fp(self, checks):
        import asyncio

        from app.services.remediation_detector import remediation_scoring_fingerprint

        return asyncio.run(remediation_scoring_fingerprint(None, 0, {"remediation": {"checks": checks}}, {"properties": {}}))

    def test_revisions(self):
        self.assertEqual(svc.check_revisions(["yaml_parse", "python_ast", "json_parse"]), {"yaml_parse": 2})
        self.assertEqual(svc.check_revisions(["python_ast", "unknown_check"]), {})

    def test_fingerprint_names_the_revised_check_only(self):
        from unittest import mock

        from app.services.code_checks.builtin import YamlParse

        with_yaml = self.fp(["python_ast", "yaml_parse"])
        without_yaml = self.fp(["python_ast"])
        with mock.patch.object(YamlParse, "revision", 1):  # the check as it was before this rule
            old_with_yaml = self.fp(["python_ast", "yaml_parse"])
            old_without_yaml = self.fp(["python_ast"])
        self.assertNotEqual(with_yaml, old_with_yaml)
        self.assertEqual(without_yaml, old_without_yaml)

    def test_compare_explains_a_revised_check(self):
        from types import SimpleNamespace

        from app.services.compare_row_evidence_service import _scoring, compare_provenance

        def er(fp, revisions):
            metrics = {"scoring_mode": "remediation", "remediation_fingerprint": fp}
            if revisions:
                metrics["remediation_check_revisions"] = revisions
            return {"scoring": _scoring(SimpleNamespace(eval_type="exact_match", metrics=metrics, details={}))}

        diff = compare_provenance(er("old0000000000000", None), er("new0000000000000", {"yaml_parse": 2}))
        self.assertFalse(diff["comparable"])
        self.assertEqual(diff["differences"][0]["message"],
                         "built-in check revised (yaml_parse v2: duplicate keys now fail) — re-score the older run")
        other = compare_provenance(er("aaaa000000000000", {"yaml_parse": 2}), er("bbbb000000000000", {"yaml_parse": 2}))
        self.assertEqual(other["differences"][0]["message"], "different scoring settings (remediation fingerprint)")


if __name__ == "__main__":
    unittest.main()
