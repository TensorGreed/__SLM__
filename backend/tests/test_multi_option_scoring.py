"""``multi_option`` scoring over canned verifier verdicts (pure; no subprocess, no DB), the prompt
never carrying the verifier payload, the checked-decoding shape verifier and its safety rule, and
defaults unchanged for every other mode. pytest-only."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import checked_decoding_service as cds
from app.services import multi_option_scoring as mos
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler

FORMATS = ["format_a", "format_b"]
TAXONOMY = ["guard-clause", "validate-at-boundary", "total-api"]
PAYLOAD = {"start": "def f(xs): return xs[0]", "hidden_assertion": "f([]) must not raise"}


def schema(**over) -> dict:
    raw = {"formats": FORMATS, "min_options": 2, "max_options": 3, "strategy_taxonomy": TAXONOMY,
           "verifier": {"check_id": "canned"}, "require_agreement": True, "row_pass": "top_option"}
    raw.update(over)
    cfg, errors = mos.normalize_config(raw)
    assert not errors, errors
    return {"fields": ["explanation", "options"], "required": ["options"], "scoring_mode": "multi_option",
            "multi_option": cfg, "verifier_label": {"id": "canned", "revision": 1}}


def opt(strategy: str, **fixes: str) -> dict:
    return {"strategy": strategy, "title": f"{strategy} fix", "rationale": "r",
            "fixes": fixes or {"format_a": "fix a", "format_b": "fix b"}}


def verdict(passes_a=True, passes_b=True, agree=True, option_passes=None) -> dict:
    v = {"formats": {
        "format_a": {"checks": {"parses": True, "behaviour_changed": passes_a}, "passes": passes_a, "messages": []},
        "format_b": {"checks": {"parses": True, "behaviour_changed": passes_b}, "passes": passes_b, "messages": []},
    }, "agreement": {"agree": agree, "differences": [] if agree else ["end states differ"]}}
    if option_passes is not None:
        v["option_passes"] = option_passes
    return v


# strategy -> what the canned verifier says
CANNED = {
    "guard-clause": verdict(),
    "validate-at-boundary": verdict(),
    "total-api": verdict(passes_b=False),
    "disagree": verdict(agree=False),
    "veto": verdict(option_passes=False),
    "no-agreement": {"formats": verdict()["formats"], "agreement": None},
}


def canned_verify(calls: list | None = None):
    def verify(payload, option, formats):
        if calls is not None:
            calls.append((payload, option, list(formats)))
        strategy = option.get("strategy")
        if strategy == "crash":
            return {"error": "exit 3: boom"}  # no error_kind: infrastructure (unverified)
        if strategy == "option-timeout":
            return {"error": "timed out after 5s", "error_kind": "option", "reason": "verifier_timeout"}
        if strategy == "option-crash":
            return {"error": "exit 3: boom", "error_kind": "option", "reason": "verifier_crash"}
        if strategy == "option-too-large":
            return {"error": "output was truncated", "error_kind": "option", "reason": "verifier_output_too_large"}
        if strategy == "option-garbage":
            return {"error": "stdout was not a JSON object", "error_kind": "option", "reason": "verifier_bad_output"}
        if strategy == "raise":
            raise RuntimeError("kaput")
        from app.services.external_checks import normalize_verdict

        return normalize_verdict(json.loads(json.dumps(CANNED.get(strategy, verdict()))), formats)

    return verify


def pred(*options, raw: str | None = None) -> dict:
    return {
        "prompt": "p", "reference": "",
        "prediction": raw if raw is not None else json.dumps({"explanation": "e", "options": list(options)}),
        "multi_option_row": {"context": "def f(xs): return xs[0]", "verifier_payload": PAYLOAD, "reference_options": None},
    }


def parse_json(value):
    return StructuredExtractionHandler()._parse_json_safely(value)


def score(predictions, sch=None, verify=None):
    return mos.score_multi_option(predictions, sch or schema(), parse_json, verify or canned_verify())


class ScoringTests(unittest.TestCase):
    def test_rates_over_rows_and_options(self):
        rows = [
            pred(opt("guard-clause"), opt("validate-at-boundary")),  # both pass, 2 distinct
            pred(opt("total-api"), opt("guard-clause")),  # top fails, 2nd passes
            pred(raw="not json"),  # malformed
        ]
        m = score(rows)
        self.assertEqual(m["scoring_mode"], "multi_option")
        self.assertEqual(m["total"], 3)
        self.assertEqual(m["multi_option_pass_rate"], round(1 / 3, 4))
        self.assertEqual(m["top_option_pass_rate"], round(1 / 3, 4))
        self.assertEqual(m["rows_with_passing_option_rate"], round(2 / 3, 4))
        self.assertEqual(m["all_options_pass_rate"], round(1 / 3, 4))
        self.assertEqual(m["option_pass_rate"], 0.75)  # 3 of 4 options
        self.assertEqual(m["options_scored"], 4)
        self.assertEqual(m["rows_with_two_distinct_passing_rate"], round(1 / 3, 4))
        self.assertEqual(m["distinct_strategies_mean"], 1.0)  # (2 + 1 + 0) / 3
        self.assertEqual(m["malformed_rate"], round(1 / 3, 4))
        self.assertEqual(m["option_count_in_range_rate"], round(2 / 3, 4))
        self.assertEqual(m["format_pass_rate"], {"format_a": 1.0, "format_b": 0.75})
        self.assertEqual(m["check_pass_rate"]["parses"], 1.0)
        self.assertEqual(m["cross_format_agreement_rate"], 1.0)
        self.assertEqual(m["out_of_taxonomy_rate"], 0.0)
        self.assertEqual(m["verifier_error_rows"], 0)
        self.assertEqual(m["multi_option"]["verifier"], {"id": "canned", "revision": 1})

        first, second, bad = rows
        self.assertTrue(first["row_multi_option_pass"])
        self.assertEqual(first["row_exact_match"], 1.0)
        self.assertEqual(first["row_options_passing"], 2)
        self.assertEqual(first["row_distinct_strategies"], 2)
        self.assertEqual([o["rank"] for o in first["row_options"]], [0, 1])
        self.assertEqual(first["row_options"][0]["fixes"], {"format_a": "fix a", "format_b": "fix b"})
        self.assertFalse(second["row_multi_option_pass"])
        self.assertTrue(second["row_any_option_pass"])
        self.assertFalse(second["row_options"][0]["formats"]["format_b"]["passes"])
        self.assertTrue(bad["row_malformed"])
        self.assertFalse(bad["row_multi_option_pass"])
        self.assertEqual(bad["row_option_count"], 0)

    def test_row_pass_modes(self):
        options = (opt("total-api"), opt("guard-clause"))
        for mode, expected in (("top_option", False), ("any_option", True), ("all_options", False)):
            row = pred(*options)
            score([row], schema(row_pass=mode))
            self.assertIs(row["row_multi_option_pass"], expected, mode)
        row = pred(opt("guard-clause"), opt("validate-at-boundary"))
        score([row], schema(row_pass="all_options"))
        self.assertTrue(row["row_multi_option_pass"])
        # all_options needs at least min_options options
        row = pred(opt("guard-clause"))
        score([row], schema(row_pass="all_options"))
        self.assertFalse(row["row_multi_option_pass"])

    def test_out_of_taxonomy_and_duplicate_strategies(self):
        row = pred(opt("guard-clause"), opt("Guard-Clause"), opt("rewrite-everything"))
        m = score([row])
        self.assertEqual(m["out_of_taxonomy_rate"], round(1 / 3, 4))
        self.assertIs(row["row_options"][2]["in_taxonomy"], False)
        # the out-of-taxonomy option passes the verifier but never counts as a distinct strategy;
        # the duplicate counts once.
        self.assertEqual(row["row_options_passing"], 3)
        self.assertEqual(row["row_distinct_strategies"], 1)
        self.assertEqual(m["rows_with_two_distinct_passing_rate"], 0.0)

    def test_no_taxonomy_counts_every_named_strategy(self):
        row = pred(opt("a"), opt("b"))
        m = score([row], schema(strategy_taxonomy=[]))
        self.assertNotIn("out_of_taxonomy_rate", m)
        self.assertEqual(row["row_distinct_strategies"], 2)

    def test_too_few_options_fail_every_row_rule(self):
        # min_options=2 and one (passing) option: the row fails under EVERY rule, not just all_options.
        for mode in mos.ROW_PASS_MODES:
            row = pred(opt("guard-clause"))
            m = score([row], schema(row_pass=mode))
            self.assertIs(row["row_multi_option_pass"], False, mode)
            self.assertEqual(m["multi_option_pass_rate"], 0.0, mode)
            self.assertEqual(m["option_count_in_range_rate"], 0.0)
            self.assertFalse(row["row_option_count_in_range"])
            self.assertTrue(row["row_options"][0]["passes"])  # the option itself still passes

    def test_too_many_options_are_reported_not_failed(self):
        row = pred(opt("guard-clause"), opt("validate-at-boundary"), opt("total-api"), opt("guard-clause"))
        m = score([row])
        self.assertEqual(m["option_count_in_range_rate"], 0.0)
        self.assertTrue(row["row_multi_option_pass"])  # top_option passes

    def test_agreement_required_vs_not(self):
        row = pred(opt("disagree"))
        score([row], schema(min_options=1))
        self.assertFalse(row["row_options"][0]["passes"])
        row = pred(opt("disagree"))
        m = score([row], schema(min_options=1, require_agreement=False))
        self.assertTrue(row["row_options"][0]["passes"])
        self.assertEqual(m["cross_format_agreement_rate"], 0.0)  # still reported
        # required but not reported = undecided: the option fails (a verifier verdict, not infrastructure)
        row = pred(opt("no-agreement"))
        m = score([row], schema(min_options=1))
        self.assertIs(row["row_options"][0]["passes"], False)
        self.assertEqual(row["row_options"][0]["failure_reason"], "verifier_undecided")
        self.assertEqual(m["verifier_error_rows"], 0)
        self.assertEqual(m["option_verifier_failures"], 1)

    def test_verifier_can_veto(self):
        row = pred(opt("veto"))
        score([row], schema(min_options=1))
        self.assertFalse(row["row_options"][0]["passes"])

    def test_missing_format_fails_without_asking_the_verifier_for_nothing(self):
        calls: list = []
        row = pred({"strategy": "guard-clause", "fixes": {}}, opt("validate-at-boundary", format_a="only a"))
        score([row], verify=canned_verify(calls))
        first = row["row_options"][0]
        self.assertFalse(first["passes"])
        self.assertEqual(first["formats"]["format_a"]["checks"], {"present": False})
        self.assertEqual(len(calls), 1)  # only the option that has a fix
        second = row["row_options"][1]
        self.assertFalse(second["formats"]["format_b"]["passes"])  # missing format_b
        self.assertFalse(second["passes"])

    def test_verifier_errors_are_excluded_and_noted_never_raised(self):
        rows = [pred(opt("crash"), opt("guard-clause")), pred(opt("raise"), opt("guard-clause")),
                pred(opt("guard-clause"), opt("total-api"))]
        m = score(rows)
        self.assertEqual(m["verifier_error_rows"], 2)
        self.assertEqual(m["total"], 1)
        self.assertEqual(m["multi_option_pass_rate"], 1.0)
        self.assertIn("could not be verified", m["verifier_error"])
        self.assertTrue(rows[0]["row_verifier_error"])
        self.assertIsNone(rows[0]["row_multi_option_pass"])
        self.assertIn("exit 3", rows[0]["row_verifier_note"])
        self.assertIn("RuntimeError", rows[1]["row_options"][0]["error"])

    def test_option_level_verifier_errors_fail_the_option_not_the_row(self):
        reasons = {"option-timeout": "verifier_timeout", "option-crash": "verifier_crash",
                   "option-too-large": "verifier_output_too_large", "option-garbage": "verifier_bad_output"}
        for strategy, reason in reasons.items():
            row = pred(opt(strategy), opt("guard-clause"))
            m = score([row], schema(strategy_taxonomy=[]))
            first = row["row_options"][0]
            self.assertIs(first["passes"], False, strategy)
            self.assertEqual(first["failure_reason"], reason)
            self.assertIsNone(first["error"])
            self.assertIs(row["row_multi_option_pass"], False)  # top option failed
            self.assertFalse(row["row_verifier_error"])
            self.assertEqual(row["row_option_verifier_failures"], 1)
            self.assertEqual(m["verifier_error_rows"], 0)
            self.assertEqual(m["option_verifier_failures"], 1)
            self.assertEqual(m["total"], 1)
            self.assertEqual(m["multi_option_pass_rate"], 0.0)
            self.assertEqual(row["row_exact_match"], 0.0)

    def test_top_option_is_decided_by_option_zero_alone(self):
        # An infrastructure error on option 1 never excludes a top_option row.
        rows = [pred(opt("guard-clause"), opt("crash")), pred(opt("total-api"), opt("crash"))]
        m = score(rows)
        self.assertEqual(m["verifier_error_rows"], 0)
        self.assertEqual(m["total"], 2)
        self.assertIs(rows[0]["row_multi_option_pass"], True)
        self.assertIs(rows[1]["row_multi_option_pass"], False)
        self.assertFalse(rows[0]["row_verifier_error"])
        self.assertIn("exit 3", rows[0]["row_verifier_note"])  # still noted
        # any_option: a passing option decides; all_options: a failing option decides
        row = pred(opt("crash"), opt("guard-clause"))
        score([row], schema(row_pass="any_option"))
        self.assertIs(row["row_multi_option_pass"], True)
        row = pred(opt("crash"), opt("total-api"))
        score([row], schema(row_pass="any_option"))
        self.assertIsNone(row["row_multi_option_pass"])  # undecidable without the unverified option
        row = pred(opt("crash"), opt("total-api"))
        score([row], schema(row_pass="all_options"))
        self.assertIs(row["row_multi_option_pass"], False)

    def test_no_formats_never_pass_vacuously(self):
        cfg, errors = mos.normalize_config({"formats": [], "verifier": {"check_id": "canned"}, "row_pass": "top_option"})
        self.assertEqual(errors, [])  # not enabled: allowed, but the scorer must not pass anything for free
        sch = {"fields": ["explanation", "options"], "required": ["options"], "scoring_mode": "multi_option",
               "multi_option": cfg}

        def null_verdict(payload, option, formats):
            from app.services.external_checks import normalize_verdict

            self.assertEqual(formats, [])
            return normalize_verdict({"formats": {}, "option_passes": None}, formats)

        rows = [pred(opt("a")), pred(opt("b"))]
        m = mos.score_multi_option(rows, sch, parse_json, null_verdict)
        self.assertNotEqual(m["multi_option_pass_rate"], 1.0)
        self.assertEqual(m["multi_option_pass_rate"], 0.0)
        self.assertEqual(rows[0]["row_options"][0]["failure_reason"], "verifier_undecided")
        # an explicit option_passes decides
        row = pred(opt("a"))
        mos.score_multi_option([row], sch, parse_json, lambda *_a: {"formats": {}, "agreement": None, "option_passes": True})
        self.assertIs(row["row_multi_option_pass"], True)

    def test_enabled_config_needs_a_format(self):
        _cfg, errors = mos.normalize_config({"enabled": True, "formats": []})
        self.assertTrue(any("at least one format" in e for e in errors), errors)
        _cfg, errors = mos.normalize_config({"enabled": True, "formats": ["a"]})
        self.assertEqual(errors, [])

    def test_excluded_rows_are_unscored_per_row_and_in_the_pairing(self):
        from app.services import evaluation_service as es

        rows = [pred(opt("crash"), opt("guard-clause")), pred(opt("guard-clause"), opt("total-api")),
                pred(opt("total-api"), opt("guard-clause"))]
        for i, r in enumerate(rows):
            r["prompt"] = f"p{i}"
        m = score(rows)
        self.assertEqual(m["verifier_error_rows"], 1)
        self.assertIsNone(rows[0]["row_exact_match"])
        scores = es._row_scores_for_pairing(rows)
        self.assertEqual(scores["correct"], [None, 1, 0])  # never 1 for an excluded row
        self.assertEqual(scores["multi_option_pass"], [None, 1.0, 0.0])
        self.assertFalse(es._prediction_failed(rows[0]))
        from app.services.paired_comparison_stats import pair_row_scores

        other = {"keys": scores["keys"], "correct": [1, 1, 1]}
        paired = pair_row_scores(scores, other, "correct")
        self.assertEqual(len(paired["before"]), 2)  # the excluded row is skipped

    def test_every_row_excluded_reports_not_scored(self):
        rows = [pred(opt("crash")), pred(opt("raise"))]
        m = score(rows, schema(min_options=1))
        self.assertEqual(m["verifier_error_rows"], 2)
        self.assertIsNone(m["multi_option_pass_rate"])
        self.assertIsNone(m["exact_match"])

    def test_preview_caps_fix_text_and_option_count(self):
        from app.services import evaluation_service as es

        long_fix = "x" * (mos.PREVIEW_FIX_CHARS + 500)
        options = [opt("guard-clause", format_a=long_fix, format_b="short")] + [opt(f"alt-{i}") for i in range(7)]
        row = pred(*options)
        score([row], schema(max_options=10, strategy_taxonomy=[]))
        self.assertEqual(len(row["row_options"]), 8)  # the scorer keeps everything
        preview = es._multi_option_preview(row)
        self.assertEqual(len(preview["row_options"]), mos.PREVIEW_MAX_OPTIONS)
        self.assertTrue(preview["row_options_truncated"])
        first = preview["row_options"][0]
        self.assertEqual(len(first["fixes"]["format_a"]), mos.PREVIEW_FIX_CHARS)
        self.assertEqual(first["fixes"]["format_b"], "short")
        self.assertEqual(first["fixes_truncated"], ["format_a"])
        self.assertNotIn("fixes_truncated", preview["row_options"][1])
        # the stored analysis is untouched
        self.assertEqual(len(row["row_options"][0]["fixes"]["format_a"]), len(long_fix))
        small = pred(opt("guard-clause"), opt("total-api"))
        score([small])
        self.assertNotIn("row_options_truncated", es._multi_option_preview(small))

    def test_no_verifier_means_every_row_unverified(self):
        rows = [pred(opt("guard-clause"))]
        m = mos.score_multi_option(rows, schema(min_options=1), parse_json, None)
        self.assertEqual(m["verifier_error_rows"], 1)
        self.assertEqual(rows[0]["row_options"][0]["error"], "the option verifier did not run")

    def test_verifier_sees_payload_option_and_formats(self):
        calls: list = []
        row = pred(opt("guard-clause"))
        score([row], verify=canned_verify(calls))
        payload, option, formats = calls[0]
        self.assertEqual(payload, PAYLOAD)
        self.assertEqual(option["strategy"], "guard-clause")
        self.assertEqual(formats, FORMATS)

    def test_analysis_is_cached(self):
        row = pred(opt("guard-clause"), opt("total-api"))
        mos.precompute([row], schema(), parse_json, canned_verify())

        def boom(*_a):
            raise AssertionError("re-verified")

        score([row], verify=boom)
        self.assertTrue(row["row_multi_option_pass"])

    def test_non_dict_option_and_skipped_rows(self):
        row = pred("just a string", opt("guard-clause"))
        other = {"prompt": "p", "reference": "", "prediction": "x"}
        m = score([row, other])
        self.assertTrue(row["row_options"][0]["malformed"])
        self.assertFalse(row["row_options"][0]["passes"])
        self.assertTrue(other["row_skipped"])
        self.assertEqual(m["skipped_non_multi_option_rows"], 1)

    def test_empty(self):
        m = score([])
        self.assertIsNone(m["multi_option_pass_rate"])  # not scored, never 0%
        self.assertIsNone(m["exact_match"])
        self.assertEqual(m["total"], 0)


class ConfigAndFingerprintTests(unittest.TestCase):
    def test_normalize_config(self):
        cfg, errors = mos.normalize_config(None)
        self.assertEqual(cfg, mos.DEFAULTS)
        self.assertEqual(errors, [])
        _cfg, errors = mos.normalize_config({"min_options": 4, "max_options": 2, "row_pass": "best", "formats": "a"})
        self.assertEqual(len(errors), 3)

    def test_a_project_cannot_supply_a_command_line(self):
        cfg, _ = mos.normalize_config({"verifier": {"check_id": "x", "argv": ["/bin/sh", "-c", "anything"]}})
        self.assertEqual(cfg["verifier"], {"check_id": "x"})

    def test_effective_config_overlays_the_manifest(self):
        manifest_schema = {"scoring_mode": "multi_option", "multi_option": {"formats": ["a"], "max_options": 2}}
        cfg = mos.effective_config({"multi_option": {"formats": ["b", "c"]}}, manifest_schema)
        self.assertEqual(cfg["formats"], ["b", "c"])
        self.assertEqual(cfg["max_options"], 2)
        self.assertTrue(mos.uses_multi_option(None, {"output_schema": manifest_schema}))
        self.assertFalse(mos.uses_multi_option({}, {"output_schema": {"scoring_mode": "field_match"}}))
        self.assertTrue(mos.uses_multi_option({"multi_option": {"enabled": True}}, {}))

    def test_fingerprint_is_stable_and_tracks_the_verifier_revision(self):
        cfg = schema()["multi_option"]
        with patch("app.services.external_checks.verifier_identity", return_value="canned:p1:r1"):
            a = mos.multi_option_fingerprint(cfg)
            self.assertEqual(a, mos.multi_option_fingerprint(dict(cfg)))
            self.assertNotEqual(a, mos.multi_option_fingerprint({**cfg, "row_pass": "any_option"}))
            self.assertNotEqual(a, mos.multi_option_fingerprint({**cfg, "strategy_taxonomy": TAXONOMY[:2]}))
        with patch("app.services.external_checks.verifier_identity", return_value="canned:p1:r2"):
            self.assertNotEqual(a, mos.multi_option_fingerprint(cfg))

    def test_fingerprint_tracks_the_instruction(self):
        cfg = schema()["multi_option"]
        with patch("app.services.external_checks.verifier_identity", return_value="canned:p1:r1"):
            a = mos.multi_option_fingerprint(cfg)
            b = mos.multi_option_fingerprint({**cfg, "instruction": "Propose three alternative plans."})
            self.assertNotEqual(a, b)
            # the default instruction, spelled out, is the same effective prompt
            self.assertEqual(a, mos.multi_option_fingerprint({**cfg, "instruction": mos.DEFAULT_INSTRUCTION}))


class PromptAndSafetyTests(unittest.TestCase):
    def handler(self, sch=None):
        h = StructuredExtractionHandler()
        manifest = {"output_schema": {"scoring_mode": "multi_option", "multi_option": (sch or schema())["multi_option"]}}
        ctx = EvalContext(project_id=1, experiment_id=1, eval_type="exact_match", dataset_name="test",
                          task_profile="structured_extraction", handler_id="structured_extraction",
                          prepared_dir=Path("."), manifest=manifest)
        return h, ctx

    def row(self) -> dict:
        return {"context": "def f(xs): return xs[0]", "verifier_payload": PAYLOAD,
                "options_json": {"explanation": "e", "options": [opt("guard-clause")]}}

    def test_prompt_renders_config_and_context_never_the_payload(self):
        h, ctx = self.handler()
        built = h.build_prompts([self.row()], ctx)[0]
        self.assertIn("def f(xs)", built.prompt)
        for name in FORMATS + TAXONOMY:
            self.assertIn(name, built.prompt)
        self.assertIn("between 2 and 3", built.prompt)
        self.assertNotIn("hidden_assertion", built.prompt)
        self.assertNotIn("must not raise", built.prompt)
        self.assertEqual(built.extras["multi_option_row"]["verifier_payload"], PAYLOAD)
        self.assertEqual(json.loads(built.reference)["options"][0]["strategy"], "guard-clause")
        self.assertFalse(h.wraps_own_prompt(ctx))
        self.assertGreaterEqual(h.max_new_tokens_override(256), 2048)

    def test_checked_decoding_never_reads_the_row(self):
        h, ctx = self.handler()
        built = h.build_prompts([self.row()], ctx)[0]
        pair = {"prompt": built.prompt, "reference": built.reference, **built.extras}
        spec = {"multi_option_shape": cds.output_option_shape(ctx.manifest)}
        vin = cds.build_verifier_input(pair, spec)
        flat = json.dumps({k: (dict(v) if hasattr(v, "keys") else v) for k, v in vin.__dict__.items()}, default=str)
        self.assertNotIn("hidden_assertion", flat)
        self.assertNotIn("guard-clause fix", flat)  # the reference options never reach a verifier
        self.assertIn("multi_option_shape", cds.applicable_verifiers(vin, cds.normalize_policy({"enabled": True})))

    def test_shape_verifier(self):
        h, ctx = self.handler()
        vin = cds.build_verifier_input({"prompt": "p"}, {"multi_option_shape": cds.output_option_shape(ctx.manifest)})
        ok = cds.verify_multi_option_shape(vin, json.dumps({"options": [opt("guard-clause"), opt("total-api")]}))
        self.assertTrue(ok["ok"], ok)
        bad = cds.verify_multi_option_shape(vin, json.dumps({"options": [opt("guard-clause"), opt("guard-clause", format_a="x"), opt("nope"), opt("total-api")]}))
        self.assertFalse(bad["ok"])
        text = " ".join(bad["messages"])
        self.assertIn("expected 2-3", text)
        self.assertIn("repeats strategy", text)
        self.assertIn("not one of the listed strategy ids", text)
        self.assertIn("no fix in: format_b", text)
        self.assertFalse(cds.verify_multi_option_shape(vin, "plain prose")["ok"])


class DefaultsUnchangedTests(unittest.TestCase):
    def test_other_modes_get_no_shape_verifier(self):
        for manifest in ({}, {"output_schema": {"scoring_mode": "remediation"}}, {"output_schema": {"properties": {"a": {}}}}):
            self.assertIsNone(cds.output_option_shape(manifest))
        vin = cds.build_verifier_input({"prompt": "p", "multi_option_row": {"verifier_payload": PAYLOAD}}, {})
        self.assertEqual(cds.applicable_verifiers(vin, cds.normalize_policy({"enabled": True})), [])

    def test_field_match_handler_is_untouched(self):
        h = StructuredExtractionHandler()
        ctx = EvalContext(project_id=1, experiment_id=1, eval_type="exact_match", dataset_name="test",
                          task_profile="structured_extraction", handler_id="structured_extraction",
                          prepared_dir=Path("."), manifest={"output_schema": {"properties": {"total": {"type": "number"}}}})
        built = h.build_prompts([{"text": "total 5", "reference": {"total": 5}}], ctx)[0]
        self.assertNotIn("multi_option_row", built.extras)
        m = h.score([{"prompt": built.prompt, "reference": built.reference, "prediction": '{"total": 5}'}], ctx)
        self.assertNotEqual(m.get("scoring_mode"), "multi_option")
        self.assertEqual(m["exact_match"], 1.0)

    def test_option_verifiers_never_join_the_default_check_selection(self):
        from app.services import code_check_service as ccs

        class Fake:
            id = "fake_option_verifier"
            kind = "option_verifier"
            languages = frozenset({"python"})
            binary = None
            default_enabled = True

            def available(self):
                return True

            def run(self, *a, **k):
                raise AssertionError("never run as a code check")

        with patch("app.config.settings.CODE_CHECK_ALLOWED", "fake_option_verifier"):
            self.assertFalse(ccs.in_default_selection(Fake()))


if __name__ == "__main__":
    unittest.main()
