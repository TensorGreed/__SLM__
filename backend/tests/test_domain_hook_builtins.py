"""Unit tests for the Gap-#1/#2 slice 1 built-in domain hooks.

Slice 1 ships 5 new normalizers + 2 new evaluators so users can turn
the "configure a normalizer / evaluator" promise into a no-plugin-needed
knob. These tests pin the per-hook behaviour + catalog registration:

  * Each new normalizer rewrites the right fields, leaves others alone,
    and falls back gracefully when the canonical record is malformed.
  * The ``safe-cleanup`` bundle composes html-decode + whitespace-collapse
    in that order (not the reverse — order matters for entity expansion).
  * Each new evaluator emits the documented enrichment keys without
    blowing away the underlying metrics dict.
  * Catalog registration is consistent: every hook callable lives in
    its ``BUILTIN_*_HOOKS`` registry AND has a description entry in
    ``BUILTIN_HOOK_CATALOG``. A drift between the two would make the
    catalog endpoint lie about what's actually available.
"""

from __future__ import annotations

import unittest

from app.services.domain_hook_service import (
    BUILTIN_EVALUATOR_HOOKS,
    BUILTIN_HOOK_CATALOG,
    BUILTIN_NORMALIZER_HOOKS,
    BUILTIN_VALIDATOR_HOOKS,
    list_domain_hook_catalog,
)


# ─────────────────────────────────────────────────────────────────────
# New normalizers
# ─────────────────────────────────────────────────────────────────────


class WhitespaceCollapseNormalizerTests(unittest.TestCase):
    def test_collapses_runs_and_trims_default_fields(self):
        hook = BUILTIN_NORMALIZER_HOOKS["whitespace-collapse-normalizer"]
        out = hook(
            {},
            {"text": "  hello    world\n\n", "question": "  who?  ", "other": "  leave me  "},
            {},
        )
        assert out is not None
        self.assertEqual(out["text"], "hello world")
        self.assertEqual(out["question"], "who?")
        # ``other`` isn't in the default target field list — left alone.
        self.assertEqual(out["other"], "  leave me  ")

    def test_target_fields_config_lets_caller_scope_to_one_field(self):
        hook = BUILTIN_NORMALIZER_HOOKS["whitespace-collapse-normalizer"]
        out = hook(
            {},
            {"text": "  hello  world  ", "answer": "  not  touched  "},
            {"target_fields": ["text"]},
        )
        assert out is not None
        self.assertEqual(out["text"], "hello world")
        # ``answer`` was excluded from the target_fields override.
        self.assertEqual(out["answer"], "  not  touched  ")

    def test_non_dict_canonical_returns_none(self):
        hook = BUILTIN_NORMALIZER_HOOKS["whitespace-collapse-normalizer"]
        self.assertIsNone(hook({}, "not a dict", {}))  # type: ignore[arg-type]


class HtmlEntityDecodeNormalizerTests(unittest.TestCase):
    def test_decodes_named_and_numeric_entities(self):
        hook = BUILTIN_NORMALIZER_HOOKS["html-entity-decode-normalizer"]
        out = hook(
            {},
            {"text": "Salt &amp; pepper &nbsp;&#x2014; over &quot;easy&quot;"},
            {},
        )
        assert out is not None
        # Named + numeric entities decoded; collapsing whitespace is
        # NOT this hook's job (use safe-cleanup for the bundle).
        self.assertIn("Salt & pepper", out["text"])
        self.assertIn("—", out["text"])
        self.assertIn('"easy"', out["text"])

    def test_leaves_non_string_fields_alone(self):
        hook = BUILTIN_NORMALIZER_HOOKS["html-entity-decode-normalizer"]
        out = hook(
            {},
            {"text": "&amp;", "count": 42, "meta": {"nested": "&amp;"}},
            {},
        )
        assert out is not None
        self.assertEqual(out["text"], "&")
        self.assertEqual(out["count"], 42)
        # Nested dicts aren't recursed — this normalizer is shallow by
        # design. Plug in a custom one if you need deep rewrites.
        self.assertEqual(out["meta"], {"nested": "&amp;"})


class LowercaseTextNormalizerTests(unittest.TestCase):
    def test_lowercases_default_text_fields(self):
        hook = BUILTIN_NORMALIZER_HOOKS["lowercase-text-normalizer"]
        out = hook(
            {},
            {"text": "Mixed CASE", "answer": "ANSWER", "label": "Unchanged"},
            {},
        )
        assert out is not None
        self.assertEqual(out["text"], "mixed case")
        self.assertEqual(out["answer"], "answer")
        # ``label`` isn't in the default target field list.
        self.assertEqual(out["label"], "Unchanged")

    def test_target_fields_lets_caller_pick_label_only(self):
        # Classification recipes typically want labels lowercased
        # without case-folding the prose. The config lets them scope
        # to just that field.
        hook = BUILTIN_NORMALIZER_HOOKS["lowercase-text-normalizer"]
        out = hook(
            {},
            {"text": "Leave THIS Alone", "label": "Spam"},
            {"target_fields": ["label"]},
        )
        assert out is not None
        self.assertEqual(out["text"], "Leave THIS Alone")
        self.assertEqual(out["label"], "spam")


class CurrencyCanonicalNormalizerTests(unittest.TestCase):
    def test_uppercases_three_letter_currency_codes(self):
        hook = BUILTIN_NORMALIZER_HOOKS["currency-canonical-normalizer"]
        out = hook(
            {},
            {"text": "Charge in usd or eur"},
            {},
        )
        assert out is not None
        # USD + EUR uppercased; non-currency 3-letter strings (e.g.
        # 'and', 'the', 'foo') also get uppercased — the regex only
        # gates on the 3-letter shape. Documented tradeoff in the
        # docstring; users wanting stricter behaviour write a plugin.
        self.assertIn("USD", out["text"])
        self.assertIn("EUR", out["text"])

    def test_maps_symbols_to_iso_codes_when_enabled(self):
        hook = BUILTIN_NORMALIZER_HOOKS["currency-canonical-normalizer"]
        out = hook(
            {},
            {"text": "$100 or €85 — same as £75"},
            {},  # default: symbol_to_code=True
        )
        assert out is not None
        self.assertIn("USD100", out["text"])
        self.assertIn("EUR85", out["text"])
        self.assertIn("GBP75", out["text"])

    def test_symbol_to_code_false_leaves_symbols_alone(self):
        hook = BUILTIN_NORMALIZER_HOOKS["currency-canonical-normalizer"]
        out = hook(
            {},
            {"text": "$100 in usd"},
            {"symbol_to_code": False},
        )
        assert out is not None
        # Symbol untouched; code uppercased.
        self.assertIn("$100", out["text"])
        self.assertIn("USD", out["text"])


class SafeCleanupNormalizerTests(unittest.TestCase):
    def test_decodes_entities_then_collapses_whitespace(self):
        # Order matters: if whitespace-collapse runs first, ``&nbsp;``
        # stays literal (the ampersand isn't whitespace). Bundle runs
        # decode first, then collapse, so &nbsp; → space gets folded.
        hook = BUILTIN_NORMALIZER_HOOKS["safe-cleanup-normalizer"]
        out = hook(
            {},
            {"text": "Salt&nbsp;&amp;&nbsp;pepper\n\n  over  easy"},
            {},
        )
        assert out is not None
        self.assertEqual(out["text"], "Salt & pepper over easy")


# ─────────────────────────────────────────────────────────────────────
# New evaluators
# ─────────────────────────────────────────────────────────────────────


class MetricCoverageEvaluatorTests(unittest.TestCase):
    def test_tags_missing_and_present_metrics(self):
        hook = BUILTIN_EVALUATOR_HOOKS["metric-coverage-evaluator"]
        out = hook(
            "qa-sft",
            {"f1": 0.7, "exact_match": 0.5, "noise": "string-not-a-metric"},
            {"expected_metric_ids": ["f1", "exact_match", "llm_judge_pass_rate"]},
            {},
        )
        # Untouched original metrics preserved.
        self.assertEqual(out["f1"], 0.7)
        self.assertEqual(out["exact_match"], 0.5)
        # Coverage enrichment.
        self.assertEqual(out["expected_metric_count"], 3)
        self.assertEqual(out["present_metric_count"], 2)
        self.assertEqual(out["missing_metric_ids"], ["llm_judge_pass_rate"])
        self.assertAlmostEqual(out["metric_coverage"], 2 / 3, places=4)

    def test_no_op_when_expected_list_is_empty_or_missing(self):
        hook = BUILTIN_EVALUATOR_HOOKS["metric-coverage-evaluator"]
        # No config → enrichment skipped (no expectations to check).
        out = hook("qa-sft", {"f1": 0.7}, {}, {})
        self.assertEqual(out, {"f1": 0.7})
        # Empty list → same.
        out = hook("qa-sft", {"f1": 0.7}, {"expected_metric_ids": []}, {})
        self.assertEqual(out, {"f1": 0.7})


class ThresholdCountsEvaluatorTests(unittest.TestCase):
    def test_counts_metrics_above_and_below_default_threshold(self):
        hook = BUILTIN_EVALUATOR_HOOKS["threshold-counts-evaluator"]
        out = hook(
            "qa-sft",
            {"f1": 0.7, "exact_match": 0.4, "llm_judge_pass_rate": 0.8},
            {},  # default threshold=0.5
            {},
        )
        # Original metrics preserved.
        self.assertEqual(out["f1"], 0.7)
        # Counts: f1 + llm_judge_pass_rate above; exact_match below.
        self.assertEqual(out["threshold"], 0.5)
        self.assertEqual(out["metrics_above_threshold"], 2)
        self.assertEqual(out["metrics_below_threshold"], 1)

    def test_metric_ids_config_lets_caller_scope_to_subset(self):
        hook = BUILTIN_EVALUATOR_HOOKS["threshold-counts-evaluator"]
        out = hook(
            "qa-sft",
            {"f1": 0.7, "exact_match": 0.4, "llm_judge_pass_rate": 0.8},
            {"threshold": 0.5, "metric_ids": ["f1"]},
            {},
        )
        # Only ``f1`` evaluated → 1 above, 0 below.
        self.assertEqual(out["metrics_above_threshold"], 1)
        self.assertEqual(out["metrics_below_threshold"], 0)

    def test_ignores_non_numeric_values_silently(self):
        hook = BUILTIN_EVALUATOR_HOOKS["threshold-counts-evaluator"]
        # Boolean values are intentionally NOT counted (they'd otherwise
        # be picked up because bool is a subclass of int in Python).
        out = hook(
            "qa-sft",
            {"f1": 0.7, "ok": True, "label": "high"},
            {"threshold": 0.5},
            {},
        )
        # Only ``f1`` got counted.
        self.assertEqual(out["metrics_above_threshold"], 1)
        self.assertEqual(out["metrics_below_threshold"], 0)


# ─────────────────────────────────────────────────────────────────────
# Catalog consistency — every callable has a description
# ─────────────────────────────────────────────────────────────────────


class CatalogConsistencyTests(unittest.TestCase):
    def test_every_normalizer_has_a_catalog_description(self):
        # Drift between BUILTIN_NORMALIZER_HOOKS and
        # BUILTIN_HOOK_CATALOG["normalizers"] would make the catalog
        # endpoint lie about what's available. Pin them together.
        self.assertEqual(
            set(BUILTIN_NORMALIZER_HOOKS.keys()),
            set(BUILTIN_HOOK_CATALOG["normalizers"].keys()),
        )

    def test_every_evaluator_has_a_catalog_description(self):
        self.assertEqual(
            set(BUILTIN_EVALUATOR_HOOKS.keys()),
            set(BUILTIN_HOOK_CATALOG["evaluators"].keys()),
        )

    def test_every_validator_has_a_catalog_description(self):
        self.assertEqual(
            set(BUILTIN_VALIDATOR_HOOKS.keys()),
            set(BUILTIN_HOOK_CATALOG["validators"].keys()),
        )

    def test_new_slice_1_hooks_surface_in_list_catalog_endpoint(self):
        # The public catalog endpoint reads from BUILTIN_HOOK_CATALOG
        # and merges in plugin hooks at the top level. Smoke test: the
        # 5 new normalizer ids + 2 new evaluator ids all appear.
        catalog = list_domain_hook_catalog()
        normalizer_ids = set(catalog["normalizers"].keys())
        evaluator_ids = set(catalog["evaluators"].keys())
        for new_id in (
            "whitespace-collapse-normalizer",
            "html-entity-decode-normalizer",
            "lowercase-text-normalizer",
            "currency-canonical-normalizer",
            "safe-cleanup-normalizer",
        ):
            self.assertIn(new_id, normalizer_ids)
        for new_id in (
            "metric-coverage-evaluator",
            "threshold-counts-evaluator",
        ):
            self.assertIn(new_id, evaluator_ids)


if __name__ == "__main__":
    unittest.main()
