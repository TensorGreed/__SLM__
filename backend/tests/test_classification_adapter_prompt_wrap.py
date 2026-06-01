"""β-fix tests — classification-label adapter writes the production
prompt format into training rows.

Pre-β: the adapter wrote raw ``source_text = text`` and ``target_text
= label``. ClassificationHandler at eval time wrapped inputs with
``"Classify the following text. Reply with exactly one of: A, B.\\n
Text: …\\nLabel:"`` — a format the model had never seen, so it
produced unparseable completions.

Post-β (this commit's tests pin):

  1. ``_map_classification`` wraps ``source_text`` with the same
     instruction template ClassificationHandler builds. Candidates
     in ``adapter_config['candidates']`` are inlined when present
     (≤ cap); otherwise the no-list fallback variant is used.
  2. ``target_text`` carries a leading space (`" benign"`) so the
     tokenizer decodes the label as a clean BPE continuation of
     the prompt's trailing `Label:`.
  3. ``text`` + ``label`` + ``answer`` stay raw so downstream
     surfaces (data health, gold diagnostics, smoke test row peek)
     can introspect classification labels without parsing the
     wrapped prompt back apart.
  4. ``_normalize_rows_for_training`` pre-scans rows for unique
     labels and injects them into adapter_config so the per-row
     map step gets the candidate list automatically.
  5. The handler's expected prefix (``"Classify the following
     text"``) appears in every wrapped row — the smoke check's
     γ′ peek now reports ``match`` instead of ``mismatch``.
"""

from __future__ import annotations

import unittest

from app.services.data_adapter_service import (
    _CLASSIFICATION_LABEL_LIST_PROMPT_CAP,
    _build_classification_training_prompt,
    _map_classification,
)
from app.services.dataset_service import _scan_classification_labels


class ClassificationAdapterWrapTests(unittest.TestCase):
    def test_wrap_with_candidates_renders_list_in_prompt_form(self):
        out = _map_classification(
            {"text": "this is a test", "label": "injection"},
            {"candidates": ["benign", "injection"]},
        )
        assert out is not None
        # Matches the eval handler's list-in-prompt variant.
        self.assertIn("Classify the following text", out["source_text"])
        self.assertIn("Reply with exactly one of: benign, injection", out["source_text"])
        self.assertIn("Text: this is a test", out["source_text"])
        self.assertTrue(out["source_text"].endswith("Label:"))

    def test_wrap_without_candidates_renders_no_list_fallback(self):
        out = _map_classification(
            {"text": "simple", "label": "x"},
            {},
        )
        assert out is not None
        self.assertIn("Classify the following text", out["source_text"])
        self.assertIn("Reply with just the class label", out["source_text"])
        self.assertNotIn("Reply with exactly one of:", out["source_text"])

    def test_target_text_has_leading_space_for_clean_bpe_continuation(self):
        """The leading space matters — `" benign"` tokenizes to a single
        clean token in most BPE vocabs, while `"benign"` adjacent to
        the colon in `Label:` tokenizes differently. The trainer's
        target sees a single boundary the eval can match exactly."""
        out = _map_classification(
            {"text": "x", "label": "benign"},
            {"candidates": ["benign", "injection"]},
        )
        assert out is not None
        self.assertEqual(out["target_text"], " benign")

    def test_text_and_label_fields_stay_raw(self):
        """Downstream surfaces (data health, gold diagnostics, smoke
        check row peek) need raw ``text``/``label`` to do their jobs.
        The wrapping happens only in source_text/target_text."""
        out = _map_classification(
            {"text": "raw input here", "label": "spam"},
            {},
        )
        assert out is not None
        self.assertEqual(out["text"], "raw input here")
        self.assertEqual(out["label"], "spam")
        self.assertEqual(out["answer"], "spam")

    def test_oversize_candidate_set_falls_back_to_no_list_form(self):
        """When candidates > cap, the prompt format must match
        ClassificationHandler's no-list fallback (which it switches
        to under the same condition). Trainer and eval need to agree
        on the format under both branches."""
        big = [f"label_{i}" for i in range(_CLASSIFICATION_LABEL_LIST_PROMPT_CAP + 5)]
        out = _map_classification(
            {"text": "x", "label": "label_0"},
            {"candidates": big},
        )
        assert out is not None
        self.assertIn("Reply with just the class label", out["source_text"])

    def test_record_with_only_text_returns_none(self):
        """Unchanged from pre-β: rows without a recoverable label
        return None so the trainer skips them."""
        self.assertIsNone(_map_classification({"text": "x"}, {}))


class ClassificationLabelScanTests(unittest.TestCase):
    """The pre-scan inside ``_normalize_rows_for_training`` — finds
    the candidate set without the caller having to compute it."""

    def test_scan_returns_sorted_deduped_label_set(self):
        rows = [
            {"text": "a", "label": "injection"},
            {"text": "b", "label": "benign"},
            {"text": "c", "label": "injection"},
            {"text": "d", "label": "benign"},
        ]
        out = _scan_classification_labels(rows, field_mapping=None, adapter_config=None)
        self.assertEqual(out, ["benign", "injection"])

    def test_scan_returns_none_when_no_labels_present(self):
        out = _scan_classification_labels(
            [{"text": "a"}, {"text": "b"}],
            field_mapping=None,
            adapter_config=None,
        )
        self.assertIsNone(out)

    def test_scan_respects_label_fields_override(self):
        out = _scan_classification_labels(
            [{"text": "a", "custom_tag": "good"}, {"text": "b", "custom_tag": "bad"}],
            field_mapping=None,
            adapter_config={"label_fields": ["custom_tag"]},
        )
        self.assertEqual(out, ["bad", "good"])

    def test_scan_bails_out_for_huge_label_vocabularies(self):
        """If we've already seen too many distinct labels there's no
        point inlining a list in the prompt — fall back to the
        no-list variant by returning None."""
        rows = [
            {"text": str(i), "label": f"label_{i}"} for i in range(100)
        ]
        out = _scan_classification_labels(rows, field_mapping=None, adapter_config=None)
        self.assertIsNone(out)


class ClassificationHandlerSmokeCompatibilityTests(unittest.TestCase):
    """The whole point of β: trainer + eval produce identical prompts.
    Verify by building both and asserting the structural overlap."""

    def test_adapter_prompt_matches_handler_prompt_prefix(self):
        """ClassificationHandler.expected_prompt_prefixes (γ′-fix)
        returns ["Classify the following text"]. Every wrapped row
        must contain this prefix, or the γ′ smoke check rightly
        flags a train/eval mismatch."""
        from app.services.eval_task_handler_service import ClassificationHandler

        out = _map_classification(
            {"text": "abc", "label": "benign"},
            {"candidates": ["benign", "injection"]},
        )
        assert out is not None
        for prefix in ClassificationHandler().expected_prompt_prefixes():
            self.assertIn(prefix, out["source_text"], f"prefix {prefix!r}")

    def test_adapter_prompt_matches_handler_build_prompt_text_byte_for_byte(self):
        """The strongest possible contract: the adapter's training
        prompt and the handler's eval prompt are the *same string*
        modulo the trailing target_text. Without this property,
        even subtle differences (extra newline, different label
        joiner) would cause subtle generalization drift."""
        from app.services.eval_task_handler_service import ClassificationHandler

        candidates = ["benign", "injection"]
        text = "this is a test"
        adapter_prompt = _build_classification_training_prompt(text, candidates)
        handler_prompt = ClassificationHandler()._build_prompt_text(text, candidates)
        self.assertEqual(adapter_prompt, handler_prompt)

    def test_no_list_fallback_also_matches_handler(self):
        """Same byte-for-byte equality under the no-list branch (both
        sides fall back to it when the candidate set is empty or
        oversized)."""
        from app.services.eval_task_handler_service import ClassificationHandler

        text = "x"
        adapter_prompt = _build_classification_training_prompt(text, None)
        handler_prompt = ClassificationHandler()._build_prompt_text(text, [])
        self.assertEqual(adapter_prompt, handler_prompt)


if __name__ == "__main__":
    unittest.main()
