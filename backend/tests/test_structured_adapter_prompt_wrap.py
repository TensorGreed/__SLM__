"""ζ-fix tests — structured-extraction adapter writes the
production prompt format into training rows.

Pre-ζ: the adapter wrote raw ``source_text = input`` and
``target_text = json``. StructuredExtractionHandler at eval time
wrapped inputs with ``"Extract the following fields as JSON: …\\n
Reply with a single JSON object, nothing else.\\nInput: …\\n
Output:"`` — a format the model had never seen, so it produced
JSON-validity-rate + field-level EM/F1 artificially low (the same
shape of bug β surfaced for classification-label).

Post-ζ (this commit's tests pin):

  1. ``_build_structured_extraction_training_prompt`` and
     ``StructuredExtractionHandler._build_prompt_text`` produce
     IDENTICAL strings under both branches (with field list,
     and the no-list fallback).
  2. ``_map_structured_extraction`` writes ``source_text`` as the
     wrapped prompt and ``target_text`` as ``f" {json}"`` so the
     decoder treats the JSON as a clean continuation of ``Output:``.
  3. ``_normalize_rows_for_training`` pre-scans rows for the union
     of field keys in the target JSON and injects them into
     ``adapter_config['fields']`` so the per-row map step gets the
     field list automatically (mirrors β's
     ``_scan_classification_labels``).
  4. ``scripts/train.py:_adapt_record_to_text`` passes
     adapter-wrapped ``source_text`` through untouched — the
     β-tail bug that bit classification also exists for any other
     adapter writing wrapped prompts under the causal_lm task type.
"""

from __future__ import annotations

import unittest

from app.services.data_adapter_service import (
    _build_structured_extraction_training_prompt,
    _map_structured_extraction,
)
from app.services.dataset_service import _scan_structured_extraction_fields


class StructuredExtractionAdapterWrapTests(unittest.TestCase):
    def test_wrap_with_fields_renders_list_in_prompt_form(self):
        out = _map_structured_extraction(
            {"text": "John Smith works at Acme", "structured_output": {
                "name": "John Smith", "company": "Acme",
            }},
            {"fields": ["company", "name"]},
        )
        assert out is not None
        self.assertIn(
            "Extract the following fields as JSON",
            out["source_text"],
        )
        self.assertIn(
            "Reply with a single JSON object, nothing else",
            out["source_text"],
        )
        self.assertIn("company, name", out["source_text"])
        self.assertIn("Input: John Smith works at Acme", out["source_text"])
        self.assertTrue(out["source_text"].endswith("Output:"))

    def test_wrap_without_fields_renders_no_list_fallback(self):
        out = _map_structured_extraction(
            {"text": "input", "structured_output": {"k": "v"}},
            {},
        )
        assert out is not None
        self.assertIn(
            "Extract the relevant fields from the input",
            out["source_text"],
        )
        self.assertNotIn("Extract the following fields as JSON", out["source_text"])

    def test_target_text_has_leading_space_for_clean_continuation(self):
        out = _map_structured_extraction(
            {"text": "x", "structured_output": {"k": "v"}},
            {"fields": ["k"]},
        )
        assert out is not None
        self.assertTrue(out["target_text"].startswith(" "))
        # The remainder is the JSON-serialised payload.
        self.assertIn('"k": "v"', out["target_text"])

    def test_raw_fields_remain_for_downstream_introspection(self):
        # Downstream surfaces (data health, gold diagnostics, smoke
        # peek) read raw ``text``/``answer``/``structured_output``
        # to do their jobs. ζ keeps those untouched.
        out = _map_structured_extraction(
            {"text": "raw doc", "structured_output": {"name": "x"}},
            {},
        )
        assert out is not None
        self.assertEqual(out["text"], "raw doc")
        self.assertIn('"name": "x"', out["answer"])
        self.assertEqual(out["structured_output"], {"name": "x"})


class ScanStructuredExtractionFieldsTests(unittest.TestCase):
    def test_scan_returns_sorted_deduped_field_union(self):
        rows = [
            {"text": "a", "structured_output": {"name": "x", "age": 1}},
            {"text": "b", "structured_output": {"name": "y", "company": "acme"}},
        ]
        out = _scan_structured_extraction_fields(rows, adapter_config=None)
        self.assertEqual(out, ["age", "company", "name"])

    def test_scan_accepts_json_string_payloads(self):
        # Adapters / sources often store target JSON as a string
        # rather than a parsed dict. The pre-scan must json.loads
        # them transparently or we'd silently miss field lists.
        rows = [
            {"text": "a", "structured_output": '{"name": "x", "age": 1}'},
            {"text": "b", "structured_output": '{"age": 2}'},
        ]
        out = _scan_structured_extraction_fields(rows, adapter_config=None)
        self.assertEqual(out, ["age", "name"])

    def test_scan_returns_none_when_no_fields(self):
        # Rows without any structured output → no fields to inline →
        # adapter falls back to the no-list prompt variant.
        out = _scan_structured_extraction_fields(
            [{"text": "a"}, {"text": "b"}], adapter_config=None,
        )
        self.assertIsNone(out)

    def test_scan_ignores_non_dict_payloads(self):
        # Lists / scalars at the top level can't be field-extracted;
        # the handler only emits field-list prompts for object
        # outputs, so the scanner mirrors that.
        rows = [
            {"text": "a", "structured_output": ["alpha", "beta"]},
            {"text": "b", "structured_output": "not-json"},
        ]
        self.assertIsNone(
            _scan_structured_extraction_fields(rows, adapter_config=None),
        )

    def test_scan_caps_at_sample_size(self):
        # Mirrors the handler's SCHEMA_SAMPLE_SIZE=20 — we shouldn't
        # scan thousands of rows when 20 are enough to surface a
        # stable field union. Use a row with a unique key past
        # row 20 to prove we stop early.
        rows = [
            {"text": f"row-{i}", "structured_output": {"k": "v"}}
            for i in range(20)
        ]
        rows.append({"text": "late", "structured_output": {"surprise": "x"}})
        out = _scan_structured_extraction_fields(rows, adapter_config=None)
        self.assertEqual(out, ["k"])  # "surprise" wasn't reached.

    def test_scan_respects_output_fields_override(self):
        # Caller can pin which target field to read from when the
        # data isn't under one of the default aliases.
        rows = [{"text": "a", "result_json": {"custom_field": "x"}}]
        out = _scan_structured_extraction_fields(
            rows, adapter_config={"output_fields": ["result_json"]},
        )
        self.assertEqual(out, ["custom_field"])


class StructuredHandlerByteForByteCompatibilityTests(unittest.TestCase):
    """The load-bearing test: ζ guarantees adapter + handler emit
    identical strings so the trained model sees the same prompt at
    train and eval time."""

    def test_adapter_prompt_matches_handler_with_field_list(self):
        from app.services.eval_task_handler_service import (
            StructuredExtractionHandler,
        )
        input_text = "John Smith works at Acme."
        fields = ["company", "name"]
        adapter_prompt = _build_structured_extraction_training_prompt(
            input_text, fields
        )
        handler_prompt = StructuredExtractionHandler()._build_prompt_text(
            input_text, fields
        )
        self.assertEqual(adapter_prompt, handler_prompt)

    def test_adapter_prompt_matches_handler_no_list_fallback(self):
        from app.services.eval_task_handler_service import (
            StructuredExtractionHandler,
        )
        input_text = "free-form"
        adapter_prompt = _build_structured_extraction_training_prompt(
            input_text, None
        )
        handler_prompt = StructuredExtractionHandler()._build_prompt_text(
            input_text, []
        )
        self.assertEqual(adapter_prompt, handler_prompt)

    def test_adapter_source_text_carries_handler_expected_prefix(self):
        # γ′ smoke check peeks the first prepared row and checks
        # whether any of ``expected_prompt_prefixes`` appears in the
        # source_text. ζ keeps this invariant for structured rows.
        from app.services.eval_task_handler_service import (
            StructuredExtractionHandler,
        )
        out = _map_structured_extraction(
            {"text": "doc", "structured_output": {"k": "v"}},
            {"fields": ["k"]},
        )
        assert out is not None
        for prefix in StructuredExtractionHandler().expected_prompt_prefixes():
            if prefix in out["source_text"]:
                return
        self.fail(
            "Adapter source_text doesn't carry any of the handler's "
            f"expected prefixes: {out['source_text']!r}"
        )


class StructuredAdaptRecordPassthroughTests(unittest.TestCase):
    """ζ-tail — scripts/train.py:_adapt_record_to_text must pass
    adapter-wrapped source_text through untouched. Without this the
    direct-text branch overwrites the wrapped prompt with the raw
    input field, same shape as the β-tail bug for classification."""

    def test_wrapped_structured_row_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = (
            "Extract the following fields as JSON: name.\n"
            "Reply with a single JSON object, nothing else.\n"
            "Input: John Smith\n"
            "Output:"
        )
        adapted = train_script._adapt_record_to_text(
            {
                "text": "John Smith",
                "answer": '{"name": "John Smith"}',
                "structured_output": {"name": "John Smith"},
                "source_text": wrapped,
                "target_text": ' {"name": "John Smith"}',
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)
        self.assertEqual(adapted["target_text"], ' {"name": "John Smith"}')
        # Tokenized "text" = prompt + target — the trainer sees one
        # sequence with the same scaffold as eval.
        self.assertEqual(
            adapted["text"], f"{wrapped} " + '{"name": "John Smith"}',
        )

    def test_legacy_structured_row_uses_existing_fallback(self):
        # A row without an adapter-wrapped ``source_text`` should
        # still go through the direct-text branch so pre-ζ projects
        # keep training the same way.
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        adapted = train_script._adapt_record_to_text(
            {"text": "raw"},
            contract,
            "chatml",
        )
        self.assertEqual(adapted["text"], "raw")
        self.assertEqual(adapted["source_text"], "raw")


if __name__ == "__main__":
    unittest.main()
