"""θ-fix tests — seq2seq-pair adapter writes the production
prompt format into training rows.

Pre-θ: ``_map_seq2seq`` wrote ``source_text = src`` and
``text = "Input: {src}\\nOutput: {tgt}"`` — no subtask
instruction, no ``Text:`` / ``Summary:`` / ``Translation:`` /
``Paraphrase:`` blocks. ``Seq2SeqHandler.build_prompts`` at eval
time wraps inputs with one of three instruction-prefixed
scaffolds (``"Summarize the following text concisely.\\nText:
…\\nSummary:"`` etc.). The model never saw the eval-time scaffold
so BLEU/chrF/ROUGE on held-out data came in artificially low —
same shape as the bug β closed for classification-label.

Post-θ (this commit's tests pin):

  1. ``_build_seq2seq_training_prompt`` and
     ``Seq2SeqHandler._build_prompt_text`` produce IDENTICAL
     strings across all three subtasks (translation /
     summarization / paraphrase). Equality enforced byte-for-byte.
  2. The adapter reads ``subtask`` + ``tgt_lang`` from
     ``adapter_config`` — both injected by the subtask
     infrastructure committed earlier this session
     (``_resolve_adapter_subtask`` + the ``tgt_lang`` manifest
     propagation in ``_normalize_rows_for_training``).
  3. Unknown / missing ``subtask`` falls back to ``summarization``
     (the handler's ``DEFAULT_SUBTASK``). Missing ``tgt_lang``
     falls back to "the target language" (the handler's
     ``_resolve_tgt_lang`` default).
  4. ``target_text = f" {target}"`` (leading space — same trick
     as β / ζ / η for clean BPE continuation of the trailing
     ``Summary:`` / ``Translation:`` / ``Paraphrase:`` cue).
  5. Raw ``question`` / ``answer`` aliases preserved for
     downstream consumers.
  6. ``scripts/train.py:_adapt_record_to_text`` passes
     seq2seq-wrapped ``source_text`` through untouched (mirrors
     β-tail / ζ-tail / η-tail).
"""

from __future__ import annotations

import unittest

from app.services.data_adapter_service import (
    _build_seq2seq_training_prompt,
    _map_seq2seq,
)


# ── Adapter wrap, per subtask ────────────────────────────────────────


class Seq2SeqAdapterWrapTests(unittest.TestCase):
    def test_summarization_default_when_no_subtask_in_config(self):
        # Missing subtask → handler's DEFAULT_SUBTASK ("summarization")
        # — wrap MUST match that branch so train/eval agree.
        out = _map_seq2seq(
            {"source": "Long article text here.", "target": "Short."},
            {},
        )
        assert out is not None
        self.assertTrue(
            out["source_text"].startswith("Summarize the following text"),
            out["source_text"],
        )
        self.assertIn("Text: Long article text here.", out["source_text"])
        self.assertTrue(out["source_text"].endswith("Summary:"))

    def test_translation_uses_tgt_lang_from_config(self):
        # Translation branch needs tgt_lang from the manifest. The
        # subtask infra injects it into adapter_config; here we
        # pass it directly to assert the adapter reads from
        # config.tgt_lang and stitches it into the prompt.
        out = _map_seq2seq(
            {"source": "Hello.", "target": "Bonjour."},
            {"subtask": "translation", "tgt_lang": "French"},
        )
        assert out is not None
        self.assertIn(
            "Translate the following to French.", out["source_text"]
        )
        self.assertTrue(out["source_text"].endswith("Translation:"))

    def test_translation_falls_back_to_default_tgt_lang(self):
        # Missing tgt_lang → handler's "_resolve_tgt_lang" default
        # ("the target language"). Adapter must mirror so eval +
        # train agree on the no-tgt_lang case.
        out = _map_seq2seq(
            {"source": "x", "target": "y"},
            {"subtask": "translation"},
        )
        assert out is not None
        self.assertIn(
            "Translate the following to the target language.",
            out["source_text"],
        )

    def test_paraphrase_subtask(self):
        out = _map_seq2seq(
            {"source": "The cat sat on the mat.", "target": "A cat was on the mat."},
            {"subtask": "paraphrase"},
        )
        assert out is not None
        self.assertTrue(
            out["source_text"].startswith(
                "Paraphrase the following text in different words"
            )
        )
        self.assertTrue(out["source_text"].endswith("Paraphrase:"))

    def test_target_text_has_leading_space(self):
        out = _map_seq2seq(
            {"source": "x", "target": "y"},
            {"subtask": "summarization"},
        )
        assert out is not None
        self.assertEqual(out["target_text"], " y")

    def test_text_field_matches_source_plus_target_byte_for_byte(self):
        # The audit's open question — pre-θ ``text`` rendered a
        # different scaffold ("Input: …\\nOutput: …") than the
        # handler's wrap. θ aligns both so trainer paths that
        # consume ``text`` directly see the same prompt the
        # handler will rebuild at eval.
        out = _map_seq2seq(
            {"source": "Long doc.", "target": "Short summary."},
            {"subtask": "summarization"},
        )
        assert out is not None
        self.assertEqual(out["text"], f"{out['source_text']}{out['target_text']}")
        # Specifically: no "Input:" / "Output:" pre-θ leftover.
        self.assertNotIn("Input:", out["text"])
        self.assertNotIn("Output:", out["text"])

    def test_raw_question_answer_aliases_preserved(self):
        out = _map_seq2seq(
            {"source": "raw src", "target": "raw tgt"},
            {"subtask": "summarization"},
        )
        assert out is not None
        self.assertEqual(out["question"], "raw src")
        self.assertEqual(out["answer"], "raw tgt")

    def test_invalid_subtask_falls_back_to_default(self):
        # Defensive: a bogus subtask in adapter_config (drift across
        # versions) should fall through to summarization rather
        # than crash. Mirrors the resolver's invalid-value
        # tolerance.
        out = _map_seq2seq(
            {"source": "x", "target": "y"},
            {"subtask": "  🐛bogus "},
        )
        assert out is not None
        self.assertTrue(
            out["source_text"].startswith("Summarize the following text")
        )

    def test_record_missing_source_returns_none(self):
        # Pre-θ behaviour preserved. Note: the symmetric case
        # ``{"target": "y"}`` returns a mapped record (not None)
        # because ``canonicalize_record`` promotes ``target`` →
        # ``answer`` and then the fallback path treats the
        # answer as both source and target — pre-existing
        # adapter behaviour (same shape as the β
        # ``test_record_with_only_text_returns_none`` carve-out).
        self.assertIsNone(_map_seq2seq({"source": "x"}, {}))


# ── Byte-for-byte equality with the handler ──────────────────────────


class Seq2SeqHandlerByteForByteTests(unittest.TestCase):
    """The load-bearing pin: adapter + handler emit identical
    strings across all three subtasks. A drift would silently
    re-introduce the train/eval mismatch the audit closed."""

    def test_summarization_matches_handler(self):
        from app.services.eval_task_handler_service import Seq2SeqHandler
        adapter = _build_seq2seq_training_prompt(
            "doc body", "summarization", "ignored-for-summary"
        )
        handler = Seq2SeqHandler()._build_prompt_text(
            "doc body", "summarization", "ignored-for-summary"
        )
        self.assertEqual(adapter, handler)

    def test_translation_matches_handler(self):
        from app.services.eval_task_handler_service import Seq2SeqHandler
        adapter = _build_seq2seq_training_prompt("Hi.", "translation", "Spanish")
        handler = Seq2SeqHandler()._build_prompt_text(
            "Hi.", "translation", "Spanish"
        )
        self.assertEqual(adapter, handler)

    def test_paraphrase_matches_handler(self):
        from app.services.eval_task_handler_service import Seq2SeqHandler
        adapter = _build_seq2seq_training_prompt(
            "Original sentence.", "paraphrase", "ignored",
        )
        handler = Seq2SeqHandler()._build_prompt_text(
            "Original sentence.", "paraphrase", "ignored",
        )
        self.assertEqual(adapter, handler)

    def test_unknown_subtask_routes_to_summarization_branch_both_sides(self):
        # Both adapter helper and handler treat unknown subtask
        # the same (the handler's ``_build_prompt_text`` falls
        # through to the summarization branch via Python's
        # if/return order).
        from app.services.eval_task_handler_service import Seq2SeqHandler
        adapter = _build_seq2seq_training_prompt("x", "unknown", "irrelevant")
        handler = Seq2SeqHandler()._build_prompt_text("x", "unknown", "irrelevant")
        self.assertEqual(adapter, handler)

    def test_adapter_source_text_carries_handler_expected_prefixes(self):
        # γ′ smoke check / row peek surfaces these prefixes. Make
        # sure each subtask's wrapped source_text carries at least
        # one of the handler's declared prefixes.
        from app.services.eval_task_handler_service import Seq2SeqHandler
        prefixes = Seq2SeqHandler().expected_prompt_prefixes()
        for subtask in ("translation", "summarization", "paraphrase"):
            out = _map_seq2seq(
                {"source": "x", "target": "y"},
                {"subtask": subtask, "tgt_lang": "en"},
            )
            assert out is not None
            self.assertTrue(
                any(p in out["source_text"] for p in prefixes),
                f"subtask={subtask} prefixes={prefixes!r} "
                f"source_text={out['source_text']!r}",
            )


# ── train.py tail pass-through ───────────────────────────────────────


class Seq2SeqAdaptRecordPassthroughTests(unittest.TestCase):
    def test_summarization_wrap_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = (
            "Summarize the following text concisely.\n"
            "Text: Long article.\n"
            "Summary:"
        )
        adapted = train_script._adapt_record_to_text(
            {
                "question": "Long article.",
                "answer": "Short.",
                "text": f"{wrapped} Short.",
                "source_text": wrapped,
                "target_text": " Short.",
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)
        self.assertEqual(adapted["target_text"], " Short.")
        self.assertEqual(adapted["text"], f"{wrapped} Short.")

    def test_translation_wrap_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = (
            "Translate the following to French.\n"
            "Text: Hello.\n"
            "Translation:"
        )
        adapted = train_script._adapt_record_to_text(
            {
                "question": "Hello.",
                "answer": "Bonjour.",
                "text": f"{wrapped} Bonjour.",
                "source_text": wrapped,
                "target_text": " Bonjour.",
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)
        self.assertEqual(adapted["target_text"], " Bonjour.")

    def test_paraphrase_wrap_passes_through_untouched(self):
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        wrapped = (
            "Paraphrase the following text in different words.\n"
            "Text: The cat sat.\n"
            "Paraphrase:"
        )
        adapted = train_script._adapt_record_to_text(
            {
                "question": "The cat sat.",
                "answer": "A cat was sitting.",
                "text": f"{wrapped} A cat was sitting.",
                "source_text": wrapped,
                "target_text": " A cat was sitting.",
            },
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], wrapped)

    def test_legacy_qa_row_unaffected_by_seq2seq_tail(self):
        # Regression guard — a plain Q/A row (no Translate / Summarize:
        # / Paraphrase prefix) should still go through the existing
        # question+answer branch.
        from scripts import train as train_script
        contract = train_script._build_data_adapter_contract(
            "causal_lm", "chatml",
        )
        adapted = train_script._adapt_record_to_text(
            {"question": "What is 2+2?", "answer": "4"},
            contract,
            "chatml",
        )
        self.assertEqual(adapted["source_text"], "What is 2+2?")
        self.assertEqual(adapted["target_text"], "4")


if __name__ == "__main__":
    unittest.main()
