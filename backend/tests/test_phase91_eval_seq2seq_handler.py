"""Phase 5.3.3 — Seq2SeqHandler.

Pins every contract the plan made:

- Dispatcher routes ``task_profile == "seq2seq"`` to Seq2SeqHandler.
- Sub-task is read from ``manifest.subtask``: translation /
  summarization / paraphrase. Missing / unknown → summarization.
- Translation prompt template names the target language (read from
  ``manifest.tgt_lang``).
- Summarization + paraphrase use their own prompt templates.
- max_new_tokens_override raises a too-small caller value to ~1.5×
  the longest reference but never reduces below caller's value and
  never exceeds the 512 hardcap.
- Translation scoring produces ``bleu`` + ``chrf`` (normalized to
  0–1 to align with our other 0–1 metric IDs) + legacy ``f1`` /
  ``exact_match`` aliases.
- Summarization scoring produces ``rouge_1`` / ``rouge_2`` /
  ``rouge_l`` + legacy aliases.
- Paraphrase scoring produces BOTH (BLEU + ROUGE) since paraphrase
  quality has both lexical and structural components.
- All sub-tasks report ``length_ratio`` (pred-tokens / ref-tokens).
- Sub-task value lands in the metrics dict so the UI can show it.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.eval_task_handler_service import (  # noqa: E402
    EvalContext,
    GenericHandler,
    Seq2SeqHandler,
    resolve_task_handler,
)


def _ctx(manifest: dict | None = None) -> EvalContext:
    return EvalContext(
        project_id=0,
        experiment_id=0,
        eval_type="f1",
        task_profile="seq2seq",
        handler_id="seq2seq",
        prepared_dir=Path("."),
        dataset_name="test",
        manifest=manifest or {},
    )


class DispatcherRoutingTests(unittest.TestCase):
    def test_seq2seq_profile_routes_to_seq2seq_handler(self):
        handler = resolve_task_handler("seq2seq")
        self.assertIsInstance(handler, Seq2SeqHandler)
        self.assertEqual(handler.profile_id, "seq2seq")

    def test_other_profiles_unaffected(self):
        # By Phase 5.3.2, qa routes to QAHandler — generic now only
        # captures unhandled profiles and None.
        self.assertIsInstance(resolve_task_handler(None), GenericHandler)
        self.assertIsInstance(
            resolve_task_handler("rag_qa"), GenericHandler
        )


class SubtaskResolutionTests(unittest.TestCase):
    def test_defaults_to_summarization_when_subtask_missing(self):
        handler = Seq2SeqHandler()
        self.assertEqual(handler._resolve_subtask(_ctx()), "summarization")

    def test_reads_explicit_subtask(self):
        handler = Seq2SeqHandler()
        self.assertEqual(
            handler._resolve_subtask(_ctx({"subtask": "translation"})),
            "translation",
        )

    def test_unknown_subtask_falls_back_to_summarization(self):
        handler = Seq2SeqHandler()
        self.assertEqual(
            handler._resolve_subtask(_ctx({"subtask": "totally-fake"})),
            "summarization",
        )

    def test_subtask_normalized_to_lowercase(self):
        handler = Seq2SeqHandler()
        self.assertEqual(
            handler._resolve_subtask(_ctx({"subtask": "TRANSLATION"})),
            "translation",
        )


class PromptAssemblyTests(unittest.TestCase):
    def test_translation_prompt_names_target_language(self):
        handler = Seq2SeqHandler()
        ctx = _ctx({"subtask": "translation", "tgt_lang": "French"})
        built = handler.build_prompts(
            [{"source_text": "Good morning.", "target_text": "Bonjour."}], ctx
        )
        self.assertIn("Translate", built[0].prompt)
        self.assertIn("French", built[0].prompt)
        self.assertIn("Good morning.", built[0].prompt)
        self.assertTrue(built[0].prompt.endswith("Translation:"))
        self.assertEqual(built[0].reference, "Bonjour.")
        self.assertEqual(built[0].extras["seq2seq_subtask"], "translation")
        self.assertEqual(built[0].extras["seq2seq_tgt_lang"], "French")

    def test_summarization_prompt_uses_summary_template(self):
        handler = Seq2SeqHandler()
        ctx = _ctx({"subtask": "summarization"})
        built = handler.build_prompts(
            [{"text": "Long article body...", "summary": "Short."}], ctx
        )
        self.assertIn("Summarize", built[0].prompt)
        self.assertTrue(built[0].prompt.endswith("Summary:"))
        self.assertEqual(built[0].reference, "Short.")

    def test_paraphrase_prompt_uses_paraphrase_template(self):
        handler = Seq2SeqHandler()
        built = handler.build_prompts(
            [{"text": "The dog sat.", "paraphrase": "A dog was sitting."}],
            _ctx({"subtask": "paraphrase"}),
        )
        self.assertIn("Paraphrase", built[0].prompt)
        self.assertTrue(built[0].prompt.endswith("Paraphrase:"))

    def test_default_subtask_uses_summarization_prompt(self):
        handler = Seq2SeqHandler()
        # No subtask in manifest → defaults to summarization.
        built = handler.build_prompts(
            [{"text": "hello", "target_text": "hi"}], _ctx()
        )
        self.assertIn("Summarize", built[0].prompt)


class MaxNewTokensOverrideTests(unittest.TestCase):
    def test_caller_default_used_when_no_rows_seen(self):
        handler = Seq2SeqHandler()
        self.assertEqual(handler.max_new_tokens_override(128), 128)

    def test_raises_default_to_one_and_a_half_x_longest_reference(self):
        handler = Seq2SeqHandler()
        # 100-token reference; caller asked for 32 → should be raised to ~150.
        long_ref = " ".join(["word"] * 100)
        handler.build_prompts(
            [{"text": "x", "target_text": long_ref}], _ctx()
        )
        self.assertEqual(handler.max_new_tokens_override(32), 150)

    def test_never_reduces_below_caller_default(self):
        handler = Seq2SeqHandler()
        # Short reference; caller asked for 256 — keep 256 (don't reduce).
        handler.build_prompts(
            [{"text": "x", "target_text": "short"}], _ctx()
        )
        self.assertEqual(handler.max_new_tokens_override(256), 256)

    def test_hardcaps_at_512(self):
        handler = Seq2SeqHandler()
        # 1000-token reference → 1.5x = 1500 → should clamp to 512.
        long_ref = " ".join(["word"] * 1000)
        handler.build_prompts(
            [{"text": "x", "target_text": long_ref}], _ctx()
        )
        self.assertEqual(handler.max_new_tokens_override(64), 512)


class TranslationScoringTests(unittest.TestCase):
    def test_translation_produces_bleu_chrf_and_legacy_aliases(self):
        handler = Seq2SeqHandler()
        ctx = _ctx({"subtask": "translation", "tgt_lang": "French"})
        # Perfect predictions should max out every metric.
        predictions = [
            {"prediction": "Bonjour le monde", "reference": "Bonjour le monde"},
            {"prediction": "Bonsoir", "reference": "Bonsoir"},
        ]
        out = handler.score(predictions, ctx)
        self.assertEqual(out["subtask"], "translation")
        self.assertGreater(out["bleu"], 0.9)
        self.assertGreater(out["chrf"], 0.9)
        # Legacy aliases also high.
        self.assertGreater(out["f1"], 0.9)
        self.assertGreater(out["exact_match"], 0.9)
        # Translation does NOT report ROUGE.
        self.assertNotIn("rouge_l", out)

    def test_translation_bleu_normalized_to_zero_to_one_range(self):
        handler = Seq2SeqHandler()
        # Random mismatch — BLEU should be small (and definitely ≤ 1.0,
        # the assertion that catches "we forgot to /100").
        out = handler.score(
            [
                {"prediction": "totally unrelated output", "reference": "ground truth"},
            ],
            _ctx({"subtask": "translation"}),
        )
        self.assertGreaterEqual(out["bleu"], 0.0)
        self.assertLessEqual(out["bleu"], 1.0)
        self.assertLessEqual(out["chrf"], 1.0)


class SummarizationScoringTests(unittest.TestCase):
    def test_summarization_produces_rouge_and_legacy_aliases(self):
        handler = Seq2SeqHandler()
        # Predictions that share tokens with refs but aren't identical.
        predictions = [
            {
                "prediction": "the quick brown fox",
                "reference": "the quick brown fox jumped",
            },
            {
                "prediction": "summary of the article",
                "reference": "a brief summary of the news article",
            },
        ]
        out = handler.score(predictions, _ctx({"subtask": "summarization"}))
        self.assertEqual(out["subtask"], "summarization")
        for key in ("rouge_1", "rouge_2", "rouge_l"):
            self.assertIn(key, out)
            self.assertGreater(out[key], 0.0)
            self.assertLessEqual(out[key], 1.0)
        # Legacy aliases.
        self.assertIn("f1", out)
        self.assertIn("exact_match", out)
        # No translation metrics for summarization.
        self.assertNotIn("bleu", out)

    def test_summarization_zero_overlap_scores_zero(self):
        out = Seq2SeqHandler().score(
            [{"prediction": "foo bar baz", "reference": "qux quux corge"}],
            _ctx({"subtask": "summarization"}),
        )
        self.assertEqual(out["rouge_1"], 0.0)


class ParaphraseScoringTests(unittest.TestCase):
    def test_paraphrase_produces_both_bleu_and_rouge(self):
        handler = Seq2SeqHandler()
        out = handler.score(
            [
                {
                    "prediction": "the cat sat on the mat",
                    "reference": "a cat was sitting on a mat",
                },
            ],
            _ctx({"subtask": "paraphrase"}),
        )
        # Paraphrase emits both lexical (BLEU) and structural (ROUGE) signal.
        self.assertIn("bleu", out)
        self.assertIn("chrf", out)
        self.assertIn("rouge_1", out)
        self.assertIn("rouge_l", out)


class CrossCuttingScoringTests(unittest.TestCase):
    def test_length_ratio_always_reported(self):
        handler = Seq2SeqHandler()
        out = handler.score(
            [
                # 6 prediction tokens, 3 reference tokens → ratio 2.0
                {"prediction": "one two three four five six", "reference": "one two three"},
            ],
            _ctx({"subtask": "summarization"}),
        )
        self.assertEqual(out["length_ratio"], 2.0)

    def test_subtask_value_lands_in_metrics_dict(self):
        for subtask in ("translation", "summarization", "paraphrase"):
            out = Seq2SeqHandler().score(
                [{"prediction": "x", "reference": "y"}],
                _ctx({"subtask": subtask}),
            )
            self.assertEqual(out["subtask"], subtask)

    def test_empty_predictions_returns_zeroed_metrics(self):
        out = Seq2SeqHandler().score([], _ctx({"subtask": "translation"}))
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["bleu"], 0.0)
        self.assertEqual(out["f1"], 0.0)
        self.assertEqual(out["length_ratio"], 0.0)


class EndToEndIntegrationTests(unittest.TestCase):
    """build_prompts → mock-infer → score with a near-perfect model."""

    def test_summarization_full_pipeline(self):
        handler = Seq2SeqHandler()
        ctx = _ctx({"subtask": "summarization"})
        rows = [
            {
                "text": (
                    "The quick brown fox jumps over the lazy dog. "
                    "It was a sunny day in the meadow."
                ),
                "summary": "Fox jumps over dog on a sunny day.",
            },
        ]
        built = handler.build_prompts(rows, ctx)
        self.assertEqual(len(built), 1)
        # A perfect model emits the reference verbatim.
        predictions = [
            {"prediction": bp.reference, "reference": bp.reference}
            for bp in built
        ]
        out = handler.score(predictions, ctx)
        self.assertEqual(out["rouge_l"], 1.0)
        self.assertEqual(out["f1"], 1.0)

    def test_translation_full_pipeline(self):
        handler = Seq2SeqHandler()
        ctx = _ctx({"subtask": "translation", "tgt_lang": "French"})
        rows = [{"source_text": "Hello world.", "target_text": "Bonjour le monde."}]
        built = handler.build_prompts(rows, ctx)
        predictions = [
            {"prediction": bp.reference, "reference": bp.reference}
            for bp in built
        ]
        out = handler.score(predictions, ctx)
        self.assertGreater(out["bleu"], 0.9)
        self.assertGreater(out["chrf"], 0.9)


if __name__ == "__main__":
    unittest.main()
