"""Memory defaults for long-prompt runs: gradient checkpointing "auto" looks
at sequence length, and passage-prompt runs default to LoRA (2026-10-19: a
1.5B full fine-tune on ~1k-token passage prompts, no checkpointing, ~90 GB)."""

import unittest

from app.services.gradient_checkpointing_policy import (
    AUTO_CHECKPOINT_MAX_PARAMS,
    AUTO_CHECKPOINT_MAX_SEQ_TOKENS,
    resolve_gradient_checkpointing,
)
from scripts.train import (
    LONG_ROW_TOKENS,
    PASSAGE_MAX_SEQ_CAP,
    _fit_passage_seq_length,
    _longest_row_tokens,
    _resolve_micro_batch,
    _resolve_use_lora,
)


class _Tok:
    def __call__(self, text, **_kw):
        return {"input_ids": text.split()}


class GradientCheckpointingSequenceTests(unittest.TestCase):
    def test_long_rows_turn_it_on_for_a_small_model(self):
        plan = resolve_gradient_checkpointing("auto", model_params=1_500_000_000, sequence_tokens=1400)
        self.assertTrue(plan["enabled"])
        self.assertEqual(plan["sequence_tokens"], 1400)
        self.assertIn("1400 tokens", plan["reason"])

    def test_short_rows_keep_the_size_rule(self):
        small = resolve_gradient_checkpointing("auto", model_params=1_500_000_000, sequence_tokens=AUTO_CHECKPOINT_MAX_SEQ_TOKENS)
        self.assertFalse(small["enabled"])
        big = resolve_gradient_checkpointing("auto", model_params=AUTO_CHECKPOINT_MAX_PARAMS + 1, sequence_tokens=200)
        self.assertTrue(big["enabled"])
        self.assertFalse(resolve_gradient_checkpointing("auto", model_params=135_000_000)["enabled"])

    def test_explicit_setting_wins(self):
        self.assertFalse(resolve_gradient_checkpointing(False, model_params=10, sequence_tokens=4000)["enabled"])

    def test_longest_row_is_capped_by_max_seq_length(self):
        rows = {"text": ["a b c", " ".join(["w"] * 3000)]}
        self.assertEqual(_longest_row_tokens(rows, _Tok(), 2048), 2048)
        self.assertEqual(_longest_row_tokens({"text": ["a b c"]}, _Tok(), 2048), 3)
        self.assertIsNone(_longest_row_tokens(None, _Tok(), 2048))


class PassagePromptLoraDefaultTests(unittest.TestCase):
    def test_passage_prompt_run_defaults_to_lora(self):
        use_lora, reason = _resolve_use_lora({"passage_prompts": True})
        self.assertTrue(use_lora)
        self.assertIn("passage-prompt", reason)

    def test_explicit_full_fine_tune_is_honoured(self):
        self.assertEqual(_resolve_use_lora({"passage_prompts": True, "use_lora": False}), (False, None))

    def test_other_runs_keep_the_unset_default(self):
        self.assertEqual(_resolve_use_lora({}), (False, None))
        self.assertEqual(_resolve_use_lora({"use_lora": True}), (True, None))


class MicroBatchTests(unittest.TestCase):
    def test_long_rows_keep_the_effective_batch_in_micro_batches(self):
        plan = _resolve_micro_batch({}, batch_size=4, gradient_accumulation_steps=1, sequence_tokens=2048)
        self.assertEqual((plan["batch_size"], plan["gradient_accumulation_steps"], plan["effective_batch"]), (1, 4, 4))

    def test_short_rows_explicit_batch_or_batch_one_are_left_alone(self):
        self.assertIsNone(_resolve_micro_batch({}, batch_size=4, gradient_accumulation_steps=4, sequence_tokens=LONG_ROW_TOKENS))
        self.assertIsNone(_resolve_micro_batch({"batch_size": 4}, batch_size=4, gradient_accumulation_steps=4, sequence_tokens=3000))
        self.assertIsNone(_resolve_micro_batch({}, batch_size=1, gradient_accumulation_steps=4, sequence_tokens=3000))
        self.assertIsNone(_resolve_micro_batch({}, batch_size=4, gradient_accumulation_steps=4, sequence_tokens=None))


if __name__ == "__main__":
    unittest.main()


class PassageSeqLengthFitTests(unittest.TestCase):
    """Passage-prompt rows past max_seq_length lose their answer (26 of 323
    on 2026-10-19): the window rises to fit the longest row, bounded."""

    def test_raises_to_the_longest_row_rounded_up(self):
        rows = {"text": ["a b", " ".join(["w"] * 2985)]}
        fit = _fit_passage_seq_length({"passage_prompts": True}, rows, _Tok(), 2048)
        self.assertEqual(fit["to"], 3008)  # 2985 + EOS, rounded up to 64
        self.assertEqual(fit["longest_row_tokens"], 2985)
        self.assertEqual(fit["rows_over_cap"], 0)

    def test_never_lowers_and_caps_with_a_count(self):
        short = _fit_passage_seq_length({"passage_prompts": True}, {"text": ["a b c"]}, _Tok(), 2048)
        self.assertEqual(short["to"], 2048)
        huge = {"text": [" ".join(["w"] * 5000), " ".join(["w"] * 6000), "a"]}
        capped = _fit_passage_seq_length({"passage_prompts": True}, huge, _Tok(), 2048)
        self.assertEqual(capped["to"], PASSAGE_MAX_SEQ_CAP)
        self.assertEqual(capped["rows_over_cap"], 2)

    def test_only_passage_runs_and_can_be_turned_off(self):
        rows = {"text": [" ".join(["w"] * 3000)]}
        self.assertIsNone(_fit_passage_seq_length({}, rows, _Tok(), 2048))
        self.assertIsNone(_fit_passage_seq_length({"passage_prompts": True, "passage_prompts_fit_max_seq_length": False}, rows, _Tok(), 2048))
