"""Training-correctness regression net (fresh-look audit, Wave 1).

Pins the core-loop fixes in ``scripts/train.py``:

  F0  prompt→answer pairs win over a display ``text`` field (answers
      were silently dropped / trained as plain text).
  F1  QA-family rows train in the eval prompt format
      (``tokenizer.apply_chat_template``) by default.
  F2  loss lands on the completion only (prompt tokens → -100).
  F3  every row ends in a real, unmasked EOS — even when
      ``pad_token == eos_token``.
  F4  epochs scale with dataset size (``auto_epochs``).

The pure tests run everywhere with a fake tokenizer. ``RealTrainingLiftTests``
fine-tunes SmolLM2-135M-Instruct through the real ``run_training`` entry
point and asserts the fine-tuned model beats the base model on a held-out
split, using the same prompt builder as ``evaluation_service``. It skips when
torch/transformers/peft or the cached model aren't available.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from app.schemas.training import TrainingConfig
from app.services.training_epoch_policy import (
    MIN_STEPS_PER_EPOCH,
    max_epochs_for_rows,
    resolve_auto_epochs,
)
from scripts.train import (
    CausalLMCompletionCollator,
    _adapt_record_to_text,
    _build_data_adapter_contract,
    _encode_causal_lm_example,
    _rewrap_prompt_completion_row,
)

CAUSAL = _build_data_adapter_contract("causal_lm", "llama3")


class _CharTokenizer:
    """One token per character; pad == eos like most small chat models."""

    eos_token_id = 0
    pad_token_id = 0
    chat_template = "fake"

    def __call__(self, text, truncation=False, max_length=None, padding=False):
        ids = [ord(ch) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


class PromptCompletionAdapterTests(unittest.TestCase):
    def test_qa_pair_row_keeps_answer_despite_display_text(self):
        row = {
            "text": "Question: What is the PTO accrual?\nAnswer: 1.25 days/month",
            "question": "What is the PTO accrual?",
            "answer": "1.25 days/month",
            "source_text": "What is the PTO accrual?",
            "target_text": "1.25 days/month",
        }
        out = _adapt_record_to_text(row, CAUSAL, "llama3")
        self.assertEqual(out["source_text"], "What is the PTO accrual?")
        self.assertEqual(out["target_text"], "1.25 days/month")

    def test_default_canonical_row_with_question_as_text_keeps_answer(self):
        # Shape seen in real prepared splits: ``text`` is just the question.
        row = {"text": "How much PTO?", "question": "How much PTO?", "answer": "15 days"}
        out = _adapt_record_to_text(row, CAUSAL, "llama3")
        self.assertEqual(out["target_text"], "15 days")
        self.assertIn("15 days", out["text"])

    def test_alpaca_input_is_context_not_answer(self):
        row = {"instruction": "Translate to English", "input": "hola", "output": "hello"}
        out = _adapt_record_to_text(row, CAUSAL, "llama3")
        self.assertEqual(out["source_text"], "Translate to English\n\nhola")
        self.assertEqual(out["target_text"], "hello")

    def test_alpaca_without_output_is_plain_text(self):
        out = _adapt_record_to_text({"instruction": "Say hi", "input": "x"}, CAUSAL, "llama3")
        self.assertEqual(out["target_text"], "")

    def test_text_answer_row_uses_text_as_prompt(self):
        out = _adapt_record_to_text({"text": "rm -rf /", "answer": "injection"}, CAUSAL, "llama3")
        self.assertEqual((out["source_text"], out["target_text"]), ("rm -rf /", "injection"))

    def test_plain_document_row_stays_language_modelling(self):
        out = _adapt_record_to_text({"text": "Section 4.2 covers refunds."}, CAUSAL, "llama3")
        self.assertEqual(out["text"], "Section 4.2 covers refunds.")
        self.assertEqual(out["target_text"], "")

    def test_classification_wrap_passes_through_under_causal_lm(self):
        prompt = "Classify the following text. Reply with exactly one of: a, b.\nText: hi\nLabel:"
        row = {"text": "hi", "source_text": prompt, "target_text": " billing"}
        out = _adapt_record_to_text(row, CAUSAL, "llama3")
        self.assertEqual(out["text"], prompt + " billing")

    def test_multi_turn_messages_keep_full_rendering(self):
        row = {
            "text": "user: a\nassistant: b\nuser: c\nassistant: d",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ],
            "source_text": "c",
            "target_text": "d",
        }
        out = _adapt_record_to_text(row, CAUSAL, "llama3")
        self.assertEqual(out["text"], row["text"])


class ChatTemplateDefaultTests(unittest.TestCase):
    def test_schema_defaults_on(self):
        cfg = TrainingConfig(base_model="x")
        self.assertTrue(cfg.use_tokenizer_chat_template)
        self.assertTrue(cfg.auto_epochs)

    def test_explicit_num_epochs_disables_auto(self):
        self.assertFalse(TrainingConfig(base_model="x", num_epochs=7).auto_epochs)
        self.assertTrue(
            TrainingConfig(base_model="x", num_epochs=7, auto_epochs=True).auto_epochs
        )

    def test_rewrap_uses_tokenizer_template(self):
        row = {"text": "ignored", "source_text": "Q?", "target_text": "A."}
        out = _rewrap_prompt_completion_row(row, _CharTokenizer())
        self.assertEqual(out["text"], "<user>Q?</user><assistant>A.")
        self.assertEqual(out["target_text"], "A.")

    def test_rewrap_without_template_uses_raw_prompt(self):
        tok = _CharTokenizer()
        tok.chat_template = None
        out = _rewrap_prompt_completion_row({"source_text": "Q?", "target_text": "A."}, tok)
        self.assertEqual(out["text"], "Q?\nA.")

    def test_rewrap_skips_wrapped_rows(self):
        row = {"source_text": "Classify the following text.\nText: x\nLabel:", "target_text": " a"}
        self.assertIs(_rewrap_prompt_completion_row(row, _CharTokenizer()), row)


class CompletionOnlyLossTests(unittest.TestCase):
    def test_prompt_masked_answer_and_eos_trainable(self):
        tok = _CharTokenizer()
        ex = _encode_causal_lm_example(tok, "<user>Q?</user><assistant>A.", "A.", 256)
        trainable = [i for i, label in zip(ex["input_ids"], ex["labels"]) if label != -100]
        self.assertEqual(trainable, [ord("A"), ord("."), tok.eos_token_id])

    def test_plain_text_trains_every_token_plus_eos(self):
        ex = _encode_causal_lm_example(_CharTokenizer(), "doc", "", 256)
        self.assertEqual(ex["labels"], [ord("d"), ord("o"), ord("c"), 0])

    def test_truncated_row_gets_no_eos(self):
        ex = _encode_causal_lm_example(_CharTokenizer(), "x" * 50, "", 10)
        self.assertNotEqual(ex["input_ids"][-1], 0)

    def test_answer_truncated_away_is_fully_masked(self):
        ex = _encode_causal_lm_example(_CharTokenizer(), "p" * 50 + "ANSWER", "ANSWER", 20)
        self.assertTrue(all(label == -100 for label in ex["labels"]))

    def test_collator_pads_by_position_and_keeps_eos_label(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")
        import torch

        tok = _CharTokenizer()
        a = _encode_causal_lm_example(tok, "PQ" + "A", "A", 64)
        b = _encode_causal_lm_example(tok, "PQRSTU" + "B", "B", 64)
        batch = CausalLMCompletionCollator(pad_token_id=0, torch_module=torch)([a, b])
        labels = batch["labels"].tolist()
        # Row a: real EOS (id 0 == pad id) is still a label; padding is -100.
        self.assertEqual(labels[0][:4], [-100, -100, ord("A"), 0])
        self.assertTrue(all(x == -100 for x in labels[0][4:]))
        self.assertEqual(batch["attention_mask"].tolist()[0], [1, 1, 1, 1, 0, 0, 0, 0])


class AutoEpochPolicyTests(unittest.TestCase):
    def test_tiny_dataset_shrinks_accumulation_before_adding_epochs(self):
        plan = resolve_auto_epochs(train_rows=40, batch_size=4, gradient_accumulation_steps=4)
        self.assertEqual(plan["gradient_accumulation_steps"], 1)
        self.assertGreaterEqual(plan["steps_per_epoch"], MIN_STEPS_PER_EPOCH)
        self.assertLessEqual(plan["num_epochs"], max_epochs_for_rows(40))

    def test_large_dataset_gets_one_epoch(self):
        plan = resolve_auto_epochs(train_rows=50_000, batch_size=4, gradient_accumulation_steps=4)
        self.assertEqual(plan["num_epochs"], 1)
        self.assertEqual(plan["gradient_accumulation_steps"], 4)

    def test_epochs_non_increasing_with_more_data(self):
        epochs = [
            resolve_auto_epochs(train_rows=n, batch_size=4, gradient_accumulation_steps=4)["num_epochs"]
            for n in (200, 1_000, 5_000, 50_000)
        ]
        self.assertEqual(epochs, sorted(epochs, reverse=True))

    def test_small_data_stays_below_gap_scanner_warn_threshold(self):
        from app.services.training_config_gap_service import EPOCHS_WARN_FOR_SMALL

        for n in (10, 50, 99):
            plan = resolve_auto_epochs(train_rows=n, batch_size=4, gradient_accumulation_steps=4)
            self.assertLess(plan["num_epochs"], EPOCHS_WARN_FOR_SMALL)


# ── Real training ───────────────────────────────────────────────────

REAL_MODEL = os.environ.get("BREWSLM_REAL_TRAIN_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

_TEAMS = {
    "refund": "Billing",
    "charged twice": "Billing",
    "arrived broken": "Returns",
    "cracked": "Returns",
    "is late": "Shipping",
    "never arrived": "Shipping",
    "password": "Accounts",
    "locked out": "Accounts",
}
_OPENERS = ["Hi,", "Hello team,", "Urgent:", "Please help -", "Good morning,"]


def _ticket_rows(n: int, rng: random.Random, id_range: tuple[int, int]) -> list[dict[str, str]]:
    rows = []
    phrases = list(_TEAMS)
    for _ in range(n):
        order = rng.randint(*id_range)
        phrase = rng.choice(phrases)
        question = f"{rng.choice(_OPENERS)} order {order}: my request is about '{phrase}'."
        rows.append({
            "question": question,
            "answer": f"Ticket ORD-{order}: route to {_TEAMS[phrase]}.",
        })
    return rows


def _real_training_available() -> str | None:
    if os.environ.get("BREWSLM_SKIP_REAL_TRAINING"):
        return "BREWSLM_SKIP_REAL_TRAINING set"
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return f"ML deps unavailable: {exc}"
    try:
        AutoTokenizer.from_pretrained(REAL_MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        return f"{REAL_MODEL} not in the local HF cache"
    return None


_SKIP_REASON = _real_training_available()


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class RealTrainingLiftTests(unittest.TestCase):
    """Fine-tuned SmolLM2 must beat its base on a held-out split.

    Task: route support tickets to a team in a house format
    (``Ticket ORD-<id>: route to <Team>.``). The base model can't know the
    format or routing table; held-out tickets use unseen order ids, so a
    pass means the model learned the rule, not the rows.
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        rng = random.Random(7)
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        train = _ticket_rows(96, rng, (1000, 4999))
        val = _ticket_rows(8, rng, (5000, 5999))
        cls.heldout = _ticket_rows(24, rng, (6000, 9999))
        for name, rows in (("train", train), ("val", val)):
            with open(root / f"{name}.jsonl", "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        config = {
            "training_mode": "sft",
            "task_type": "causal_lm",
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "learning_rate": 1e-3,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 256,
            "gradient_checkpointing": False,
            "flash_attention": False,
            "observability_enabled": False,
            "save_steps": 10_000,
            "eval_steps": 10_000,
            "auto_oom_retry": False,
        }
        (root / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
        args = argparse.Namespace(
            project=0,
            experiment=0,
            output=str(root / "run"),
            base_model=REAL_MODEL,
            config=str(root / "training_config.json"),
            train_file=str(root / "train.jsonl"),
            val_file=str(root / "val.jsonl"),
            data_dir=str(root),
            max_train_samples=0,
            max_eval_samples=0,
            seed=42,
        )
        from scripts.train import run_training

        cls.report = run_training(args)
        cls.model_dir = Path(cls.report["model_dir"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _score(self, model, tokenizer):
        """(exact-match rate, stop rate, mean answer NLL) on the held-out
        split, prompting exactly as ``evaluation_service`` does."""
        import torch

        from app.services.evaluation_service import _apply_chat_template_if_present

        device = next(model.parameters()).device
        em = stopped = 0
        nll_total = 0.0
        for row in self.heldout:
            prompt, _ = _apply_chat_template_if_present(tokenizer, row["question"])
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = out[0][enc["input_ids"].shape[-1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            em += int(text == row["answer"])
            stopped += int(tokenizer.eos_token_id in new_tokens.tolist())

            full = tokenizer(prompt + row["answer"], return_tensors="pt").to(device)
            labels = full["input_ids"].clone()
            labels[:, : enc["input_ids"].shape[-1]] = -100
            with torch.no_grad():
                nll_total += float(model(**full, labels=labels).loss)
        n = len(self.heldout)
        return em / n, stopped / n, nll_total / n

    def test_finetuned_beats_base_on_heldout(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        env = self.report["runtime_environment"]
        self.assertEqual(env.get("chat_template_rewrap"), "tokenizer_chat_template")
        self.assertEqual(env.get("loss_masking"), "completion_only")
        self.assertIn("num_epochs", env.get("auto_epochs") or {})

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
        base = AutoModelForCausalLM.from_pretrained(REAL_MODEL).to(device).eval()
        base_em, _, base_nll = self._score(base, tokenizer)

        tuned = PeftModel.from_pretrained(base, str(self.model_dir)).to(device).eval()
        tuned_em, tuned_stop, tuned_nll = self._score(tuned, tokenizer)

        summary = (
            f"base EM={base_em:.2f} NLL={base_nll:.3f} | "
            f"tuned EM={tuned_em:.2f} NLL={tuned_nll:.3f} stop={tuned_stop:.2f}"
        )
        print(summary)
        self.assertLess(tuned_nll, base_nll * 0.5, summary)
        self.assertGreaterEqual(tuned_em, base_em + 0.5, summary)
        # F3: the model learned to end its answer.
        self.assertGreaterEqual(tuned_stop, 0.9, summary)


if __name__ == "__main__":
    unittest.main()
