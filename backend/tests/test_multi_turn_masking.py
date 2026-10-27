"""Multi-turn chat SFT: assistant-only loss in ``scripts/train.py`` (G8 lane A).

Pure tests on a tiny in-process tokenizer (``_tiny_chat_tokenizer``, no
downloads): which rows take the assistant-only path, mask correctness on a
ChatML-style template that injects ``<|im_end|>\\n`` after every message,
fallbacks, truncation by whole exchanges, the ``multi_turn`` report block, the
simulate-run approximation, the ORPO trainer resolver — and a golden fixture
proving single-exchange and non-messages rows encode byte-identically to the
pre-change code (``fixtures/train_single_turn_golden.json`` was generated from
the trainer before the change).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _tiny_chat_tokenizer import CHATML, CHATML_DEFAULT_SYSTEM, build_tiny_tokenizer
from scripts.train import (
    CHAT_MESSAGES_COLUMN,
    _adapt_record_to_text,
    _build_data_adapter_contract,
    _encode_multi_turn_example,
    _encode_text_row,
    _multi_turn_messages,
    _normalize_chat_messages,
    _resolve_pairwise_trainer,
    _rewrap_prompt_completion_row,
    _summarize_multi_turn,
)

FIXTURE = Path(__file__).parent / "fixtures" / "train_single_turn_golden.json"
CAUSAL = _build_data_adapter_contract("causal_lm", "chatml")

DIALOGUE = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Pick a colour."},
    {"role": "assistant", "content": "Green."},
    {"role": "user", "content": "And another?"},
    {"role": "assistant", "content": "Blue."},
]


def _trained_text(tok, example) -> str:
    ids = [i for i, label in zip(example["input_ids"], example["labels"]) if label != -100]
    return tok.decode(ids).replace(" ", "")


class GoldenSingleTurnTests(unittest.TestCase):
    """A4: rows that aren't multi-turn encode exactly as before."""

    def test_single_exchange_and_non_messages_rows_are_byte_identical(self):
        golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
        rows = golden["rows"]
        for cfg in golden["configs"]:
            tok = build_tiny_tokenizer(CHATML if cfg["template"] else None)
            for index, (row, expected) in enumerate(zip(rows, cfg["expected"])):
                mapped = _adapt_record_to_text(row, CAUSAL, "chatml")
                mapped["system_prompt"] = str(row.get("system_prompt") or "")
                mapped.setdefault(CHAT_MESSAGES_COLUMN, "")
                if cfg["rewrap"]:
                    mapped = _rewrap_prompt_completion_row(mapped, tok)
                example, info = _encode_text_row(
                    tok, mapped["text"], mapped.get("target_text") or "",
                    mapped[CHAT_MESSAGES_COLUMN], cfg["max_seq_length"],
                    use_template=cfg["rewrap"],
                )
                with self.subTest(config=cfg["name"], row=index):
                    self.assertIsNone(info)
                    self.assertEqual(example["input_ids"], expected["input_ids"])
                    self.assertEqual(example["labels"], expected["labels"])


class RowRuleTests(unittest.TestCase):
    def test_single_exchange_without_system_stays_single_turn(self):
        row = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
        self.assertIsNone(_multi_turn_messages(row))

    def test_system_plus_one_exchange_is_multi_turn(self):
        row = {"messages": DIALOGUE[:3]}
        self.assertEqual(_multi_turn_messages(row), DIALOGUE[:3])

    def test_two_assistant_turns_is_multi_turn(self):
        self.assertEqual(len(_multi_turn_messages({"messages": DIALOGUE[1:]})), 4)

    def test_trailing_user_turn_is_trimmed(self):
        row = {"messages": DIALOGUE[1:3] + [{"role": "user", "content": "thanks"}]}
        self.assertEqual(_multi_turn_messages(row), DIALOGUE[1:3])

    def test_no_assistant_turn_is_not_trainable(self):
        self.assertIsNone(_multi_turn_messages({"messages": DIALOGUE[:2]}))

    def test_conversations_aliases_normalise(self):
        row = {"conversations": [
            {"from": "system", "value": "Be brief."},
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]}
        self.assertEqual([m["role"] for m in _multi_turn_messages(row)], ["system", "user", "assistant"])

    def test_normalisation_agrees_with_the_chat_adapter(self):
        from app.services.data_adapter_service import _messages_from_record

        samples = [
            {"messages": DIALOGUE},
            {"messages": [{"role": "USER", "content": "  x "}, {"role": "model", "content": "y"},
                          {"role": "tool", "content": "z"}, {"role": "assistant", "content": ""},
                          "junk", {"content": "no role"}]},
            {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "text": "a"}]},
            {"messages": "not a list", "conversations": [{"role": "user", "content": 3}]},
            {"text": "no messages"},
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(_normalize_chat_messages(sample), _messages_from_record(sample))

    def test_adapter_carries_messages_for_multi_turn_rows_only(self):
        multi = _adapt_record_to_text({"messages": DIALOGUE}, CAUSAL, "chatml")
        self.assertEqual(json.loads(multi[CHAT_MESSAGES_COLUMN]), DIALOGUE)
        self.assertEqual(multi["target_text"], "")
        single = _adapt_record_to_text(
            {"messages": DIALOGUE[1:3], "source_text": "Pick a colour.", "target_text": "Green."},
            CAUSAL, "chatml",
        )
        self.assertNotIn(CHAT_MESSAGES_COLUMN, single)
        self.assertEqual(single["target_text"], "Green.")


class AssistantOnlyMaskTests(unittest.TestCase):
    def setUp(self):
        self.tok = build_tiny_tokenizer()

    def test_system_and_one_assistant_turn(self):
        ex, info = _encode_multi_turn_example(self.tok, DIALOGUE[:3], 512)
        self.assertEqual(_trained_text(self.tok, ex), "Green.<|im_end|>\n".replace(" ", ""))
        self.assertEqual(info["assistant_turns"], 1)
        self.assertFalse(info["fallback"])

    def test_every_assistant_token_and_end_of_turn_trainable_everything_else_masked(self):
        ex, info = _encode_multi_turn_example(self.tok, DIALOGUE, 512)
        rendered = self.tok.apply_chat_template(DIALOGUE, tokenize=False)
        self.assertEqual(ex["input_ids"], self.tok(rendered, add_special_tokens=False)["input_ids"])
        self.assertEqual(_trained_text(self.tok, ex), "Green.<|im_end|>\nBlue.<|im_end|>\n".replace(" ", ""))
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        end_labels = [label for i, label in zip(ex["input_ids"], ex["labels"]) if i == im_end]
        # system / user / user ends masked; both assistant ends trained.
        self.assertEqual([label != -100 for label in end_labels], [False, False, True, False, True])
        self.assertTrue(all(label == -100 for i, label in zip(ex["input_ids"], ex["labels"]) if i == im_start))
        self.assertEqual(info["trained_tokens"], sum(1 for label in ex["labels"] if label != -100))
        self.assertEqual(info["total_tokens"], len(ex["input_ids"]))
        self.assertEqual(info["assistant_turns"], 2)
        # No EOS appended: the template's end-of-turn ends the row.
        self.assertNotEqual(ex["input_ids"][-1], self.tok.eos_token_id)

    def test_tool_messages_are_masked(self):
        messages = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "Calling."},
            {"role": "tool", "content": "SUNNY"},
            {"role": "assistant", "content": "Sunny."},
        ]
        ex, _ = _encode_multi_turn_example(self.tok, messages, 512)
        trained = _trained_text(self.tok, ex)
        self.assertNotIn("SUNNY", trained)
        self.assertIn("Calling.", trained)
        self.assertIn("Sunny.", trained)

    def test_template_with_injected_default_system(self):
        tok = build_tiny_tokenizer(CHATML_DEFAULT_SYSTEM)
        ex, info = _encode_multi_turn_example(tok, DIALOGUE[1:], 512)
        self.assertFalse(info["fallback"])
        self.assertEqual(_trained_text(tok, ex), "Green.<|im_end|>\nBlue.<|im_end|>\n".replace(" ", ""))

    def test_template_that_rewrites_history_falls_back_to_final_turn(self):
        rewriting = (
            "{% for message in messages %}"
            "{% if message['role'] == 'assistant' and not loop.last %}"
            "{{ '<|im_start|>assistant\n[earlier]<|im_end|>\n' }}"
            "{% else %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        tok = build_tiny_tokenizer(rewriting)
        ex, info = _encode_multi_turn_example(tok, DIALOGUE, 512)
        self.assertTrue(info["fallback"])
        self.assertEqual(info["assistant_turns"], 1)
        self.assertEqual(_trained_text(tok, ex), "Blue.<|im_end|>\n".replace(" ", ""))

    def test_no_chat_template_masks_role_lines(self):
        tok = build_tiny_tokenizer(None)
        ex, info = _encode_multi_turn_example(tok, DIALOGUE, 512)
        self.assertEqual(
            tok.decode(ex["input_ids"]).replace(" ", ""),
            ("system: Be brief.\nuser: Pick a colour.\nassistant: Green.\nuser: And another?\nassistant: Blue.\n"
             "<|endoftext|>").replace(" ", ""),
        )
        # No template supplies an end-of-turn, so the final assistant turn gets a trained EOS (as a
        # single-turn row does); the earlier turn keeps just its own text + newline.
        self.assertEqual(_trained_text(tok, ex), "Green.\nBlue.\n<|endoftext|>".replace(" ", ""))
        self.assertEqual(ex["input_ids"][-1], tok.eos_token_id)
        self.assertEqual(ex["labels"][-1], tok.eos_token_id)
        self.assertFalse(info["fallback"])
        self.assertEqual(info["total_tokens"], len(ex["input_ids"]))

    def test_use_template_off_uses_role_lines(self):
        ex, _ = _encode_multi_turn_example(self.tok, DIALOGUE, 512, use_template=False)
        self.assertNotIn("<|im_start|>", self.tok.decode(ex["input_ids"]))

    def test_slow_tokenizer_segments_match_fast_offsets(self):
        fast = self.tok

        class _Slow:
            is_fast = False
            chat_template = CHATML

            def __call__(self, text, truncation=False, padding=False, add_special_tokens=True):
                return {"input_ids": fast(text, add_special_tokens=False)["input_ids"]}

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return fast.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

        slow_ex, _ = _encode_multi_turn_example(_Slow(), DIALOGUE, 512)
        fast_ex, _ = _encode_multi_turn_example(fast, DIALOGUE, 512)
        self.assertEqual(slow_ex["labels"], fast_ex["labels"])


EOS_APPENDING = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% if loop.last and message['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


class EosAppendingTemplateTests(unittest.TestCase):
    """Phi-3-style templates add the EOS token after the LAST message only, so a mid-dialogue
    prefix render ends in an EOS the full render does not contain at that point."""

    def test_every_assistant_turn_is_trained_without_falling_back(self):
        tok = build_tiny_tokenizer(EOS_APPENDING)
        ex, info = _encode_multi_turn_example(tok, DIALOGUE, 512)
        self.assertFalse(info["fallback"])
        self.assertEqual(info["assistant_turns"], 2)
        # Turn 1 ends at its own end-of-turn (the not-yet-final EOS is not part of it); the final
        # turn keeps the template's EOS.
        self.assertEqual(
            _trained_text(tok, ex), "Green.<|im_end|>\nBlue.<|im_end|>\n<|endoftext|>".replace(" ", "")
        )
        im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.assertTrue(all(label == -100 for i, label in zip(ex["input_ids"], ex["labels"]) if i == im_start))
        # user text stays masked
        self.assertNotIn("Pick", tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100]))

    def test_template_rewriting_earlier_turns_still_falls_back(self):
        rewriting = EOS_APPENDING.replace("message['content']", "message['content'][:2]", 1)
        tok = build_tiny_tokenizer(
            "{% for message in messages %}{% if message['role'] == 'assistant' and not loop.last %}"
            "{{ '<|im_start|>assistant\n[x]<|im_end|>\n' }}{% else %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endif %}"
            "{% if loop.last and message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        self.assertNotEqual(rewriting, EOS_APPENDING)
        _, info = _encode_multi_turn_example(tok, DIALOGUE, 512)
        self.assertTrue(info["fallback"])


def _build_bpe_tokenizer():
    """A byte-pair tokenizer trained in-process (no download) on repeated dialogue renders, so
    multi-character tokens exist and some straddle a span boundary."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    import string

    from _tiny_chat_tokenizer import SPECIAL_TOKENS

    core = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=400, special_tokens=SPECIAL_TOKENS + ["<unk>"], initial_alphabet=sorted(set(string.printable)),
    )
    plain = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in DIALOGUE
    )
    core.train_from_iterator([plain] * 200, trainer)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="<unk>", pad_token="<|pad|>", eos_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = CHATML
    return tok


class MultiCharTokenBoundaryTests(unittest.TestCase):
    def test_a_token_straddling_a_span_boundary_stays_trainable(self):
        from scripts.train import _assistant_char_spans

        tok = _build_bpe_tokenizer()
        full, spans, fell_back = _assistant_char_spans(tok, DIALOGUE, use_template=True)
        self.assertFalse(fell_back)
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        straddlers = [
            (s, e) for s, e in enc["offset_mapping"]
            if e - s > 1 and any(s < a < e or s < b < e for a, b in spans)
        ]
        self.assertTrue(straddlers, "the tokenizer must merge across at least one boundary")
        ex, info = _encode_multi_turn_example(tok, DIALOGUE, 512)
        self.assertEqual(ex["input_ids"], enc["input_ids"])
        for (s, e), label in zip(enc["offset_mapping"], ex["labels"]):
            overlaps = any(max(s, a) < min(e, b) for a, b in spans)
            self.assertEqual(label != -100, overlaps, (s, e))
        # Every assistant character is covered by a trained token.
        covered = {i for (s, e), label in zip(enc["offset_mapping"], ex["labels"]) if label != -100 for i in range(s, e)}
        for a, b in spans:
            self.assertTrue(set(range(a, b)) <= covered)


class TruncationTests(unittest.TestCase):
    def setUp(self):
        self.tok = build_tiny_tokenizer()
        self.full = len(self.tok(self.tok.apply_chat_template(DIALOGUE, tokenize=False))["input_ids"])

    def test_oldest_exchange_dropped_system_kept(self):
        ex, info = _encode_multi_turn_example(self.tok, DIALOGUE, self.full - 1)
        text = self.tok.decode(ex["input_ids"]).replace(" ", "")
        self.assertIn("Bebrief.", text)
        self.assertNotIn("Green.", text)
        self.assertIn("Blue.", text)
        self.assertEqual(info["dropped_turns"], 2)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["dropped"])
        self.assertEqual(info["assistant_turns"], 1)

    def test_fits_means_untouched(self):
        _, info = _encode_multi_turn_example(self.tok, DIALOGUE, self.full)
        self.assertFalse(info["truncated"])
        self.assertEqual(info["dropped_turns"], 0)

    def test_final_exchange_too_long_drops_row(self):
        ex, info = _encode_multi_turn_example(self.tok, DIALOGUE, 10)
        self.assertTrue(info["dropped"])
        self.assertTrue(all(label == -100 for label in ex["labels"]))
        self.assertLessEqual(len(ex["input_ids"]), 10)


class ReportBlockTests(unittest.TestCase):
    def test_summary_counts(self):
        tok = build_tiny_tokenizer()
        full = len(tok(tok.apply_chat_template(DIALOGUE, tokenize=False))["input_ids"])
        infos = [
            _encode_multi_turn_example(tok, DIALOGUE, full)[1],
            _encode_multi_turn_example(tok, DIALOGUE, full - 1)[1],
            _encode_multi_turn_example(tok, DIALOGUE, 10)[1],
        ]
        block = _summarize_multi_turn(infos)
        trained = infos[0]["trained_tokens"] + infos[1]["trained_tokens"]
        total = infos[0]["total_tokens"] + infos[1]["total_tokens"]
        self.assertEqual(block, {
            "rows": 2,
            "assistant_token_share": round(trained / total, 4),
            "assistant_turns": 3,
            "truncated_rows": 1,
            "dropped_turns": 4,
            "dropped_rows": 1,
            "mask_fallback_rows": 0,
        })

    def test_no_multi_turn_rows_means_no_block(self):
        self.assertIsNone(_summarize_multi_turn([]))


class SimulateBlockTests(unittest.TestCase):
    def test_simulate_block_is_approximate_and_counts_rows(self):
        from app.services.training_service import _simulate_multi_turn_block

        rows = [
            {"messages": DIALOGUE},
            {"messages": DIALOGUE[1:3]},  # single exchange: not counted
            {"question": "q", "answer": "a"},
            {"messages": DIALOGUE[:3]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            block = _simulate_multi_turn_block(path, 2048)
            self.assertTrue(block["approximate"])
            self.assertEqual(block["rows"], 2)
            self.assertEqual(block["assistant_turns"], 3)
            self.assertGreater(block["assistant_token_share"], 0)
            self.assertLess(block["assistant_token_share"], 1)
            # A tiny window drops the oldest exchange (chars / 4 ≈ tokens).
            small = _simulate_multi_turn_block(path, 18)
            self.assertEqual(small["truncated_rows"], 1)
            self.assertEqual(small["dropped_turns"], 2)
            self.assertIsNone(_simulate_multi_turn_block(Path(tmp) / "missing.jsonl", 2048))

    def test_no_multi_turn_rows_gives_none(self):
        from app.services.training_service import _simulate_multi_turn_block

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(json.dumps({"question": "q", "answer": "a"}) + "\n", encoding="utf-8")
            self.assertIsNone(_simulate_multi_turn_block(path, 2048))


class PairwiseTrainerResolverTests(unittest.TestCase):
    def test_orpo_resolves_on_installed_trl(self):
        try:
            import trl
        except ImportError:
            self.skipTest("trl not installed")
        trainer_cls, config_cls = _resolve_pairwise_trainer(trl, "orpo")
        self.assertIsNotNone(trainer_cls)
        self.assertIsNotNone(config_cls)
        self.assertEqual(trainer_cls.__name__, "ORPOTrainer")
        self.assertEqual(config_cls.__name__, "ORPOConfig")

    def test_dpo_resolves_top_level(self):
        try:
            import trl
        except ImportError:
            self.skipTest("trl not installed")
        trainer_cls, _ = _resolve_pairwise_trainer(trl, "dpo")
        self.assertIs(trainer_cls, trl.DPOTrainer)

    def test_top_level_name_wins(self):
        class _Fake:
            ORPOTrainer = object
            ORPOConfig = dict

        self.assertEqual(_resolve_pairwise_trainer(_Fake, "orpo"), (object, dict))


class PairsFileFeedsPairwiseLoaderTests(unittest.TestCase):
    """The pairs file built from an eval is what the DPO / ORPO loader reads."""

    def test_adapter_reads_single_turn_and_chat_pairs(self):
        from datasets import load_dataset

        from app.services.preference_pairs_from_eval_service import build_pairs
        from scripts.train import _adapt_record_to_preference, _is_valid_preference_row

        records = [
            {"row_index": 0, "prompt": "What is two plus two?", "reference": "Four.", "prediction": "Five.",
             "row_judge_verdict": "wrong"},
            {"row_index": 1, "history": [{"role": "system", "content": "Be brief."},
                                          {"role": "user", "content": "Pick a colour."}],
             "reference": "Green.", "prediction": "Purple.", "row_judge_verdict": "partial"},
        ]
        pairs, _ = build_pairs(records, include=("wrong", "partial"), eval_result_id=7, experiment_id=3,
                               source_split="validation")
        self.assertEqual(len(pairs), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
            ds = load_dataset("json", data_files=str(path), split="train")
            mapped = ds.map(_adapt_record_to_preference, remove_columns=ds.column_names)
        self.assertEqual(mapped.column_names, ["prompt", "chosen", "rejected"])
        for row, pair in zip(mapped, pairs):
            self.assertTrue(_is_valid_preference_row(row))
            self.assertEqual((row["prompt"], row["chosen"], row["rejected"]),
                (pair["prompt"].strip(), pair["chosen"], pair["rejected"]),  # the loader strips the prompt
            )
        self.assertEqual(mapped[1]["prompt"], "system: Be brief.\nuser: Pick a colour.\nassistant:")


class OrpoTrainerConstructsTests(unittest.TestCase):
    """The resolved ORPO classes actually build a trainer on the installed trl (tiny cached model)."""

    def test_orpo_trainer_constructs(self):
        try:
            import trl  # noqa: F401
            import torch
            from datasets import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self.skipTest(f"missing dependency: {exc}")
        from test_training_correctness import REAL_MODEL

        try:
            tok = AutoTokenizer.from_pretrained(REAL_MODEL, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(REAL_MODEL, local_files_only=True, dtype=torch.float32)
        except Exception as exc:  # noqa: BLE001 - model not in the HF cache
            self.skipTest(f"{REAL_MODEL} is not in the local HF cache: {exc}")
        trainer_cls, config_cls = _resolve_pairwise_trainer(trl, "orpo")
        with tempfile.TemporaryDirectory() as tmp:
            args = config_cls(output_dir=tmp, per_device_train_batch_size=1, max_length=64, report_to=[], use_cpu=True)
            data = Dataset.from_list([{"prompt": "Q?", "chosen": "Good.", "rejected": "Bad."}] * 2)
            trainer = trainer_cls(model=model, args=args, train_dataset=data, processing_class=tok)
        self.assertEqual(type(trainer).__name__, "ORPOTrainer")


def _real_skip_reason() -> str | None:
    from test_training_correctness import _SKIP_REASON

    return _SKIP_REASON


class RealRuntimeMultiTurnTests(unittest.TestCase):
    """End to end through ``run_training`` on SmolLM2: mixed single-turn and
    multi-turn rows → ``loss_masking`` mixed + a ``multi_turn`` report.
    Skips without a GPU / BREWSLM_REAL_TRAINING=1 or the cached model."""

    def setUp(self):
        reason = _real_skip_reason()
        if reason:
            self.skipTest(reason)

    def test_real_tokenizer_masks_without_fallback(self):
        from test_training_correctness import REAL_MODEL
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REAL_MODEL, local_files_only=True)
        ex, info = _encode_multi_turn_example(tok, DIALOGUE, 512)
        self.assertFalse(info["fallback"])
        self.assertEqual(info["assistant_turns"], 2)
        trained = tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
        self.assertIn("Green.", trained)
        self.assertIn("Blue.", trained)
        self.assertNotIn("Pick a colour", trained)
        self.assertNotIn("Be brief", trained)

    def test_run_training_reports_multi_turn(self):
        import argparse
        import os

        from test_training_correctness import REAL_MODEL

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        rows = []
        for n in range(8):
            rows.append({"messages": [
                {"role": "system", "content": "Answer with one word."},
                {"role": "user", "content": f"Item {n}: colour?"},
                {"role": "assistant", "content": "Green."},
                {"role": "user", "content": "And size?"},
                {"role": "assistant", "content": "Small."},
            ]})
            rows.append({
                "messages": [{"role": "user", "content": f"Shape {n}?"}, {"role": "assistant", "content": "Round."}],
                "question": f"Shape {n}?", "answer": "Round.",
            })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            config = {
                "training_mode": "sft", "task_type": "causal_lm", "use_lora": True,
                "num_epochs": 1, "batch_size": 4, "gradient_accumulation_steps": 1,
                "max_seq_length": 256, "gradient_checkpointing": False, "flash_attention": False,
                "observability_enabled": False, "save_steps": 10_000, "eval_steps": 10_000,
                "auto_oom_retry": False,
            }
            (root / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
            args = argparse.Namespace(
                project=0, experiment=0, output=str(root / "run"), base_model=REAL_MODEL,
                config=str(root / "training_config.json"), train_file=str(root / "train.jsonl"),
                val_file=None, data_dir=str(root), max_train_samples=0, max_eval_samples=0, seed=42,
            )
            from scripts.train import run_training

            report = run_training(args)
        self.assertEqual(report["runtime_environment"]["loss_masking"], "mixed:completion_only+assistant_only")
        block = report["multi_turn"]
        self.assertEqual(block["rows"], 8)
        self.assertEqual(block["assistant_turns"], 16)
        self.assertEqual(block["mask_fallback_rows"], 0)
        self.assertTrue(0 < block["assistant_token_share"] < 1)


class SimulateLoopStoresBlockTests(unittest.TestCase):
    """A simulate run on multi-turn rows reports the assistant-token share
    in ``config._runtime.multi_turn`` (one event loop for the whole test —
    the StaticPool engine doesn't survive loop swaps)."""

    def _simulate(self, training_mode=None):
        import asyncio
        from unittest import mock
        from uuid import uuid4

        from app.config import settings
        from app.database import async_session_factory, init_db
        from app.models.experiment import Experiment, ExperimentStatus
        from app.models.project import Project
        from app.services import training_service

        async def _noop(*_a, **_k):
            return None

        async def _scenario():
            await init_db()
            async with async_session_factory() as db:
                project = Project(name=f"mt-{uuid4().hex[:8]}", description="multi-turn simulate")
                db.add(project)
                await db.flush()
                prepared = settings.DATA_DIR / "projects" / str(project.id) / "prepared"
                prepared.mkdir(parents=True, exist_ok=True)
                (prepared / "train.jsonl").write_text(
                    json.dumps({"messages": DIALOGUE}) + "\n" + json.dumps({"question": "q", "answer": "a"}) + "\n",
                    encoding="utf-8",
                )
                exp = Experiment(
                    project_id=project.id, name="mt-sim", status=ExperimentStatus.RUNNING,
                    base_model="tiny", output_dir=str(prepared.parent / "experiments" / "mt"),
                    config={"_runtime": {"backend": "simulate"}},
                )
                db.add(exp)
                await db.commit()
                exp_id = exp.id
            with mock.patch.object(training_service.asyncio, "sleep", _noop), \
                    mock.patch.object(training_service, "broadcast_metric", _noop), \
                    mock.patch.object(training_service, "broadcast_event", _noop), \
                    mock.patch.object(training_service, "load_project_vibe_check_config", lambda *a, **k: {"enabled": False}), \
                    mock.patch.object(training_service, "_safe_build_auto_rag_index", _noop), \
                    mock.patch.object(training_service, "_safe_score_unlabeled_pool", _noop), \
                    mock.patch.object(training_service, "_fire_seed_group_aggregation", _noop), \
                    mock.patch("app.services.training_telemetry_service.record_observability_event", lambda *a, **k: None):
                await training_service._simulate_training_loop(
                    exp_id, {"num_epochs": 1, "save_steps": 100, "max_seq_length": 2048,
                             **({"training_mode": training_mode} if training_mode else {})},
                )
            async with async_session_factory() as db:
                return await db.get(Experiment, exp_id)

        return asyncio.run(_scenario())

    def test_simulate_run_reports_multi_turn(self):
        from app.models.experiment import ExperimentStatus

        exp = self._simulate()
        self.assertEqual(exp.status, ExperimentStatus.COMPLETED)
        block = exp.config["_runtime"]["multi_turn"]
        self.assertTrue(block["approximate"])
        self.assertEqual(block["rows"], 1)
        self.assertEqual(block["assistant_turns"], 2)
        self.assertTrue(0 < block["assistant_token_share"] < 1)

    def test_preference_modes_report_no_multi_turn_block(self):
        for mode in ("dpo", "orpo"):
            with self.subTest(mode=mode):
                exp = self._simulate(mode)
                self.assertNotIn("multi_turn", exp.config["_runtime"])


if __name__ == "__main__":
    unittest.main()
