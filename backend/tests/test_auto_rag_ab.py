"""Tests for the Phase 9c A/B harness (pure-function pieces).

End-to-end training + eval isn't unit-tested here — that's what the
live run produces. These tests exercise the harness's correctness-
critical pure pieces:

  * qa-sft flatten reads ``input.question`` + ``expected.answer`` from
    the template's nested shape.
  * 70/15/15 split matches ``demo_project_service`` (consistent with
    what a real demo project would train + eval on).
  * llama3 prompt format matches train.py's ``_qa_to_chat_text`` —
    inference uses the same chat template training used.
  * RAG-prompt builder prepends a system message preamble before
    the user turn (mirrors the playground's insert-after-system
    behavior).
  * Generated-answer cleaner strips the prompt prefix + the trailing
    ``<|eot_id|>`` so token-level F1 scores the answer only.
  * Gate criterion enforces both the lift threshold AND the non-
    overlapping bands check on the one QA-SFT template.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.auto_rag_ab import (  # noqa: E402
    GATE_MIN_LIFT_PCT,
    QA_SFT_TEMPLATES,
    RunResult,
    TemplateSummary,
    _build_rag_preamble,
    _clean_generated_answer,
    _flatten_qa_row,
    _format_llama3_inference_prompt,
    _format_llama3_rag_prompt,
    _split_70_15_15,
    aggregate_results,
    apply_gate,
    format_markdown_block,
    prepare_template_splits,
)


# ─────────────────────────────────────────────────────────────────────
# Flatten + split (lightly different from curriculum_ab's variants)
# ─────────────────────────────────────────────────────────────────────


class FlattenAndSplitTests(unittest.TestCase):
    def test_flatten_qa_row_from_template_shape(self):
        row = {
            "key": "g001",
            "input": {"question": "Can I roll over PTO?"},
            "expected": {"answer": "Up to 5 days."},
            "rationale": "ignored",
        }
        flat = _flatten_qa_row(row)
        self.assertEqual(flat, {"question": "Can I roll over PTO?", "answer": "Up to 5 days."})

    def test_flatten_drops_rows_missing_question_or_answer(self):
        # Missing answer.
        self.assertIsNone(_flatten_qa_row(
            {"input": {"question": "Q?"}, "expected": {}}
        ))
        # Missing question.
        self.assertIsNone(_flatten_qa_row(
            {"input": {}, "expected": {"answer": "A."}}
        ))
        # Wrong shape entirely.
        self.assertIsNone(_flatten_qa_row(
            {"question": "flat shape (the flatten expects nested)", "answer": "x"}
        ))

    def test_split_70_15_15_on_200_rows(self):
        rows = [{"i": i} for i in range(200)]
        train, val, test = _split_70_15_15(rows)
        # 200/7 = 28; matches demo_project_service._split_rows.
        self.assertEqual(len(train), 144)
        self.assertEqual(len(val), 28)
        self.assertEqual(len(test), 28)
        # Deterministic + non-overlapping (no shuffle).
        self.assertEqual(train[0]["i"], 0)
        self.assertEqual(val[0]["i"], 144)
        self.assertEqual(test[0]["i"], 172)


# ─────────────────────────────────────────────────────────────────────
# Live template prep — catches drift if the QA template ever changes
# shape away from what the harness's flatten() expects.
# ─────────────────────────────────────────────────────────────────────


class LiveTemplatePrepTests(unittest.TestCase):
    def test_policy_qa_style_flattens_at_least_95pct(self):
        for slug in QA_SFT_TEMPLATES:
            with self.subTest(template=slug), TemporaryDirectory() as td:
                counts = prepare_template_splits(slug, Path(td))
                # 200 gold rows; ≥ 95% must survive the flatten.
                self.assertGreaterEqual(
                    counts["train"], 130,
                    f"{slug}: train rows dropped low ({counts['train']}/200)",
                )
                self.assertGreater(counts["val"], 0)
                self.assertGreater(counts["test"], 0)
                train_path = Path(td) / "train.jsonl"
                with train_path.open() as f:
                    row = json.loads(f.readline())
                self.assertIn("question", row)
                self.assertIn("answer", row)


# ─────────────────────────────────────────────────────────────────────
# Prompt builders — without-RAG and with-RAG must produce the same
# user turn shape so the A/B compares like with like, modulo the
# prepended preamble.
# ─────────────────────────────────────────────────────────────────────


class PromptBuilderTests(unittest.TestCase):
    def test_inference_prompt_uses_llama3_headers_and_ends_with_assistant_marker(self):
        prompt = _format_llama3_inference_prompt("What's the policy?")
        # User turn opens with the user header.
        self.assertIn("<|start_header_id|>user<|end_header_id|>", prompt)
        # User turn ends with eot, then the assistant header opens.
        self.assertIn("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", prompt)
        # Ends in the assistant header's body (model fills in here).
        self.assertTrue(prompt.endswith("\n\n"))

    def test_rag_prompt_inserts_system_preamble_before_user_turn(self):
        retrieved = [
            {"question": "How many PTO?", "answer": "Up to 5."},
            {"question": "Can I email PII?", "answer": "No."},
        ]
        prompt = _format_llama3_rag_prompt("What about today?", retrieved)
        # System block opens FIRST (before the user block).
        sys_pos = prompt.find("<|start_header_id|>system<|end_header_id|>")
        usr_pos = prompt.find("<|start_header_id|>user<|end_header_id|>")
        self.assertGreaterEqual(sys_pos, 0)
        self.assertGreater(usr_pos, sys_pos)
        # Both retrieved Q&As surface in the preamble body.
        self.assertIn("How many PTO?", prompt)
        self.assertIn("Up to 5.", prompt)
        self.assertIn("Can I email PII?", prompt)
        # The user's actual question is still in the user turn.
        self.assertIn("What about today?", prompt)

    def test_rag_preamble_numbers_pairs_for_citation(self):
        retrieved = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
        preamble = _build_rag_preamble(retrieved)
        # Pair numbering [1] [2] [3] — the prompt asks the model
        # to cite by number, so the format must be consistent.
        self.assertIn("[1]", preamble)
        self.assertIn("[2]", preamble)
        self.assertIn("[3]", preamble)
        self.assertIn("Reference Q&A pairs", preamble)


class GeneratedAnswerCleanerTests(unittest.TestCase):
    def test_strips_prompt_prefix_and_eot_tail(self):
        decoded = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What's the policy?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "Up to five PTO days carry over.<|eot_id|>"
        )
        self.assertEqual(_clean_generated_answer(decoded), "Up to five PTO days carry over.")

    def test_handles_missing_eot_tail(self):
        # When generate hits max_new_tokens before emitting eot, the
        # cleaner still returns whatever the assistant emitted.
        decoded = (
            "<|start_header_id|>user<|end_header_id|>\n\nQ?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "Partial answer with no eot"
        )
        self.assertEqual(_clean_generated_answer(decoded), "Partial answer with no eot")

    def test_only_returns_the_final_assistant_turn_in_multi_turn_decode(self):
        """If the model echoed the prompt + a stray prior assistant
        turn, the cleaner pulls the LAST assistant header's body."""
        decoded = (
            "<|start_header_id|>system<|end_header_id|>\n\nReference Q&A pairs ...<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nQ?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "The real answer<|eot_id|>"
        )
        self.assertEqual(_clean_generated_answer(decoded), "The real answer")


# ─────────────────────────────────────────────────────────────────────
# Gate + aggregation
# ─────────────────────────────────────────────────────────────────────


class GateAndAggregationTests(unittest.TestCase):
    def _summary(self, slug: str, on: list[float], off: list[float]) -> TemplateSummary:
        return TemplateSummary(template=slug, on_f1s=on, off_f1s=off)

    def test_gate_passes_with_5pct_lift_and_separated_bands(self):
        summaries = {
            "policy-qa-style": self._summary(
                "policy-qa-style",
                on=[0.65, 0.66, 0.67, 0.68, 0.69],
                off=[0.55, 0.56, 0.57, 0.58, 0.59],
            ),
        }
        decision = apply_gate(summaries)
        self.assertTrue(decision.passed, decision.reason)

    def test_gate_fails_on_lift_below_threshold(self):
        summaries = {
            "policy-qa-style": self._summary(
                "policy-qa-style",
                on=[0.61, 0.62, 0.63, 0.61, 0.62],   # ~+3% over off
                off=[0.59, 0.60, 0.61, 0.59, 0.60],
            ),
        }
        decision = apply_gate(summaries)
        self.assertFalse(decision.passed)
        self.assertIn(f"< {GATE_MIN_LIFT_PCT}%", decision.reason)

    def test_gate_fails_on_overlapping_bands_despite_lift(self):
        summaries = {
            "policy-qa-style": self._summary(
                "policy-qa-style",
                # 30% lift in mean but huge variance → bands overlap.
                on=[0.30, 0.95, 0.45, 0.85, 0.55],
                off=[0.40, 0.50, 0.45, 0.55, 0.50],
            ),
        }
        decision = apply_gate(summaries)
        self.assertFalse(decision.passed)
        self.assertIn("bands overlap", decision.reason)

    def test_gate_fails_on_empty_summaries(self):
        decision = apply_gate({})
        self.assertFalse(decision.passed)
        self.assertIn("no successful runs", decision.reason)

    def test_aggregate_drops_failed_runs(self):
        runs = [
            RunResult(template="t", seed=0, without_rag_f1=None, with_rag_f1=None,
                      train_runtime_seconds=10, eval_runtime_seconds=10,
                      output_dir="", error="OOM"),
            RunResult(template="t", seed=0, without_rag_f1=0.5, with_rag_f1=0.7,
                      train_runtime_seconds=10, eval_runtime_seconds=10, output_dir=""),
        ]
        summaries = aggregate_results(runs)
        # Failed run dropped silently; surviving run shows up.
        self.assertEqual(summaries["t"].off_f1s, [0.5])
        self.assertEqual(summaries["t"].on_f1s, [0.7])


class MarkdownFormatterTests(unittest.TestCase):
    def test_pass_message_and_table_render(self):
        summaries = {
            "policy-qa-style": TemplateSummary(
                template="policy-qa-style",
                on_f1s=[0.65, 0.66, 0.67, 0.68, 0.69],
                off_f1s=[0.55, 0.56, 0.57, 0.58, 0.59],
            ),
        }
        gate = apply_gate(summaries)
        md = format_markdown_block(
            [], summaries, gate,
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            num_epochs=3, seeds=[0, 1, 2, 3, 4],
        )
        self.assertIn("policy-qa-style", md)
        self.assertIn("Gate: PASS", md)
        self.assertIn("Phase 9d cleared to ship", md)
        # 1-template caveat is in the markdown for auditability.
        self.assertIn("one** QA-SFT template", md)

    def test_fail_message_keeps_phase_9b_alive(self):
        summaries = {
            "policy-qa-style": TemplateSummary(
                template="policy-qa-style",
                on_f1s=[0.61, 0.62, 0.63, 0.61, 0.62],
                off_f1s=[0.59, 0.60, 0.61, 0.59, 0.60],
            ),
        }
        gate = apply_gate(summaries)
        md = format_markdown_block(
            [], summaries, gate, base_model="m", num_epochs=3, seeds=[0, 1, 2, 3, 4],
        )
        self.assertIn("Gate: FAIL", md)
        # Surfacing that auto-RAG remains usable as opt-in.
        self.assertIn("power-user", md)


if __name__ == "__main__":
    unittest.main()
