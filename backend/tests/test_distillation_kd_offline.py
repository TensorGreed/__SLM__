"""Offline-KD data prep + readiness gate + recipes — Track 1, Epic A, slice 2.

Pure helpers tested with a fake tokenizer (no torch / transformers needed for
the capture+alignment layer).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.services.distillation.kd_capture import (
    build_offline_kd_records,
    build_teacher_target_topk,
    load_teacher_capture,
    verify_capture_artifact,
)
from app.services.training_recipe_service import (
    get_training_recipe,
    list_training_recipes,
    resolve_training_recipe,
)


# A tiny deterministic "tokenizer": each whitespace word is one token.
_VOCAB = {"the": 1, "cat": 2, "sat": 3, "dog": 4, "ran": 5, "fast": 6}


def _encode(text: str) -> list[int]:
    return [_VOCAB[w] for w in text.split() if w in _VOCAB]


def _token_to_id(token: str):
    return _VOCAB.get(token)


def _capture_row(question: str, completion: str, positions: list[list[list]]) -> dict:
    return {
        "question": question,
        "answer": completion,
        "teacher_completion": completion,
        "teacher_logits": [{"token": "x", "top_k": p} for p in positions],
        "source": "teacher_capture",
        "status": "accepted",
    }


class CaptureGateTests(unittest.TestCase):
    def test_missing_file_not_ok(self):
        gate = verify_capture_artifact("/no/such/path/teacher_capture.jsonl")
        self.assertFalse(gate["ok"])
        self.assertIn("No teacher-capture artifact", gate["message"])

    def test_empty_file_not_ok(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cap.jsonl"
            p.write_text("", encoding="utf-8")
            gate = verify_capture_artifact(p)
            self.assertFalse(gate["ok"])
            self.assertEqual(gate["row_count"], 0)

    def test_rows_without_logits_not_ok(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cap.jsonl"
            p.write_text(json.dumps({"question": "q", "teacher_logits": []}) + "\n", encoding="utf-8")
            gate = verify_capture_artifact(p)
            self.assertFalse(gate["ok"])
            self.assertEqual(gate["row_count"], 1)
            self.assertEqual(gate["rows_with_logits"], 0)

    def test_ok_when_logits_present(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cap.jsonl"
            row = _capture_row("the cat", "sat", [[["sat", -0.1], ["ran", -2.0]]])
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            gate = verify_capture_artifact(p)
            self.assertTrue(gate["ok"])
            self.assertEqual(gate["rows_with_logits"], 1)
            self.assertEqual(len(load_teacher_capture(p)), 1)


class TeacherTargetTopkTests(unittest.TestCase):
    def test_maps_tokens_drops_unknown_and_pads(self):
        row = _capture_row(
            "the cat",
            "sat",
            [[["sat", -0.1], ["UNKNOWN", -0.5], ["dog", -1.0]]],
        )
        ids, logprobs, stats = build_teacher_target_topk(row, _token_to_id, top_k=3)
        self.assertEqual(len(ids), 1)
        # "sat"->3, "UNKNOWN" dropped, "dog"->4, then pad (-1) to width 3.
        self.assertEqual(ids[0], [3, 4, -1])
        self.assertAlmostEqual(logprobs[0][0], -0.1, places=5)
        self.assertEqual(stats["mapped"], 2)
        self.assertEqual(stats["dropped"], 1)

    def test_top_k_truncates(self):
        row = _capture_row("the", "cat", [[["cat", -0.1], ["dog", -0.2], ["ran", -0.3]]])
        ids, _logprobs, _stats = build_teacher_target_topk(row, _token_to_id, top_k=2)
        self.assertEqual(len(ids[0]), 2)


class OfflineRecordTests(unittest.TestCase):
    def test_prompt_masked_completion_aligned(self):
        # prompt "the cat" -> [1,2]; completion "sat ran" -> [3,5]; two teacher positions.
        row = _capture_row(
            "the cat",
            "sat ran",
            [[["sat", -0.1], ["dog", -2.0]], [["ran", -0.2], ["fast", -1.0]]],
        )
        records, stats = build_offline_kd_records(
            [row], _encode, _token_to_id, top_k=2, max_seq_length=64
        )
        self.assertEqual(stats["built"], 1)
        rec = records[0]
        self.assertEqual(rec["input_ids"], [1, 2, 3, 5])
        # Prompt positions masked; completion positions carry the gold token ids.
        self.assertEqual(rec["labels"], [-100, -100, 3, 5])
        # Teacher rows: pad for prompt, mapped for completion.
        self.assertEqual(rec["teacher_topk_ids"][0], [-1, -1])
        self.assertEqual(rec["teacher_topk_ids"][2], [3, 4])  # sat, dog
        self.assertEqual(rec["teacher_topk_ids"][3], [5, 6])  # ran, fast
        # Alignment width matches input length.
        self.assertEqual(len(rec["teacher_topk_ids"]), len(rec["input_ids"]))

    def test_extra_completion_tokens_without_teacher_are_masked(self):
        # completion re-tokenizes to 2 tokens but only 1 captured position.
        row = _capture_row("the", "sat ran", [[["sat", -0.1], ["dog", -2.0]]])
        records, stats = build_offline_kd_records(
            [row], _encode, _token_to_id, top_k=2, max_seq_length=64
        )
        rec = records[0]
        # prompt "the"->[1]; completion "sat ran"->[3,5]; only pos0 has teacher.
        self.assertEqual(rec["input_ids"], [1, 3, 5])
        self.assertEqual(rec["labels"], [-100, 3, -100])  # 2nd completion tok masked
        self.assertEqual(stats["positions_without_teacher"], 1)

    def test_truncation_to_max_seq_length(self):
        row = _capture_row("the cat sat", "dog ran fast", [
            [["dog", -0.1]], [["ran", -0.2]], [["fast", -0.3]],
        ])
        records, stats = build_offline_kd_records(
            [row], _encode, _token_to_id, top_k=1, max_seq_length=4
        )
        rec = records[0]
        self.assertEqual(len(rec["input_ids"]), 4)
        self.assertEqual(len(rec["teacher_topk_ids"]), 4)
        self.assertEqual(stats["truncated"], 1)

    def test_rows_without_prompt_or_completion_skipped(self):
        row = {"question": "", "teacher_completion": "", "teacher_logits": []}
        records, stats = build_offline_kd_records([row], _encode, _token_to_id, top_k=2)
        self.assertEqual(records, [])
        self.assertEqual(stats["skipped"], 1)


class KDRecipeTests(unittest.TestCase):
    def test_kd_recipes_present_with_offline_defaults(self):
        for rid in ("recipe.kd.classification", "recipe.kd.qa", "recipe.kd.span_extraction"):
            recipe = get_training_recipe(rid)
            self.assertIsNotNone(recipe, rid)
            patch = recipe["config_patch"]
            self.assertEqual(patch["training_mode"], "distillation")
            self.assertTrue(patch["distillation_offline"])
            self.assertEqual(patch["distillation_alpha"], 0.5)
            self.assertEqual(patch["distillation_temperature"], 2.0)
            self.assertEqual(patch["task_type"], "causal_lm")

    def test_kd_recipes_listed(self):
        ids = {r["recipe_id"] for r in list_training_recipes()}
        self.assertTrue(
            {"recipe.kd.classification", "recipe.kd.qa", "recipe.kd.span_extraction"} <= ids
        )

    def test_resolve_layers_base_and_overrides(self):
        resolved = resolve_training_recipe(
            "recipe.kd.qa",
            base_config={"base_model": "HuggingFaceTB/SmolLM2-135M-Instruct"},
            overrides={"distillation_temperature": 3.0},
        )
        cfg = resolved["resolved_config"]
        self.assertEqual(cfg["base_model"], "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertEqual(cfg["distillation_temperature"], 3.0)  # override wins
        self.assertEqual(cfg["training_mode"], "distillation")
        self.assertEqual(resolved["missing_required_fields"], [])


if __name__ == "__main__":
    unittest.main()
