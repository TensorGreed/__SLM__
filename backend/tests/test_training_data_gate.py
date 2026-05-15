"""Pre-training data-shape gate (training_data_gate).

Pins:
- SFT with rows carrying ``answer``/``completion``/``output``/
  ``response`` passes the gate.
- SFT with text-only rows (the domain-pretrain shape that bit the
  Qwen-PII-V2 run) is refused with an actionable message.
- DOMAIN_PRETRAIN doesn't gate; text-only is legitimate there.
- DPO / ORPO don't gate (alignment path has its own contract checks).
- Missing file / empty file / malformed JSONL all surface as
  ``ok=False`` rather than crashing the runtime startup.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.services.training_data_gate import (
    DEFAULT_TARGET_FIELDS,
    count_rows_with_any_field,
    sample_train_rows,
    verify_training_data_has_targets,
)


def _write_jsonl(rows: list[dict]) -> Path:
    fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    try:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    finally:
        fh.close()
    return Path(fh.name)


class TrainingDataGateTests(unittest.TestCase):
    # ── Happy path ─────────────────────────────────────────────────

    def test_sft_with_answer_field_passes(self):
        path = _write_jsonl(
            [
                {"question": "what is 2+2?", "answer": "4"},
                {"prompt": "summarize", "completion": "ok"},
                {"text": "extract entities", "response": "{...}"},
            ]
        )
        try:
            report = verify_training_data_has_targets(
                path, training_mode="sft"
            )
            self.assertTrue(report["ok"], msg=report["message"])
            self.assertTrue(report["gate_applied"])
            self.assertEqual(report["rows_with_target"], 3)
            self.assertEqual(report["ratio"], 1.0)
        finally:
            path.unlink(missing_ok=True)

    def test_sft_with_mixed_rows_passes_if_any_have_target(self):
        path = _write_jsonl(
            [
                {"text": "raw chunk only"},
                {"text": "raw chunk only"},
                {"question": "good row", "answer": "the answer"},
            ]
        )
        try:
            report = verify_training_data_has_targets(
                path, training_mode="sft"
            )
            self.assertTrue(report["ok"], msg=report["message"])
            self.assertEqual(report["rows_with_target"], 1)
            self.assertAlmostEqual(report["ratio"], 1 / 3)
        finally:
            path.unlink(missing_ok=True)

    # ── The Qwen-PII-V2 incident ──────────────────────────────────

    def test_sft_with_only_text_field_is_refused(self):
        """The exact shape of the experiment-10 train.jsonl: rows
        carry ``text`` (raw HF chunks) and no answer / completion /
        output / response field. The gate must catch this before the
        runtime burns hours on a no-target run."""
        path = _write_jsonl(
            [
                {
                    "source_doc": "hf_ai4privacy_pii-masking-200k_train.jsonl",
                    "chunk_id": i,
                    "text": f"mixed chunk {i}",
                    "_task_profile": "instruction_sft",
                }
                for i in range(50)
            ]
        )
        try:
            report = verify_training_data_has_targets(
                path, training_mode="sft"
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["rows_with_target"], 0)
            self.assertEqual(report["ratio"], 0.0)
            # Message names the missing fields + the likely cause so
            # the operator can act on it without re-reading code.
            msg = report["message"]
            self.assertIn("answer", msg)
            self.assertIn("eval F1 will be 0", msg)
            self.assertIn("dataset_import mapper", msg)
        finally:
            path.unlink(missing_ok=True)

    def test_sft_with_empty_string_answer_counts_as_missing(self):
        """Whitespace-only answer is functionally the same as a
        missing target field — caught alongside the obvious case."""
        path = _write_jsonl(
            [
                {"text": "q", "answer": ""},
                {"text": "q", "answer": "   "},
            ]
        )
        try:
            report = verify_training_data_has_targets(
                path, training_mode="sft"
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["rows_with_target"], 0)
        finally:
            path.unlink(missing_ok=True)

    # ── Gate skipped for non-SFT modes ─────────────────────────────

    def test_domain_pretrain_is_not_gated(self):
        path = _write_jsonl(
            [{"text": f"chunk {i}"} for i in range(10)]
        )
        try:
            report = verify_training_data_has_targets(
                path, training_mode="domain_pretrain"
            )
            self.assertTrue(report["ok"])
            self.assertFalse(report["gate_applied"])
        finally:
            path.unlink(missing_ok=True)

    def test_dpo_is_not_gated(self):
        path = _write_jsonl(
            [{"prompt": "x", "chosen": "a", "rejected": "b"}]
        )
        try:
            report = verify_training_data_has_targets(
                path, training_mode="dpo"
            )
            self.assertTrue(report["ok"])
            self.assertFalse(report["gate_applied"])
        finally:
            path.unlink(missing_ok=True)

    # ── Failure modes the gate must survive ────────────────────────

    def test_missing_file_is_reported_not_crashed(self):
        report = verify_training_data_has_targets(
            Path("/tmp/this-file-absolutely-does-not-exist.jsonl"),
            training_mode="sft",
        )
        self.assertFalse(report["ok"])
        self.assertIn("not found", report["message"])

    def test_empty_file_is_reported(self):
        fh = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        fh.close()
        path = Path(fh.name)
        try:
            report = verify_training_data_has_targets(
                path, training_mode="sft"
            )
            self.assertFalse(report["ok"])
            self.assertIn("empty", report["message"])
        finally:
            path.unlink(missing_ok=True)

    def test_malformed_jsonl_lines_dont_crash(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        path.write('{"question": "ok", "answer": "fine"}\n')
        path.write("not a json line\n")
        path.write('{"text": "no target"}\n')
        path.close()
        path_obj = Path(path.name)
        try:
            report = verify_training_data_has_targets(
                path_obj, training_mode="sft"
            )
            # Two parseable rows; one has a target → passes.
            self.assertTrue(report["ok"])
            self.assertEqual(report["sample_size"], 2)
            self.assertEqual(report["rows_with_target"], 1)
        finally:
            path_obj.unlink(missing_ok=True)

    # ── Helper coverage ────────────────────────────────────────────

    def test_count_rows_with_any_field_picks_first_match(self):
        rows = [
            {"answer": "x"},
            {"completion": "y"},
            {"output": "z"},
            {"text": "no target"},
        ]
        count = count_rows_with_any_field(rows, DEFAULT_TARGET_FIELDS)
        self.assertEqual(count, 3)

    def test_sample_train_rows_caps_at_sample_n(self):
        path = _write_jsonl(
            [{"text": f"row {i}"} for i in range(100)]
        )
        try:
            scanned, rows = sample_train_rows(path, sample_n=10)
            self.assertEqual(len(rows), 10)
            # scanned reflects what we read before hitting the cap,
            # so it should equal the row count (10), not the file
            # total (100).
            self.assertEqual(scanned, 10)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
