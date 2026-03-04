"""Phase 2 tests: generic record normalization and schema profiling."""

import unittest

from app.services.record_normalization import (
    build_schema_profile,
    canonicalize_record,
    normalize_records,
)


class RecordNormalizationTests(unittest.TestCase):
    def test_explicit_field_mapping(self):
        row = {
            "payload": {
                "prompt": "What is SOC2?",
                "completion": "A security compliance framework.",
            }
        }
        mapped = canonicalize_record(
            row,
            field_mapping={
                "question": "payload.prompt",
                "answer": "payload.completion",
            },
        )
        self.assertIsNotNone(mapped)
        self.assertEqual(mapped["question"], "What is SOC2?")
        self.assertEqual(mapped["answer"], "A security compliance framework.")
        self.assertIn("Question:", mapped["text"])

    def test_heuristic_text_fallback(self):
        row = {"content": "Internal escalation runbook for incident response."}
        normalized = canonicalize_record(row)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["text"], row["content"])

    def test_normalize_records_drops_unusable_rows(self):
        rows = [{"text": "valid"}, {}, {"value": None}]
        normalized, dropped = normalize_records(rows)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(dropped, 2)

    def test_schema_profile_coverage(self):
        rows = [
            {"prompt": "Q1", "completion": "A1"},
            {"prompt": "Q2", "completion": "A2"},
            {"x": "fallback text"},
            {},
        ]
        profile = build_schema_profile(rows)
        self.assertEqual(profile["total_records"], 4)
        self.assertEqual(profile["normalized_records"], 3)
        self.assertEqual(profile["dropped_records"], 1)
        self.assertGreater(profile["normalization_coverage"], 70.0)


if __name__ == "__main__":
    unittest.main()
