"""Phase 99 — synthetic span-extraction generation.

User report (from PII demo screenshot): clicking "Generate" produced
Q&A pairs instead of `{text, entities: […]}` rows shaped for the
span_set scoring mode. Root cause was the SyntheticPanel having only
hardcoded `qa` / `conversation` modes; the new `span_extraction` mode
generates the right shape end-to-end with regex-based demo fallback +
teacher-model prompt + offset validation.

Pins the backend contract:

- Regex extractor catches the common PII patterns (email / phone /
  SSN / credit_card / ip_address) with correct offsets.
- Demo fallback emits `{text, entities, confidence, source: 'demo_heuristic'}`
  rows from source-text sentences.
- Teacher-output validator drops entities whose offsets don't match
  the claimed text — these are hallucinated spans that would poison
  training data.
- Validator filters to declared entity_types when supplied.
- Save endpoint writes JSONL in the canonical shape.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services.synthetic_service import (  # noqa: E402
    _extract_entities_via_regex,
    _generate_demo_span_rows,
    _validate_span_rows,
)


class RegexExtractorTests(unittest.TestCase):
    def test_extracts_email_with_correct_offsets(self):
        text = "Email me at jane@example.com please."
        ents = _extract_entities_via_regex(text)
        emails = [e for e in ents if e["type"] == "email"]
        self.assertEqual(len(emails), 1)
        e = emails[0]
        self.assertEqual(e["text"], "jane@example.com")
        self.assertEqual(text[e["start"] : e["end"]], e["text"])

    def test_extracts_seven_and_ten_digit_phones(self):
        text = "Call 555-0199 or 555-123-4567 today."
        ents = _extract_entities_via_regex(text)
        phones = [e for e in ents if e["type"] == "phone"]
        # Both formats covered.
        self.assertGreaterEqual(len(phones), 2)
        for e in phones:
            self.assertEqual(text[e["start"] : e["end"]], e["text"])

    def test_extracts_ssn(self):
        text = "SSN on file: 000-12-3456."
        ents = _extract_entities_via_regex(text)
        ssns = [e for e in ents if e["type"] == "ssn"]
        self.assertEqual(len(ssns), 1)
        self.assertEqual(ssns[0]["text"], "000-12-3456")

    def test_extracts_credit_card(self):
        text = "Card on file: 4242424242424242 (test)."
        ents = _extract_entities_via_regex(text)
        cards = [e for e in ents if e["type"] == "credit_card"]
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["text"], "4242424242424242")

    def test_extracts_ip_address(self):
        text = "Login from 192.0.2.55 detected."
        ents = _extract_entities_via_regex(text)
        ips = [e for e in ents if e["type"] == "ip_address"]
        self.assertEqual(len(ips), 1)
        self.assertEqual(ips[0]["text"], "192.0.2.55")

    def test_filters_by_allowed_entity_types(self):
        text = "Email me at jane@example.com or call 555-0100."
        ents = _extract_entities_via_regex(text, entity_types=["email"])
        types = {e["type"] for e in ents}
        self.assertEqual(types, {"email"})

    def test_entities_sorted_by_start_offset(self):
        text = "Call 555-0100 and email jane@example.com please."
        ents = _extract_entities_via_regex(text)
        starts = [e["start"] for e in ents]
        self.assertEqual(starts, sorted(starts))


class DemoFallbackTests(unittest.TestCase):
    def test_emits_text_plus_entities_with_correct_offsets(self):
        source = (
            "Customer Jane at jane@example.com called 555-0100. "
            "Another contact: 555-0199."
        )
        rows = _generate_demo_span_rows(source, num_rows=2, entity_types=None)
        self.assertEqual(len(rows), 2)
        for row in rows:
            text = row["text"]
            for ent in row["entities"]:
                self.assertEqual(text[ent["start"] : ent["end"]], ent["text"])

    def test_marks_source_as_demo_heuristic(self):
        rows = _generate_demo_span_rows(
            "Email: foo@bar.com.", num_rows=1, entity_types=None
        )
        self.assertEqual(rows[0]["source"], "demo_heuristic")
        self.assertEqual(rows[0]["model"], "regex")

    def test_confidence_higher_when_entities_found(self):
        with_pii = _generate_demo_span_rows(
            "Email me at a@b.com.", num_rows=1, entity_types=None
        )
        clean = _generate_demo_span_rows(
            "Release notes are clean.", num_rows=1, entity_types=None
        )
        self.assertGreater(with_pii[0]["confidence"], clean[0]["confidence"])

    def test_respects_entity_type_filter(self):
        rows = _generate_demo_span_rows(
            "Call 555-0100 and email a@b.com.",
            num_rows=1,
            entity_types=["email"],
        )
        for row in rows:
            for ent in row["entities"]:
                self.assertEqual(ent["type"], "email")


class ValidatorTests(unittest.TestCase):
    def test_keeps_valid_entities(self):
        raw = [
            {
                "text": "Email is foo@bar.com.",
                "entities": [
                    {"type": "email", "start": 9, "end": 20, "text": "foo@bar.com"}
                ],
            }
        ]
        out = _validate_span_rows(raw, entity_types=None)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]["entities"]), 1)
        self.assertEqual(out[0]["entities"][0]["text"], "foo@bar.com")

    def test_drops_entities_with_mismatched_offsets(self):
        # Teacher claimed text="foo@bar.com" but text[9:20] is "foo@bar.co"
        # — exactly the kind of off-by-one hallucination that poisons
        # training data.
        raw = [
            {
                "text": "Email is foo@bar.com.",
                "entities": [
                    {"type": "email", "start": 9, "end": 19, "text": "foo@bar.com"}
                ],
            }
        ]
        out = _validate_span_rows(raw, entity_types=None)
        # Row kept (text present) but the bad entity is dropped.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["entities"], [])

    def test_drops_entities_outside_text_range(self):
        raw = [
            {
                "text": "short",
                "entities": [
                    {"type": "email", "start": 100, "end": 200, "text": "x"}
                ],
            }
        ]
        out = _validate_span_rows(raw, entity_types=None)
        self.assertEqual(out[0]["entities"], [])

    def test_filters_to_allowed_entity_types(self):
        raw = [
            {
                "text": "Email a@b.com or call 555-0100",
                "entities": [
                    {"type": "email", "start": 6, "end": 13, "text": "a@b.com"},
                    {"type": "phone", "start": 22, "end": 30, "text": "555-0100"},
                ],
            }
        ]
        out = _validate_span_rows(raw, entity_types=["email"])
        types = {e["type"] for e in out[0]["entities"]}
        self.assertEqual(types, {"email"})

    def test_drops_rows_with_empty_text(self):
        raw = [{"text": "", "entities": []}, {"text": "real text", "entities": []}]
        out = _validate_span_rows(raw, entity_types=None)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "real text")

    def test_drops_entities_without_type_or_offsets(self):
        raw = [
            {
                "text": "Some text",
                "entities": [
                    {"type": "", "start": 0, "end": 4, "text": "Some"},
                    {"type": "email", "text": "Some"},  # no offsets
                    {"start": 0, "end": 4},  # no type
                ],
            }
        ]
        out = _validate_span_rows(raw, entity_types=None)
        self.assertEqual(out[0]["entities"], [])

    def test_handles_non_dict_payloads_gracefully(self):
        out = _validate_span_rows(["not a dict", None, 42], entity_types=None)
        self.assertEqual(out, [])


class SaveBatchShapeTests(unittest.TestCase):
    """End-to-end smoke test: the span save endpoint writes the JSONL
    in the canonical `{text, entities}` shape that
    `StructuredExtractionHandler`'s span_set scoring mode consumes."""

    def test_save_persists_canonical_shape(self):
        import asyncio
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        from app.services.synthetic_service import save_synthetic_span_batch

        rows = [
            {
                "text": "Email me at jane@example.com.",
                "entities": [
                    {
                        "type": "email",
                        "start": 12,
                        "end": 28,
                        "text": "jane@example.com",
                    }
                ],
                "confidence": 0.8,
                "source": "teacher_llm",
                "model": "llama3",
                "generated_at": "2026-05-13T00:00:00Z",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            fake_ds = MagicMock(record_count=0, file_path="")
            fake_db = MagicMock()
            fake_db.flush = AsyncMock()

            with patch(
                "app.services.synthetic_service.get_or_create_synthetic_dataset",
                AsyncMock(return_value=fake_ds),
            ), patch(
                "app.services.synthetic_service._synthetic_dir",
                return_value=tmp_path,
            ):
                result = asyncio.run(
                    save_synthetic_span_batch(fake_db, 1, rows, min_confidence=0.4)
                )

            self.assertEqual(result["accepted"], 1)
            self.assertEqual(result["rejected"], 0)

            # The JSONL on disk carries the canonical shape.
            written = (tmp_path / "synthetic.jsonl").read_text(encoding="utf-8")
            entry = json.loads(written.strip())
            self.assertEqual(entry["text"], "Email me at jane@example.com.")
            self.assertEqual(len(entry["entities"]), 1)
            self.assertEqual(entry["entities"][0]["type"], "email")
            self.assertEqual(entry["status"], "accepted")

    def test_save_rejects_low_confidence_rows(self):
        import asyncio
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        from app.services.synthetic_service import save_synthetic_span_batch

        rows = [
            {"text": "low", "entities": [], "confidence": 0.2},
            {"text": "high", "entities": [], "confidence": 0.9},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            fake_ds = MagicMock(record_count=0, file_path="")
            fake_db = MagicMock()
            fake_db.flush = AsyncMock()
            with patch(
                "app.services.synthetic_service.get_or_create_synthetic_dataset",
                AsyncMock(return_value=fake_ds),
            ), patch(
                "app.services.synthetic_service._synthetic_dir",
                return_value=Path(tmp),
            ):
                result = asyncio.run(
                    save_synthetic_span_batch(fake_db, 1, rows, min_confidence=0.4)
                )
            self.assertEqual(result["accepted"], 1)
            self.assertEqual(result["rejected"], 1)


if __name__ == "__main__":
    unittest.main()
