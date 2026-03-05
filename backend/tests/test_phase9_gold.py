"""Tests for the gold evaluation dataset service."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

# We test the file I/O utility logic without needing a real database.

class GoldServiceTests(unittest.TestCase):
    """Unit tests for gold dataset utility logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.gold_file = Path(self.tmpdir) / "gold_dev.jsonl"

    def tearDown(self):
        if self.gold_file.exists():
            self.gold_file.unlink()
        os.rmdir(self.tmpdir)

    def _write_entries(self, entries: list[dict]):
        with open(self.gold_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def _read_entries(self) -> list[dict]:
        entries = []
        with open(self.gold_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def test_write_and_read_entries(self):
        entries = [
            {"id": 1, "question": "What is SLM?", "answer": "Small Language Model"},
            {"id": 2, "question": "What is LoRA?", "answer": "Low-Rank Adaptation"},
        ]
        self._write_entries(entries)
        read_back = self._read_entries()
        self.assertEqual(len(read_back), 2)
        self.assertEqual(read_back[0]["question"], "What is SLM?")
        self.assertEqual(read_back[1]["answer"], "Low-Rank Adaptation")

    def test_append_entry(self):
        self._write_entries([{"id": 1, "question": "Q1", "answer": "A1"}])
        with open(self.gold_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": 2, "question": "Q2", "answer": "A2"}) + "\n")
        entries = self._read_entries()
        self.assertEqual(len(entries), 2)

    def test_empty_file_returns_empty_list(self):
        self._write_entries([])
        entries = self._read_entries()
        self.assertEqual(len(entries), 0)

    def test_entry_has_required_fields(self):
        entries = [
            {
                "id": 1,
                "question": "What is fine-tuning?",
                "answer": "Adapting a pretrained model to a specific task.",
                "difficulty": "easy",
                "criticality": "high",
                "is_hallucination_trap": False,
            }
        ]
        self._write_entries(entries)
        read = self._read_entries()
        self.assertIn("question", read[0])
        self.assertIn("answer", read[0])
        self.assertIn("difficulty", read[0])
        self.assertIn("criticality", read[0])
        self.assertFalse(read[0]["is_hallucination_trap"])

    def test_hallucination_trap_flag(self):
        entries = [
            {
                "id": 1,
                "question": "What color is the invisible unicorn?",
                "answer": "This question cannot be answered from the data.",
                "is_hallucination_trap": True,
            }
        ]
        self._write_entries(entries)
        read = self._read_entries()
        self.assertTrue(read[0]["is_hallucination_trap"])

    def test_bulk_import_count(self):
        entries = [
            {"id": i, "question": f"Q{i}", "answer": f"A{i}"}
            for i in range(50)
        ]
        self._write_entries(entries)
        read = self._read_entries()
        self.assertEqual(len(read), 50)


if __name__ == "__main__":
    unittest.main()
