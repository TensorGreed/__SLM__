"""The Support FAQ demo bundle must be big enough to measure, and honest.

The bundle shipped 20 tickets → a 16 / 2 / 2 split, so the lift check and the
auto-RAG comparison ran on 2 rows (one answer moved the result by tens of
percent), and one answer-key row was an exact copy of a ticket. Pins:
  * the ordered 70/15/15 demo split yields >= 20 val and >= 20 test rows;
  * every question in val/test is a new phrasing, not a copy of a train row
    (no exact / near-duplicate overlap between splits — the leakage matcher);
  * no answer-key row is an exact or near copy of a ticket.
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from app.services.data_health_service import _build_leakage_index, _match_row_against_index
from app.services.demo_project_service import _canonical_prepared_row, _split_rows
from app.services.trainability_forecast_service import _row_to_text

BUNDLE = Path(__file__).resolve().parents[1] / "data" / "demo_samples" / "support-faq"


def _leaks(rows: list[dict], haystack: list[dict]) -> list[str]:
    index = _build_leakage_index(haystack)
    return [text for text in (_row_to_text(r) for r in rows) if _match_row_against_index(text, index)[0]]


class SupportFaqDemoBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (BUNDLE / "tickets.csv").open(encoding="utf-8", newline="") as handle:
            tickets = list(csv.DictReader(handle))
        cls.rows = [
            _canonical_prepared_row(r, input_field="question", output_field="answer", task_profile="instruction_sft")
            for r in tickets
        ]
        cls.train, cls.val, cls.test = _split_rows(cls.rows)
        cls.gold = [json.loads(line) for line in (BUNDLE / "gold.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_split_is_large_enough_to_measure(self):
        self.assertGreaterEqual(len(self.val), 20)
        self.assertGreaterEqual(len(self.test), 20)
        self.assertGreaterEqual(len(self.train), 100)

    def test_heldout_answers_are_learnable_from_train(self):
        # Rows are ordered so val/test hold later phrasings of topics the train
        # split already covers (shared "Settings → X" style anchors), never a
        # topic seen nowhere in train — otherwise neither fine-tuning nor
        # retrieval could answer them and every comparison would read ~0.
        train_questions = {r["question"] for r in self.train}
        self.assertFalse(train_questions & {r["question"] for r in self.val + self.test})
        self.assertEqual(len({r["question"] for r in self.rows}), len(self.rows))

    def test_no_leakage_between_splits(self):
        self.assertEqual(_leaks(self.val + self.test, self.train), [])
        self.assertEqual(_leaks(self.test, self.val), [])

    def test_answer_key_is_not_a_copy_of_the_tickets(self):
        self.assertEqual(_leaks(self.gold, self.rows), [])
