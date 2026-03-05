"""Tests for the synthetic data generation service — demo/heuristic mode."""

import unittest

from app.services.synthetic_service import (
    _generate_demo_pairs,
    _compute_confidence,
)


class SyntheticServiceTests(unittest.TestCase):
    """Unit tests for synthetic generation utility functions."""

    # ── Demo QA Generation ─────────────────────────────────────────────

    SAMPLE_TEXT = (
        "Machine learning is a subset of artificial intelligence. "
        "It uses algorithms to learn from data and make predictions. "
        "Deep learning is a further subset that uses neural networks with many layers. "
        "Training requires large datasets and significant compute resources. "
        "Common frameworks include PyTorch and TensorFlow."
    )

    def test_demo_pairs_returns_list(self):
        pairs = _generate_demo_pairs(self.SAMPLE_TEXT, num_pairs=3)
        self.assertIsInstance(pairs, list)
        self.assertGreater(len(pairs), 0)

    def test_demo_pair_has_required_fields(self):
        pairs = _generate_demo_pairs(self.SAMPLE_TEXT, num_pairs=2)
        for pair in pairs:
            self.assertIn("question", pair)
            self.assertIn("answer", pair)
            self.assertTrue(len(pair["question"]) > 0)
            self.assertTrue(len(pair["answer"]) > 0)

    def test_demo_pairs_empty_text(self):
        pairs = _generate_demo_pairs("", num_pairs=5)
        self.assertIsInstance(pairs, list)
        # Should return empty or very few pairs for empty text
        self.assertEqual(len(pairs), 0)

    def test_demo_pairs_short_text(self):
        pairs = _generate_demo_pairs("Short.", num_pairs=5)
        self.assertIsInstance(pairs, list)

    def test_demo_pairs_respects_num_pairs(self):
        pairs = _generate_demo_pairs(self.SAMPLE_TEXT, num_pairs=2)
        self.assertLessEqual(len(pairs), 2)

    # ── Confidence Scoring ─────────────────────────────────────────────

    def test_confidence_score_returns_float(self):
        pair = {"question": "What is ML?", "answer": "Machine learning is a method."}
        score = _compute_confidence(pair)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_confidence_higher_for_longer_answers(self):
        short = _compute_confidence({"question": "What?", "answer": "Yes."})
        long = _compute_confidence({
            "question": "What is machine learning?",
            "answer": "Machine learning is a method of data analysis that automates analytical model building.",
        })
        self.assertLessEqual(short, long)

    def test_confidence_low_for_empty(self):
        score = _compute_confidence({"question": "", "answer": ""})
        self.assertLessEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
