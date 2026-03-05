"""Tests for the synthetic data generation service — demo/heuristic mode."""

import unittest

from app.services.synthetic_service import (
    _generate_demo_pairs,
    _compute_confidence,
    _parse_teacher_pairs,
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

    # ── Teacher Response Parsing ──────────────────────────────────────

    def test_parse_teacher_pairs_accepts_json_array(self):
        content = (
            '[{"question":"What is ML?","answer":"Machine learning is a subset of AI."},'
            '{"question":"What is deep learning?","answer":"A subset of ML using neural networks."}]'
        )
        pairs = _parse_teacher_pairs(content)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["question"], "What is ML?")

    def test_parse_teacher_pairs_accepts_fenced_json(self):
        content = (
            "Here are your pairs:\n"
            "```json\n"
            '[{"question":"Q1","answer":"A1"},{"question":"Q2","answer":"A2"}]\n'
            "```"
        )
        pairs = _parse_teacher_pairs(content)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[1]["answer"], "A2")

    def test_parse_teacher_pairs_accepts_wrapped_object(self):
        content = (
            '{"pairs":[{"question":"What is AI?","answer":"AI stands for artificial intelligence."}]}'
        )
        pairs = _parse_teacher_pairs(content)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "What is AI?")

    def test_parse_teacher_pairs_falls_back_to_plaintext_qa(self):
        content = (
            "Q: What is supervised learning?\n"
            "A: It is learning from labeled data.\n\n"
            "Q: Name one ML framework.\n"
            "A: PyTorch."
        )
        pairs = _parse_teacher_pairs(content)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["question"], "What is supervised learning?")
        self.assertEqual(pairs[1]["answer"], "PyTorch.")


if __name__ == "__main__":
    unittest.main()
