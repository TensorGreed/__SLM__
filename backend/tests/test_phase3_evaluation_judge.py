"""Phase 3 tests: judge scoring and provider endpoint parsing."""

import unittest

from app.services.evaluation_service import (
    _build_judge_endpoint,
    _heuristic_judge_score,
    _parse_api_judge_content,
)


class EvaluationJudgeTests(unittest.TestCase):
    def test_heuristic_judge_exact_match(self):
        score, rationale = _heuristic_judge_score(
            reference="Rotate API keys every 90 days.",
            prediction="Rotate API keys every 90 days.",
        )
        self.assertEqual(score, 5)
        self.assertIn("match", rationale.lower())

    def test_heuristic_judge_irrelevant(self):
        score, _ = _heuristic_judge_score(
            reference="Use HTTPS for all public endpoints.",
            prediction="Bananas are yellow.",
        )
        self.assertEqual(score, 1)

    def test_parse_api_judge_content_json(self):
        score, rationale = _parse_api_judge_content('{"score": 4, "rationale": "Mostly correct"}')
        self.assertEqual(score, 4)
        self.assertIn("Mostly correct", rationale)

    def test_build_judge_endpoint(self):
        self.assertEqual(
            _build_judge_endpoint("https://judge.example.com"),
            "https://judge.example.com/v1/chat/completions",
        )
        self.assertEqual(
            _build_judge_endpoint("https://judge.example.com/v1"),
            "https://judge.example.com/v1/chat/completions",
        )
        self.assertEqual(
            _build_judge_endpoint("https://judge.example.com/v1/chat/completions"),
            "https://judge.example.com/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
