"""Unit tests for the model-sweep Pareto frontier annotation (Epic C).

Pure function — no DB, no network. Verifies dominance on (quality ↑, latency ↓).
"""

import unittest

from app.services.model_benchmark_service import annotate_pareto_frontier


class ParetoFrontierTests(unittest.TestCase):
    def test_clear_dominance(self):
        # B is worse than A on BOTH axes → dominated. C trades quality for speed
        # (lower quality, lower latency) → on the frontier alongside A.
        matrix = [
            {"model_id": "A", "estimated_quality_score": 0.9, "estimated_latency_ms": 100.0},
            {"model_id": "B", "estimated_quality_score": 0.7, "estimated_latency_ms": 150.0},
            {"model_id": "C", "estimated_quality_score": 0.6, "estimated_latency_ms": 40.0},
        ]
        annotate_pareto_frontier(matrix)
        by_id = {r["model_id"]: r for r in matrix}
        self.assertTrue(by_id["A"]["pareto_optimal"])
        self.assertTrue(by_id["C"]["pareto_optimal"])
        self.assertFalse(by_id["B"]["pareto_optimal"])
        self.assertIn("A", by_id["B"]["dominated_by"])

    def test_identical_points_are_both_optimal(self):
        matrix = [
            {"model_id": "A", "estimated_quality_score": 0.8, "estimated_latency_ms": 90.0},
            {"model_id": "B", "estimated_quality_score": 0.8, "estimated_latency_ms": 90.0},
        ]
        annotate_pareto_frontier(matrix)
        self.assertTrue(all(r["pareto_optimal"] for r in matrix))

    def test_strictly_better_on_one_axis_dominates(self):
        # Same quality, lower latency → A dominates B.
        matrix = [
            {"model_id": "A", "estimated_quality_score": 0.8, "estimated_latency_ms": 50.0},
            {"model_id": "B", "estimated_quality_score": 0.8, "estimated_latency_ms": 80.0},
        ]
        annotate_pareto_frontier(matrix)
        by_id = {r["model_id"]: r for r in matrix}
        self.assertTrue(by_id["A"]["pareto_optimal"])
        self.assertFalse(by_id["B"]["pareto_optimal"])

    def test_single_row_is_optimal(self):
        matrix = [{"model_id": "A", "estimated_quality_score": 0.5, "estimated_latency_ms": 10.0}]
        annotate_pareto_frontier(matrix)
        self.assertTrue(matrix[0]["pareto_optimal"])
        self.assertEqual(matrix[0]["dominated_by"], [])


if __name__ == "__main__":
    unittest.main()
