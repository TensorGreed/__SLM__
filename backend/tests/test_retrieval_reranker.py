"""Cross-encoder reranking + per-project retrieval settings.

Pins (no model download — the scorer is patched):
  * rerank reorders BM25 hits by the scorer and keeps top-k, stamping
    rerank_score / bm25_rank; without a model it keeps BM25 order; a
    scorer failure keeps BM25 order;
  * retrieval_settings reads runtime_config.auto_rag_retrieval with
    defaults + clamps; retrieve_ranked pulls a candidate pool before
    reranking;
  * the sweep picks the judge's best, ties → the cheaper arm.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("DEBUG", "false")

from app.services import retrieval_reranker as rr
from app.services.auto_rag_service import retrieval_settings


def _hits(n: int) -> list[dict]:
    return [{"payload": {"text": f"passage {i}", "chunk_id": i}, "score": float(n - i)} for i in range(n)]


class RerankTests(unittest.TestCase):
    def test_reorders_by_score_and_keeps_top_k(self):
        with mock.patch.object(rr, "score_pairs", lambda q, texts, model_name: [float(len(texts) - 1 - i) if i != 3 else 100.0 for i in range(len(texts))]):
            out = rr.rerank("q", _hits(6), top_k=2, model_name="default")
        self.assertEqual([h["payload"]["chunk_id"] for h in out], [3, 0])
        self.assertEqual(out[0]["bm25_rank"], 4)
        self.assertEqual(out[0]["rerank_score"], 100.0)

    def test_no_model_or_failure_keeps_bm25_order(self):
        self.assertEqual([h["payload"]["chunk_id"] for h in rr.rerank("q", _hits(5), top_k=3, model_name=None)], [0, 1, 2])
        self.assertEqual([h["payload"]["chunk_id"] for h in rr.rerank("q", _hits(5), top_k=3, model_name="none")], [0, 1, 2])

        def _boom(*a, **k):
            raise RuntimeError("no model")

        with mock.patch.object(rr, "score_pairs", _boom):
            self.assertEqual([h["payload"]["chunk_id"] for h in rr.rerank("q", _hits(5), top_k=3, model_name="default")], [0, 1, 2])
        self.assertEqual(rr.rerank("q", [], top_k=3), [])

    def test_normalize_and_pool(self):
        self.assertIsNone(rr.normalize_reranker(None))
        self.assertIsNone(rr.normalize_reranker("off"))
        self.assertEqual(rr.normalize_reranker("default"), rr.DEFAULT_RERANKER)
        self.assertEqual(rr.normalize_reranker(True), rr.DEFAULT_RERANKER)
        self.assertEqual(rr.normalize_reranker("BAAI/bge-reranker-base"), "BAAI/bge-reranker-base")
        self.assertEqual(rr.candidate_pool(3), 12)
        self.assertEqual(rr.candidate_pool(1), rr.MIN_CANDIDATES)
        self.assertEqual(rr.candidate_pool(50), rr.MAX_CANDIDATES)


class SettingsTests(unittest.TestCase):
    def test_settings_defaults_and_clamps(self):
        class P:
            runtime_config = None

        self.assertEqual(retrieval_settings(P()), {"k": 3, "reranker": None})
        P.runtime_config = {"auto_rag_retrieval": {"k": 5, "reranker": "default", "source": "retrieval_sweep"}}
        self.assertEqual(retrieval_settings(P()), {"k": 5, "reranker": rr.DEFAULT_RERANKER})
        P.runtime_config = {"auto_rag_retrieval": {"k": "99", "reranker": "none"}}
        self.assertEqual(retrieval_settings(P()), {"k": 10, "reranker": None})

    def test_retrieve_ranked_pulls_a_candidate_pool_when_reranking(self):
        from app.services import auto_rag_service as svc

        seen: dict = {}

        def _retrieve(query, *, index_dir, k):
            seen["k"] = k
            return _hits(k)

        with mock.patch.object(svc, "retrieve", _retrieve), \
                mock.patch.object(rr, "score_pairs", lambda q, texts, model_name: [float(i) for i in range(len(texts))]):
            out = svc.retrieve_ranked("q", index_dir=None, k=3, reranker="default")
            self.assertEqual(seen["k"], 12)
            self.assertEqual([h["payload"]["chunk_id"] for h in out], [11, 10, 9])
            out = svc.retrieve_ranked("q", index_dir=None, k=3, reranker=None)
            self.assertEqual(seen["k"], 3)
            self.assertEqual(len(out), 3)


class SweepPickTests(unittest.TestCase):
    def test_judge_wins_ties_go_to_the_cheaper_arm(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import auto_rag_ab as harness

        arms = [
            {"label": "top-3", "judge_score": 0.6, "on_mean_f1": 0.30},
            {"label": "top-5", "judge_score": 0.6, "on_mean_f1": 0.45},
            {"label": "top-3 + reranker", "judge_score": 0.71, "on_mean_f1": 0.20},
            {"label": "top-5 + reranker", "judge_score": 0.71, "on_mean_f1": 0.50},
        ]
        self.assertEqual(harness.pick_best_retrieval(arms)["label"], "top-3 + reranker")
        unjudged = [{"label": "a", "judge_score": None, "on_mean_f1": 0.3}, {"label": "b", "judge_score": None, "on_mean_f1": 0.4}]
        self.assertEqual(harness.pick_best_retrieval(unjudged)["label"], "b")
        self.assertEqual(harness.retrieval_label({"k": 5, "reranker": "default"}), "top-5 + reranker ms-marco-MiniLM-L-6-v2")


if __name__ == "__main__":
    unittest.main()
