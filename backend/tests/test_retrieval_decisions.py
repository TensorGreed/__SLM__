"""Pin decisions (retrieval_decisions_service): append-only record of what
users chose when comparing retrievals, the per-retrieval tally, and the
"preferred over" rule the sweep uses as a tie-break."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("DEBUG", "false")

from app.services import retrieval_decisions_service as svc

BM25 = {"k": 3, "reranker": None}
RERANK = {"k": 3, "reranker": "default"}
RERANK_ID = {"k": 3, "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
TOP5 = {"k": 5, "reranker": None}


class DecisionsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(svc.settings, "DATA_DIR", Path(self._tmp.name))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _record(self, pinned, candidates=(BM25, RERANK), question="Q?"):
        return svc.record_decision(
            7, question=question,
            candidates=[{"retrieval": c, "reply_excerpt": "x" * 500, "latency_ms": 10, "passages": 3} for c in candidates],
            pinned=pinned,
        )

    def test_record_list_and_keys(self):
        entry = self._record(RERANK)
        self.assertEqual(entry["pinned"], RERANK_ID)
        self.assertEqual(len(entry["candidates"][0]["reply_excerpt"]), svc.MAX_EXCERPT_CHARS)
        self.assertEqual(svc.retrieval_key(RERANK), svc.retrieval_key(RERANK_ID))
        self.assertEqual(svc.retrieval_key(BM25), "3|")
        self.assertEqual(len(svc.list_decisions(7)), 1)
        self.assertEqual(svc.list_decisions(99), [])
        with self.assertRaises(ValueError):
            self._record(TOP5)  # not among the candidates
        with self.assertRaises(ValueError):
            self._record(BM25, candidates=(BM25,))
        with self.assertRaises(ValueError):
            self._record({"reranker": "default"})

    def test_tally_and_preference(self):
        self._record(RERANK)
        self._record(RERANK)
        self._record(BM25)
        self._record(TOP5, candidates=(BM25, TOP5))
        tally = svc.project_preference(7)
        self.assertEqual((tally["3|"]["offered"], tally["3|"]["chosen"]), (4, 1))
        rerank = tally[svc.retrieval_key(RERANK_ID)]
        self.assertEqual((rerank["offered"], rerank["chosen"], rerank["share"]), (3, 2, 0.667))
        self.assertTrue(svc.preferred_over(tally, RERANK, BM25))
        self.assertFalse(svc.preferred_over(tally, BM25, RERANK))
        # One pin is a click, not a preference.
        self.assertFalse(svc.preferred_over(tally, TOP5, BM25))
        # A retrieval never offered has share 0 — a preferred one beats it.
        self.assertTrue(svc.preferred_over(tally, RERANK, {"k": 5, "reranker": "default"}))
        self.assertEqual(svc.preference_tally([]), {})


if __name__ == "__main__":
    unittest.main()
