"""Epic F Phase 1 — active-learning row ranker for label-jobs.

Pins the contract for :func:`assign_next` when called with
``strategy="active"``:

  * The **pure ranker** (``rank_rows_by_uncertainty``) orders rows by
    descending score and trails any ``None``-scored row after rows
    with real scores. This is the heart of the active-learning
    surface — tested without a GPU or any torch import.
  * ``extract_row_text`` reads the canonical text aliases used by
    the classification-label adapter so the ranker sees the same
    text the model would see at eval.
  * ``softmax_entropy`` is numerically stable on large logits and
    correctly maximises at a uniform distribution.
  * The next-row endpoint accepts ``strategy`` and rejects unknown
    values with HTTP 422 rather than silently falling back. The
    fallback path (no scoreable experiment) still hands out a row in
    insertion order so a labeler on a fresh project isn't blocked.
  * Active strategy + a stubbed score function actually changes the
    handed-out order — verifies the wiring all the way through
    ``assign_next``.

The classifier-head scorer (``score_classification_rows``) imports
torch lazily and runs a real forward pass; we don't exercise that
path in unit tests because the CI environment may not have a GPU.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import Dataset, DatasetType  # noqa: E402
from app.models.experiment import Experiment  # noqa: E402
from app.models.label_job import LabelRow  # noqa: E402
from app.services.annotation.active_learning import (  # noqa: E402
    extract_row_text,
    rank_rows_by_uncertainty,
    softmax_entropy,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir()) / f"brewslm-active-learning-{uuid.uuid4().hex[:8]}"
)


def _write_jsonl(rows: list[dict]) -> str:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    try:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    finally:
        fh.close()
    return fh.name


class _Row:
    """Stub stand-in for ``LabelRow`` — the pure ranker only reads
    ``id`` + ``raw_payload`` so we don't need to spin up a session."""

    def __init__(self, row_id: int, payload: dict | None) -> None:
        self.id = row_id
        self.raw_payload = payload


# ── Pure helpers ─────────────────────────────────────────────────────


class ExtractRowTextTests(unittest.TestCase):
    def test_picks_first_non_empty_alias_in_priority_order(self):
        # ``text`` wins over ``content`` over ``input``.
        self.assertEqual(extract_row_text({"text": "a", "content": "b"}), "a")
        self.assertEqual(extract_row_text({"content": "b", "input": "c"}), "b")
        self.assertEqual(extract_row_text({"input": "c"}), "c")

    def test_skips_blank_strings_continues_to_next_alias(self):
        # Empty / whitespace ``text`` shouldn't shadow a real ``content``.
        out = extract_row_text({"text": "   ", "content": "real"})
        self.assertEqual(out, "real")

    def test_returns_none_for_payloads_without_text(self):
        self.assertIsNone(extract_row_text({"label": "x"}))
        self.assertIsNone(extract_row_text({}))
        self.assertIsNone(extract_row_text(None))


class SoftmaxEntropyTests(unittest.TestCase):
    def test_uniform_distribution_maximises_entropy(self):
        # 3 classes, all equally likely → ln(3) nats. The function
        # operates on logits, so all-zero logits → uniform softmax.
        self.assertAlmostEqual(softmax_entropy([0.0, 0.0, 0.0]), math.log(3))

    def test_one_hot_logits_give_zero_entropy(self):
        # Logit gap of 100 effectively pins all mass on one class.
        self.assertAlmostEqual(softmax_entropy([100.0, 0.0, 0.0]), 0.0, places=5)

    def test_stable_under_large_logits(self):
        # Without the max-subtract trick this would overflow exp().
        # Result should still be defined and positive (mild spread).
        value = softmax_entropy([1e3, 1e3 - 1.0, 1e3 - 2.0])
        self.assertGreater(value, 0.0)
        self.assertLess(value, math.log(3))

    def test_empty_logits_returns_zero(self):
        self.assertEqual(softmax_entropy([]), 0.0)


class RankRowsByUncertaintyTests(unittest.TestCase):
    def test_sorts_descending_by_score(self):
        rows = [_Row(1, {"text": "a"}), _Row(2, {"text": "b"}), _Row(3, {"text": "c"})]
        out = rank_rows_by_uncertainty(
            rows, score_fn=lambda _batch: [0.1, 0.9, 0.5]
        )
        # Highest score first: 2 (0.9) → 3 (0.5) → 1 (0.1).
        self.assertEqual(out, [2, 3, 1])

    def test_none_scored_rows_trail_real_scored_rows(self):
        rows = [_Row(1, {"text": "a"}), _Row(2, None), _Row(3, {"text": "c"})]
        out = rank_rows_by_uncertainty(
            rows, score_fn=lambda _batch: [0.5, None, 0.9]
        )
        # Real scores come first (3, 1), unscored row trails (2).
        self.assertEqual(out, [3, 1, 2])

    def test_score_fn_length_mismatch_raises(self):
        rows = [_Row(1, {"text": "a"}), _Row(2, {"text": "b"})]
        with self.assertRaises(ValueError):
            rank_rows_by_uncertainty(rows, score_fn=lambda _batch: [0.5])

    def test_empty_input_returns_empty_list_without_calling_score_fn(self):
        called: list[bool] = []

        def _trap(_batch):
            called.append(True)
            return []

        self.assertEqual(rank_rows_by_uncertainty([], score_fn=_trap), [])
        self.assertEqual(called, [])

    def test_tied_scores_preserve_insertion_order(self):
        # Ties should resolve by earlier-first so a labeler's queue
        # is deterministic across calls with the same model state.
        rows = [_Row(10, {"text": "a"}), _Row(11, {"text": "b"}), _Row(12, {"text": "c"})]
        out = rank_rows_by_uncertainty(
            rows, score_fn=lambda _batch: [0.5, 0.5, 0.5]
        )
        self.assertEqual(out, [10, 11, 12])


# ── End-to-end: assign_next + strategy parameter ─────────────────────


class AssignNextStrategyEndToEndTests(unittest.TestCase):
    """Drives the real FastAPI app + database. Active-learning's
    classifier-head scorer is stubbed out so we don't need a GPU."""

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_project(self, label: str) -> int:
        resp = self.client.post(
            "/api/projects", json={"name": f"{label}-{uuid.uuid4().hex[:6]}"}
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_classification_job(
        self, project_id: int, texts: list[str]
    ) -> int:
        async def _go() -> int:
            async with async_session_factory() as session:
                rows = [
                    {"id": i + 1, "text": txt, "label": "benign"}
                    for i, txt in enumerate(texts)
                ]
                jsonl_path = _write_jsonl(rows)
                ds = Dataset(
                    project_id=project_id,
                    name="al-fixture",
                    dataset_type=DatasetType.SYNTHETIC,
                    record_count=len(rows),
                    file_path=jsonl_path,
                )
                session.add(ds)
                await session.commit()
                return int(ds.id)

        dataset_id = asyncio.run(_go())

        job_resp = self.client.post(
            f"/api/projects/{project_id}/label-jobs/",
            json={
                "name": "al-job",
                "label_type": "classification",
                "label_schema": {"allowed_labels": ["benign", "injection"]},
            },
        )
        self.assertEqual(job_resp.status_code, 201, job_resp.text)
        job_id = int(job_resp.json()["id"])
        seed_resp = self.client.post(
            f"/api/projects/{project_id}/label-jobs/{job_id}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": len(texts)},
        )
        self.assertEqual(seed_resp.status_code, 200, seed_resp.text)
        return job_id

    def _seed_completed_classification_experiment(self, project_id: int) -> int:
        """Drop a fake completed Experiment so the AL path's
        ``latest_scoreable_classification_experiment`` returns it."""

        async def _go() -> int:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=project_id,
                    name="al-fake-exp",
                    description="active-learning test fixture",
                    status="completed",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config={
                        "task_type": "classification",
                        "label_space": ["benign", "injection"],
                        "model_path": "/tmp/does/not/exist",
                    },
                )
                session.add(exp)
                await session.commit()
                return int(exp.id)

        return asyncio.run(_go())

    def test_unknown_strategy_rejected_with_422(self):
        pid = self._create_project("al-unknown-strategy")
        job_id = self._seed_classification_job(pid, ["a"])
        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/next-row",
            json={"user_id": None, "strategy": "random-walk"},
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertIn("unknown_assign_strategy", resp.text)

    def test_active_falls_back_to_fifo_without_completed_experiment(self):
        pid = self._create_project("al-no-experiment")
        job_id = self._seed_classification_job(
            pid, ["row-A", "row-B", "row-C"]
        )
        # No experiment seeded → AL path returns None and the
        # endpoint hands out the lowest-id row, matching FIFO.
        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/next-row",
            json={"user_id": None, "strategy": "active"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["queue_empty"])
        self.assertEqual(body["strategy"], "active")
        # FIFO would hand out the first-inserted row.
        self.assertEqual(body["row"]["raw_payload"]["text"], "row-A")

    def test_active_strategy_picks_highest_entropy_row(self):
        pid = self._create_project("al-active-pick")
        job_id = self._seed_classification_job(
            pid, ["confident-A", "uncertain-B", "confident-C"]
        )
        self._seed_completed_classification_experiment(pid)

        # Stub the entropy scorer so row B (id 2) ranks first by
        # uncertainty. The wiring under test is whether the API
        # actually picks the high-entropy row, not the math.
        def _fake_score(rows, *, model_path, label_space):
            del model_path, label_space
            scored: list[float | None] = []
            for r in rows:
                text = (r.raw_payload or {}).get("text", "")
                scored.append(0.95 if "uncertain" in text else 0.05)
            return scored

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            new=_fake_score,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/label-jobs/{job_id}/next-row",
                json={"user_id": None, "strategy": "active"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Active strategy should reach past row A (FIFO order) to
        # row B because B's entropy is highest.
        self.assertEqual(body["row"]["raw_payload"]["text"], "uncertain-B")
        self.assertEqual(body["strategy"], "active")

    def test_fifo_default_unchanged_by_active_wiring(self):
        # Regression guard: the default strategy is still FIFO and
        # doesn't load any model. Even with an experiment that would
        # rank row C first under "active", a missing/default strategy
        # should hand out row A.
        pid = self._create_project("al-default-fifo")
        job_id = self._seed_classification_job(
            pid, ["row-A", "row-B", "row-C"]
        )
        self._seed_completed_classification_experiment(pid)

        def _explode(rows, *, model_path, label_space):
            raise AssertionError(
                "score_classification_rows should not be called for fifo"
            )

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            new=_explode,
        ):
            # No ``strategy`` field → defaults to fifo.
            resp = self.client.post(
                f"/api/projects/{pid}/label-jobs/{job_id}/next-row",
                json={"user_id": None},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["row"]["raw_payload"]["text"], "row-A")


if __name__ == "__main__":
    unittest.main()
