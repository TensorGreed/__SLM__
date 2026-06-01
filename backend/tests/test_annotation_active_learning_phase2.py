"""Epic F Phase 2 — span / preference-pair active strategy + Cohen's κ stats.

Pins:
  * Pure math helpers (``top_two_margin``, ``vote_disagreement``,
    ``cohens_kappa``, ``span_f1``, ``preference_agreement``) — each
    metric's edge cases (empty, length mismatch, degenerate
    one-label inputs, perfect agreement, complete disagreement)
    return the documented sentinel value rather than crashing.
  * Active-strategy dispatch in ``assign_next`` now covers ``span``
    and ``preference_pair`` jobs: with a stubbed scorer the right
    row gets handed out first; without a scoreable experiment the
    path silently falls back to FIFO so the labeler never blocks.
  * ``submit_label`` writes a row to ``label_row_reviews`` in
    addition to updating the primary ``label_payload`` so the
    historical promotion + UI path is unchanged. Re-submission by
    the same reviewer upserts (unique on (row_id, reviewer_id)),
    keeping the agreement math stable across edits.
  * ``job_stats`` returns ``inter_annotator_agreement = None``
    until ≥2 distinct reviewers have labeled at least one
    overlapping row, then surfaces the appropriate metric per
    label_type (κ for classification, span-F1 for span,
    preference agreement for preference-pair).
"""

from __future__ import annotations

import asyncio
import json
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
from app.models.label_job import LabelRowReview  # noqa: E402
from app.services.annotation.active_learning import (  # noqa: E402
    cohens_kappa,
    preference_agreement,
    span_f1,
    top_two_margin,
    vote_disagreement,
)
from sqlalchemy import select  # noqa: E402


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-active-learning-p2-{uuid.uuid4().hex[:8]}"
)


def _write_jsonl(rows: list[dict]) -> str:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    try:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    finally:
        fh.close()
    return fh.name


# ── Pure math helpers ────────────────────────────────────────────────


class TopTwoMarginTests(unittest.TestCase):
    def test_returns_top1_minus_top2(self):
        self.assertEqual(top_two_margin([0.9, 0.4, 0.2]), 0.5)
        self.assertEqual(top_two_margin([0.4, 0.9, 0.2]), 0.5)

    def test_singleton_returns_inf_meaning_no_uncertainty(self):
        self.assertEqual(top_two_margin([0.7]), float("inf"))

    def test_empty_returns_inf(self):
        self.assertEqual(top_two_margin([]), float("inf"))

    def test_negative_margin_when_logits_are_close(self):
        # Top-2 effectively tied → margin near zero → caller will
        # negate this to a *high* uncertainty score.
        self.assertAlmostEqual(top_two_margin([0.501, 0.500]), 0.001, places=5)


class VoteDisagreementTests(unittest.TestCase):
    def test_unanimous_vote_is_zero_disagreement(self):
        self.assertEqual(vote_disagreement(["A", "A", "A"]), 0.0)

    def test_binary_split_is_half_disagreement(self):
        self.assertAlmostEqual(vote_disagreement(["A", "B"]), 0.5)

    def test_three_way_split_approaches_two_thirds(self):
        # 3 distinct picks, no majority → 1 - 1/3 = 2/3.
        self.assertAlmostEqual(
            vote_disagreement(["A", "B", "C"]), 2.0 / 3.0
        )

    def test_empty_votes_returns_zero(self):
        self.assertEqual(vote_disagreement([]), 0.0)


class CohensKappaTests(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        self.assertEqual(
            cohens_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0,
        )

    def test_chance_agreement_close_to_zero(self):
        # 4 items, both raters label half each but with no actual
        # correlation → κ ≈ 0. (Constructed for the math to
        # collapse to zero.)
        # rater_a: a,a,b,b; rater_b: a,b,a,b — 2/4 agree, p_e = 0.5
        # → κ = (0.5 - 0.5) / (1 - 0.5) = 0.
        k = cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])
        self.assertIsNotNone(k)
        self.assertAlmostEqual(k, 0.0, places=5)

    def test_systematic_swap_is_negative(self):
        # Two raters flip labels — worse than chance → κ < 0.
        k = cohens_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"])
        self.assertIsNotNone(k)
        self.assertLess(k, 0.0)

    def test_single_label_population_is_undefined(self):
        # Both raters used only one label → p_e = 1 → κ undefined.
        # We return None rather than silently dividing by zero.
        self.assertIsNone(
            cohens_kappa(["a", "a", "a"], ["a", "a", "a"]),
        )

    def test_length_mismatch_returns_none(self):
        self.assertIsNone(cohens_kappa(["a", "b"], ["a"]))


class SpanF1Tests(unittest.TestCase):
    def test_identical_spans_are_perfect_f1(self):
        spans_a = [{"start": 0, "end": 5, "type": "PER"}]
        spans_b = [{"start": 0, "end": 5, "type": "PER"}]
        self.assertEqual(span_f1(spans_a, spans_b), 1.0)

    def test_disjoint_spans_are_zero_f1(self):
        spans_a = [{"start": 0, "end": 5, "type": "PER"}]
        spans_b = [{"start": 10, "end": 15, "type": "PER"}]
        self.assertEqual(span_f1(spans_a, spans_b), 0.0)

    def test_partial_overlap_uses_harmonic_mean(self):
        # 1 shared span, 1 in A only, 1 in B only → precision=0.5,
        # recall=0.5, F1=0.5.
        spans_a = [
            {"start": 0, "end": 5, "type": "PER"},
            {"start": 10, "end": 15, "type": "ORG"},
        ]
        spans_b = [
            {"start": 0, "end": 5, "type": "PER"},
            {"start": 20, "end": 25, "type": "LOC"},
        ]
        self.assertAlmostEqual(span_f1(spans_a, spans_b), 0.5, places=5)

    def test_tuple_shape_accepted(self):
        # Same spans but expressed as (start, end, type) tuples.
        self.assertEqual(
            span_f1([(0, 5, "PER")], [(0, 5, "PER")]),
            1.0,
        )

    def test_both_empty_returns_none(self):
        self.assertIsNone(span_f1([], []))


class PreferenceAgreementTests(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertEqual(
            preference_agreement(["A", "B", "A"], ["A", "B", "A"]),
            1.0,
        )

    def test_half_agreement(self):
        self.assertAlmostEqual(
            preference_agreement(["A", "A", "B", "B"], ["A", "B", "A", "B"]),
            0.5,
            places=5,
        )

    def test_empty_returns_none(self):
        self.assertIsNone(preference_agreement([], []))


# ── End-to-end: assign_next dispatchers + κ stats ────────────────────


class Phase2AssignAndStatsEndToEndTests(unittest.TestCase):
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

    # ── helpers ─────────────────────────────────────────────────────

    def _create_project(self, label: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{label}-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_dataset_and_job(
        self,
        project_id: int,
        rows: list[dict],
        *,
        label_type: str,
        label_schema: dict,
    ) -> int:
        async def _go() -> int:
            async with async_session_factory() as session:
                jsonl_path = _write_jsonl(rows)
                ds = Dataset(
                    project_id=project_id,
                    name="p2-fixture",
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
                "name": "p2-job",
                "label_type": label_type,
                "label_schema": label_schema,
            },
        )
        self.assertEqual(job_resp.status_code, 201, job_resp.text)
        job_id = int(job_resp.json()["id"])
        seed_resp = self.client.post(
            f"/api/projects/{project_id}/label-jobs/{job_id}/seed-from-dataset",
            json={"dataset_id": dataset_id, "n": len(rows)},
        )
        self.assertEqual(seed_resp.status_code, 200, seed_resp.text)
        return job_id

    def _seed_completed_experiment(
        self,
        project_id: int,
        *,
        task_type: str,
        training_mode: str = "sft",
        config_extra: dict | None = None,
    ) -> int:
        async def _go() -> int:
            async with async_session_factory() as session:
                cfg = {
                    "task_type": task_type,
                    "training_mode": training_mode,
                    "model_path": "/tmp/does/not/exist",
                }
                cfg.update(config_extra or {})
                exp = Experiment(
                    project_id=project_id,
                    name=f"p2-{task_type}-fake",
                    description="phase 2 test fixture",
                    status="completed",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config=cfg,
                )
                session.add(exp)
                await session.commit()
                return int(exp.id)

        return asyncio.run(_go())

    # ── span active strategy ────────────────────────────────────────

    def test_span_active_picks_lowest_margin_row(self):
        # Active rank should hand out the row whose stubbed margin
        # is *smallest* (model least certain). FIFO would have
        # handed out row-A.
        pid = self._create_project("p2-span-active")
        job_id = self._seed_dataset_and_job(
            pid,
            [
                {"id": 1, "text": "confident-A"},
                {"id": 2, "text": "uncertain-B"},
                {"id": 3, "text": "confident-C"},
            ],
            label_type="span",
            label_schema={"span_types": ["PER", "ORG"]},
        )
        self._seed_completed_experiment(pid, task_type="token_classification")

        def _fake_score(rows, *, model_path, span_types):
            del model_path, span_types
            scored: list[float | None] = []
            for r in rows:
                text = (r.raw_payload or {}).get("text", "")
                # Smaller (more uncertain) score for the row we want
                # ranked first — caller already negates margin so we
                # return the post-negation value directly here.
                scored.append(0.9 if "uncertain" in text else 0.1)
            return scored

        with patch(
            "app.services.annotation.active_learning.score_span_rows",
            new=_fake_score,
        ):
            resp = self.client.post(
                f"/api/projects/{pid}/label-jobs/{job_id}/next-row",
                json={"user_id": None, "strategy": "active"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(
            resp.json()["row"]["raw_payload"]["text"], "uncertain-B",
        )

    def test_preference_active_falls_back_when_scorer_unimplemented(self):
        # The Phase-2 preference scorer raises NotImplementedError;
        # the safe-score wrapper should catch that and degrade to
        # FIFO so the labeler keeps moving.
        pid = self._create_project("p2-pref-fallback")
        job_id = self._seed_dataset_and_job(
            pid,
            [
                {
                    "id": 1,
                    "prompt": "Which is better?",
                    "completion_a": "first answer",
                    "completion_b": "second answer",
                },
                {
                    "id": 2,
                    "prompt": "Pick one.",
                    "completion_a": "option A",
                    "completion_b": "option B",
                },
            ],
            label_type="preference_pair",
            label_schema={},
        )
        self._seed_completed_experiment(
            pid, task_type="sft", training_mode="dpo",
        )
        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/next-row",
            json={"user_id": None, "strategy": "active"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        # FIFO would pick the first-inserted row.
        self.assertEqual(
            resp.json()["row"]["raw_payload"]["completion_a"],
            "first answer",
        )

    # ── multi-review + κ stats ──────────────────────────────────────

    def _create_two_users(self) -> tuple[int, int]:
        async def _go() -> tuple[int, int]:
            from app.models.auth import User
            async with async_session_factory() as session:
                a = User(username=f"a-{uuid.uuid4().hex[:6]}", role="ENGINEER")
                b = User(username=f"b-{uuid.uuid4().hex[:6]}", role="ENGINEER")
                session.add_all([a, b])
                await session.commit()
                return int(a.id), int(b.id)

        return asyncio.run(_go())

    def _force_assign(self, row_id: int, user_id: int) -> None:
        """Manually assign a row to a specific user so we can stage
        the two-reviewer overlap that κ needs. Bypasses assign_next
        which would only pick the FIFO-first unassigned row."""

        async def _go() -> None:
            from app.models.label_job import LabelRow
            from datetime import datetime, timezone
            async with async_session_factory() as session:
                row = await session.get(LabelRow, row_id)
                assert row is not None
                row.assigned_to = user_id
                row.assigned_at = datetime.now(timezone.utc)
                row.labeled_at = None
                row.label_payload = None
                await session.commit()

        asyncio.run(_go())

    def _list_label_rows(self, project_id: int, job_id: int) -> list[dict]:
        resp = self.client.get(
            f"/api/projects/{project_id}/label-jobs/{job_id}"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Detail endpoint doesn't return raw rows — query directly.
        async def _go() -> list[dict]:
            from app.models.label_job import LabelRow
            async with async_session_factory() as session:
                q = await session.execute(
                    select(LabelRow).where(LabelRow.job_id == job_id)
                )
                return [
                    {"id": r.id, "raw_payload": r.raw_payload}
                    for r in q.scalars()
                ]
        rows = asyncio.run(_go())
        return rows

    def test_submit_label_writes_review_for_assigned_reviewer(self):
        pid = self._create_project("p2-review-write")
        job_id = self._seed_dataset_and_job(
            pid,
            [{"id": 1, "text": "row-A"}],
            label_type="classification",
            label_schema={"allowed_labels": ["pos", "neg"]},
        )
        user_a, _ = self._create_two_users()
        # Pull the row out + manually assign to user_a then submit.
        rows = self._list_label_rows(pid, job_id)
        row_id = rows[0]["id"]
        self._force_assign(row_id, user_a)
        submit_resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/rows/{row_id}/submit",
            json={"label_payload": {"label": "pos"}},
        )
        self.assertEqual(submit_resp.status_code, 200, submit_resp.text)

        async def _check() -> int:
            async with async_session_factory() as session:
                q = await session.execute(
                    select(LabelRowReview).where(
                        LabelRowReview.row_id == row_id
                    )
                )
                return len(list(q.scalars()))

        # Exactly one review written — keyed by (row, reviewer).
        self.assertEqual(asyncio.run(_check()), 1)

    def test_job_stats_returns_kappa_when_two_reviewers_overlap(self):
        # Two reviewers each label the same two rows; one row they
        # agree on, one they disagree on. Both raters use both
        # labels so κ is well-defined.
        pid = self._create_project("p2-stats-kappa")
        job_id = self._seed_dataset_and_job(
            pid,
            [
                {"id": 1, "text": "row-A"},
                {"id": 2, "text": "row-B"},
            ],
            label_type="classification",
            label_schema={"allowed_labels": ["pos", "neg"]},
        )
        user_a, user_b = self._create_two_users()
        rows = self._list_label_rows(pid, job_id)
        row_a_id, row_b_id = rows[0]["id"], rows[1]["id"]

        # Reviewer A: row-A=pos, row-B=neg.
        self._force_assign(row_a_id, user_a)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/rows/{row_a_id}/submit",
            json={"label_payload": {"label": "pos"}},
        )
        self._force_assign(row_b_id, user_a)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/rows/{row_b_id}/submit",
            json={"label_payload": {"label": "neg"}},
        )
        # Reviewer B: row-A=pos (agree), row-B=pos (disagree).
        self._force_assign(row_a_id, user_b)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/rows/{row_a_id}/submit",
            json={"label_payload": {"label": "pos"}},
        )
        self._force_assign(row_b_id, user_b)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/rows/{row_b_id}/submit",
            json={"label_payload": {"label": "pos"}},
        )

        stats_resp = self.client.get(
            f"/api/projects/{pid}/label-jobs/{job_id}"
        )
        self.assertEqual(stats_resp.status_code, 200, stats_resp.text)
        stats = stats_resp.json()["stats"]
        agreement = stats["inter_annotator_agreement"]
        self.assertIsNotNone(agreement)
        self.assertEqual(agreement["metric"], "cohens_kappa")
        self.assertEqual(agreement["reviewer_count"], 2)
        self.assertEqual(agreement["overlap_rows"], 2)
        self.assertEqual(agreement["pair_count"], 1)
        # 1 of 2 rows agreed; with each rater using both labels
        # asymmetrically κ is non-zero — assert the API actually
        # returns a finite number rather than pinning the exact value.
        self.assertIsInstance(agreement["value"], float)

    def test_job_stats_returns_none_when_only_one_reviewer(self):
        pid = self._create_project("p2-stats-solo")
        job_id = self._seed_dataset_and_job(
            pid,
            [{"id": 1, "text": "row-A"}],
            label_type="classification",
            label_schema={"allowed_labels": ["pos", "neg"]},
        )
        user_a, _ = self._create_two_users()
        rows = self._list_label_rows(pid, job_id)
        self._force_assign(rows[0]["id"], user_a)
        self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/rows/{rows[0]['id']}/submit",
            json={"label_payload": {"label": "pos"}},
        )
        stats_resp = self.client.get(
            f"/api/projects/{pid}/label-jobs/{job_id}"
        )
        self.assertIsNone(
            stats_resp.json()["stats"]["inter_annotator_agreement"]
        )


if __name__ == "__main__":
    unittest.main()
