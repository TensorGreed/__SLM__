"""Quality-Lift phase 3 slice 1 — Unlabeled-pool scoring service.

Pins (slice 1: scoring service + post-training hook + seed-group
leader semantics. Coach nudge + Data Studio surface land in slices
2 and 3):

  Skipped reasons (every path returns a valid snapshot — never
  raises — so the post-training hook is safe):
    * unsupported_task_type — non-classification experiments
      (QA / seq2seq / language_modeling) skip cleanly so the
      Coach nudge falls silent.
    * empty_pool — project has zero unlabeled label_rows.
    * no_label_space_configured — project has no classification
      label_job to pull the allowed_labels from.
    * checkpoint_path_missing — exp.output_dir is empty or doesn't
      exist on disk (typical for simulate-runner experiments).
    * scoring_failed — score_classification_rows raised; error
      string and checkpoint path land in the snapshot for
      debugging.

  Happy path:
    * Sampling cap clamps pool to sample_cap; remaining rows are
      not scored.
    * Top-K result preserves entropy ranking (highest first).
    * Rows with None entropy (text-extraction failure) drop out
      of the top_k entirely rather than crowding genuinely
      uncertain rows.
    * Snapshot carries scored_at / model_experiment_id /
      uncertainty_metric / pool_size_total / pool_size_scored.
    * Stamp helper writes to _runtime["active_learning"]
      preserving any existing _runtime entries (auto_rag_build,
      etc.).

  Multi-seed semantics:
    * _safe_score_unlabeled_pool skips when the experiment is a
      seed-group child (seed_value is set + seed_group_id is set);
      returns the seed_group_child_defers_to_leader skip reason
      so the runner-side hook is a no-op for children.
    * Leader (seed_value is None) goes through the normal scoring
      path.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode  # noqa: E402
from app.models.label_job import LabelJob, LabelRow  # noqa: E402
from app.services.unlabeled_pool_scoring_service import (  # noqa: E402
    DEFAULT_SAMPLE_CAP,
    DEFAULT_TOP_K,
    score_unlabeled_pool_for_experiment,
    stamp_snapshot_on_experiment,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-alscoring-{uuid.uuid4().hex[:8]}"
)


def setUpModule() -> None:
    settings.AUTH_ENABLED = False
    settings.DEBUG = False
    settings.DATA_DIR = TEST_DATA_DIR.resolve()
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    global _CLIENT_CM, CLIENT
    _CLIENT_CM = TestClient(app)
    CLIENT = _CLIENT_CM.__enter__()


def tearDownModule() -> None:
    _CLIENT_CM.__exit__(None, None, None)


def _create_project() -> int:
    resp = CLIENT.post(
        "/api/projects",
        json={"name": f"al-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _make_checkpoint_dir() -> Path:
    """Real directory for the checkpoint_path existence check.
    Empty — score_classification_rows is patched in tests, so we
    don't actually load a model."""
    d = TEST_DATA_DIR / f"checkpoint-{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed_experiment(
    project_id: int,
    *,
    task_type: str = "classification",
    output_dir: str | None = None,
    seed_value: int | None = None,
    seed_group_id: str | None = None,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
) -> int:
    async def _go() -> int:
        async with async_session_factory() as session:
            exp = Experiment(
                project_id=project_id,
                name=f"al-exp-{uuid.uuid4().hex[:4]}",
                status=status,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                training_mode=TrainingMode.SFT,
                config={"task_type": task_type},
                output_dir=output_dir or "",
                seed_value=seed_value,
                seed_group_id=seed_group_id,
            )
            session.add(exp)
            await session.commit()
            return int(exp.id)

    return asyncio.run(_go())


def _seed_classification_label_job(
    project_id: int,
    *,
    allowed_labels: list[str] | None = None,
    row_count: int = 10,
) -> tuple[int, list[int]]:
    """Create a classification label_job + N unlabeled rows. Returns
    (job_id, row_ids)."""
    async def _go() -> tuple[int, list[int]]:
        async with async_session_factory() as session:
            # Use ``is None`` instead of truthiness so an explicit
            # empty list passes through (the "no label space configured"
            # test relies on this).
            effective_labels = (
                allowed_labels if allowed_labels is not None else ["A", "B", "C"]
            )
            job = LabelJob(
                project_id=project_id,
                name=f"al-job-{uuid.uuid4().hex[:4]}",
                label_type="classification",
                label_schema={
                    "allowed_labels": effective_labels,
                },
            )
            session.add(job)
            await session.flush()
            job_id = int(job.id)
            row_ids: list[int] = []
            for i in range(row_count):
                row = LabelRow(
                    job_id=job_id,
                    raw_payload={"text": f"row-{i}", "id": i},
                )
                session.add(row)
                await session.flush()
                row_ids.append(int(row.id))
            await session.commit()
            return job_id, row_ids

    return asyncio.run(_go())


def _label_some_rows(row_ids: list[int]) -> None:
    """Mark these rows as labeled so they drop out of the unlabeled pool."""
    async def _go() -> None:
        async with async_session_factory() as session:
            for rid in row_ids:
                row = (await session.execute(
                    select(LabelRow).where(LabelRow.id == rid)
                )).scalar_one()
                row.labeled_at = datetime.now(timezone.utc)
                row.label_payload = {"label": "A"}
            await session.commit()

    asyncio.run(_go())


def _run_scoring(
    project_id: int,
    experiment_id: int,
    *,
    top_k: int = DEFAULT_TOP_K,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
) -> dict:
    async def _go() -> dict:
        async with async_session_factory() as session:
            return await score_unlabeled_pool_for_experiment(
                session,
                project_id=project_id,
                experiment_id=experiment_id,
                top_k=top_k,
                sample_cap=sample_cap,
            )

    return asyncio.run(_go())


# ────────────────────────────────────────────────────────────────────────
# Skipped reasons
# ────────────────────────────────────────────────────────────────────────


class SkippedReasonTests(unittest.TestCase):

    def test_unsupported_task_type_skipped(self):
        pid = _create_project()
        eid = _seed_experiment(pid, task_type="causal_lm")
        snapshot = _run_scoring(pid, eid)
        self.assertEqual(snapshot["skipped_reason"], "unsupported_task_type")
        self.assertEqual(snapshot["task_type"], "causal_lm")
        self.assertEqual(snapshot["top_k"], [])
        # Slice 2's Coach nudge gates on top_k.length > 0; verify the
        # empty-pool case is well-formed for that check.
        self.assertEqual(snapshot["pool_size_total"], 0)

    def test_empty_pool_skipped(self):
        # Classification experiment + no label_jobs → empty pool.
        pid = _create_project()
        eid = _seed_experiment(pid, task_type="classification", output_dir=str(_make_checkpoint_dir()))
        snapshot = _run_scoring(pid, eid)
        self.assertEqual(snapshot["skipped_reason"], "empty_pool")

    def test_no_label_space_configured_skipped(self):
        # Has rows, but the label_job has no allowed_labels in schema.
        pid = _create_project()
        eid = _seed_experiment(pid, task_type="classification", output_dir=str(_make_checkpoint_dir()))
        # Empty allowed_labels → service can't load the head.
        _seed_classification_label_job(pid, allowed_labels=[])
        snapshot = _run_scoring(pid, eid)
        self.assertEqual(snapshot["skipped_reason"], "no_label_space_configured")
        # Pool size still surfaced — useful diagnostic for the user.
        self.assertGreater(snapshot["pool_size_total"], 0)

    def test_checkpoint_path_missing_skipped(self):
        pid = _create_project()
        # Empty output_dir — score_classification_rows would crash without
        # this guard, so we surface a clean skip.
        eid = _seed_experiment(pid, task_type="classification", output_dir="")
        _seed_classification_label_job(pid)
        snapshot = _run_scoring(pid, eid)
        self.assertEqual(snapshot["skipped_reason"], "checkpoint_path_missing")

    def test_checkpoint_path_nonexistent_skipped(self):
        pid = _create_project()
        eid = _seed_experiment(pid, task_type="classification", output_dir="/tmp/this/path/does/not/exist")
        _seed_classification_label_job(pid)
        snapshot = _run_scoring(pid, eid)
        self.assertEqual(snapshot["skipped_reason"], "checkpoint_path_missing")

    def test_scoring_failed_skip_carries_error(self):
        # Force score_classification_rows to raise — common when torch
        # isn't installed or CUDA OOMs. Snapshot must carry the error
        # so the user / Coach can show a useful diagnostic.
        pid = _create_project()
        ckpt = _make_checkpoint_dir()
        eid = _seed_experiment(pid, task_type="classification", output_dir=str(ckpt))
        _seed_classification_label_job(pid)

        def _boom(*args, **kwargs):
            raise RuntimeError("torch not available")

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            side_effect=_boom,
        ):
            snapshot = _run_scoring(pid, eid)

        self.assertEqual(snapshot["skipped_reason"], "scoring_failed")
        self.assertIn("torch not available", snapshot["error"])
        self.assertEqual(snapshot["checkpoint_path"], str(ckpt))


# ────────────────────────────────────────────────────────────────────────
# Happy path
# ────────────────────────────────────────────────────────────────────────


class HappyPathTests(unittest.TestCase):

    def _seed_setup(self, *, row_count: int = 10):
        pid = _create_project()
        ckpt = _make_checkpoint_dir()
        eid = _seed_experiment(pid, task_type="classification", output_dir=str(ckpt))
        job_id, row_ids = _seed_classification_label_job(pid, row_count=row_count)
        return pid, eid, job_id, row_ids

    def test_top_k_preserves_entropy_ranking(self):
        # 5 rows with monotonically increasing entropy → top_k should
        # come back in descending uncertainty order.
        pid, eid, job_id, row_ids = self._seed_setup(row_count=5)
        entropies = [0.10, 0.50, 0.30, 0.90, 0.70]

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            return_value=entropies,
        ):
            snapshot = _run_scoring(pid, eid, top_k=3)

        self.assertIsNone(snapshot["skipped_reason"])
        self.assertEqual(len(snapshot["top_k"]), 3)
        scores = [entry["uncertainty_score"] for entry in snapshot["top_k"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # The three highest entropies were 0.90, 0.70, 0.50 — verify.
        self.assertEqual(scores, [0.90, 0.70, 0.50])
        # Every entry carries label_row_id + label_job_id so the UI
        # can build a deep-link without a second fetch.
        for entry in snapshot["top_k"]:
            self.assertIn("label_row_id", entry)
            self.assertEqual(entry["label_job_id"], job_id)

    def test_none_scores_drop_out_of_top_k(self):
        # Rows whose text extraction failed get None entropy — they
        # must NOT crowd out genuinely uncertain rows by sorting first.
        pid, eid, job_id, row_ids = self._seed_setup(row_count=5)
        entropies = [None, 0.10, None, 0.20, 0.50]

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            return_value=entropies,
        ):
            snapshot = _run_scoring(pid, eid, top_k=5)

        # Only 3 entries survive (the non-None ones). Slice 1 contract
        # is "top_k entries that actually have an entropy."
        self.assertEqual(len(snapshot["top_k"]), 3)
        for entry in snapshot["top_k"]:
            self.assertGreater(entry["uncertainty_score"], 0)

    def test_sample_cap_clamps_pool(self):
        # Pool of 100 rows, sample_cap=10 → score_classification_rows
        # gets 10 rows; pool_size_scored reports 10, total reports 100.
        pid, eid, _, _ = self._seed_setup(row_count=100)
        captured = {}

        def _capture(rows, **kwargs):
            captured["count"] = len(rows)
            return [0.5] * len(rows)

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            side_effect=_capture,
        ):
            snapshot = _run_scoring(pid, eid, sample_cap=10)

        self.assertEqual(captured["count"], 10)
        self.assertEqual(snapshot["pool_size_total"], 100)
        self.assertEqual(snapshot["pool_size_scored"], 10)

    def test_labeled_rows_excluded_from_pool(self):
        pid, eid, _, row_ids = self._seed_setup(row_count=10)
        # Mark 7 of 10 rows as labeled → pool shrinks to 3.
        _label_some_rows(row_ids[:7])

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            return_value=[0.5, 0.5, 0.5],
        ) as patched:
            snapshot = _run_scoring(pid, eid)

        self.assertEqual(snapshot["pool_size_total"], 3)
        # Only the unlabeled 3 rows got passed to score_classification_rows.
        args, kwargs = patched.call_args
        self.assertEqual(len(args[0]), 3)

    def test_snapshot_contract_complete(self):
        pid, eid, _, _ = self._seed_setup(row_count=5)
        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            return_value=[0.10, 0.50, 0.30, 0.90, 0.70],
        ):
            snapshot = _run_scoring(pid, eid)

        # Fields the Coach nudge (slice 2) + Data Studio card (slice 3)
        # read directly — any drift here is a slice-2/3-breaking change.
        for key in (
            "scored_at", "model_experiment_id", "task_type",
            "uncertainty_metric", "pool_size_total", "pool_size_scored",
            "top_k", "skipped_reason", "checkpoint_path", "label_space_size",
        ):
            self.assertIn(key, snapshot, f"missing {key}")
        self.assertEqual(snapshot["model_experiment_id"], eid)
        self.assertEqual(snapshot["uncertainty_metric"], "entropy")
        self.assertEqual(snapshot["task_type"], "classification")


# ────────────────────────────────────────────────────────────────────────
# Stamp helper
# ────────────────────────────────────────────────────────────────────────


class StampSnapshotTests(unittest.TestCase):

    def test_stamp_writes_to_runtime_active_learning(self):
        pid = _create_project()
        eid = _seed_experiment(pid, task_type="classification", output_dir=str(_make_checkpoint_dir()))
        snapshot = {"top_k": [], "skipped_reason": "empty_pool"}

        async def _go() -> dict:
            async with async_session_factory() as session:
                await stamp_snapshot_on_experiment(
                    session, experiment_id=eid, snapshot=snapshot,
                )
            async with async_session_factory() as session:
                exp = (await session.execute(
                    select(Experiment).where(Experiment.id == eid)
                )).scalar_one()
                return dict(exp.config or {})

        cfg = asyncio.run(_go())
        # Slice 2 + slice 3 both read from this exact path; any drift
        # breaks them.
        self.assertEqual(
            cfg["_runtime"]["active_learning"], snapshot,
        )

    def test_stamp_preserves_existing_runtime_entries(self):
        # The COMPLETED branches stamp ``auto_rag_build`` BEFORE
        # the active_learning snapshot. Ensure we don't squash it.
        pid = _create_project()
        eid = _seed_experiment(pid, task_type="classification", output_dir=str(_make_checkpoint_dir()))

        async def _seed() -> None:
            async with async_session_factory() as session:
                exp = (await session.execute(
                    select(Experiment).where(Experiment.id == eid)
                )).scalar_one()
                cfg = dict(exp.config or {})
                cfg["_runtime"] = {"auto_rag_build": {"built": True, "docs": 42}}
                exp.config = cfg
                await session.commit()
        asyncio.run(_seed())

        async def _stamp_and_read() -> dict:
            async with async_session_factory() as session:
                await stamp_snapshot_on_experiment(
                    session,
                    experiment_id=eid,
                    snapshot={"top_k": [{"label_row_id": 1, "label_job_id": 2, "uncertainty_score": 0.9}]},
                )
            async with async_session_factory() as session:
                exp = (await session.execute(
                    select(Experiment).where(Experiment.id == eid)
                )).scalar_one()
                return dict(exp.config or {})

        cfg = asyncio.run(_stamp_and_read())
        # Both entries present.
        self.assertEqual(cfg["_runtime"]["auto_rag_build"]["docs"], 42)
        self.assertEqual(len(cfg["_runtime"]["active_learning"]["top_k"]), 1)


# ────────────────────────────────────────────────────────────────────────
# Multi-seed safe-hook semantics
# ────────────────────────────────────────────────────────────────────────


class SafeScoreSeedGroupTests(unittest.TestCase):
    """The runner-side hook (_safe_score_unlabeled_pool) must skip
    when the experiment is a seed-group child — the leader handles
    scoring once after the aggregator runs."""

    def test_child_skips_with_defer_reason(self):
        from app.services.training_service import _safe_score_unlabeled_pool

        pid = _create_project()
        eid = _seed_experiment(
            pid,
            task_type="classification",
            output_dir=str(_make_checkpoint_dir()),
            seed_value=42,
            seed_group_id="group-abc",
        )

        async def _go() -> dict:
            async with async_session_factory() as session:
                return await _safe_score_unlabeled_pool(
                    session, project_id=pid, experiment_id=eid,
                )

        result = asyncio.run(_go())
        self.assertEqual(result["skipped_reason"], "seed_group_child_defers_to_leader")
        self.assertEqual(result["seed_group_id"], "group-abc")

    def test_leader_runs_normal_path(self):
        from app.services.training_service import _safe_score_unlabeled_pool

        pid = _create_project()
        # Leader: seed_group_id set, seed_value NULL.
        eid = _seed_experiment(
            pid,
            task_type="classification",
            output_dir=str(_make_checkpoint_dir()),
            seed_value=None,
            seed_group_id="group-xyz",
        )
        _seed_classification_label_job(pid, row_count=3)

        async def _go() -> dict:
            async with async_session_factory() as session:
                return await _safe_score_unlabeled_pool(
                    session, project_id=pid, experiment_id=eid,
                )

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            return_value=[0.5, 0.5, 0.5],
        ):
            result = asyncio.run(_go())

        # Leader gets a real scoring snapshot — not the child skip
        # reason — even though it has a seed_group_id.
        self.assertNotEqual(result.get("skipped_reason"), "seed_group_child_defers_to_leader")
        self.assertEqual(result.get("model_experiment_id"), eid)

    def test_single_seed_runs_normal_path(self):
        # No seed group at all — neither seed_value nor seed_group_id.
        # Hook runs the regular scoring path.
        from app.services.training_service import _safe_score_unlabeled_pool

        pid = _create_project()
        eid = _seed_experiment(
            pid,
            task_type="classification",
            output_dir=str(_make_checkpoint_dir()),
        )
        _seed_classification_label_job(pid, row_count=3)

        async def _go() -> dict:
            async with async_session_factory() as session:
                return await _safe_score_unlabeled_pool(
                    session, project_id=pid, experiment_id=eid,
                )

        with patch(
            "app.services.annotation.active_learning.score_classification_rows",
            return_value=[0.5, 0.5, 0.5],
        ):
            result = asyncio.run(_go())

        self.assertIsNone(result.get("skipped_reason"))
        self.assertEqual(result.get("model_experiment_id"), eid)


if __name__ == "__main__":
    unittest.main()
