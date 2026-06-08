"""Tests for Quality-Lift phase 1, slice 2 — multi-seed dispatch + aggregation.

Pins for slice 2 (the dispatch + aggregator wiring; UI surfacing lands
in slice 3):

  Pure variance stats (no DB):
    * Single-seed degenerate case: n=1, std=0.0, min=max=mean.
    * Three-seed numeric metrics aggregate into {mean,std,min,max,n}.
    * Per-class nested dict shape recurses (matches the Gap-#6 shape).
    * Non-numeric leaves (strings, bools) pass through unchanged.
    * Missing keys across seeds — present-only seeds contribute.
    * NaN values are filtered, not propagated into the mean.

  Resolver:
    * Explicit seeds list wins over num_seeds.
    * num_seeds derives [seed, seed+1, ...] deterministically.
    * num_seeds=1 → single-seed list.

  Aggregator end-to-end (with DB):
    * 3-of-3 succeeded: leader → COMPLETED, one aggregate row per
      (dataset, eval_type), per-seed rows untouched, details carry
      provenance (which experiment_id + seed produced what).
    * 2-of-3 succeeded, 1 failed: leader → COMPLETED with warning
      counters; aggregate rolls over only the 2 successful children
      (n=2 in the stat block).
    * 0-of-3 succeeded: leader → FAILED, no aggregate row.
    * Partial completion (1 of 3 still RUNNING): no aggregate yet,
      leader stays RUNNING.
    * Idempotent: re-firing the hook after the first aggregation
      pass is a no-op (doesn't duplicate aggregate rows).
    * Drill-down: aggregate.details.per_seed lists each contributing
      (experiment_id, seed_value, eval_result_id) so the slice 3 UI
      can render the picked-data-provenance footer.
"""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import (  # noqa: E402
    EvalResult,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.services.experiment_aggregation_service import (  # noqa: E402
    compute_variance_stats,
    maybe_aggregate_seed_group,
)
from app.services.training_service import _resolve_seeds  # noqa: E402


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-multiseed-agg-{uuid.uuid4().hex[:8]}"
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
        json={"name": f"agg-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


class ComputeVarianceStatsTests(unittest.TestCase):
    """Pure function — no DB, no async."""

    def test_empty_input_returns_empty(self):
        self.assertEqual(compute_variance_stats([]), {})

    def test_single_seed_degenerate(self):
        # With n=1 there is no meaningful spread — but we still emit a
        # well-formed block so downstream UI never has to special-case
        # "no std field." std=0.0 by definition.
        result = compute_variance_stats([{"macro_f1": 0.83}])
        self.assertIn("macro_f1", result)
        block = result["macro_f1"]
        self.assertEqual(block["n"], 1)
        self.assertEqual(block["mean"], 0.83)
        self.assertEqual(block["std"], 0.0)
        self.assertEqual(block["min"], 0.83)
        self.assertEqual(block["max"], 0.83)

    def test_three_seed_numeric_metrics(self):
        per_seed = [
            {"macro_f1": 0.80, "accuracy": 0.85},
            {"macro_f1": 0.83, "accuracy": 0.87},
            {"macro_f1": 0.86, "accuracy": 0.89},
        ]
        result = compute_variance_stats(per_seed)
        f1 = result["macro_f1"]
        self.assertEqual(f1["n"], 3)
        self.assertAlmostEqual(f1["mean"], 0.83, places=5)
        # Population std: sqrt(((0.80-0.83)² + 0 + (0.86-0.83)²) / 3)
        expected_std = math.sqrt((0.0009 + 0.0 + 0.0009) / 3)
        self.assertAlmostEqual(f1["std"], expected_std, places=5)
        self.assertAlmostEqual(f1["min"], 0.80, places=5)
        self.assertAlmostEqual(f1["max"], 0.86, places=5)

    def test_per_class_nested_dict_recurses(self):
        # Mirrors the per-class shape introduced by Gap-#6 — the
        # aggregator must NOT flatten this back into scalar means at
        # the class level; users compare per-class spreads.
        per_seed = [
            {"per_class": {"A": {"precision": 0.80, "recall": 0.70, "support": 100}}},
            {"per_class": {"A": {"precision": 0.82, "recall": 0.75, "support": 100}}},
            {"per_class": {"A": {"precision": 0.84, "recall": 0.80, "support": 100}}},
        ]
        result = compute_variance_stats(per_seed)
        per_class_A = result["per_class"]["A"]
        self.assertIn("precision", per_class_A)
        self.assertIn("recall", per_class_A)
        self.assertAlmostEqual(per_class_A["precision"]["mean"], 0.82, places=5)
        self.assertEqual(per_class_A["recall"]["n"], 3)
        # Support — constant 100 across seeds → std=0, mean=100.
        self.assertEqual(per_class_A["support"]["mean"], 100.0)
        self.assertEqual(per_class_A["support"]["std"], 0.0)

    def test_non_numeric_leaves_pass_through(self):
        # Strings, booleans, None: nonsense to "mean" them, so they
        # pass through from the first seed that reports them.
        per_seed = [
            {"label": "positive", "is_valid": True},
            {"label": "positive", "is_valid": True},
        ]
        result = compute_variance_stats(per_seed)
        self.assertEqual(result["label"], "positive")
        self.assertEqual(result["is_valid"], True)
        # Critically: no "mean" key under is_valid. If a future change
        # accidentally treats bool as int, the next assertion fails.
        self.assertNotIsInstance(result["is_valid"], dict)

    def test_missing_keys_partial_seeds(self):
        # Seed 1 has 'extra', seeds 2-3 don't. Aggregate over what's
        # present. n=1 for 'extra', n=3 for 'shared'.
        per_seed = [
            {"shared": 0.5, "extra": 0.9},
            {"shared": 0.6},
            {"shared": 0.7},
        ]
        result = compute_variance_stats(per_seed)
        self.assertEqual(result["shared"]["n"], 3)
        self.assertEqual(result["extra"]["n"], 1)
        self.assertEqual(result["extra"]["mean"], 0.9)

    def test_nan_values_filtered(self):
        # NaN propagates through arithmetic; if it slips into the mean,
        # the whole aggregate is poison. We drop them entirely.
        per_seed = [
            {"f1": 0.80},
            {"f1": float("nan")},
            {"f1": 0.90},
        ]
        result = compute_variance_stats(per_seed)
        self.assertEqual(result["f1"]["n"], 2)
        self.assertAlmostEqual(result["f1"]["mean"], 0.85, places=5)


class ResolveSeedsTests(unittest.TestCase):
    """Resolver — no DB, no async."""

    def test_explicit_seeds_wins(self):
        self.assertEqual(
            _resolve_seeds({"seed": 42, "num_seeds": 99, "seeds": [7, 11]}),
            [7, 11],
        )

    def test_num_seeds_derives_deterministically(self):
        # Determinism is load-bearing: a re-launch after partial failure
        # must rebuild the same seed list to attach to the right group.
        self.assertEqual(
            _resolve_seeds({"seed": 42, "num_seeds": 3}),
            [42, 43, 44],
        )

    def test_single_seed_default(self):
        self.assertEqual(_resolve_seeds({}), [42])

    def test_num_seeds_one_collapses(self):
        self.assertEqual(_resolve_seeds({"seed": 99, "num_seeds": 1}), [99])


# ────────────────────────────────────────────────────────────────────────
# End-to-end aggregator tests (with DB)
# ────────────────────────────────────────────────────────────────────────


def _seed_group(
    project_id: int,
    n_children: int,
    group_id: str | None = None,
) -> tuple[str, int, list[int]]:
    """Create a leader + n_children child experiments under a fresh
    seed group. Returns (group_id, leader_id, [child_id, ...])."""
    gid = group_id or uuid.uuid4().hex

    async def _go() -> tuple[int, list[int]]:
        async with async_session_factory() as session:
            leader = Experiment(
                project_id=project_id,
                name="leader",
                status=ExperimentStatus.RUNNING,
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                training_mode=TrainingMode.SFT,
                config={"task_type": "classification"},
                seed_group_id=gid,
                # seed_value stays None on leader
            )
            session.add(leader)
            await session.flush()
            leader_id = int(leader.id)
            child_ids: list[int] = []
            for i in range(n_children):
                child = Experiment(
                    project_id=project_id,
                    name=f"child-seed-{42 + i}",
                    status=ExperimentStatus.RUNNING,
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    training_mode=TrainingMode.SFT,
                    config={"task_type": "classification", "seed": 42 + i},
                    seed_value=42 + i,
                    seed_group_id=gid,
                )
                session.add(child)
                await session.flush()
                child_ids.append(int(child.id))
            await session.commit()
            return leader_id, child_ids

    leader_id, child_ids = asyncio.run(_go())
    return gid, leader_id, child_ids


def _set_child_terminal(
    child_id: int,
    status: ExperimentStatus,
    metrics: dict | None = None,
    pass_rate: float | None = None,
) -> None:
    """Mark a child as terminal with optional eval results."""

    async def _go() -> None:
        async with async_session_factory() as session:
            exp = (await session.execute(
                select(Experiment).where(Experiment.id == child_id)
            )).scalar_one()
            exp.status = status
            if status == ExperimentStatus.COMPLETED and metrics is not None:
                er = EvalResult(
                    experiment_id=child_id,
                    dataset_name="held_out",
                    eval_type="classification",
                    metrics=metrics,
                    pass_rate=pass_rate,
                )
                session.add(er)
            await session.commit()

    asyncio.run(_go())


def _fire_aggregator(experiment_id: int) -> dict | None:
    """Drive the aggregator in a fresh session, returning its summary."""

    async def _go() -> dict | None:
        async with async_session_factory() as session:
            return await maybe_aggregate_seed_group(session, experiment_id)

    return asyncio.run(_go())


def _load_aggregate_rows(leader_id: int) -> list[EvalResult]:
    async def _go() -> list[EvalResult]:
        async with async_session_factory() as session:
            rows = (await session.execute(
                select(EvalResult).where(
                    EvalResult.experiment_id == leader_id,
                    EvalResult.is_aggregate.is_(True),
                )
            )).scalars().all()
            return list(rows)

    return asyncio.run(_go())


def _load_leader(leader_id: int) -> Experiment:
    async def _go() -> Experiment:
        async with async_session_factory() as session:
            return (await session.execute(
                select(Experiment).where(Experiment.id == leader_id)
            )).scalar_one()

    return asyncio.run(_go())


class AggregatorEndToEndTests(unittest.TestCase):

    def test_three_of_three_succeeded_rolls_up(self):
        pid = _create_project()
        gid, leader_id, child_ids = _seed_group(pid, n_children=3)
        for i, cid in enumerate(child_ids):
            _set_child_terminal(
                cid,
                ExperimentStatus.COMPLETED,
                metrics={"macro_f1": 0.80 + 0.03 * i, "accuracy": 0.85 + 0.02 * i},
                pass_rate=1.0,
            )

        # Fire the hook on the last child (mirrors the runtime hook
        # call site). The aggregator must walk the whole group.
        summary = _fire_aggregator(child_ids[-1])

        self.assertIsNotNone(summary)
        self.assertEqual(summary["n_succeeded"], 3)
        self.assertEqual(summary["n_failed"], 0)
        self.assertEqual(len(summary["aggregates_created"]), 1)

        leader = _load_leader(leader_id)
        self.assertEqual(leader.status, ExperimentStatus.COMPLETED)
        # Seed-group summary stamped onto leader.config for the UI.
        sg = (leader.config or {})["_seed_group"]
        self.assertEqual(sg["n_total"], 3)
        self.assertEqual(sg["n_succeeded"], 3)
        self.assertEqual(sg["n_failed"], 0)
        self.assertEqual(sg["succeeded_seeds"], sorted(c for c in [42, 43, 44]))

        aggs = _load_aggregate_rows(leader_id)
        self.assertEqual(len(aggs), 1)
        agg = aggs[0]
        self.assertEqual(agg.dataset_name, "held_out")
        self.assertEqual(agg.eval_type, "classification")
        self.assertTrue(agg.is_aggregate)
        # Variance-stat shape:
        self.assertEqual(agg.metrics["macro_f1"]["n"], 3)
        self.assertAlmostEqual(agg.metrics["macro_f1"]["mean"], 0.83, places=5)
        # Drill-down provenance: slice 3 UI will click through this.
        per_seed = agg.details["per_seed"]
        self.assertEqual(len(per_seed), 3)
        for entry in per_seed:
            self.assertIn(entry["experiment_id"], child_ids)
            self.assertIn(entry["seed_value"], [42, 43, 44])
            self.assertIsNotNone(entry["eval_result_id"])

    def test_two_of_three_succeeded_aggregates_over_two(self):
        pid = _create_project()
        gid, leader_id, child_ids = _seed_group(pid, n_children=3)
        _set_child_terminal(
            child_ids[0], ExperimentStatus.COMPLETED,
            metrics={"macro_f1": 0.80}, pass_rate=1.0,
        )
        _set_child_terminal(
            child_ids[1], ExperimentStatus.FAILED,
        )
        _set_child_terminal(
            child_ids[2], ExperimentStatus.COMPLETED,
            metrics={"macro_f1": 0.86}, pass_rate=1.0,
        )

        summary = _fire_aggregator(child_ids[-1])
        self.assertEqual(summary["n_succeeded"], 2)
        self.assertEqual(summary["n_failed"], 1)

        leader = _load_leader(leader_id)
        # Leader still COMPLETED — at least one child made it.
        self.assertEqual(leader.status, ExperimentStatus.COMPLETED)
        sg = (leader.config or {})["_seed_group"]
        self.assertEqual(sg["failed_seeds"], [43])

        aggs = _load_aggregate_rows(leader_id)
        self.assertEqual(len(aggs), 1)
        # n=2 in the variance stat block — the failed child must NOT
        # contaminate the mean.
        self.assertEqual(aggs[0].metrics["macro_f1"]["n"], 2)
        self.assertAlmostEqual(aggs[0].metrics["macro_f1"]["mean"], 0.83, places=5)

    def test_all_failed_leader_is_failed_no_aggregate(self):
        pid = _create_project()
        gid, leader_id, child_ids = _seed_group(pid, n_children=2)
        for cid in child_ids:
            _set_child_terminal(cid, ExperimentStatus.FAILED)

        summary = _fire_aggregator(child_ids[-1])
        self.assertEqual(summary["n_succeeded"], 0)
        self.assertEqual(summary["n_failed"], 2)
        self.assertEqual(summary["aggregates_created"], [])

        leader = _load_leader(leader_id)
        self.assertEqual(leader.status, ExperimentStatus.FAILED)

        aggs = _load_aggregate_rows(leader_id)
        self.assertEqual(len(aggs), 0)

    def test_partial_completion_no_aggregate(self):
        # Two of three terminal — one still RUNNING. Aggregator must
        # not fire until the last child finishes.
        pid = _create_project()
        gid, leader_id, child_ids = _seed_group(pid, n_children=3)
        _set_child_terminal(
            child_ids[0], ExperimentStatus.COMPLETED, metrics={"f1": 0.8},
        )
        _set_child_terminal(
            child_ids[1], ExperimentStatus.COMPLETED, metrics={"f1": 0.85},
        )
        # child_ids[2] stays RUNNING

        summary = _fire_aggregator(child_ids[1])
        self.assertIsNone(summary)

        leader = _load_leader(leader_id)
        self.assertEqual(leader.status, ExperimentStatus.RUNNING)
        self.assertEqual(len(_load_aggregate_rows(leader_id)), 0)

    def test_aggregation_is_idempotent(self):
        # Race between two children's terminal hooks: both fire
        # maybe_aggregate_seed_group. The second call must not insert
        # duplicate aggregate rows.
        pid = _create_project()
        gid, leader_id, child_ids = _seed_group(pid, n_children=2)
        for i, cid in enumerate(child_ids):
            _set_child_terminal(
                cid, ExperimentStatus.COMPLETED,
                metrics={"f1": 0.80 + 0.05 * i}, pass_rate=1.0,
            )

        first = _fire_aggregator(child_ids[-1])
        self.assertEqual(len(first["aggregates_created"]), 1)
        # Second call: aggregation already done, must produce no new rows.
        second = _fire_aggregator(child_ids[-1])
        # Note: leader status still resolves on second call (idempotent
        # transition COMPLETED → COMPLETED), but no new aggregate row.
        self.assertEqual(second["aggregates_created"], [])

        self.assertEqual(len(_load_aggregate_rows(leader_id)), 1)

    def test_single_seed_experiment_is_no_op(self):
        # A non-seed-group experiment (no seed_group_id) firing the
        # hook must be a silent no-op — single-seed flows go through
        # the same runtime hook sites.
        pid = _create_project()

        async def _seed() -> int:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=pid,
                    name="solo",
                    status=ExperimentStatus.COMPLETED,
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    training_mode=TrainingMode.SFT,
                    config={"seed": 42},
                )
                session.add(exp)
                await session.commit()
                return int(exp.id)

        eid = asyncio.run(_seed())
        result = _fire_aggregator(eid)
        self.assertIsNone(result)

    def test_leader_firing_hook_is_no_op(self):
        # The hook is called for child experiments; if it somehow gets
        # called for the leader (seed_value=NULL) it must be a no-op.
        pid = _create_project()
        gid, leader_id, _children = _seed_group(pid, n_children=2)
        result = _fire_aggregator(leader_id)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
