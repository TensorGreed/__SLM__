"""Tests for Quality-Lift phase 1, slice 1 — multi-seed schema foundation.

Pins for slice 1 (schema + migration + model round-trip only — no
dispatch / aggregation behavior yet, those land in slice 2/3):

  * TrainingConfig defaults preserve single-seed behavior:
    seeds=None, num_seeds=1, parallel_seeds=False.
  * TrainingConfig validates an explicit seeds list and a num_seeds
    bounded to [1, 8].
  * Experiment can be created with seed_value + seed_group_id and the
    columns round-trip through the DB.
  * EvalResult can be created with is_aggregate=True and a metrics
    payload whose values are {mean,std,min,max,n} dicts; the JSON
    column round-trips that shape unchanged.
  * Legacy Experiment / EvalResult rows (no seed_* columns set, no
    is_aggregate set) still construct and read back correctly — the
    additive migration must not break existing single-seed flows.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import EvalResult, Experiment, TrainingMode  # noqa: E402
from app.schemas.training import TrainingConfig  # noqa: E402


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-multiseed-{uuid.uuid4().hex[:8]}"
)


def setUpModule() -> None:
    settings.AUTH_ENABLED = False
    settings.DEBUG = False
    settings.DATA_DIR = TEST_DATA_DIR.resolve()
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    # Open a TestClient at module scope so the FastAPI lifespan runs
    # init_db once — same pattern as test_jobs_service.
    global _CLIENT_CM, CLIENT
    _CLIENT_CM = TestClient(app)
    CLIENT = _CLIENT_CM.__enter__()


def tearDownModule() -> None:
    _CLIENT_CM.__exit__(None, None, None)


def _create_project() -> int:
    resp = CLIENT.post(
        "/api/projects",
        json={"name": f"multiseed-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


class TrainingConfigSchemaTests(unittest.TestCase):
    """Pydantic-level checks on the new seeds/num_seeds/parallel_seeds fields."""

    def test_defaults_preserve_single_seed_behavior(self):
        # The whole point of defaulting num_seeds=1 + seeds=None is that
        # existing single-seed flows see zero behavior change. If this
        # ever changes, the rollout becomes a breaking change to every
        # in-flight recipe.
        cfg = TrainingConfig(base_model="HuggingFaceTB/SmolLM2-135M")
        self.assertIsNone(cfg.seeds)
        self.assertEqual(cfg.num_seeds, 1)
        self.assertFalse(cfg.parallel_seeds)
        self.assertEqual(cfg.seed, 42)

    def test_explicit_seed_list_validates(self):
        cfg = TrainingConfig(
            base_model="HuggingFaceTB/SmolLM2-135M",
            seeds=[7, 11, 13],
        )
        self.assertEqual(cfg.seeds, [7, 11, 13])
        # num_seeds default stays 1 — resolution logic (slice 2) chooses
        # ``seeds`` over ``num_seeds`` when both are set, so the field
        # shapes are independent at the schema layer.
        self.assertEqual(cfg.num_seeds, 1)

    def test_num_seeds_lower_bound_rejected(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(
                base_model="HuggingFaceTB/SmolLM2-135M",
                num_seeds=0,
            )

    def test_num_seeds_upper_bound_rejected(self):
        # 8 is the cap — anything above and a single user can saturate
        # the box. If you ever raise this, raise the GPU scheduler limit
        # in lockstep.
        with self.assertRaises(ValidationError):
            TrainingConfig(
                base_model="HuggingFaceTB/SmolLM2-135M",
                num_seeds=9,
            )

    def test_num_seeds_upper_bound_accepted_at_edge(self):
        cfg = TrainingConfig(
            base_model="HuggingFaceTB/SmolLM2-135M",
            num_seeds=8,
        )
        self.assertEqual(cfg.num_seeds, 8)

    def test_parallel_seeds_toggle(self):
        cfg = TrainingConfig(
            base_model="HuggingFaceTB/SmolLM2-135M",
            num_seeds=3,
            parallel_seeds=True,
        )
        self.assertTrue(cfg.parallel_seeds)


class ExperimentSeedColumnTests(unittest.TestCase):
    """ORM round-trip checks on the new seed_value / seed_group_id columns."""

    def test_experiment_with_seed_columns_round_trips(self):
        pid = _create_project()
        group_id = str(uuid.uuid4())

        async def _go() -> tuple[int, int | None, str | None]:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=pid,
                    name="seed-7-child",
                    description="child of a num_seeds=3 group",
                    status="completed",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    training_mode=TrainingMode.SFT,
                    config={"task_type": "classification", "seed": 7},
                    seed_value=7,
                    seed_group_id=group_id,
                )
                session.add(exp)
                await session.commit()
                eid = int(exp.id)

            async with async_session_factory() as session:
                fresh = (await session.execute(
                    select(Experiment).where(Experiment.id == eid)
                )).scalar_one()
                return eid, fresh.seed_value, fresh.seed_group_id

        eid, seed_value, fetched_group = asyncio.run(_go())
        self.assertGreater(eid, 0)
        self.assertEqual(seed_value, 7)
        self.assertEqual(fetched_group, group_id)

    def test_legacy_experiment_without_seed_columns_still_works(self):
        # The additive migration must not break the (currently dominant)
        # single-seed path. If this test fails after the migration, the
        # nullable defaults regressed somewhere.
        pid = _create_project()

        async def _go() -> tuple[int | None, str | None]:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=pid,
                    name="legacy-single-seed",
                    status="completed",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    training_mode=TrainingMode.SFT,
                    config={"task_type": "classification"},
                )
                session.add(exp)
                await session.commit()
                eid = int(exp.id)

            async with async_session_factory() as session:
                fresh = (await session.execute(
                    select(Experiment).where(Experiment.id == eid)
                )).scalar_one()
                return fresh.seed_value, fresh.seed_group_id

        seed_value, group_id = asyncio.run(_go())
        self.assertIsNone(seed_value)
        self.assertIsNone(group_id)

    def test_seed_group_id_indexed_lookup(self):
        # Two children sharing a seed_group_id must both be findable by
        # the group id — the aggregator (slice 2) will rely on this.
        pid = _create_project()
        group_id = str(uuid.uuid4())

        async def _go() -> int:
            async with async_session_factory() as session:
                for seed in (42, 43):
                    session.add(Experiment(
                        project_id=pid,
                        name=f"child-seed-{seed}",
                        status="completed",
                        base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                        training_mode=TrainingMode.SFT,
                        config={"seed": seed},
                        seed_value=seed,
                        seed_group_id=group_id,
                    ))
                await session.commit()

            async with async_session_factory() as session:
                rows = (await session.execute(
                    select(Experiment).where(
                        Experiment.seed_group_id == group_id
                    )
                )).scalars().all()
                return len(rows)

        count = asyncio.run(_go())
        self.assertEqual(count, 2)


class EvalResultAggregateColumnTests(unittest.TestCase):
    """ORM + JSON round-trip checks on is_aggregate + the new metrics shape."""

    def _seed_experiment(self, project_id: int) -> int:
        async def _go() -> int:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=project_id,
                    name="parent-leader",
                    status="completed",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    training_mode=TrainingMode.SFT,
                    config={"task_type": "classification"},
                )
                session.add(exp)
                await session.commit()
                return int(exp.id)
        return asyncio.run(_go())

    def test_aggregate_eval_result_with_variance_metrics_round_trips(self):
        pid = _create_project()
        eid = self._seed_experiment(pid)
        group_id = str(uuid.uuid4())
        agg_metrics = {
            "macro_f1": {
                "mean": 0.832, "std": 0.0125, "min": 0.81,
                "max": 0.85, "n": 3,
            },
            "accuracy": {
                "mean": 0.871, "std": 0.0080, "min": 0.86,
                "max": 0.88, "n": 3,
            },
        }

        async def _go() -> tuple[bool, dict, str | None]:
            async with async_session_factory() as session:
                er = EvalResult(
                    experiment_id=eid,
                    dataset_name="held_out",
                    eval_type="classification",
                    metrics=agg_metrics,
                    pass_rate=1.0,
                    is_aggregate=True,
                    seed_group_id=group_id,
                )
                session.add(er)
                await session.commit()
                rid = int(er.id)

            async with async_session_factory() as session:
                fresh = (await session.execute(
                    select(EvalResult).where(EvalResult.id == rid)
                )).scalar_one()
                return fresh.is_aggregate, fresh.metrics, fresh.seed_group_id

        is_agg, fetched, fetched_group = asyncio.run(_go())
        self.assertTrue(is_agg)
        self.assertEqual(fetched_group, group_id)
        # JSON column must round-trip nested dicts byte-equivalent.
        self.assertEqual(fetched, agg_metrics)
        self.assertEqual(fetched["macro_f1"]["mean"], 0.832)
        self.assertEqual(fetched["macro_f1"]["n"], 3)

    def test_legacy_scalar_eval_result_still_works(self):
        # Single-seed flow keeps writing scalar metrics + is_aggregate=False
        # by default. The aggregator (slice 2) is the only writer that
        # flips is_aggregate=True. If this test breaks, it means the
        # default flipped — that's a behavior-change masquerading as a
        # schema migration.
        pid = _create_project()
        eid = self._seed_experiment(pid)

        async def _go() -> tuple[bool, dict, str | None]:
            async with async_session_factory() as session:
                er = EvalResult(
                    experiment_id=eid,
                    dataset_name="held_out",
                    eval_type="classification",
                    metrics={"macro_f1": 0.83, "accuracy": 0.87},
                    pass_rate=1.0,
                )
                session.add(er)
                await session.commit()
                rid = int(er.id)

            async with async_session_factory() as session:
                fresh = (await session.execute(
                    select(EvalResult).where(EvalResult.id == rid)
                )).scalar_one()
                return fresh.is_aggregate, fresh.metrics, fresh.seed_group_id

        is_agg, fetched, group_id = asyncio.run(_go())
        self.assertFalse(is_agg)
        self.assertIsNone(group_id)
        self.assertEqual(fetched, {"macro_f1": 0.83, "accuracy": 0.87})


if __name__ == "__main__":
    unittest.main()
