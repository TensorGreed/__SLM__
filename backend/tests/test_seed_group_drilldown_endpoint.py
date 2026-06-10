"""Quality-Lift phase 8 slice 1 — seed-group drill-down endpoint.

Pins:
  * GET /evaluation/results/{exp_id} returns ``is_aggregate`` +
    ``seed_group_id`` so the EvalPanel can decide whether to render
    the AggregateRunBadge at all.
  * GET /evaluation/seed-group/{group_id} returns the per-seed child
    experiments + their scalar EvalResults so the badge's "drill
    into the 3 individual runs" expander has data to render.
  * Filter by ``?dataset_name=&eval_type=`` so the badge can scope
    the drill-down to the same row the user clicked from.
  * Cross-project isolation: a seed_group_id that lives in project A
    is 404 from project B's URL (don't leak siblings).
  * Pending children with no EvalResult yet are still surfaced (with
    empty metrics) so the user understands why a seed is missing
    from the mean — picked-data-provenance rule.
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

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import (  # noqa: E402
    EvalResult,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-seed-drill-{uuid.uuid4().hex[:8]}"
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
        json={"name": f"seed-drill-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


async def _seed_group(
    project_id: int,
    *,
    seed_group_id: str,
    child_seeds: list[int],
    child_metrics: list[dict | None],
    aggregate_metrics: dict | None,
    dataset: str = "goldset",
    eval_type: str = "classification",
) -> int:
    """Seed a leader + N children + optional aggregate EvalResult.

    Returns the leader's experiment_id. Caller can then exercise the
    endpoint against the seed_group_id.
    """
    async with async_session_factory() as session:
        leader = Experiment(
            project_id=project_id,
            name=f"leader-{uuid.uuid4().hex[:6]}",
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            training_mode=TrainingMode.SFT,
            status=ExperimentStatus.COMPLETED,
            seed_group_id=seed_group_id,
            seed_value=None,  # leader has NULL seed_value
            config={"num_seeds": len(child_seeds)},
        )
        session.add(leader)
        await session.flush()
        leader_id = int(leader.id)

        for seed_val, metrics in zip(child_seeds, child_metrics):
            child = Experiment(
                project_id=project_id,
                name=f"child-{seed_val}-{uuid.uuid4().hex[:4]}",
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                training_mode=TrainingMode.SFT,
                status=(
                    ExperimentStatus.COMPLETED
                    if metrics is not None else ExperimentStatus.FAILED
                ),
                seed_group_id=seed_group_id,
                seed_value=seed_val,
                config={"num_seeds": 1},
            )
            session.add(child)
            await session.flush()
            if metrics is not None:
                session.add(EvalResult(
                    experiment_id=int(child.id),
                    dataset_name=dataset,
                    eval_type=eval_type,
                    metrics=metrics,
                    pass_rate=float(metrics.get("pass_rate", 0.0)),
                    is_aggregate=False,
                ))

        if aggregate_metrics is not None:
            session.add(EvalResult(
                experiment_id=leader_id,
                dataset_name=dataset,
                eval_type=eval_type,
                metrics=aggregate_metrics,
                pass_rate=None,
                is_aggregate=True,
                seed_group_id=seed_group_id,
            ))

        await session.commit()
        return leader_id


class EvalResultResponseAggregateFieldsTests(unittest.TestCase):
    """The EvalResultResponse schema MUST expose is_aggregate +
    seed_group_id so the frontend EvalPanel can render the badge."""

    def test_results_endpoint_surfaces_is_aggregate_and_seed_group_id(self):
        pid = _create_project()
        group_id = "grp-" + uuid.uuid4().hex[:8]
        leader_id = asyncio.run(_seed_group(
            pid,
            seed_group_id=group_id,
            child_seeds=[42, 43, 44],
            child_metrics=[
                {"accuracy": 0.82, "f1": 0.81, "pass_rate": 0.82},
                {"accuracy": 0.87, "f1": 0.86, "pass_rate": 0.87},
                {"accuracy": 0.84, "f1": 0.83, "pass_rate": 0.84},
            ],
            aggregate_metrics={
                "accuracy": {
                    "mean": 0.843, "std": 0.021, "min": 0.82, "max": 0.87, "n": 3,
                },
                "f1": {
                    "mean": 0.833, "std": 0.022, "min": 0.81, "max": 0.86, "n": 3,
                },
            },
        ))
        resp = CLIENT.get(f"/api/projects/{pid}/evaluation/results/{leader_id}")
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = resp.json()
        # The leader has one aggregate row; serializer must include
        # the new fields.
        agg_rows = [r for r in rows if r["is_aggregate"]]
        self.assertEqual(len(agg_rows), 1)
        self.assertEqual(agg_rows[0]["seed_group_id"], group_id)
        # Spot-check the variance dict shape made it through Pydantic
        # without being scalar-coerced (this would break the badge).
        self.assertIsInstance(agg_rows[0]["metrics"]["accuracy"], dict)
        self.assertIn("mean", agg_rows[0]["metrics"]["accuracy"])
        self.assertIn("std", agg_rows[0]["metrics"]["accuracy"])


class SeedGroupDrillDownEndpointTests(unittest.TestCase):

    def test_drilldown_returns_per_seed_children_sorted_by_seed_value(self):
        pid = _create_project()
        group_id = "grp-" + uuid.uuid4().hex[:8]
        asyncio.run(_seed_group(
            pid,
            seed_group_id=group_id,
            child_seeds=[44, 42, 43],  # deliberately unordered
            child_metrics=[
                {"accuracy": 0.84, "f1": 0.83, "pass_rate": 0.84},
                {"accuracy": 0.82, "f1": 0.81, "pass_rate": 0.82},
                {"accuracy": 0.87, "f1": 0.86, "pass_rate": 0.87},
            ],
            aggregate_metrics={
                "accuracy": {"mean": 0.843, "std": 0.021, "min": 0.82, "max": 0.87, "n": 3},
            },
        ))
        resp = CLIENT.get(
            f"/api/projects/{pid}/evaluation/seed-group/{group_id}"
            "?dataset_name=goldset&eval_type=classification"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["seed_group_id"], group_id)
        # aggregate_eval_result_id is non-null when filters match an
        # existing aggregate — lets the frontend link the badge row
        # back to the underlying EvalResult.
        self.assertIsNotNone(body["aggregate_eval_result_id"])
        children = body["children"]
        self.assertEqual(len(children), 3)
        self.assertEqual([c["seed_value"] for c in children], [42, 43, 44])
        # Each child carries its scalar metrics (no variance dict
        # wrapping — these are the per-seed scalars).
        for child in children:
            self.assertIsInstance(child["metrics"]["accuracy"], float)
            self.assertEqual(child["experiment_status"], "completed")

    def test_drilldown_filters_to_specified_dataset_eval_type(self):
        # When the same children have results for multiple eval
        # types, the filter scopes the drill-down to just one.
        pid = _create_project()
        group_id = "grp-" + uuid.uuid4().hex[:8]
        leader_id = asyncio.run(_seed_group(
            pid,
            seed_group_id=group_id,
            child_seeds=[42, 43],
            child_metrics=[
                {"accuracy": 0.80, "pass_rate": 0.80},
                {"accuracy": 0.85, "pass_rate": 0.85},
            ],
            aggregate_metrics={
                "accuracy": {"mean": 0.825, "std": 0.025, "min": 0.80, "max": 0.85, "n": 2},
            },
            dataset="goldset",
            eval_type="classification",
        ))
        # Seed an extra child eval row in a different dataset.
        async def _extra() -> None:
            async with async_session_factory() as session:
                children = (await session.execute(
                    __import__("sqlalchemy").select(Experiment).where(
                        Experiment.seed_group_id == group_id,
                        Experiment.seed_value.is_not(None),
                    )
                )).scalars().all()
                for c in children:
                    session.add(EvalResult(
                        experiment_id=int(c.id),
                        dataset_name="otherset",
                        eval_type="safety",
                        metrics={"safety_pass_rate": 1.0, "pass_rate": 1.0},
                        pass_rate=1.0,
                        is_aggregate=False,
                    ))
                await session.commit()
        asyncio.run(_extra())
        # Without the filter the response carries 4 rows (2 children
        # × 2 eval types).
        resp_all = CLIENT.get(f"/api/projects/{pid}/evaluation/seed-group/{group_id}")
        self.assertEqual(len(resp_all.json()["children"]), 4)
        # With the filter we get back exactly the two classification
        # rows the badge clicked from.
        resp_filtered = CLIENT.get(
            f"/api/projects/{pid}/evaluation/seed-group/{group_id}"
            "?dataset_name=goldset&eval_type=classification"
        )
        children = resp_filtered.json()["children"]
        self.assertEqual(len(children), 2)
        for c in children:
            self.assertIn("accuracy", c["metrics"])
            self.assertNotIn("safety_pass_rate", c["metrics"])
        # Sanity-check the leader id round-trips so the UI can
        # navigate up to the leader experiment.
        self.assertEqual(resp_filtered.json()["leader_experiment_id"], leader_id)

    def test_unknown_seed_group_returns_404(self):
        pid = _create_project()
        resp = CLIENT.get(
            f"/api/projects/{pid}/evaluation/seed-group/no-such-group"
        )
        self.assertEqual(resp.status_code, 404)

    def test_drilldown_isolated_to_project(self):
        # Seed group A in project A → project B must not be able to
        # read it via the URL path.
        pid_a = _create_project()
        pid_b = _create_project()
        group_id = "grp-" + uuid.uuid4().hex[:8]
        asyncio.run(_seed_group(
            pid_a,
            seed_group_id=group_id,
            child_seeds=[42, 43],
            child_metrics=[
                {"accuracy": 0.81, "pass_rate": 0.81},
                {"accuracy": 0.84, "pass_rate": 0.84},
            ],
            aggregate_metrics=None,
        ))
        resp_correct = CLIENT.get(
            f"/api/projects/{pid_a}/evaluation/seed-group/{group_id}"
        )
        self.assertEqual(resp_correct.status_code, 200)
        resp_wrong = CLIENT.get(
            f"/api/projects/{pid_b}/evaluation/seed-group/{group_id}"
        )
        self.assertEqual(resp_wrong.status_code, 404)

    def test_pending_child_with_no_eval_row_still_surfaces(self):
        # Mid-run state: one child finished + has eval, one is still
        # going (FAILED with no EvalResult). The drill-down surfaces
        # both so the badge can explain a 2-of-3 mean honestly.
        pid = _create_project()
        group_id = "grp-" + uuid.uuid4().hex[:8]
        asyncio.run(_seed_group(
            pid,
            seed_group_id=group_id,
            child_seeds=[42, 43],
            child_metrics=[
                {"accuracy": 0.83, "pass_rate": 0.83},
                None,  # second child failed before eval ran
            ],
            aggregate_metrics=None,
        ))
        body = CLIENT.get(
            f"/api/projects/{pid}/evaluation/seed-group/{group_id}"
        ).json()
        children = body["children"]
        self.assertEqual(len(children), 2)
        # The failed child still appears in the list with empty
        # metrics + its status, so the UI can render a row that says
        # "seed 43 — failed, no eval".
        failed = [c for c in children if c["seed_value"] == 43][0]
        self.assertEqual(failed["metrics"], {})
        self.assertEqual(failed["experiment_status"], "failed")


if __name__ == "__main__":
    unittest.main()
