"""Quality-Lift phase 7 slice 3 — Coach Mode multi-seed variance nudge.

Pins:
  * No experiments yet → no nudge (training surface itself is enough).
  * Latest run is multi-seed → no nudge (user is doing the right thing).
  * Single-seed latest + prior aggregate with ≥10% relative std on one
    metric → ``training:variance-hidden`` (warning), body cites the
    actual mean + std so the claim is falsifiable.
  * Single-seed latest + EvalResult exists + no prior aggregate → soft
    ``training:variance-unknown`` (info).
  * Single-seed latest + NO EvalResult anywhere → no nudge (nothing
    to compare against — coach can't claim variance matters yet).
  * Action payload deep-links to ``training-config`` with
    ``expand_multi_seed=True``.
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
from app.services.coach_service import _multi_seed_variance_nudge  # noqa: E402


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-ms-variance-{uuid.uuid4().hex[:8]}"
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
        json={"name": f"variance-nudge-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


async def _seed_experiment(
    project_id: int,
    *,
    seed_group_id: str | None,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    eval_metrics: dict | None = None,
    is_aggregate: bool = False,
) -> int:
    async with async_session_factory() as session:
        exp = Experiment(
            project_id=project_id,
            name=f"exp-{uuid.uuid4().hex[:6]}",
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            training_mode=TrainingMode.SFT,
            status=status,
            seed_group_id=seed_group_id,
            config={"num_seeds": 3 if seed_group_id else 1},
        )
        session.add(exp)
        await session.flush()
        if eval_metrics is not None:
            session.add(EvalResult(
                experiment_id=exp.id,
                dataset_name="goldset",
                eval_type="classification",
                metrics=eval_metrics,
                is_aggregate=is_aggregate,
                seed_group_id=seed_group_id if is_aggregate else None,
            ))
        await session.commit()
        return int(exp.id)


async def _call_nudge(project_id: int) -> dict | None:
    async with async_session_factory() as session:
        return await _multi_seed_variance_nudge(session, project_id)


class MultiSeedVarianceNudgeTests(unittest.TestCase):

    def test_no_experiments_returns_none(self):
        pid = _create_project()
        self.assertIsNone(asyncio.run(_call_nudge(pid)))

    def test_latest_multi_seed_returns_none(self):
        # Project's most recent terminal run is multi-seed → coach
        # silences (user is doing the right thing already).
        pid = _create_project()
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id="group-abc",
            eval_metrics={"accuracy": {"mean": 0.85, "std": 0.02, "n": 3}},
            is_aggregate=True,
        ))
        self.assertIsNone(asyncio.run(_call_nudge(pid)))

    def test_high_std_aggregate_with_single_seed_latest_returns_warning(self):
        # Older multi-seed aggregate measured 12% relative std on
        # accuracy → variance is *real* for this project. Latest run
        # is single-seed → user has gone back to one seed and is now
        # hiding the variance. Coach warns + names the metric.
        pid = _create_project()
        # Seed the older aggregate first so the latest-experiment
        # query picks up the single-seed run as most recent.
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id="older-group",
            eval_metrics={
                "accuracy": {
                    "mean": 0.85, "std": 0.105, "min": 0.71, "max": 0.93, "n": 3,
                }
            },
            is_aggregate=True,
        ))
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id=None,  # single-seed latest run
            eval_metrics={"accuracy": 0.84},
            is_aggregate=False,
        ))
        result = asyncio.run(_call_nudge(pid))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["id"], "training:variance-hidden")
        self.assertEqual(result["severity"], "warning")
        # Body cites the metric name + std so the claim is auditable.
        self.assertIn("accuracy", result["title"])
        self.assertIn("std=0.105", result["body"])
        self.assertIn("mean=0.850", result["body"])
        # Action deep-links to the training config page with the
        # multi-seed section pre-expanded.
        self.assertEqual(result["action"]["kind"], "navigate")
        self.assertEqual(result["action"]["params"]["target"], "training-config")
        self.assertTrue(result["action"]["params"]["expand_multi_seed"])
        self.assertEqual(result["action"]["params"]["suggested_num_seeds"], 3)
        # Context carries the metric name + raw values for downstream
        # telemetry (no vanity — the same numbers the body claims).
        self.assertEqual(result["context"]["worst_metric"], "accuracy")
        self.assertAlmostEqual(result["context"]["std"], 0.105, places=3)
        self.assertAlmostEqual(result["context"]["mean"], 0.85, places=3)

    def test_low_std_aggregate_with_single_seed_latest_returns_unknown(self):
        # Aggregate exists but std is below the 10% relative threshold,
        # so the *hidden* variance claim is not justified. We fall
        # through to the *unknown* nudge (info severity) because the
        # latest run was single-seed and there's eval to compare to.
        # This is the discovery-mode nudge: project has run eval but
        # variance wasn't a problem before — still worth measuring.
        pid = _create_project()
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id="quiet-group",
            eval_metrics={
                "accuracy": {
                    "mean": 0.92, "std": 0.01, "min": 0.91, "max": 0.93, "n": 3,
                }
            },
            is_aggregate=True,
        ))
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id=None,
            eval_metrics={"accuracy": 0.91},
            is_aggregate=False,
        ))
        result = asyncio.run(_call_nudge(pid))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["id"], "training:variance-unknown")
        self.assertEqual(result["severity"], "info")

    def test_single_seed_with_eval_but_no_aggregate_returns_unknown(self):
        # First-ever training is single-seed and produced eval results
        # — soft nudge ("variance is unmeasured") prompts the user
        # to try num_seeds=3 next time.
        pid = _create_project()
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id=None,
            eval_metrics={"accuracy": 0.81},
            is_aggregate=False,
        ))
        result = asyncio.run(_call_nudge(pid))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["id"], "training:variance-unknown")
        self.assertEqual(result["severity"], "info")
        self.assertTrue(result["action"]["params"]["expand_multi_seed"])

    def test_single_seed_without_eval_returns_none(self):
        # The training stage might fire before eval has run at all;
        # there's nothing to compare against → no nudge (the training
        # surface itself already shows the multi-seed section in the
        # power-tools tab; coach doesn't need to repeat that).
        pid = _create_project()
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id=None,
            eval_metrics=None,
            is_aggregate=False,
        ))
        self.assertIsNone(asyncio.run(_call_nudge(pid)))

    def test_per_class_metric_variance_walks_nested_dict(self):
        # The Gap-#6 work introduced nested per-class metrics on
        # aggregate rows — verify the walker descends into them and
        # surfaces the worst-offending leaf path.
        pid = _create_project()
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id="older-group",
            eval_metrics={
                "accuracy": {"mean": 0.92, "std": 0.005, "n": 3},
                "classes": {
                    "negative": {
                        "f1": {
                            "mean": 0.62, "std": 0.18, "n": 3,
                        },
                    },
                },
            },
            is_aggregate=True,
        ))
        asyncio.run(_seed_experiment(
            pid,
            seed_group_id=None,
            eval_metrics={"accuracy": 0.92},
            is_aggregate=False,
        ))
        result = asyncio.run(_call_nudge(pid))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["id"], "training:variance-hidden")
        # Leaf path includes the per-class nesting so the user can
        # find it in the scorecard ("classes.negative.f1").
        self.assertIn("classes.negative.f1", result["context"]["worst_metric"])


if __name__ == "__main__":
    unittest.main()
