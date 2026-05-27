"""Tests for forecast vs reality calibration (USER-SUCCESS Epic 1, T5).

Covers:
  - ``record_forecast_observation`` pairs an experiment with the
    latest snapshot at launch time; no-ops when there's no snapshot.
  - ``resolve_forecast_observation`` flips actual_passed + resolved_at
    when eval gates evaluate.
  - ``compute_calibration_buckets`` produces decile-bucketed
    pass-rate aggregation with recipe filtering.
  - ``GET /api/admin/forecast/calibration`` exposes the bucketed data.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "forecast_calibration_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "forecast_calibration_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode
from app.models.forecast_calibration_observation import (
    ForecastCalibrationObservation,
)
from app.models.project import Project, ProjectStatus
from app.models.training_forecast_snapshot import TrainingForecastSnapshot
from app.services.trainability_forecast_service import (
    compute_calibration_buckets,
    record_forecast_observation,
    resolve_forecast_observation,
)


class ForecastCalibrationServiceTests(unittest.TestCase):
    """Direct service-level tests — exercise the helpers without
    going through training/eval endpoints. The endpoint-level wiring
    is covered by the integration suite below."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    _seed_counter = 0

    def _seed_project_with_snapshot(
        self,
        *,
        recipe_id: str = "classification",
        confidence_pct: int = 72,
        overall: str = "likely_pass",
    ) -> tuple[int, int]:
        """Insert a Project + Experiment + TrainingForecastSnapshot
        directly via the DB, bypassing endpoints. Returns
        (project_id, experiment_id). Project names are uniqued
        per-call via a class counter — the projects table has a
        UNIQUE(name) and multiple tests seed at confidence_pct=72."""
        ForecastCalibrationServiceTests._seed_counter += 1
        tag = ForecastCalibrationServiceTests._seed_counter

        async def _seed():
            async with async_session_factory() as session:
                proj = Project(
                    name=f"Calib Project {confidence_pct} #{tag}",
                    status=ProjectStatus.DRAFT,
                    selected_recipe={"recipe_id": recipe_id},
                )
                session.add(proj)
                await session.flush()
                snap = TrainingForecastSnapshot(
                    project_id=proj.id,
                    cache_key="seed-key-00000000",
                    computed_at=datetime.now(timezone.utc),
                    overall=overall,
                    confidence_pct=confidence_pct,
                    signals=[],
                )
                session.add(snap)
                exp = Experiment(
                    project_id=proj.id,
                    name=f"Calib Exp {confidence_pct}",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config={},
                    training_mode=TrainingMode.SFT,
                    status=ExperimentStatus.PENDING,
                )
                session.add(exp)
                await session.commit()
                return proj.id, exp.id
        return asyncio.run(_seed())

    def _run(self, coro):
        return asyncio.run(coro)

    def test_record_pairs_experiment_with_latest_snapshot(self):
        pid, eid = self._seed_project_with_snapshot(confidence_pct=72)

        async def _record():
            async with async_session_factory() as session:
                return await record_forecast_observation(session, eid)
        obs = self._run(_record())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.experiment_id, eid)
        self.assertEqual(obs.predicted_confidence_pct, 72)
        self.assertEqual(obs.predicted_overall, "likely_pass")
        self.assertEqual(obs.recipe_id, "classification")
        self.assertIsNone(obs.actual_passed)

    def test_record_picks_the_NEWEST_snapshot_not_the_oldest(self):
        # User computes the forecast twice; we should pair the
        # experiment with the second (most recent) snapshot — that's
        # the one they were actually looking at when they hit Train.
        pid, eid = self._seed_project_with_snapshot(confidence_pct=40)

        async def _seed_newer():
            async with async_session_factory() as session:
                session.add(
                    TrainingForecastSnapshot(
                        project_id=pid,
                        cache_key="seed-key-newer",
                        computed_at=datetime.now(timezone.utc) + timedelta(seconds=10),
                        overall="borderline",
                        confidence_pct=58,
                        signals=[],
                    )
                )
                await session.commit()
        self._run(_seed_newer())

        async def _record():
            async with async_session_factory() as session:
                return await record_forecast_observation(session, eid)
        obs = self._run(_record())
        self.assertIsNotNone(obs)
        # Newer snapshot wins.
        self.assertEqual(obs.predicted_confidence_pct, 58)
        self.assertEqual(obs.predicted_overall, "borderline")

    def test_record_is_idempotent_per_experiment(self):
        # Two record_forecast_observation calls for the same
        # experiment must NOT create a second observation. The
        # second call returns None.
        pid, eid = self._seed_project_with_snapshot(confidence_pct=72)

        async def _record_twice():
            async with async_session_factory() as session:
                first = await record_forecast_observation(session, eid)
                second = await record_forecast_observation(session, eid)
                return first, second
        first, second = self._run(_record_twice())
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_record_returns_None_when_no_snapshot_exists(self):
        # Training without ever viewing the forecast: there's no
        # prediction to pair against, so we silently skip.
        async def _seed_no_snap():
            async with async_session_factory() as session:
                proj = Project(
                    name="No Snapshot Project",
                    status=ProjectStatus.DRAFT,
                    selected_recipe={"recipe_id": "classification"},
                )
                session.add(proj)
                await session.flush()
                exp = Experiment(
                    project_id=proj.id,
                    name="No Snapshot Exp",
                    base_model="x",
                    training_mode=TrainingMode.SFT,
                    status=ExperimentStatus.PENDING,
                )
                session.add(exp)
                await session.commit()
                return exp.id
        eid = self._run(_seed_no_snap())

        async def _record():
            async with async_session_factory() as session:
                return await record_forecast_observation(session, eid)
        self.assertIsNone(self._run(_record()))

    def test_record_returns_None_when_project_has_no_recipe(self):
        # Calibration aggregates by recipe; a recipe-less run can't
        # bucket cleanly. Excluded silently.
        async def _seed_no_recipe():
            async with async_session_factory() as session:
                proj = Project(name="No Recipe Project", status=ProjectStatus.DRAFT)
                session.add(proj)
                await session.flush()
                session.add(
                    TrainingForecastSnapshot(
                        project_id=proj.id,
                        cache_key="x",
                        computed_at=datetime.now(timezone.utc),
                        overall="borderline",
                        confidence_pct=50,
                        signals=[],
                    )
                )
                exp = Experiment(
                    project_id=proj.id,
                    name="X",
                    base_model="x",
                    training_mode=TrainingMode.SFT,
                    status=ExperimentStatus.PENDING,
                )
                session.add(exp)
                await session.commit()
                return exp.id
        eid = self._run(_seed_no_recipe())

        async def _record():
            async with async_session_factory() as session:
                return await record_forecast_observation(session, eid)
        self.assertIsNone(self._run(_record()))

    def test_resolve_sets_actual_passed_and_resolved_at(self):
        pid, eid = self._seed_project_with_snapshot(confidence_pct=72)

        async def _record_then_resolve():
            async with async_session_factory() as session:
                await record_forecast_observation(session, eid)
                obs = await resolve_forecast_observation(session, eid, passed=True)
                await session.commit()
                return obs
        obs = self._run(_record_then_resolve())
        self.assertIsNotNone(obs)
        self.assertEqual(obs.actual_passed, True)
        self.assertIsNotNone(obs.resolved_at)

    def test_resolve_returns_None_when_no_observation_exists(self):
        # If the experiment was created before T5 landed (or training
        # bypassed the recording path), eval-gate evaluation must not
        # fail loudly.
        async def _resolve_missing():
            async with async_session_factory() as session:
                return await resolve_forecast_observation(session, 99999, passed=True)
        self.assertIsNone(self._run(_resolve_missing()))

    def test_resolve_overwrites_on_re_eval(self):
        # Eval can re-run against the same experiment (e.g. after a
        # gate config tweak). The freshest signal wins so the
        # calibration aggregate reflects the user's actual final
        # verdict.
        pid, eid = self._seed_project_with_snapshot(confidence_pct=72)

        async def _record_resolve_twice():
            async with async_session_factory() as session:
                await record_forecast_observation(session, eid)
                first = await resolve_forecast_observation(session, eid, passed=False)
                first_resolved_at = first.resolved_at
                await session.commit()
                second = await resolve_forecast_observation(session, eid, passed=True)
                await session.commit()
                return first_resolved_at, second
        first_ts, second = self._run(_record_resolve_twice())
        self.assertEqual(second.actual_passed, True)
        self.assertGreaterEqual(second.resolved_at, first_ts)

    def test_compute_buckets_returns_decile_payload(self):
        # Fresh DB → 10 decile buckets with zeroed counts.
        async def _compute_empty():
            async with async_session_factory() as session:
                # Wipe any leftover observations from sibling tests.
                from sqlalchemy import delete
                await session.execute(delete(ForecastCalibrationObservation))
                await session.commit()
                return await compute_calibration_buckets(session)
        payload = self._run(_compute_empty())
        self.assertEqual(len(payload["buckets"]), 10)
        self.assertEqual(payload["total_observations"], 0)
        self.assertEqual(payload["resolved_observations"], 0)
        for bucket in payload["buckets"]:
            self.assertEqual(bucket["predicted_count"], 0)
            self.assertIsNone(bucket["actual_pass_rate"])

    def test_compute_buckets_groups_predictions_by_decile_and_filters_by_recipe(self):
        # Seed a hand-built distribution:
        #   classification recipe:
        #     - confidence 72 → passes
        #     - confidence 76 → passes
        #     - confidence 35 → fails
        #   span-extraction recipe:
        #     - confidence 78 → fails  (per-recipe miscalibration anchor)
        #     - confidence 82 → fails
        # Unresolved row in each recipe to confirm null filtering.
        async def _seed():
            async with async_session_factory() as session:
                from sqlalchemy import delete
                await session.execute(delete(ForecastCalibrationObservation))
                rows = [
                    (1, "classification", "likely_pass", 72, True),
                    (2, "classification", "likely_pass", 76, True),
                    (3, "classification", "likely_fail", 35, False),
                    (4, "classification", "likely_pass", 80, None),  # unresolved
                    (5, "span-extraction", "likely_pass", 78, False),
                    (6, "span-extraction", "likely_pass", 82, False),
                ]
                for exp_id, recipe, overall, conf, passed in rows:
                    session.add(
                        ForecastCalibrationObservation(
                            experiment_id=exp_id,
                            project_id=exp_id,  # 1-to-1 stand-in for the test
                            snapshot_id=1,
                            predicted_confidence_pct=conf,
                            predicted_overall=overall,
                            recipe_id=recipe,
                            actual_passed=passed,
                            resolved_at=None if passed is None else datetime.now(timezone.utc),
                        )
                    )
                await session.commit()

            async with async_session_factory() as session:
                cls = await compute_calibration_buckets(session, recipe_id="classification")
                span = await compute_calibration_buckets(session, recipe_id="span-extraction")
                everything = await compute_calibration_buckets(session)
                return cls, span, everything
        cls, span, everything = self._run(_seed())

        # classification bucket [70, 80): 2 predictions, both passed → rate=1.0.
        # classification bucket [30, 40): 1 prediction, didn't pass → rate=0.0.
        # classification bucket [80, 90): 1 unresolved → excluded from
        #   predicted_count (only resolved rows enter the math).
        cls_70 = next(b for b in cls["buckets"] if b["range"] == [70, 80])
        cls_30 = next(b for b in cls["buckets"] if b["range"] == [30, 40])
        cls_80 = next(b for b in cls["buckets"] if b["range"] == [80, 90])
        self.assertEqual(cls_70["predicted_count"], 2)
        self.assertEqual(cls_70["actual_pass_count"], 2)
        self.assertEqual(cls_70["actual_pass_rate"], 1.0)
        self.assertEqual(cls_30["predicted_count"], 1)
        self.assertEqual(cls_30["actual_pass_rate"], 0.0)
        self.assertEqual(cls_80["predicted_count"], 0)
        self.assertIsNone(cls_80["actual_pass_rate"])  # no-data preserves None
        # classification recipe sums: 4 total (3 resolved + 1 pending).
        self.assertEqual(cls["total_observations"], 4)
        self.assertEqual(cls["resolved_observations"], 3)

        # span-extraction recipe shows the miscalibration: two
        # predictions in the 70-90 range, both fail → 0% actual.
        span_70 = next(b for b in span["buckets"] if b["range"] == [70, 80])
        span_80 = next(b for b in span["buckets"] if b["range"] == [80, 90])
        self.assertEqual(span_70["predicted_count"], 1)
        self.assertEqual(span_70["actual_pass_count"], 0)
        self.assertEqual(span_80["actual_pass_rate"], 0.0)

        # Without a recipe filter, classification + span-extraction
        # combine — the 70-80 bucket now has 3 predictions (2 cls + 1 span).
        unfiltered_70 = next(b for b in everything["buckets"] if b["range"] == [70, 80])
        self.assertEqual(unfiltered_70["predicted_count"], 3)
        self.assertEqual(unfiltered_70["actual_pass_count"], 2)


class ForecastCalibrationApiTests(unittest.TestCase):
    """Integration tests for ``GET /api/admin/forecast/calibration``
    and end-to-end recording through create_experiment +
    evaluate_experiment_auto_gates."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    def test_admin_endpoint_returns_empty_buckets_on_a_fresh_db(self):
        async def _wipe():
            async with async_session_factory() as session:
                from sqlalchemy import delete
                await session.execute(delete(ForecastCalibrationObservation))
                await session.commit()
        asyncio.run(_wipe())

        resp = self.client.get("/api/admin/forecast/calibration")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["total_observations"], 0)
        self.assertEqual(len(body["buckets"]), 10)
        self.assertEqual(body["recipe_id"], None)

    def test_admin_endpoint_filters_by_recipe(self):
        # Seed two recipes with different distributions.
        async def _seed():
            async with async_session_factory() as session:
                from sqlalchemy import delete
                await session.execute(delete(ForecastCalibrationObservation))
                base = 1000  # avoid collisions with any prior test data
                # classification: 2 successful in 70-80 bucket.
                for off in (0, 1):
                    session.add(
                        ForecastCalibrationObservation(
                            experiment_id=base + off,
                            project_id=base + off,
                            snapshot_id=1,
                            predicted_confidence_pct=72,
                            predicted_overall="likely_pass",
                            recipe_id="classification",
                            actual_passed=True,
                            resolved_at=datetime.now(timezone.utc),
                        )
                    )
                # summarization: 1 failure in 70-80 bucket.
                session.add(
                    ForecastCalibrationObservation(
                        experiment_id=base + 2,
                        project_id=base + 2,
                        snapshot_id=1,
                        predicted_confidence_pct=78,
                        predicted_overall="likely_pass",
                        recipe_id="summarization",
                        actual_passed=False,
                        resolved_at=datetime.now(timezone.utc),
                    )
                )
                await session.commit()
        asyncio.run(_seed())

        # Default (no filter) sees both.
        all_resp = self.client.get("/api/admin/forecast/calibration")
        self.assertEqual(all_resp.json()["total_observations"], 3)

        # Recipe filter narrows the result.
        cls_resp = self.client.get(
            "/api/admin/forecast/calibration?recipe=classification"
        )
        self.assertEqual(cls_resp.status_code, 200, cls_resp.text)
        body = cls_resp.json()
        self.assertEqual(body["recipe_id"], "classification")
        self.assertEqual(body["total_observations"], 2)
        bucket_70 = next(b for b in body["buckets"] if b["range"] == [70, 80])
        self.assertEqual(bucket_70["actual_pass_rate"], 1.0)

        # summarization filter shows the calibration-failure case.
        sum_body = self.client.get(
            "/api/admin/forecast/calibration?recipe=summarization"
        ).json()
        self.assertEqual(sum_body["total_observations"], 1)
        sum_70 = next(b for b in sum_body["buckets"] if b["range"] == [70, 80])
        self.assertEqual(sum_70["actual_pass_rate"], 0.0)

    def test_create_experiment_records_observation_when_a_snapshot_exists(self):
        # Full integration: instantiate a template, compute the
        # forecast (lands one snapshot), call the training-create
        # endpoint, then read the calibration table.
        project = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": "Calib E2E Create"},
        ).json()
        pid = project["id"]

        # Snapshot via the forecast endpoint.
        self.client.get(f"/api/projects/{pid}/training/forecast")

        # Now create an experiment. The training router mounts at
        # /api/projects/{id}/training, so the create path is
        # /api/projects/{id}/training/experiments and the body shape
        # is the ExperimentCreate Pydantic model.
        exp_resp = self.client.post(
            f"/api/projects/{pid}/training/experiments",
            json={
                "name": "Calib E2E Run",
                "description": "T5 end-to-end calibration smoke test",
                "config": {
                    "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "training_mode": "sft",
                    "num_epochs": 1,
                    "batch_size": 4,
                },
            },
        )
        self.assertEqual(exp_resp.status_code, 201, exp_resp.text)
        exp_id = exp_resp.json()["id"]

        async def _read_obs():
            async with async_session_factory() as session:
                from sqlalchemy import select
                row = await session.execute(
                    select(ForecastCalibrationObservation).where(
                        ForecastCalibrationObservation.experiment_id == exp_id
                    )
                )
                return row.scalar_one_or_none()
        obs = asyncio.run(_read_obs())
        self.assertIsNotNone(obs, "create_experiment must record a calibration observation")
        self.assertEqual(obs.recipe_id, "classification")
        self.assertIsNone(obs.actual_passed)  # unresolved until eval runs


if __name__ == "__main__":
    unittest.main()
