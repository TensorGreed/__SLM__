"""Tests for the remediation tracking service + API (E2).

Covers:
  - ``compute_params_hash`` stable hashing of (kind, params) so
    re-clicks collapse and order doesn't matter.
  - ``record_action_event`` persistence via service + via
    POST /api/projects/{id}/remediation/events.
  - ``stamp_evaluation_lift`` resolves pending events with the
    percentage-point lift between consecutive eval runs.
  - ``aggregate_outcomes_by_kind`` produces per-kind buckets with
    median/mean lift + positive-lift rate.
  - ``GET /api/admin/remediation/outcomes`` exposes the aggregation.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "remediation_tracking_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "remediation_tracking_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import (
    EvalResult,
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.models.project import Project, ProjectStatus
from app.models.remediation_action_event import (
    RemediationActionEvent,
    RemediationOutcome,
)
from app.services.remediation_tracking_service import (
    aggregate_outcomes_by_kind,
    compute_params_hash,
    record_action_event,
    stamp_evaluation_lift,
)


def _run(coro):
    return asyncio.run(coro)


class ParamsHashTests(unittest.TestCase):
    """Pure-function: compute_params_hash stability."""

    def test_same_kind_and_params_yields_same_hash(self):
        h1 = compute_params_hash("synth_augment", {"target_rows": 50})
        h2 = compute_params_hash("synth_augment", {"target_rows": 50})
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_param_key_order_doesnt_change_hash(self):
        # JSON serialisation sorts keys, so {a, b} and {b, a} collide.
        h1 = compute_params_hash("synth_balance", {"underrepresented_classes": ["a"], "target_rows_per_class": 10})
        h2 = compute_params_hash("synth_balance", {"target_rows_per_class": 10, "underrepresented_classes": ["a"]})
        self.assertEqual(h1, h2)

    def test_different_kinds_yield_different_hashes(self):
        self.assertNotEqual(
            compute_params_hash("synth_augment", {"target_rows": 50}),
            compute_params_hash("synth_balance", {"target_rows": 50}),
        )

    def test_different_params_yield_different_hashes(self):
        self.assertNotEqual(
            compute_params_hash("synth_augment", {"target_rows": 50}),
            compute_params_hash("synth_augment", {"target_rows": 60}),
        )

    def test_None_and_empty_dict_hash_differently(self):
        # The caller may legitimately distinguish "no params at all"
        # from "explicit empty params" — keep them distinguishable.
        self.assertNotEqual(
            compute_params_hash("cluster_fix", None),
            compute_params_hash("cluster_fix", {}),
        )


class RemediationTrackingServiceTests(unittest.TestCase):
    """Direct service-level tests against a real DB."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    _counter = 0

    def _new_project(self, recipe_id: str = "classification") -> int:
        """Insert a Project directly; return its id. Projects need
        unique names (UNIQUE constraint)."""
        RemediationTrackingServiceTests._counter += 1
        tag = RemediationTrackingServiceTests._counter

        async def _seed():
            async with async_session_factory() as session:
                proj = Project(
                    name=f"Remediation Test Project #{tag}",
                    status=ProjectStatus.DRAFT,
                    selected_recipe={"recipe_id": recipe_id},
                )
                session.add(proj)
                await session.commit()
                return proj.id
        return _run(_seed())

    def _new_experiment(self, project_id: int, started_at: datetime | None = None) -> int:
        """Insert an Experiment with optional started_at override —
        the lift-stamping code uses started_at to define the cutoff
        for "previous" eval lookup."""
        async def _seed():
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=project_id,
                    name=f"Exp #{datetime.now(timezone.utc).timestamp()}",
                    base_model="x",
                    training_mode=TrainingMode.SFT,
                    status=ExperimentStatus.COMPLETED,
                    started_at=started_at,
                )
                session.add(exp)
                await session.commit()
                return exp.id
        return _run(_seed())

    def _new_eval_result(
        self,
        experiment_id: int,
        pass_rate: float,
        created_at: datetime,
    ) -> int:
        async def _seed():
            async with async_session_factory() as session:
                evr = EvalResult(
                    experiment_id=experiment_id,
                    dataset_name="gold_test",
                    eval_type="f1",
                    metrics={"f1": pass_rate},
                    pass_rate=pass_rate,
                )
                session.add(evr)
                await session.flush()
                # SQLAlchemy default sets created_at on insert; force
                # the test value so the lift-stamping window covers
                # what we expect.
                evr.created_at = created_at
                await session.commit()
                return evr.id
        return _run(_seed())

    def test_record_persists_event_with_hash_and_default_outcome(self):
        pid = self._new_project()

        async def _record():
            async with async_session_factory() as session:
                ev = await record_action_event(
                    session,
                    project_id=pid,
                    kind="synth_augment",
                    params={"target_rows": 50},
                )
                await session.commit()
                return ev
        event = _run(_record())
        self.assertEqual(event.project_id, pid)
        self.assertEqual(event.action_kind, "synth_augment")
        self.assertEqual(event.outcome, RemediationOutcome.CLICKED)
        self.assertEqual(event.params_hash, compute_params_hash("synth_augment", {"target_rows": 50}))
        self.assertIsNone(event.evaluation_lift_pct)
        self.assertIsNone(event.resolved_at)

    def test_stamp_lift_resolves_pending_events_with_pct_delta(self):
        # Setup: project with two consecutive evals — pass_rate 0.40
        # then 0.55. The lift should be +15.0 pp. A click between
        # the two evals must be stamped; a click BEFORE the first
        # eval should be left alone.
        pid = self._new_project()
        eid1 = self._new_experiment(pid, started_at=datetime(2026, 5, 26, 9, 0, tzinfo=timezone.utc))
        eid2 = self._new_experiment(pid, started_at=datetime(2026, 5, 26, 11, 0, tzinfo=timezone.utc))
        # First eval lands at 10am.
        self._new_eval_result(eid1, 0.40, datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc))

        async def _record_click(observed_at: datetime):
            async with async_session_factory() as session:
                ev = await record_action_event(
                    session, project_id=pid, kind="fix_gold_rows",
                    params={"invalid_row_ids": [1, 2]},
                )
                ev.observed_at = observed_at
                await session.commit()
                return ev.id
        # Click at 10:30am — between the two evals.
        click_in_window = _run(_record_click(datetime(2026, 5, 26, 10, 30, tzinfo=timezone.utc)))
        # Click at 8am — BEFORE the first eval; should not stamp.
        click_pre = _run(_record_click(datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc)))

        # Second eval lands at 11:30am — pass_rate 0.55.
        self._new_eval_result(eid2, 0.55, datetime(2026, 5, 26, 11, 30, tzinfo=timezone.utc))

        async def _stamp():
            async with async_session_factory() as session:
                count = await stamp_evaluation_lift(
                    session,
                    project_id=pid,
                    experiment_id=eid2,
                    current_pass_rate=0.55,
                )
                await session.commit()
                return count
        stamped = _run(_stamp())
        self.assertEqual(stamped, 1, "only the click in the window should be stamped")

        async def _readback():
            from sqlalchemy import select
            async with async_session_factory() as session:
                rows = await session.execute(
                    select(RemediationActionEvent).where(
                        RemediationActionEvent.project_id == pid,
                    )
                )
                return list(rows.scalars())
        events = _run(_readback())
        by_id = {ev.id: ev for ev in events}
        # The in-window click is resolved with +15.0 lift.
        self.assertAlmostEqual(by_id[click_in_window].evaluation_lift_pct, 15.0, places=2)
        self.assertEqual(by_id[click_in_window].experiment_id, eid2)
        self.assertIsNotNone(by_id[click_in_window].resolved_at)
        # The pre-window click is left alone.
        self.assertIsNone(by_id[click_pre].evaluation_lift_pct)

    def test_stamp_lift_no_ops_when_no_previous_eval_exists(self):
        # First-ever eval — nothing to compare against. The click
        # stays pending; the next eval will resolve it.
        pid = self._new_project()
        eid = self._new_experiment(pid)

        async def _record_click():
            async with async_session_factory() as session:
                ev = await record_action_event(
                    session, project_id=pid, kind="synth_augment",
                    params={"target_rows": 30},
                )
                await session.commit()
                return ev.id
        click_id = _run(_record_click())

        async def _stamp():
            async with async_session_factory() as session:
                return await stamp_evaluation_lift(
                    session,
                    project_id=pid,
                    experiment_id=eid,
                    current_pass_rate=0.5,
                )
        self.assertEqual(_run(_stamp()), 0)

        async def _readback():
            async with async_session_factory() as session:
                from sqlalchemy import select
                return (await session.execute(
                    select(RemediationActionEvent).where(
                        RemediationActionEvent.id == click_id
                    )
                )).scalar_one()
        ev = _run(_readback())
        self.assertIsNone(ev.evaluation_lift_pct)

    def test_stamp_lift_no_ops_when_current_pass_rate_is_None(self):
        # Defensive — the eval may not have a pass_rate (e.g. a
        # safety-only eval). Stamping must not crash + must not
        # invent a lift.
        pid = self._new_project()
        eid = self._new_experiment(pid)

        async def _record_then_stamp():
            async with async_session_factory() as session:
                await record_action_event(
                    session, project_id=pid, kind="cluster_fix", params={"cluster_id": "c"},
                )
                count = await stamp_evaluation_lift(
                    session, project_id=pid, experiment_id=eid, current_pass_rate=None,
                )
                await session.commit()
                return count
        self.assertEqual(_run(_record_then_stamp()), 0)

    def test_aggregate_outcomes_groups_by_kind_with_lift_stats(self):
        # Seed a mixed set:
        #   synth_augment: 2 events, both stamped (+10, +5)
        #   fix_gold_rows: 1 event, stamped (-3)
        #   cluster_fix: 1 event, unstamped
        pid = self._new_project()

        async def _seed():
            async with async_session_factory() as session:
                from sqlalchemy import delete
                await session.execute(delete(RemediationActionEvent))
                rows = [
                    ("synth_augment", "h1", 10.0),
                    ("synth_augment", "h2", 5.0),
                    ("fix_gold_rows", "h3", -3.0),
                    ("cluster_fix", "h4", None),
                ]
                for kind, params_hash, lift in rows:
                    session.add(
                        RemediationActionEvent(
                            project_id=pid,
                            action_kind=kind,
                            params_hash=params_hash,
                            outcome=RemediationOutcome.CLICKED,
                            evaluation_lift_pct=lift,
                            resolved_at=datetime.now(timezone.utc) if lift is not None else None,
                        )
                    )
                await session.commit()

            async with async_session_factory() as session:
                return (
                    await aggregate_outcomes_by_kind(session, kind="synth_augment"),
                    await aggregate_outcomes_by_kind(session, kind="cluster_fix"),
                    await aggregate_outcomes_by_kind(session),
                )
        synth, cluster, total = _run(_seed())

        # synth_augment: 2 resolved, median=7.5, mean=7.5, both >0 → positive_lift_rate=1.0.
        self.assertEqual(synth["kind"], "synth_augment")
        self.assertEqual(synth["total_events"], 2)
        self.assertEqual(synth["resolved_count"], 2)
        self.assertEqual(synth["median_lift_pct"], 7.5)
        self.assertEqual(synth["mean_lift_pct"], 7.5)
        self.assertEqual(synth["positive_lift_count"], 2)
        self.assertEqual(synth["positive_lift_rate"], 1.0)

        # cluster_fix: 1 unresolved → lift stats are None.
        self.assertEqual(cluster["total_events"], 1)
        self.assertEqual(cluster["resolved_count"], 0)
        self.assertIsNone(cluster["median_lift_pct"])
        self.assertIsNone(cluster["positive_lift_rate"])

        # Unfiltered roll-up sees all 4 events + per-kind breakdown.
        self.assertEqual(total["total_events"], 4)
        kinds_seen = {b["kind"] for b in total["by_kind"]}
        self.assertEqual(kinds_seen, {"synth_augment", "fix_gold_rows", "cluster_fix"})


class RemediationApiTests(unittest.TestCase):
    """Endpoint contract — POST /projects/{id}/remediation/events
    and GET /admin/remediation/outcomes."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    def test_post_event_returns_201_with_hashed_payload(self):
        # Use a template project — the API guards on Project existence.
        project = self.client.post(
            "/api/project-templates/ticket-router/instantiate",
            json={"project_name": "Remediation API e2e"},
        ).json()
        pid = project["id"]
        resp = self.client.post(
            f"/api/projects/{pid}/remediation/events",
            json={
                "kind": "synth_augment",
                "params": {"target_rows": 50},
                "outcome": "clicked",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], pid)
        self.assertEqual(body["action_kind"], "synth_augment")
        self.assertEqual(body["outcome"], "clicked")
        # params_hash is computed server-side from kind+params.
        self.assertEqual(
            body["params_hash"],
            compute_params_hash("synth_augment", {"target_rows": 50}),
        )

    def test_post_event_404s_on_unknown_project(self):
        resp = self.client.post(
            "/api/projects/99999/remediation/events",
            json={"kind": "synth_augment", "params": {}, "outcome": "clicked"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_admin_outcomes_returns_per_kind_aggregation(self):
        # Wipe + seed a couple of events; admin endpoint returns them.
        async def _seed():
            async with async_session_factory() as session:
                from sqlalchemy import delete
                await session.execute(delete(RemediationActionEvent))
                session.add(
                    RemediationActionEvent(
                        project_id=1,
                        action_kind="synth_augment",
                        params_hash="abc",
                        outcome=RemediationOutcome.CLICKED,
                        evaluation_lift_pct=12.5,
                        resolved_at=datetime.now(timezone.utc),
                    )
                )
                session.add(
                    RemediationActionEvent(
                        project_id=1,
                        action_kind="fix_gold_rows",
                        params_hash="def",
                        outcome=RemediationOutcome.CLICKED,
                    )
                )
                await session.commit()
        asyncio.run(_seed())

        resp = self.client.get("/api/admin/remediation/outcomes?kind=synth_augment")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["kind"], "synth_augment")
        self.assertEqual(body["total_events"], 1)
        self.assertEqual(body["resolved_count"], 1)
        self.assertEqual(body["median_lift_pct"], 12.5)
        self.assertEqual(body["positive_lift_rate"], 1.0)

        # Unfiltered roll-up exposes both kinds with a by_kind list.
        roll_resp = self.client.get("/api/admin/remediation/outcomes")
        roll = roll_resp.json()
        self.assertEqual(roll["total_events"], 2)
        kinds = {b["kind"] for b in roll["by_kind"]}
        self.assertEqual(kinds, {"synth_augment", "fix_gold_rows"})


if __name__ == "__main__":
    unittest.main()
