"""Experiment scorecard verdict follows the gates, not the (advisory) project policy.

Before: the scorecard started from SHIP and flipped only when the project's gate_policy
enforced it; the project default is advisory, so a simulated run with every gate "Not
measured" read SHIP next to a FAIL pass/fail card. Now SHIP needs measured, passing gates.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.api.evaluation import scorecard_decision  # noqa: E402
from app.main import app  # noqa: E402

_MODULE_CLIENT_CM = TestClient(app)
client: TestClient


def setUpModule() -> None:  # noqa: N802
    global client
    client = _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__exit__(None, None, None)


PACK = "evalpack.fast.iteration"


class _ScorecardHelpers:
    def _project_and_run(self, name: str, pack: str | None = PACK) -> tuple[int, int]:
        resp = client.post("/api/projects", json={"name": name, "description": "scorecard"})
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])
        if pack:
            pref = client.put(f"/api/projects/{pid}/evaluation/pack-preference", json={"pack_id": pack})
            self.assertEqual(pref.status_code, 200, pref.text)
        exp = client.post(
            f"/api/projects/{pid}/training/experiments",
            json={"name": "simulated-run", "config": {"base_model": "microsoft/phi-2"}},
        )
        self.assertEqual(exp.status_code, 201, exp.text)
        return pid, int(exp.json()["id"])

    def _eval(self, pid: int, eid: int, eval_type: str, pairs: list[tuple[str, str]]) -> None:
        resp = client.post(
            f"/api/projects/{pid}/evaluation/run",
            json={
                "experiment_id": eid,
                "dataset_name": "gold_test",
                "eval_type": eval_type,
                "predictions": [{"prediction": p, "reference": r} for p, r in pairs],
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)

    def _scorecard(self, pid: int, eid: int) -> dict:
        resp = client.get(f"/api/projects/{pid}/evaluation/scorecard/{eid}")
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()


class ScorecardEndpointTests(_ScorecardHelpers, unittest.TestCase):
    def test_real_run_with_nothing_measured_is_not_evaluated_without_blaming_simulation(self):
        pid, eid = self._project_and_run("scorecard-unmeasured")
        card = self._scorecard(pid, eid)
        self.assertEqual(card["decision"], "NOT EVALUATED")
        self.assertIn("Evaluate it on the test split first.", card["reasons"][0])
        self.assertNotIn("simulated", " ".join(card["reasons"]).lower())
        self.assertFalse(card["is_ship"])
        self.assertFalse(card["gate_report"]["passed"])
        # The project default policy is advisory; the verdict still is not SHIP, and says so.
        self.assertFalse(card["enforced"])
        self.assertIn("does not block export", card["policy_note"])

    def test_simulated_run_says_it_cannot_be_evaluated(self):
        pid, eid = self._project_and_run("scorecard-simulated")
        asyncio.run(_set_config(eid, {"_runtime": {"backend": "simulate", "runtime_id": "builtin.simulate"}}))
        card = self._scorecard(pid, eid)
        self.assertEqual(card["decision"], "NOT EVALUATED")
        self.assertIn("simulated run", card["reasons"][0])

    def test_failing_gate_is_no_ship_under_the_advisory_default(self):
        pid, eid = self._project_and_run("scorecard-failing")
        self._eval(pid, eid, "exact_match", [("wrong", "Answer A"), ("also wrong", "Answer B")])
        self._eval(pid, eid, "f1", [("nothing alike", "Correct response")])
        card = self._scorecard(pid, eid)
        self.assertEqual(card["decision"], "NO-SHIP")
        self.assertFalse(card["is_ship"])
        self.assertTrue(card["failed_gates"])
        self.assertTrue(any("Failed" in r for r in card["reasons"]))

    def test_a_required_metric_not_measured_is_no_ship(self):
        # The default pack requires more than exact_match (e.g. judge and safety pass rates).
        pid, eid = self._project_and_run("scorecard-partial", pack=None)
        self._eval(pid, eid, "exact_match", [("Answer A", "Answer A"), ("Answer B", "Answer B")])
        card = self._scorecard(pid, eid)
        self.assertTrue(card["gate_report"]["missing_required_metrics"], card)
        self.assertTrue(any(c.get("actual") is not None for c in card["gate_report"]["checks"]))
        self.assertEqual(card["decision"], "NO-SHIP")
        self.assertFalse(card["is_ship"])
        self.assertTrue(any("Missing" in r for r in card["reasons"]))

    def test_all_gates_pass_is_ship(self):
        pid, eid = self._project_and_run("scorecard-passing")
        self._eval(pid, eid, "exact_match", [("Answer A", "Answer A"), ("Answer B", "Answer B")])
        self._eval(pid, eid, "f1", [("Correct response", "Correct response")])
        card = self._scorecard(pid, eid)
        self.assertTrue(card["gate_report"]["passed"], card)
        self.assertEqual(card["decision"], "SHIP")
        self.assertTrue(card["is_ship"])
        self.assertEqual(card["reasons"], [])


async def _set_config(experiment_id: int, config: dict) -> None:
    from app.database import async_session_factory
    from app.models.experiment import Experiment

    async with async_session_factory() as db:
        exp = await db.get(Experiment, experiment_id)
        exp.config = {**(exp.config or {}), **config}
        await db.commit()


async def _seed_group(project_id: int, n_children: int) -> tuple[int, list[int]]:
    """A multi-seed leader (seed_value None) + COMPLETED children, as fan-out writes them."""
    from app.database import async_session_factory
    from app.models.experiment import Experiment, ExperimentStatus

    group = str(uuid.uuid4())
    async with async_session_factory() as db:
        leader = Experiment(
            project_id=project_id, name="multi-seed", base_model="microsoft/phi-2",
            status=ExperimentStatus.COMPLETED, seed_group_id=group, config={},
        )
        db.add(leader)
        children = [
            Experiment(
                project_id=project_id, name=f"multi-seed seed {i}", base_model="microsoft/phi-2",
                status=ExperimentStatus.COMPLETED, seed_group_id=group, seed_value=i, config={},
            )
            for i in range(1, n_children + 1)
        ]
        db.add_all(children)
        await db.commit()
        return int(leader.id), [int(c.id) for c in children]


class SeedGroupLeaderScorecardTests(_ScorecardHelpers, unittest.TestCase):
    """A multi-seed leader trains nothing itself; its seed runs hold the evals."""

    def _leader(self, name: str) -> tuple[int, int, list[int]]:
        resp = client.post("/api/projects", json={"name": name, "description": "scorecard"})
        pid = int(resp.json()["id"])
        client.put(f"/api/projects/{pid}/evaluation/pack-preference", json={"pack_id": PACK})
        leader, children = asyncio.run(_seed_group(pid, 3))
        return pid, leader, children

    def test_leader_is_scored_on_an_evaluated_seed_run(self):
        pid, leader, children = self._leader("scorecard-seed-group")
        for child in children:
            self._eval(pid, child, "exact_match", [("Answer A", "Answer A"), ("Answer B", "Answer B")])
            self._eval(pid, child, "f1", [("Correct response", "Correct response")])
        card = self._scorecard(pid, leader)
        report = card["gate_report"]
        self.assertIn(report["evaluated_on"]["experiment_id"], children)
        self.assertEqual(report["missing_required_metrics"], [])
        self.assertTrue(report["passed"], report)
        self.assertEqual(card["decision"], "SHIP")
        gates = client.get(f"/api/projects/{pid}/evaluation/gates/{leader}").json()
        self.assertEqual(gates["evaluated_on"], report["evaluated_on"])
        self.assertTrue(gates["passed"])
        self.assertEqual(gates["failed_gate_ids"], [])

    def test_leader_with_no_evaluated_seed_points_at_its_seed_runs(self):
        pid, leader, _children = self._leader("scorecard-seed-group-unevaluated")
        card = self._scorecard(pid, leader)
        self.assertEqual(card["decision"], "NOT EVALUATED")
        self.assertIsNone(card["gate_report"]["evaluated_on"])
        self.assertIn("seed runs", card["reasons"][0])
        self.assertNotIn("simulated", card["reasons"][0].lower())

    def test_single_run_gates_have_no_seed_source(self):
        pid, eid = self._project_and_run("scorecard-single-source")
        self.assertIsNone(self._scorecard(pid, eid)["gate_report"]["evaluated_on"])


class ScorecardDecisionTests(unittest.TestCase):
    """The pure rule, including cases the endpoint can't easily reach."""

    def _check(self, actual, passed, required=True):
        return {"gate_id": "g", "metric_id": "m", "required": required, "actual": actual, "passed": passed}

    def test_required_not_measured_beside_a_measured_pass_is_no_ship(self):
        report = {
            "passed": False,
            "checks": [self._check(0.9, True), self._check(None, False)],
            "failed_gate_ids": ["g2"],
            "missing_required_metrics": ["m2"],
        }
        out = scorecard_decision(report, {"must_pass": False, "blocked_if_missing": False})
        self.assertEqual(out["decision"], "NO-SHIP")

    def test_not_evaluated_wording_follows_the_run_kind(self):
        empty = {"passed": False, "checks": [self._check(None, False)]}
        plain = scorecard_decision(empty, {})["reasons"][0]
        self.assertNotIn("simulated", plain.lower())
        self.assertIn("simulated run", scorecard_decision(empty, {}, simulated=True)["reasons"][0])
        self.assertIn("seed runs", scorecard_decision(empty, {}, seed_group_leader=True)["reasons"][0])

    def test_no_checks_at_all_is_not_evaluated(self):
        out = scorecard_decision({"passed": True, "checks": []}, {})
        self.assertEqual(out["decision"], "NOT EVALUATED")
        self.assertFalse(out["is_ship"])

    def test_enforced_policy_has_no_advisory_note(self):
        report = {"passed": True, "checks": [self._check(0.9, True)]}
        out = scorecard_decision(report, {"must_pass": True, "blocked_if_missing": True})
        self.assertEqual(out["decision"], "SHIP")
        self.assertTrue(out["enforced"])
        self.assertIsNone(out["policy_note"])

    def test_report_not_passed_without_listed_ids_is_still_no_ship(self):
        out = scorecard_decision({"passed": False, "checks": [self._check(0.1, False)]}, {})
        self.assertEqual(out["decision"], "NO-SHIP")
