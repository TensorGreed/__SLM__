"""Experiment scorecard verdict follows the gates, not the (advisory) project policy.

Before: the scorecard started from SHIP and flipped only when the project's gate_policy
enforced it; the project default is advisory, so a simulated run with every gate "Not
measured" read SHIP next to a FAIL pass/fail card. Now SHIP needs measured, passing gates.
"""

from __future__ import annotations

import os
import unittest

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


class ScorecardEndpointTests(unittest.TestCase):
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

    def test_simulated_run_with_nothing_measured_is_not_evaluated(self):
        pid, eid = self._project_and_run("scorecard-unmeasured")
        card = self._scorecard(pid, eid)
        self.assertEqual(card["decision"], "NOT EVALUATED")
        self.assertFalse(card["is_ship"])
        self.assertFalse(card["gate_report"]["passed"])
        # The project default policy is advisory; the verdict still is not SHIP, and says so.
        self.assertFalse(card["enforced"])
        self.assertIn("does not block export", card["policy_note"])

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
