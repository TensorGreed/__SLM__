"""Tests for the adopt-gate-from-cluster action (Gap #5 slice 3).

Covers:
  * Pure suggestion mapping — reason_code → recipe-aware metric pick.
  * End-to-end via TestClient: seeds a Project + FailureCluster row in
    the per-PID scratch DB, calls POST /evaluation/adopt-gate-from-cluster,
    and asserts the resulting scaffolded pack carries the new gate +
    that the gate cleared the slice-1 validator.

Drives via TestClient (NOT IsolatedAsyncioTestCase) because the
StaticPool-pinned aiosqlite engine doesn't survive event-loop swaps —
same constraint as test_jobs_service.py.
"""

from __future__ import annotations

import asyncio
import os
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.evaluation_gate_catalog import suggest_gate_for_reason_code  # noqa: E402


# Module-level TestClient runs lifespan once so init_db creates the
# projects + failure_clusters tables in the scratch DB.
_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__exit__(None, None, None)


# ─────────────────────────────────────────────────────────────────────
# Pure mapping — no DB
# ─────────────────────────────────────────────────────────────────────


class SuggestGateForReasonCodeTests(unittest.TestCase):

    def test_safety_failure_suggests_safety_pass_rate_gate(self):
        # Safety regressions have a canonical metric — make sure the
        # mapping returns it regardless of recipe.
        gate = suggest_gate_for_reason_code("safety_failure", recipe_id="qa-sft")
        self.assertIsNotNone(gate)
        assert gate is not None  # for type narrowing
        self.assertEqual(gate["metric_id"], "safety_pass_rate")
        self.assertEqual(gate["operator"], "gte")
        # Starter gates start as ``required=False`` so the new gate
        # doesn't flip a currently-passing eval the moment it lands.
        self.assertFalse(gate["required"])

    def test_hallucination_prefers_rag_metrics_when_recipe_uses_them(self):
        # rag-protocol recipe should land on hallucination_rate (lte).
        gate = suggest_gate_for_reason_code("hallucination", recipe_id="rag-protocol")
        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual(gate["metric_id"], "hallucination_rate")
        self.assertEqual(gate["operator"], "lte")

    def test_hallucination_falls_back_to_judge_for_non_rag_recipes(self):
        # qa-sft doesn't track hallucination_rate; the mapping should
        # walk to the next candidate that's in the recipe's recommended
        # set (llm_judge_pass_rate).
        gate = suggest_gate_for_reason_code("hallucination", recipe_id="qa-sft")
        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual(gate["metric_id"], "llm_judge_pass_rate")

    def test_coverage_gap_picks_recipe_headline_metric(self):
        # classification recipe → macro_f1 first (recommended); qa-sft
        # → f1 first (recommended). Same reason_code, different pick.
        cls_gate = suggest_gate_for_reason_code("coverage_gap", recipe_id="classification")
        qa_gate = suggest_gate_for_reason_code("coverage_gap", recipe_id="qa-sft")
        self.assertIsNotNone(cls_gate)
        self.assertIsNotNone(qa_gate)
        assert cls_gate is not None and qa_gate is not None
        self.assertEqual(cls_gate["metric_id"], "macro_f1")
        self.assertEqual(qa_gate["metric_id"], "f1")

    def test_unknown_reason_code_returns_none(self):
        # Defensive: the FE surfaces None as "no obvious gate — add
        # one manually" so an unknown reason_code doesn't crash.
        self.assertIsNone(suggest_gate_for_reason_code("never_seen_before"))


# ─────────────────────────────────────────────────────────────────────
# End-to-end: seed Project + FailureCluster, hit the endpoint
# ─────────────────────────────────────────────────────────────────────


_PROJECT_NAME_COUNTER = 0


def _next_project_name(reason_code: str) -> str:
    # Project.name has a UNIQUE constraint and tests share the scratch
    # DB — bump a module-level counter so every seed call gets a fresh
    # name. Avoids inter-test collisions without needing teardown.
    global _PROJECT_NAME_COUNTER
    _PROJECT_NAME_COUNTER += 1
    return f"adopt-gate-{reason_code}-{_PROJECT_NAME_COUNTER}"


class AdoptGateFromClusterEndpointTests(unittest.TestCase):

    def _make_project_and_cluster(self, *, recipe_id: str, reason_code: str) -> tuple[int, int]:
        """Seed a project with ``recipe_id`` selected + a FailureCluster
        with ``reason_code``. Returns ``(project_id, cluster_id)``."""
        from app.database import async_session_factory
        from app.models.failure_cluster import FailureCluster
        from app.models.project import Project

        async def _seed() -> tuple[int, int]:
            async with async_session_factory() as db:
                project = Project(
                    name=_next_project_name(reason_code),
                    description="seed for adopt-gate-from-cluster tests",
                    selected_recipe={"recipe_id": recipe_id},
                )
                db.add(project)
                await db.flush()
                cluster = FailureCluster(
                    project_id=project.id,
                    stage="eval",
                    reason_code=reason_code,
                    signature=f"sig-{reason_code}-{project.id}",
                    failure_count=12,
                    exemplar_event_ids=[],
                    exemplar_summaries=[],
                    exemplar_run_ids=[],
                )
                db.add(cluster)
                await db.flush()
                await db.commit()
                return int(project.id), int(cluster.id)

        return asyncio.run(_seed())

    def test_adopt_gate_creates_scaffold_and_appends_gate(self):
        # Fresh project with no scaffolded pack yet → endpoint should
        # build one from the recipe + append the new gate.
        project_id, cluster_id = self._make_project_and_cluster(
            recipe_id="qa-sft", reason_code="safety_failure",
        )
        client = TestClient(app)
        resp = client.post(
            f"/api/projects/{project_id}/evaluation/adopt-gate-from-cluster",
            json={"cluster_id": cluster_id},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], project_id)
        self.assertEqual(body["cluster_reason_code"], "safety_failure")
        # Body returns the new gate's full shape so the FE can deep-link.
        new_gate = body["new_gate"]
        self.assertEqual(new_gate["metric_id"], "safety_pass_rate")
        self.assertEqual(new_gate["operator"], "gte")
        # And the saved scaffold carries the gate at the tail of the
        # first task_spec.
        saved_gates = body["scaffolded_pack"]["task_specs"][0]["gates"]
        gate_ids = [g["gate_id"] for g in saved_gates]
        self.assertIn(new_gate["gate_id"], gate_ids)

    def test_adopt_gate_collision_suffixes_the_id(self):
        # Adopt the same gate twice. Second call should suffix the
        # gate_id (..._2) so it doesn't collide with the first.
        project_id, cluster_id = self._make_project_and_cluster(
            recipe_id="qa-sft", reason_code="safety_failure",
        )
        client = TestClient(app)
        r1 = client.post(
            f"/api/projects/{project_id}/evaluation/adopt-gate-from-cluster",
            json={"cluster_id": cluster_id},
        )
        r2 = client.post(
            f"/api/projects/{project_id}/evaluation/adopt-gate-from-cluster",
            json={"cluster_id": cluster_id},
        )
        self.assertEqual(r1.status_code, 201, r1.text)
        self.assertEqual(r2.status_code, 201, r2.text)
        first_id = r1.json()["new_gate"]["gate_id"]
        second_id = r2.json()["new_gate"]["gate_id"]
        self.assertNotEqual(first_id, second_id)
        self.assertTrue(second_id.endswith("_2"), f"expected _2 suffix, got {second_id}")

    def test_404_when_cluster_belongs_to_a_different_project(self):
        # Cross-project request — endpoint must refuse to scope the
        # action across project boundaries even if the cluster_id is
        # valid in another project.
        pid_a, _cluster_a = self._make_project_and_cluster(
            recipe_id="qa-sft", reason_code="safety_failure",
        )
        _pid_b, cluster_b = self._make_project_and_cluster(
            recipe_id="qa-sft", reason_code="hallucination",
        )
        client = TestClient(app)
        # cluster_b belongs to project B; request it under project A.
        resp = client.post(
            f"/api/projects/{pid_a}/evaluation/adopt-gate-from-cluster",
            json={"cluster_id": cluster_b},
        )
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["detail"], "cluster_not_found")

    def test_400_when_reason_code_has_no_mapping_and_no_override(self):
        # An unmapped reason_code with no client overrides should
        # surface the "no suggestion" error so the FE can prompt the
        # user to add a gate manually.
        project_id, cluster_id = self._make_project_and_cluster(
            recipe_id="qa-sft", reason_code="totally_made_up_code",
        )
        client = TestClient(app)
        resp = client.post(
            f"/api/projects/{project_id}/evaluation/adopt-gate-from-cluster",
            json={"cluster_id": cluster_id},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(
            resp.json()["detail"].startswith("no_gate_suggestion_for_reason_code:"),
            resp.json(),
        )

    def test_overrides_let_the_client_pick_metric_and_threshold(self):
        # When the FE wants to override the catalog default (e.g. the
        # user picked a different metric in a confirm dialog), the
        # overrides should win.
        project_id, cluster_id = self._make_project_and_cluster(
            recipe_id="qa-sft", reason_code="safety_failure",
        )
        client = TestClient(app)
        resp = client.post(
            f"/api/projects/{project_id}/evaluation/adopt-gate-from-cluster",
            json={
                "cluster_id": cluster_id,
                "metric_id": "f1",
                "threshold": 0.42,
                "operator": "gte",
                "required": True,
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        new_gate = resp.json()["new_gate"]
        self.assertEqual(new_gate["metric_id"], "f1")
        self.assertEqual(new_gate["threshold"], 0.42)
        self.assertTrue(new_gate["required"])


if __name__ == "__main__":
    unittest.main()
