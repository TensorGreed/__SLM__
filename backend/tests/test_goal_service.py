"""Tests for goal_service + the /goal/progress + /goal endpoints (Arc H).

Covers:
  * compute_progress on a brand-new project → has_explicit_goal=False,
    falls back to default goal (f1 ≥ 0.70), all components are
    "attention" or "pending" depending on data availability.
  * compute_progress on a project with training rows + gold rows →
    data_ready and gold_set components flip to "met"; predicted_pass
    + eval_pass_rate stay "pending" until forecast + eval run.
  * compute_progress when training_forecast_cache + EvalResult exist
    → predicted_pass + eval_pass_rate components compute their ratio
    against the goal threshold.
  * PUT /goal validates target_metric + clamps target_threshold to
    [0, 1]; bad metric → 400.
  * DELETE /goal clears the project's stated goal idempotently.
  * Every component carries a stable concept_id that matches the
    frontend Term registry (failure here means a UI surface will
    silently lose its Term tooltip).
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "goal_service_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "goal_service_test_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import Dataset, DatasetType  # noqa: E402
from app.models.experiment import EvalResult, Experiment, ExperimentStatus  # noqa: E402
from app.models.project import Project  # noqa: E402


def setUpModule() -> None:
    settings.AUTH_ENABLED = False
    settings.DATA_DIR = TEST_DATA_DIR.resolve()
    settings.ensure_dirs()
    for suffix in ("", "-shm", "-wal"):
        p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
        if p.exists():
            p.unlink()
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GoalServiceApiTests._cm = TestClient(app)
    GoalServiceApiTests.client = GoalServiceApiTests._cm.__enter__()


def tearDownModule() -> None:
    GoalServiceApiTests._cm.__exit__(None, None, None)


class GoalServiceApiTests(unittest.TestCase):
    _cm: TestClient | None = None
    client: TestClient

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"goal-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _seed_training_rows(self, project_id: int, count: int) -> None:
        async with async_session_factory() as db:
            ds = Dataset(
                project_id=project_id,
                name="training",
                dataset_type=DatasetType.CLEANED,
                record_count=count,
                file_path="",
            )
            db.add(ds)
            await db.commit()

    async def _seed_gold_rows(self, project_id: int, count: int) -> None:
        async with async_session_factory() as db:
            ds = Dataset(
                project_id=project_id,
                name="gold",
                dataset_type=DatasetType.GOLD_DEV,
                record_count=count,
                file_path="",
            )
            db.add(ds)
            await db.commit()

    async def _seed_forecast_cache(
        self, project_id: int, predicted_f1: float,
    ) -> None:
        async with async_session_factory() as db:
            project = await db.get(Project, project_id)
            assert project is not None
            project.training_forecast_cache = {
                "forecast": {"predicted_f1_confidence": predicted_f1},
            }
            await db.commit()

    async def _seed_eval_result(
        self, project_id: int, pass_rate: float,
    ) -> None:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=project_id,
                name="exp",
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                status=ExperimentStatus.COMPLETED,
            )
            db.add(exp)
            await db.flush()
            ev = EvalResult(
                experiment_id=exp.id,
                dataset_name="gold",
                eval_type="auto",
                metrics={"f1": pass_rate},
                pass_rate=pass_rate,
                created_at=datetime.now(timezone.utc),
            )
            db.add(ev)
            await db.commit()

    # ─────────────────────────────────────────────────────────────
    # Happy paths
    # ─────────────────────────────────────────────────────────────

    def test_no_goal_falls_back_to_default_f1_threshold(self):
        pid = self._create_project()
        resp = self.client.get(f"/api/projects/{pid}/goal/progress")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["has_explicit_goal"])
        self.assertEqual(body["goal"]["target_metric"], "f1")
        self.assertAlmostEqual(body["goal"]["target_threshold"], 0.70)
        # Brand-new project: no data, no gold, no forecast, no eval.
        comp = {c["id"]: c for c in body["components"]}
        self.assertEqual(comp["data_ready"]["status"], "attention")
        self.assertEqual(comp["gold_set"]["status"], "attention")
        self.assertEqual(comp["predicted_pass"]["status"], "pending")
        self.assertEqual(comp["eval_pass_rate"]["status"], "pending")
        self.assertIn("predicted_pass", body["pending_components"])
        self.assertIn("eval_pass_rate", body["pending_components"])

    def test_components_flip_to_met_when_data_and_gold_seeded(self):
        pid = self._create_project()
        asyncio.run(self._seed_training_rows(pid, 200))
        asyncio.run(self._seed_gold_rows(pid, 120))
        resp = self.client.get(f"/api/projects/{pid}/goal/progress")
        body = resp.json()
        comp = {c["id"]: c for c in body["components"]}
        self.assertEqual(comp["data_ready"]["status"], "met")
        self.assertEqual(comp["data_ready"]["value"], 1.0)
        self.assertEqual(comp["gold_set"]["status"], "met")
        self.assertEqual(comp["gold_set"]["value"], 1.0)

    def test_predicted_pass_ratio_against_target_threshold(self):
        pid = self._create_project()
        # Seed forecast = 0.85, default target = 0.70 → ratio clamped to 1.0
        asyncio.run(self._seed_forecast_cache(pid, 0.85))
        resp = self.client.get(f"/api/projects/{pid}/goal/progress")
        body = resp.json()
        comp = {c["id"]: c for c in body["components"]}
        self.assertEqual(comp["predicted_pass"]["status"], "met")
        self.assertEqual(comp["predicted_pass"]["value"], 1.0)
        # And when forecast < target, status drops to "attention".
        pid2 = self._create_project()
        asyncio.run(self._seed_forecast_cache(pid2, 0.40))
        body2 = self.client.get(f"/api/projects/{pid2}/goal/progress").json()
        comp2 = {c["id"]: c for c in body2["components"]}
        self.assertEqual(comp2["predicted_pass"]["status"], "attention")
        self.assertLess(comp2["predicted_pass"]["value"], 1.0)

    def test_eval_pass_rate_uses_latest_result(self):
        pid = self._create_project()
        asyncio.run(self._seed_eval_result(pid, 0.92))
        body = self.client.get(f"/api/projects/{pid}/goal/progress").json()
        comp = {c["id"]: c for c in body["components"]}
        self.assertEqual(comp["eval_pass_rate"]["status"], "met")
        self.assertEqual(comp["eval_pass_rate"]["value"], 1.0)
        self.assertIn("92%", comp["eval_pass_rate"]["detail"])

    def test_status_flips_to_ready_to_ship_when_all_components_met(self):
        pid = self._create_project()
        asyncio.run(self._seed_training_rows(pid, 200))
        asyncio.run(self._seed_gold_rows(pid, 120))
        asyncio.run(self._seed_forecast_cache(pid, 0.90))
        asyncio.run(self._seed_eval_result(pid, 0.92))
        body = self.client.get(f"/api/projects/{pid}/goal/progress").json()
        self.assertEqual(body["status"], "ready_to_ship")
        self.assertEqual(body["overall_progress"], 1.0)
        self.assertEqual(body["pending_components"], [])
        self.assertEqual(body["blockers"], [])

    # ─────────────────────────────────────────────────────────────
    # Goal CRUD
    # ─────────────────────────────────────────────────────────────

    def test_put_goal_persists_and_round_trips(self):
        pid = self._create_project()
        resp = self.client.put(
            f"/api/projects/{pid}/goal",
            json={
                "target_metric": "f1",
                "target_threshold": 0.85,
                "deadline": "2026-07-15",
                "title": "Ship refund classifier",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        goal = resp.json()["goal"]
        self.assertEqual(goal["target_metric"], "f1")
        self.assertEqual(goal["target_threshold"], 0.85)
        self.assertEqual(goal["deadline"], "2026-07-15")
        self.assertEqual(goal["title"], "Ship refund classifier")
        self.assertIsNotNone(goal["stated_at"])
        # Progress now reports has_explicit_goal=True and uses the
        # custom threshold.
        body = self.client.get(f"/api/projects/{pid}/goal/progress").json()
        self.assertTrue(body["has_explicit_goal"])
        self.assertEqual(body["goal"]["target_threshold"], 0.85)

    def test_put_goal_rejects_unsupported_metric(self):
        pid = self._create_project()
        resp = self.client.put(
            f"/api/projects/{pid}/goal",
            json={"target_metric": "perplexity", "target_threshold": 0.5},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("perplexity", resp.json()["detail"])

    def test_put_goal_clamps_threshold_to_unit_interval(self):
        pid = self._create_project()
        resp = self.client.put(
            f"/api/projects/{pid}/goal",
            json={"target_metric": "f1", "target_threshold": 1.5},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["goal"]["target_threshold"], 1.0)
        resp2 = self.client.put(
            f"/api/projects/{pid}/goal",
            json={"target_metric": "f1", "target_threshold": -0.3},
        )
        self.assertEqual(resp2.status_code, 200, resp2.text)
        self.assertEqual(resp2.json()["goal"]["target_threshold"], 0.0)

    def test_delete_goal_drops_the_stated_goal_idempotently(self):
        pid = self._create_project()
        self.client.put(
            f"/api/projects/{pid}/goal",
            json={"target_metric": "f1", "target_threshold": 0.85},
        )
        # First delete: 200.
        resp = self.client.delete(f"/api/projects/{pid}/goal")
        self.assertEqual(resp.status_code, 200, resp.text)
        # Idempotent second delete: still 200, project still exists.
        resp2 = self.client.delete(f"/api/projects/{pid}/goal")
        self.assertEqual(resp2.status_code, 200, resp2.text)
        # And progress now reports has_explicit_goal=False.
        body = self.client.get(f"/api/projects/{pid}/goal/progress").json()
        self.assertFalse(body["has_explicit_goal"])

    def test_progress_404_on_unknown_project(self):
        resp = self.client.get("/api/projects/9999999/goal/progress")
        self.assertEqual(resp.status_code, 404)

    # ─────────────────────────────────────────────────────────────
    # Frontend contract — every component must carry a concept_id
    # that the Term registry knows. This is a tripwire: if a future
    # commit drops or renames a concept_id without updating the
    # frontend Term registry, the UI's "Learn more" link silently
    # disappears. Better to fail loud here.
    # ─────────────────────────────────────────────────────────────

    def test_every_component_carries_a_concept_id_for_term_linking(self):
        pid = self._create_project()
        body = self.client.get(f"/api/projects/{pid}/goal/progress").json()
        # Concept IDs we *expect* to find — these MUST exist in
        # frontend/src/components/shared/glossary.ts.
        expected = {
            "data_ready": "task_shape",
            "gold_set": "gold_set",
            "predicted_pass": "predicted_f1_confidence",
            "eval_pass_rate": "pass_rate",
        }
        for component in body["components"]:
            self.assertIn("concept_id", component, f"component {component['id']} missing concept_id")
            self.assertEqual(
                component["concept_id"],
                expected[component["id"]],
                f"concept_id changed for {component['id']} — update frontend Term registry too",
            )

    # ─────────────────────────────────────────────────────────────
    # Arc R-2 slice 2 — gate_breakdown on the eval_pass_rate
    # component. Empty when no eval has run; populated when an
    # eval exists, projecting the project's eval pack's gate
    # checks through into the ledger row.
    # ─────────────────────────────────────────────────────────────

    def test_eval_pass_rate_gate_breakdown_empty_without_eval(self):
        pid = self._create_project()
        body = self.client.get(f"/api/projects/{pid}/goal/progress").json()
        comp = {c["id"]: c for c in body["components"]}
        # Field present + an empty list (frontend treats this as
        # "no breakdown to render"); never missing entirely so
        # the type contract stays stable.
        self.assertIn("gate_breakdown", comp["eval_pass_rate"])
        self.assertEqual(comp["eval_pass_rate"]["gate_breakdown"], [])

    async def _apply_recipe(self, project_id: int, recipe_id: str) -> None:
        # Use the public endpoint instead of touching project.selected_recipe
        # directly so the recipe-apply service stamps every adjacent field
        # (adapter preset, eval_pack_id, etc.).
        resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": recipe_id},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_gate_breakdown_populates_when_eval_run_on_rag_protocol_project(self):
        pid = self._create_project()
        # Apply the rag-protocol recipe; its eval_pack_id is now the
        # discipline pack (Arc R-2 slice 1).
        asyncio.run(self._apply_recipe(pid, "rag-protocol"))
        # Seed an eval result with the metric fields the discipline
        # pack reads (citation_rate alias → faithfulness_rate, etc.)
        asyncio.run(self._seed_rag_eval_result(
            pid,
            pass_rate=0.62,
            metrics={
                "f1": 0.62,
                "faithfulness_rate": 0.72,      # gates as citation_rate (≥0.75 required)
                "unsupported_token_rate_mean": 0.18,  # gates as hallucination_rate (≤0.15 required)
                "appropriate_refusal_rate": 0.85,     # ≥0.80 required → passes
            },
        ))

        body = self.client.get(f"/api/projects/{pid}/goal/progress").json()
        comp = {c["id"]: c for c in body["components"]}
        breakdown = comp["eval_pass_rate"]["gate_breakdown"]
        # The discipline pack ships 6 gates (4 required + 2 optional).
        self.assertGreaterEqual(len(breakdown), 4)
        by_gate = {g["gate_id"]: g for g in breakdown}
        self.assertIn("min_citation_rate", by_gate)
        self.assertIn("max_hallucination_rate", by_gate)
        self.assertIn("min_appropriate_refusal_rate", by_gate)
        self.assertIn("min_f1", by_gate)

        # Citation: actual 0.72 < 0.75 threshold → not passed.
        citation = by_gate["min_citation_rate"]
        self.assertEqual(citation["operator"], "gte")
        self.assertEqual(citation["threshold"], 0.75)
        self.assertAlmostEqual(citation["actual"], 0.72, places=3)
        self.assertFalse(citation["passed"])

        # Hallucination: actual 0.18 > 0.15 threshold → not passed.
        # Uses lte operator.
        hallucination = by_gate["max_hallucination_rate"]
        self.assertEqual(hallucination["operator"], "lte")
        self.assertEqual(hallucination["threshold"], 0.15)
        self.assertAlmostEqual(hallucination["actual"], 0.18, places=3)
        self.assertFalse(hallucination["passed"])

        # Refusal: actual 0.85 ≥ 0.80 → passes.
        refusal = by_gate["min_appropriate_refusal_rate"]
        self.assertEqual(refusal["operator"], "gte")
        self.assertTrue(refusal["passed"])

    def test_gate_breakdown_metric_ids_match_frontend_glossary(self):
        # Tripwire — the gate_breakdown's metric_id is what the
        # frontend wraps in <Term id={metric_id}>. If a future commit
        # renames a metric_id without updating the glossary, the UI
        # silently loses the Academy deep-link. Better to fail loud.
        pid = self._create_project()
        asyncio.run(self._apply_recipe(pid, "rag-protocol"))
        asyncio.run(self._seed_rag_eval_result(pid, pass_rate=0.62, metrics={"f1": 0.62}))

        body = self.client.get(f"/api/projects/{pid}/goal/progress").json()
        comp = {c["id"]: c for c in body["components"]}
        breakdown = comp["eval_pass_rate"]["gate_breakdown"]
        # The discipline metrics the frontend glossary MUST carry.
        expected_glossary_ids = {
            "f1",
            "citation_rate",
            "hallucination_rate",
            "appropriate_refusal_rate",
            "format_consistency",
            "safety_pass_rate",
        }
        actual_metric_ids = {g["metric_id"] for g in breakdown}
        self.assertTrue(
            actual_metric_ids.issubset(expected_glossary_ids),
            f"new metric_id detected in gate_breakdown that may lack a "
            f"frontend glossary entry: {actual_metric_ids - expected_glossary_ids}. "
            f"Add it to frontend/src/components/shared/glossary.ts.",
        )

    async def _seed_rag_eval_result(
        self, project_id: int, pass_rate: float, metrics: dict[str, float],
    ) -> None:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=project_id,
                name="exp",
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                status=ExperimentStatus.COMPLETED,
            )
            db.add(exp)
            await db.flush()
            ev = EvalResult(
                experiment_id=exp.id,
                dataset_name="gold",
                # Use a task_profile the rag-protocol recipe's pack
                # surfaces so _select_task_spec resolves correctly.
                eval_type="rag_qa",
                metrics=metrics,
                pass_rate=pass_rate,
                created_at=datetime.now(timezone.utc),
            )
            db.add(ev)
            await db.commit()


if __name__ == "__main__":
    unittest.main()
