"""Tests for the recipe-aware eval-pack scaffolder + save endpoint (E5).

Covers:
  - ``scaffold_pack`` returns recipe-appropriate metrics + gates for
    every supported recipe id; unknown recipes fall back to generic.
  - ``scaffold_pack_for_project`` reads recipe + gold-set summary,
    raises ``recipe_required`` / ``project_not_found`` correctly.
  - ``GET /api/projects/{id}/evaluation/pack-scaffold`` returns the
    full draft.
  - ``POST /api/projects/{id}/evaluation/pack-scaffold`` persists the
    edited draft to ``project.runtime_config["scaffolded_evaluation_pack"]``
    and flips ``evaluation_preferred_pack_id`` to the scaffolded id.
  - The resolver picks up the saved scaffold as the active pack on the
    next read of ``/pack-preference``.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "eval_pack_scaffold_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "eval_pack_scaffold_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.project import Project, ProjectStatus
from app.services.eval_pack_scaffold_service import (
    SCAFFOLDED_PACK_ID,
    get_scaffolded_pack,
    is_scaffolded_pack_id,
    save_scaffolded_pack,
    scaffold_pack,
    scaffold_pack_for_project,
)


def _run(coro):
    return asyncio.run(coro)


class ScaffolderUnitTests(unittest.TestCase):
    """Pure-function: no DB. One test per recipe shape."""

    def test_classification_draft_carries_macro_f1_and_class_balance_gates(self):
        draft = scaffold_pack("classification")
        self.assertEqual(draft["pack_id"], SCAFFOLDED_PACK_ID)
        task_spec = draft["task_specs"][0]
        self.assertEqual(task_spec["task_profile"], "classification")
        self.assertIn("macro_f1", task_spec["required_metric_ids"])
        gate_ids = [g["gate_id"] for g in task_spec["gates"]]
        self.assertIn("min_macro_f1", gate_ids)
        self.assertIn("min_per_class_f1", gate_ids)

    def test_span_extraction_draft_carries_span_set_metrics(self):
        draft = scaffold_pack("span-extraction")
        task_spec = draft["task_specs"][0]
        self.assertEqual(task_spec["task_profile"], "structured_extraction")
        self.assertIn("span_set_f1", task_spec["required_metric_ids"])
        gate_ids = [g["gate_id"] for g in task_spec["gates"]]
        self.assertIn("min_span_set_f1", gate_ids)
        # precision + recall gates for false-positive/negative balance.
        self.assertIn("min_span_set_precision", gate_ids)
        self.assertIn("min_span_set_recall", gate_ids)

    def test_summarization_draft_carries_rouge_l_and_groundedness(self):
        draft = scaffold_pack("summarization")
        task_spec = draft["task_specs"][0]
        self.assertEqual(task_spec["task_profile"], "summarization")
        gate_ids = [g["gate_id"] for g in task_spec["gates"]]
        self.assertIn("min_rouge_l", gate_ids)
        # Groundedness catches fabrication — must be present.
        self.assertIn("min_groundedness", gate_ids)

    def test_qa_sft_draft_carries_exact_match_and_llm_judge(self):
        draft = scaffold_pack("qa-sft")
        task_spec = draft["task_specs"][0]
        gate_ids = [g["gate_id"] for g in task_spec["gates"]]
        self.assertIn("min_exact_match", gate_ids)
        self.assertIn("min_llm_judge_pass_rate", gate_ids)

    def test_unknown_recipe_falls_back_to_generic(self):
        draft = scaffold_pack("never_heard_of_this")
        # Doesn't crash + still returns a valid pack with at least
        # one task spec the eval engine can consume.
        self.assertTrue(draft["task_specs"])
        self.assertEqual(draft["task_specs"][0]["task_profile"], "instruction_sft")

    def test_gold_set_summary_threads_into_description(self):
        draft = scaffold_pack("qa-sft", gold_set_summary={"row_count": 200})
        self.assertIn("200 rows", draft["description"])

    def test_is_scaffolded_pack_id_helper(self):
        self.assertTrue(is_scaffolded_pack_id(SCAFFOLDED_PACK_ID))
        self.assertTrue(is_scaffolded_pack_id(SCAFFOLDED_PACK_ID.upper()))
        self.assertFalse(is_scaffolded_pack_id("evalpack.general.default"))
        self.assertFalse(is_scaffolded_pack_id(None))


class ScaffoldServiceDBTests(unittest.TestCase):
    """scaffold_pack_for_project + save_scaffolded_pack against the real DB."""

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

    def _new_project(self, recipe_id: str | None = "classification") -> int:
        ScaffoldServiceDBTests._counter += 1
        tag = ScaffoldServiceDBTests._counter

        async def _seed():
            async with async_session_factory() as session:
                proj = Project(
                    name=f"Eval Pack Scaffold Project #{tag}",
                    status=ProjectStatus.DRAFT,
                    selected_recipe={"recipe_id": recipe_id} if recipe_id else None,
                )
                session.add(proj)
                await session.commit()
                return proj.id
        return _run(_seed())

    def test_scaffold_pack_for_project_returns_recipe_appropriate_draft(self):
        pid = self._new_project(recipe_id="classification")

        async def _call():
            async with async_session_factory() as session:
                return await scaffold_pack_for_project(session, project_id=pid)
        result = _run(_call())
        self.assertEqual(result["project_id"], pid)
        self.assertEqual(result["recipe_id"], "classification")
        self.assertIn("draft_pack", result)
        # macro_f1 lives in the required metric list for classification.
        task_spec = result["draft_pack"]["task_specs"][0]
        self.assertIn("macro_f1", task_spec["required_metric_ids"])

    def test_scaffold_pack_for_project_raises_recipe_required(self):
        pid = self._new_project(recipe_id=None)

        async def _call():
            async with async_session_factory() as session:
                with self.assertRaises(ValueError) as ctx:
                    await scaffold_pack_for_project(session, project_id=pid)
                return str(ctx.exception)
        self.assertEqual(_run(_call()), "recipe_required")

    def test_scaffold_pack_for_project_raises_project_not_found(self):
        async def _call():
            async with async_session_factory() as session:
                with self.assertRaises(ValueError) as ctx:
                    await scaffold_pack_for_project(session, project_id=999999)
                return str(ctx.exception)
        self.assertEqual(_run(_call()), "project_not_found")

    def test_save_persists_to_runtime_config_and_flips_preference(self):
        pid = self._new_project(recipe_id="qa-sft")
        draft = scaffold_pack("qa-sft")
        # Simulate an inline edit before save — the user dropped the
        # exact-match threshold from 0.45 to 0.35.
        edited = dict(draft)
        edited["task_specs"] = list(draft["task_specs"])
        edited_task_spec = dict(edited["task_specs"][0])
        edited_gates = [dict(g) for g in edited_task_spec["gates"]]
        for gate in edited_gates:
            if gate["gate_id"] == "min_exact_match":
                gate["threshold"] = 0.35
        edited_task_spec["gates"] = edited_gates
        edited["task_specs"][0] = edited_task_spec

        async def _save():
            async with async_session_factory() as session:
                result = await save_scaffolded_pack(
                    session, project_id=pid, draft_pack=edited,
                )
                await session.commit()
                return result
        out = _run(_save())
        self.assertEqual(out["preferred_pack_id"], SCAFFOLDED_PACK_ID)

        # Round-trip: read the project back, see the saved pack +
        # flipped preference.
        async def _readback():
            async with async_session_factory() as session:
                p = await session.get(Project, pid)
                return p.evaluation_preferred_pack_id, get_scaffolded_pack(p)
        preferred, saved = _run(_readback())
        self.assertEqual(preferred, SCAFFOLDED_PACK_ID)
        self.assertIsNotNone(saved)
        # The edit survived the roundtrip.
        gates = saved["task_specs"][0]["gates"]
        exact_match_gate = next(g for g in gates if g["gate_id"] == "min_exact_match")
        self.assertEqual(exact_match_gate["threshold"], 0.35)

    def test_save_rejects_drafts_with_no_task_specs(self):
        pid = self._new_project(recipe_id="qa-sft")

        async def _save():
            async with async_session_factory() as session:
                with self.assertRaises(ValueError) as ctx:
                    await save_scaffolded_pack(
                        session, project_id=pid, draft_pack={"pack_id": "x", "task_specs": []},
                    )
                return str(ctx.exception)
        self.assertEqual(_run(_save()), "draft_pack_missing_task_specs")

    def test_save_forces_canonical_pack_id_even_if_caller_changed_it(self):
        # The resolver keys on SCAFFOLDED_PACK_ID — letting the client
        # rename it would silently disconnect the saved blob from the
        # active-pack path. The save normalises this back.
        pid = self._new_project(recipe_id="classification")
        draft = scaffold_pack("classification")
        draft["pack_id"] = "my-custom-id"

        async def _save():
            async with async_session_factory() as session:
                result = await save_scaffolded_pack(
                    session, project_id=pid, draft_pack=draft,
                )
                await session.commit()
                return result
        out = _run(_save())
        self.assertEqual(out["scaffolded_pack"]["pack_id"], SCAFFOLDED_PACK_ID)


class ScaffoldApiTests(unittest.TestCase):
    """Endpoint contract — GET + POST + resolver pickup."""

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

    def _instantiate_template(self, slug: str, name: str) -> int:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def test_get_scaffold_endpoint_returns_draft_matching_recipe_defaults(self):
        # ticket-router is a classification template — expect
        # macro_f1 + per_class_f1 gates in the scaffold.
        pid = self._instantiate_template("ticket-router", "E5 Scaffold GET")
        resp = self.client.get(f"/api/projects/{pid}/evaluation/pack-scaffold")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["recipe_id"], "classification")
        gate_ids = [g["gate_id"] for g in body["draft_pack"]["task_specs"][0]["gates"]]
        self.assertIn("min_macro_f1", gate_ids)
        self.assertIn("min_per_class_f1", gate_ids)

    def test_get_scaffold_404s_on_unknown_project(self):
        resp = self.client.get("/api/projects/99999/evaluation/pack-scaffold")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_get_scaffold_400s_on_project_with_no_recipe(self):
        # Bare project — no recipe applied. The endpoint must 400 +
        # explain the missing-recipe state so the UI can prompt.
        create_resp = self.client.post(
            "/api/projects",
            json={"name": f"E5 Scaffold No Recipe {os.getpid()}"},
        )
        pid = create_resp.json()["id"]
        resp = self.client.get(f"/api/projects/{pid}/evaluation/pack-scaffold")
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("recipe_required", resp.text)

    def test_post_scaffold_persists_and_resolver_picks_up_saved_pack(self):
        # Full edit-then-save flow: GET → mutate gate threshold →
        # POST → confirm the active pack-preference resolves to the
        # saved scaffold with the user's threshold change.
        pid = self._instantiate_template("ticket-router", "E5 Scaffold POST")
        draft_resp = self.client.get(f"/api/projects/{pid}/evaluation/pack-scaffold")
        draft = draft_resp.json()["draft_pack"]
        # Bump the macro_f1 gate threshold so we can detect the edit
        # in the saved blob.
        for gate in draft["task_specs"][0]["gates"]:
            if gate["gate_id"] == "min_macro_f1":
                gate["threshold"] = 0.55

        save_resp = self.client.post(
            f"/api/projects/{pid}/evaluation/pack-scaffold",
            json={"draft_pack": draft},
        )
        self.assertEqual(save_resp.status_code, 201, save_resp.text)
        self.assertEqual(save_resp.json()["preferred_pack_id"], SCAFFOLDED_PACK_ID)

        # GET /pack-preference now resolves the scaffolded pack as
        # the active one.
        pref = self.client.get(f"/api/projects/{pid}/evaluation/pack-preference").json()
        self.assertEqual(pref["active_pack_id"], SCAFFOLDED_PACK_ID)
        self.assertEqual(pref["active_pack_source"], "project_scaffold")
        # The edit survived the resolver path — active pack carries
        # the 0.55 threshold the user typed in.
        active_gates = pref["active_pack"]["task_specs"][0]["gates"]
        macro_gate = next(g for g in active_gates if g["gate_id"] == "min_macro_f1")
        self.assertEqual(macro_gate["threshold"], 0.55)

    def test_post_scaffold_400s_on_missing_task_specs(self):
        pid = self._instantiate_template(
            "ticket-router", "E5 Scaffold POST Validation",
        )
        resp = self.client.post(
            f"/api/projects/{pid}/evaluation/pack-scaffold",
            json={"draft_pack": {"pack_id": SCAFFOLDED_PACK_ID, "task_specs": []}},
        )
        self.assertEqual(resp.status_code, 400, resp.text)


if __name__ == "__main__":
    unittest.main()
