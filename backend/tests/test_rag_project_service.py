"""Tests for the RAG-skeleton project service
(USER-SUCCESS Epic 7 Phase 7b).

Covers:

- ``clone_project_for_rag`` happy path: copies fields + gold rows
  + parent link; stamps ``runtime_config.rag_first``; sets
  ``target_profile_id="qa_with_auto_rag"``.
- Refusal paths: non-qa-sft source, no-recipe source, missing
  source.
- Unique-name resolution: a clone whose default name collides
  with an existing project gets a numeric suffix.
- ``is_rag_first`` helper truth table.
- ``POST /api/projects/{id}/reroute-to-rag`` end-to-end via
  TestClient: 201 happy path, 400 wrong recipe, 404 missing source.
- Playground integration: a rag_first project chat call forces
  ``auto_rag=True`` even when the request body says otherwise, and
  the response surfaces ``rag_first_active: true``.
- Training-start gate: POST .../start on a rag_first project
  returns 400 with the documented error_code.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")
os.environ.setdefault("TRAINING_BACKEND", "simulate")
os.environ.setdefault("ALLOW_SIMULATED_TRAINING", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.rag_project_service import (  # noqa: E402
    _pick_unique_name,
    _rewrite_file_path,
    is_rag_first,
)


# ─────────────────────────────────────────────────────────────────────
# Pure-function helpers
# ─────────────────────────────────────────────────────────────────────


class RewriteFilePathTests(unittest.TestCase):
    def test_swaps_project_id_segment(self):
        out = _rewrite_file_path(
            "/data/projects/42/gold/gold_dev.jsonl", 42, 100
        )
        self.assertEqual(out, "/data/projects/100/gold/gold_dev.jsonl")

    def test_leaves_unrelated_paths_alone(self):
        out = _rewrite_file_path("/etc/passwd", 42, 100)
        self.assertEqual(out, "/etc/passwd")

    def test_handles_none(self):
        self.assertIsNone(_rewrite_file_path(None, 1, 2))

    def test_handles_empty_string(self):
        self.assertIsNone(_rewrite_file_path("", 1, 2))


class IsRagFirstHelperTests(unittest.TestCase):
    def _project(self, runtime_config):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.runtime_config = runtime_config
        return m

    def test_true_when_flag_set(self):
        self.assertTrue(is_rag_first(self._project({"rag_first": True})))

    def test_false_when_flag_explicitly_false(self):
        self.assertFalse(is_rag_first(self._project({"rag_first": False})))

    def test_false_when_runtime_config_missing(self):
        self.assertFalse(is_rag_first(self._project(None)))

    def test_false_when_runtime_config_empty(self):
        self.assertFalse(is_rag_first(self._project({})))

    def test_false_when_runtime_config_not_dict(self):
        self.assertFalse(is_rag_first(self._project("rag_first")))

    def test_false_on_none_project(self):
        self.assertFalse(is_rag_first(None))


# ─────────────────────────────────────────────────────────────────────
# End-to-end via TestClient
# ─────────────────────────────────────────────────────────────────────


class RerouteToRagApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    # ── happy path ─────────────────────────────────────────────────

    def test_qa_sft_source_clones_into_rag_first_sibling(self):
        source = self._instantiate_template(
            "policy-qa-style", "Phase7b Clone Happy"
        )
        resp = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["source_project_id"], source["id"])
        new_id = body["new_project_id"]
        self.assertNotEqual(new_id, source["id"])
        self.assertTrue(body["new_project_name"].endswith(" (RAG)"))

        # New project should have rag_first flag + parent link.
        new_proj_resp = self.client.get(f"/api/projects/{new_id}")
        self.assertEqual(new_proj_resp.status_code, 200, new_proj_resp.text)
        new_proj = new_proj_resp.json()
        self.assertEqual(new_proj["parent_project_id"], source["id"])
        self.assertEqual(new_proj["target_profile_id"], "qa_with_auto_rag")
        runtime_config = new_proj["runtime_config"] or {}
        self.assertTrue(runtime_config.get("rag_first"))
        self.assertEqual(
            runtime_config.get("auto_rag"), {"enabled": True}
        )
        # Recipe carried forward.
        self.assertEqual(
            (new_proj["selected_recipe"] or {}).get("recipe_id"),
            "qa-sft",
        )

        # Clone report stamped onto runtime_config for observability.
        clone_report = runtime_config.get("clone_report") or {}
        self.assertEqual(clone_report.get("source_project_id"), source["id"])
        self.assertIn("gold", clone_report.get("copied_subdirs", []))
        # Auto-RAG index built immediately (Phase 9b hook).
        index_report = clone_report.get("auto_rag_index") or {}
        self.assertTrue(index_report.get("built"))

    def test_gold_file_is_copied_to_new_project_data_dir(self):
        source = self._instantiate_template(
            "policy-qa-style", "Phase7b Gold Copied"
        )
        resp = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        new_id = resp.json()["new_project_id"]

        # The gold/ dir under the new project's data root should exist
        # and contain at least the gold_dev jsonl.
        new_gold_dir = (
            settings.DATA_DIR / "projects" / str(new_id) / "gold"
        )
        self.assertTrue(
            new_gold_dir.exists(),
            f"expected gold dir at {new_gold_dir}",
        )
        gold_files = list(new_gold_dir.glob("*.jsonl"))
        self.assertGreater(len(gold_files), 0, f"no gold jsonl in {new_gold_dir}")
        # Sanity: file has actual content
        first_row = next(iter(gold_files))
        with first_row.open() as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        self.assertGreater(len(lines), 0)
        # Round-trip parses as JSON.
        json.loads(lines[0])

    def test_collision_resolves_with_numeric_suffix(self):
        # Phase 7d added a 1-hour idempotency window on the endpoint
        # so calling reroute-to-rag twice in quick succession now
        # returns 429. To test the unique-name path, backdate the
        # first clone past the cooldown then fire a second call.
        from datetime import datetime, timedelta, timezone

        from app.database import async_session_factory
        from app.models.project import Project

        source = self._instantiate_template(
            "policy-qa-style", "Phase7b Collision"
        )
        first = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(first.status_code, 201, first.text)
        first_clone_id = first.json()["new_project_id"]

        async def _backdate() -> None:
            async with async_session_factory() as db:
                row = await db.get(Project, first_clone_id)
                row.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
                await db.commit()

        asyncio.run(_backdate())

        second = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(second.status_code, 201, second.text)
        self.assertNotEqual(
            first.json()["new_project_name"], second.json()["new_project_name"]
        )
        # Second should carry the "2" suffix.
        self.assertTrue(second.json()["new_project_name"].endswith(" 2"))

    def test_idempotency_returns_429_with_existing_clone_id_within_window(self):
        source = self._instantiate_template(
            "policy-qa-style", "Phase7d Idempotency"
        )
        first = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(first.status_code, 201, first.text)
        first_clone_id = first.json()["new_project_id"]

        # Second call within the 1-hour window — 429 with the
        # existing clone id in metadata. The reroute path doesn't
        # match any structured-error stage in app.main, so the dict
        # detail is wrapped in {"detail": ...} (default FastAPI).
        second = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(second.status_code, 429, second.text)
        # The structured-error envelope hoists error_code + metadata to the top
        # level (`detail` is now the message string), so read the body.
        detail = second.json()
        self.assertEqual(detail["error_code"], "REROUTE_RECENTLY_CLONED")
        self.assertEqual(detail["metadata"]["existing_clone_id"], first_clone_id)
        self.assertEqual(detail["metadata"]["window_seconds"], 3600)

    def test_custom_name_suffix_respected(self):
        source = self._instantiate_template(
            "policy-qa-style", "Phase7b Custom Suffix"
        )
        resp = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={"name_suffix": " (Retrieval)"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertTrue(resp.json()["new_project_name"].endswith(" (Retrieval)"))

    # ── refusal paths ──────────────────────────────────────────────

    def test_non_qa_sft_source_returns_400(self):
        source = self._instantiate_template(
            "ticket-router", "Phase7b Wrong Recipe"
        )
        resp = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("source_recipe_not_eligible", resp.text)
        self.assertIn("classification", resp.text)

    def test_missing_source_returns_404(self):
        resp = self.client.post(
            "/api/projects/99999999/reroute-to-rag",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertIn("source_project_not_found", resp.text)

    # ── playground integration ─────────────────────────────────────

    def test_playground_chat_forces_auto_rag_on_rag_first_project(self):
        # Clone a qa-sft project, then call playground without
        # auto_rag in the body — server should force it on.
        source = self._instantiate_template(
            "policy-qa-style", "Phase7b Playground Force"
        )
        clone_resp = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(clone_resp.status_code, 201, clone_resp.text)
        new_id = clone_resp.json()["new_project_id"]

        chat = self.client.post(
            f"/api/projects/{new_id}/training/playground/chat",
            json={
                "provider": "mock",
                "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "messages": [
                    {"role": "user", "content": "What is the PTO policy?"}
                ],
                "auto_rag": False,  # explicit False — should be overridden
                "save_history": False,
            },
        )
        self.assertEqual(chat.status_code, 200, chat.text)
        body = chat.json()
        # Server surfaces the forced-on signal.
        self.assertTrue(body.get("rag_first_active"))
        # auto_rag block present (forced on) — applied=True iff the
        # index built successfully. For policy-qa-style template
        # which carries 200 gold rows, the index will be built.
        self.assertIn("auto_rag", body)

    # ── training-start gate ────────────────────────────────────────

    def test_training_start_refuses_rag_first_project(self):
        from app.database import async_session_factory
        from app.models.experiment import Experiment, ExperimentStatus, TrainingMode

        source = self._instantiate_template(
            "policy-qa-style", "Phase7b Training Gate"
        )
        clone_resp = self.client.post(
            f"/api/projects/{source['id']}/reroute-to-rag",
            json={},
        )
        self.assertEqual(clone_resp.status_code, 201, clone_resp.text)
        new_id = clone_resp.json()["new_project_id"]

        # Hand-create an experiment row directly (since create_experiment
        # API call also runs through training_service which would have
        # to opt-in to rag_first projects — out of scope for the gate
        # test). We just want to drive the start endpoint to verify it
        # refuses.
        async def _go() -> int:
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=new_id,
                    name="phase7b-gate-fixture",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config={},
                    training_mode=TrainingMode.SFT,
                    status=ExperimentStatus.PENDING,
                )
                db.add(exp)
                await db.flush()
                exp_id = exp.id
                await db.commit()
                return exp_id

        exp_id = asyncio.run(_go())
        resp = self.client.post(
            f"/api/projects/{new_id}/training/experiments/{exp_id}/start",
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        # The training stage uses a structured error payload that
        # flattens the dict into the top-level response body (see
        # app.main.http_exception_handler).
        body = resp.json()
        self.assertEqual(body["error_code"], "TRAINING_DISABLED_FOR_RAG_FIRST_PROJECT")
        # Parent project id surfaces in the error so the UI can deep-link.
        self.assertEqual(body["metadata"]["parent_project_id"], source["id"])


# ─────────────────────────────────────────────────────────────────────
# Unique-name picker (direct service-level test)
# ─────────────────────────────────────────────────────────────────────


class UniqueNamePickerTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_base_when_free(self):
        from app.database import async_session_factory

        async with async_session_factory() as db:
            name = await _pick_unique_name(db, "Phase7b Unique Pick Free Name")
        self.assertEqual(name, "Phase7b Unique Pick Free Name")


if __name__ == "__main__":
    unittest.main()
