"""Training Config Gap patch engine — Coach-stage-2 phase 2.

Covers the preview → apply contract:
- /patch/preview returns the proposed before/after without mutating.
- /patch/apply persists into ``runtime_config["training_config_overrides"]``.
- A re-scan after Apply flips the affected signal to ``ok``.
- /overrides reflects the persisted state.
- Bad inputs are rejected at the right layer (400 vs 404).
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "tcg_patch.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "tcg_patch_data"

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


def _signal_by_id(body: dict, signal_id: str) -> dict | None:
    for group in body["groups"]:
        for sig in group["signals"]:
            if sig["id"] == signal_id:
                return sig
    return None


class TrainingConfigGapPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _make_project(
        self,
        *,
        recipe_id: str,
        labelled_rows: int,
        base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    ) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"tcg-patch-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])

        async def _set():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, pid)
                proj.selected_recipe = {"recipe_id": recipe_id}
                proj.base_model_name = base_model
                if labelled_rows > 0:
                    db.add(Dataset(
                        project_id=pid,
                        name="cleaned",
                        dataset_type=DatasetType.CLEANED,
                        file_path="",
                        record_count=labelled_rows,
                    ))
                await db.commit()
        asyncio.run(_set())
        return pid

    # ── Tests ───────────────────────────────────────────────────────

    def test_preview_returns_before_after_without_mutating_runtime_config(self):
        # Tiny dataset → eval cadence signal fires with a recommendation.
        pid = self._make_project(
            recipe_id="recipe.classification.sentiment",
            labelled_rows=20,
        )
        # Sanity: signal carries apply_patch_kind.
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.eval_cadence_too_sparse")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["apply_patch_kind"], "eval_steps_recommend")

        # Preview.
        resp = self.client.post(
            f"/api/projects/{pid}/training-config-gaps/patch/preview",
            json={"signal_id": "training_config.eval_cadence_too_sparse"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        preview = resp.json()
        self.assertEqual(preview["patch_kind"], "eval_steps_recommend")
        self.assertIn("eval_steps", preview["patch"])
        self.assertEqual(preview["before"]["eval_steps"], 100)
        self.assertEqual(
            preview["after"]["eval_steps"], preview["patch"]["eval_steps"]
        )
        self.assertTrue(preview["safe_to_apply"])

        # Confirm runtime_config wasn't touched.
        ov = self.client.get(
            f"/api/projects/{pid}/training-config-gaps/overrides"
        ).json()
        self.assertEqual(ov["overrides"], {})

    def test_apply_persists_override_and_signal_flips_to_ok(self):
        pid = self._make_project(
            recipe_id="recipe.classification.sentiment",
            labelled_rows=20,
        )
        # Apply the eval-cadence patch.
        resp = self.client.post(
            f"/api/projects/{pid}/training-config-gaps/patch/apply",
            json={"signal_id": "training_config.eval_cadence_too_sparse"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        result = resp.json()
        self.assertTrue(result["applied"])
        new_eval = int(result["patch"]["eval_steps"])
        # Floor at 1 (the patch closes the gap by setting eval_steps
        # ≤ total_steps / EVAL_OBS_WARN; very short runs land at 1).
        self.assertGreaterEqual(new_eval, 1)
        self.assertLess(new_eval, 100)  # < the default 100
        self.assertEqual(result["overrides_after"]["eval_steps"], new_eval)

        # Persisted: /overrides returns the same value.
        ov = self.client.get(
            f"/api/projects/{pid}/training-config-gaps/overrides"
        ).json()
        self.assertEqual(ov["overrides"]["eval_steps"], new_eval)

        # Re-scan: the signal flips to ok (the recommended cadence is
        # always tuned to yield ≥ 3 observations, which clears the warn
        # threshold).
        body = self.client.get(
            f"/api/projects/{pid}/training-config-gaps"
        ).json()
        sig = _signal_by_id(body, "training_config.eval_cadence_too_sparse")
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig["severity"], "ok")

    def test_apply_is_idempotent(self):
        # Applying the same patch twice should land the same value
        # both times (idempotent). After the first apply the signal
        # is ``ok``, so the second apply's preview would normally
        # reject — we re-apply by directly hitting the apply endpoint,
        # which re-computes the patch off the persisted override.
        # The contract: same signal_id, same final state.
        pid = self._make_project(
            recipe_id="recipe.classification.sentiment",
            labelled_rows=20,
        )
        first = self.client.post(
            f"/api/projects/{pid}/training-config-gaps/patch/apply",
            json={"signal_id": "training_config.eval_cadence_too_sparse"},
        ).json()
        # After apply, the signal is ok → no apply_patch_kind, so a
        # re-apply rejects. That's the desired behavior: it tells the
        # caller "nothing to do." Confirm the state hasn't changed.
        retry = self.client.post(
            f"/api/projects/{pid}/training-config-gaps/patch/apply",
            json={"signal_id": "training_config.eval_cadence_too_sparse"},
        )
        self.assertEqual(retry.status_code, 400)
        ov = self.client.get(
            f"/api/projects/{pid}/training-config-gaps/overrides"
        ).json()
        self.assertEqual(
            ov["overrides"]["eval_steps"], first["patch"]["eval_steps"]
        )

    def test_apply_preserves_unrelated_runtime_config_keys(self):
        # Pre-seed a non-overrides key in runtime_config (e.g. the
        # rag_first flag the reroute-to-RAG feature writes) and confirm
        # the patch apply doesn't stomp it. This is the property that
        # makes the patch engine safe to compose with other runtime
        # surfaces.
        pid = self._make_project(
            recipe_id="recipe.classification.sentiment",
            labelled_rows=20,
        )

        async def _seed():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, pid)
                proj.runtime_config = {
                    "rag_first": True,
                    "auto_rag": {"enabled": False},
                }
                await db.commit()
        asyncio.run(_seed())

        resp = self.client.post(
            f"/api/projects/{pid}/training-config-gaps/patch/apply",
            json={"signal_id": "training_config.eval_cadence_too_sparse"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        # Reload the project and confirm both old + new keys present.
        async def _read():
            async with async_session_factory() as db:
                from app.models.project import Project
                proj = await db.get(Project, pid)
                return dict(proj.runtime_config or {})
        runtime = asyncio.run(_read())
        self.assertEqual(runtime.get("rag_first"), True)
        self.assertEqual(runtime.get("auto_rag", {}).get("enabled"), False)
        self.assertIn("training_config_overrides", runtime)
        self.assertIn("eval_steps", runtime["training_config_overrides"])

    def test_preview_rejects_signal_without_apply_patch_kind(self):
        # Base-model-undersized signal exists but has no patch.
        pid = self._make_project(
            recipe_id="recipe.structured_extraction.entity",
            labelled_rows=1000,
        )
        resp = self.client.post(
            f"/api/projects/{pid}/training-config-gaps/patch/preview",
            json={"signal_id": "training_config.base_model_undersized"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("no one-click patch", resp.text.lower())

    def test_preview_rejects_unknown_signal_id(self):
        pid = self._make_project(
            recipe_id="recipe.classification.sentiment",
            labelled_rows=20,
        )
        resp = self.client.post(
            f"/api/projects/{pid}/training-config-gaps/patch/preview",
            json={"signal_id": "training_config.does_not_exist"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_preview_404s_for_missing_project(self):
        resp = self.client.post(
            "/api/projects/9999999/training-config-gaps/patch/preview",
            json={"signal_id": "training_config.eval_cadence_too_sparse"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_overrides_endpoint_404s_for_missing_project(self):
        resp = self.client.get(
            "/api/projects/9999999/training-config-gaps/overrides",
        )
        self.assertEqual(resp.status_code, 404)
