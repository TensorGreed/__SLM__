"""Phase 62 — training preflight surfaces target-profile incompatibility early.

Before this phase, a base-model ↔ target-profile mismatch was only caught at
``POST /experiments/{id}/start`` time. The user would fully configure a run,
click "Run Experiment", and only then see "Base model X is not compatible
with target profile Y". Moving the check into the preflight endpoints (which
are already called by the frontend before submission) turns the same failure
into an upfront, actionable warning.
"""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


TEST_DB_PATH = Path(__file__).resolve().parent / "phase62_preflight_target_compat.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase62_preflight_target_compat_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

import asyncio

from fastapi.testclient import TestClient

from app.api.training import _check_target_profile_compatibility
from app.config import settings
from app.database import async_session_factory
from app.main import app


_INCOMPATIBLE_COMPAT_RESULT = {
    "compatible": False,
    "reasons": [
        "Model size (2.7B) exceeds target limit (2.0B).",
        "Estimated minimum VRAM (6 GB) exceeds target baseline (4 GB) by 2 GB.",
    ],
    "warnings": ["bitsandbytes is not installed; optimizer will fall back to adamw_torch."],
    "target": {"id": "edge_gpu"},
    "model_metadata": {"parameters_billions": 2.7, "estimated_min_vram_gb": 6.0},
    "vram_check": {"status": "blocked", "gap_gb": 2.0},
}


_COMPATIBLE_COMPAT_RESULT = {
    "compatible": True,
    "reasons": [],
    "warnings": [],
    "target": {"id": "vllm_server"},
    "model_metadata": {"parameters_billions": 2.7, "estimated_min_vram_gb": 6.0},
    "vram_check": {"status": "ok"},
}


class Phase62PreflightTargetCompatTests(unittest.TestCase):
    @classmethod
    def _cleanup_artifacts(cls):
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        for suffix in ("", "-shm", "-wal"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_artifacts()

    def _create_project(self, *, target_profile_id: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"phase62-{uuid.uuid4().hex[:8]}",
                "description": "phase62 preflight compat",
                "target_profile_id": target_profile_id,
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    # -- tests --------------------------------------------------------------

    def test_preflight_flips_to_not_ok_when_model_incompatible_with_target(self):
        project_id = self._create_project(target_profile_id="edge_gpu")
        payload = {
            "config": {
                "base_model": "microsoft/phi-2",
                "task_type": "instruction_sft",
                "training_mode": "sft",
            }
        }
        with patch(
            "app.api.training.check_compatibility",
            return_value=_INCOMPATIBLE_COMPAT_RESULT,
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/training/experiments/preflight",
                json=payload,
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        preflight = body["preflight"]
        self.assertFalse(preflight["ok"])
        joined_errors = " ".join(preflight["errors"])
        self.assertIn("not compatible with target profile", joined_errors)
        self.assertIn("edge_gpu", joined_errors)

        meta = preflight.get("target_profile_compatibility")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["target_profile_id"], "edge_gpu")
        self.assertEqual(meta["error_code"], "TARGET_INCOMPATIBLE")
        self.assertIn(
            "Model size (2.7B) exceeds target limit (2.0B).",
            meta["reasons"],
        )

    def test_preflight_stays_ok_when_model_is_compatible(self):
        project_id = self._create_project(target_profile_id="vllm_server")
        payload = {
            "config": {
                "base_model": "microsoft/phi-2",
                "task_type": "instruction_sft",
                "training_mode": "sft",
            }
        }
        with patch(
            "app.api.training.check_compatibility",
            return_value=_COMPATIBLE_COMPAT_RESULT,
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/training/experiments/preflight",
                json=payload,
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        preflight = resp.json()["preflight"]
        # May still have other errors/warnings from the regular preflight,
        # but the compatibility metadata block must not be present.
        self.assertNotIn("target_profile_compatibility", preflight)

    def test_preflight_plan_also_blocks_on_incompatible_target(self):
        project_id = self._create_project(target_profile_id="edge_gpu")
        payload = {
            "config": {
                "base_model": "microsoft/phi-2",
                "task_type": "instruction_sft",
                "training_mode": "sft",
            }
        }
        with patch(
            "app.api.training.check_compatibility",
            return_value=_INCOMPATIBLE_COMPAT_RESULT,
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/training/experiments/preflight/plan",
                json=payload,
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        plan_preflight = resp.json()["plan"]["preflight"]
        self.assertFalse(plan_preflight["ok"])
        self.assertIn(
            "target_profile_compatibility", plan_preflight,
            "plan endpoint must also surface the compatibility metadata",
        )
        self.assertEqual(
            plan_preflight["target_profile_compatibility"]["error_code"],
            "TARGET_INCOMPATIBLE",
        )

    def test_error_message_includes_compatible_alternatives(self):
        project_id = self._create_project(target_profile_id="edge_gpu")
        payload = {
            "config": {
                "base_model": "meta-llama/Llama-3.2-3B",
                "task_type": "instruction_sft",
                "training_mode": "sft",
            }
        }
        fake_suggestions = [
            {
                "source_ref": "Qwen/Qwen2.5-1.5B-Instruct",
                "display_name": "Qwen 1.5B Instruct",
                "model_key": "catalog:qwen:q15",
                "params_estimate_b": 1.5,
                "compatibility_score": 0.92,
                "compatible": True,
            },
            {
                "source_ref": "google/gemma-2-2b-it",
                "display_name": "Gemma 2 2B Instruct",
                "model_key": "catalog:gemma:g2-2b",
                "params_estimate_b": 2.0,
                "compatibility_score": 0.85,
                "compatible": True,
            },
        ]
        with patch(
            "app.api.training.check_compatibility",
            return_value=_INCOMPATIBLE_COMPAT_RESULT,
        ), patch(
            "app.services.base_model_registry_service.recommend_models_for_project",
            return_value=fake_suggestions,
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/training/experiments/preflight",
                json=payload,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        preflight = resp.json()["preflight"]
        joined = " ".join(preflight["errors"])
        # User-facing string carries the model ids so a stuck user unblocks
        # without opening another tab.
        self.assertIn("Compatible alternatives:", joined)
        self.assertIn("Qwen/Qwen2.5-1.5B-Instruct", joined)
        self.assertIn("google/gemma-2-2b-it", joined)

        meta = preflight["target_profile_compatibility"]
        self.assertEqual(len(meta["suggested_models"]), 2)
        self.assertEqual(meta["suggested_models"][0]["source_ref"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(meta["suggested_models"][0]["params_estimate_b"], 1.5)

    def test_suggestions_extract_from_nested_recommend_shape(self):
        # The real ``recommend_models_for_project`` returns rows shaped like
        # ``{"model": {...serialized record...}, "compatible": True, ...}`` —
        # NOT flat dicts. A previous iteration of the helper read
        # ``row["source_ref"]`` directly and silently dropped every real
        # suggestion, leaving users staring at a bare "not compatible"
        # message with no alternatives. Lock the real shape in.
        project_id = self._create_project(target_profile_id="edge_gpu")
        payload = {
            "config": {
                "base_model": "meta-llama/Llama-3.2-3B",
                "task_type": "instruction_sft",
                "training_mode": "sft",
            }
        }
        realistic_rows = [
            {
                "compatible": True,
                "compatibility_score": 0.92,
                "model": {
                    "source_ref": "Qwen/Qwen2.5-1.5B-Instruct",
                    "display_name": "Qwen 1.5B Instruct",
                    "model_key": "catalog:qwen:q15",
                    "params_estimate_b": 1.5,
                },
            },
        ]
        with patch(
            "app.api.training.check_compatibility",
            return_value=_INCOMPATIBLE_COMPAT_RESULT,
        ), patch(
            "app.services.base_model_registry_service.recommend_models_for_project",
            return_value=realistic_rows,
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/training/experiments/preflight",
                json=payload,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        preflight = resp.json()["preflight"]
        joined = " ".join(preflight["errors"])
        self.assertIn("Qwen/Qwen2.5-1.5B-Instruct", joined)
        meta = preflight["target_profile_compatibility"]
        self.assertEqual(len(meta["suggested_models"]), 1)
        self.assertEqual(
            meta["suggested_models"][0]["source_ref"],
            "Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertEqual(meta["suggested_models"][0]["params_estimate_b"], 1.5)

    def test_suggestions_are_gracefully_absent_when_registry_fails(self):
        project_id = self._create_project(target_profile_id="edge_gpu")
        payload = {
            "config": {
                "base_model": "meta-llama/Llama-3.2-3B",
                "task_type": "instruction_sft",
                "training_mode": "sft",
            }
        }
        with patch(
            "app.api.training.check_compatibility",
            return_value=_INCOMPATIBLE_COMPAT_RESULT,
        ), patch(
            "app.services.base_model_registry_service.recommend_models_for_project",
            side_effect=RuntimeError("registry unreachable"),
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/training/experiments/preflight",
                json=payload,
            )
        # Preflight must still return (non-500) even when the suggestion
        # lookup crashes — the core error must still be surfaced.
        self.assertEqual(resp.status_code, 200, resp.text)
        preflight = resp.json()["preflight"]
        self.assertFalse(preflight["ok"])
        joined = " ".join(preflight["errors"])
        self.assertIn("not compatible with target profile", joined)
        self.assertNotIn("Compatible alternatives:", joined)
        self.assertEqual(
            preflight["target_profile_compatibility"]["suggested_models"], []
        )

    def test_compat_helper_skips_lookup_when_base_model_is_blank(self):
        # Directly exercises the early-return branch of the helper — called
        # with an empty base_model, it must NOT touch the registry.
        project_id = self._create_project(target_profile_id="edge_gpu")

        async def _run():
            async with async_session_factory() as db:
                with patch(
                    "app.api.training.check_compatibility",
                    side_effect=AssertionError("helper should have returned early"),
                ):
                    return await _check_target_profile_compatibility(
                        db=db,
                        project_id=project_id,
                        base_model="   ",
                    )

        errors, meta = asyncio.run(_run())
        self.assertEqual(errors, [])
        self.assertIsNone(meta)


if __name__ == "__main__":
    unittest.main()
