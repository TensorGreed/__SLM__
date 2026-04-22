"""Phase 55 tests — strict-mode enforcement across autopilot auto-repairs (P5).

Covers each of the four auto-repair paths in the orchestrator:

1. ``intent_rewrite`` — autopilot suggestion refused; user-explicit still applies.
2. ``dataset_auto_prepare`` — refused; operator must prepare data first.
3. ``target_fallback`` — refused; operator must pick a compatible target.
4. ``profile_autotune`` — refused; operator must pick a runnable profile.

Plus sanity checks that strict mode is a no-op when no repair would fire, that
the decision log persists the blocker entries via P1, and that the existing
simulation-fallback strict check still composes cleanly.
"""

from __future__ import annotations

import contextlib
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

TEST_DB_PATH = Path(__file__).resolve().parent / "phase55_strict_mode_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase55_strict_mode_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"
os.environ["ALLOW_SYNTHETIC_DEMO_FALLBACK"] = "true"

from fastapi.testclient import TestClient

from app.api.training import (
    STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE,
    STRICT_MODE_REFUSED_INTENT_REWRITE,
    STRICT_MODE_REFUSED_PROFILE_AUTOTUNE,
    STRICT_MODE_REFUSED_TARGET_FALLBACK,
)
from app.config import settings
from app.main import app


class Phase55StrictModeEnforcementTests(unittest.TestCase):
    @classmethod
    def _cleanup_test_artifacts(cls):
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
        settings.TRAINING_BACKEND = "simulate"
        settings.ALLOW_SIMULATED_TRAINING = True
        settings.STRICT_EXECUTION_MODE = False
        settings.ensure_dirs()

        cls._cleanup_test_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_test_artifacts()

    # -- helpers ----------------------------------------------------------

    def _create_project(self, name: str) -> int:
        unique_name = f"{name}-{uuid.uuid4().hex[:8]}"
        resp = self.client.post(
            "/api/projects",
            json={"name": unique_name, "description": "phase55 strict-mode"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_prepared_rows(self, project_id: int, row_count: int = 32) -> None:
        prepared_dir = TEST_DATA_DIR / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        with open(prepared_dir / "train.jsonl", "w", encoding="utf-8") as handle:
            for idx in range(row_count):
                handle.write(
                    '{"text": "User: ticket %d\\nAssistant: reply %d"}\n' % (idx, idx)
                )

    def _upload_raw_text(self, project_id: int, filename: str, content: str) -> int:
        resp = self.client.post(
            f"/api/projects/{project_id}/ingestion/upload",
            files={"file": (filename, content.encode("utf-8"), "text/plain")},
            data={"source": "upload", "sensitivity": "internal", "license_info": ""},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _orchestrate_dry_run(self, project_id: int, strict: bool, **overrides) -> dict:
        payload = {
            "intent": "Build a concise support assistant.",
            "target_profile_id": "edge_gpu",
            "dry_run": True,
            "auto_prepare_data": False,
        }
        payload.update(overrides)
        patches = [patch.object(settings, "STRICT_EXECUTION_MODE", bool(strict))]
        if strict:
            # The built-in simulate runtime refuses to plan in strict mode
            # (`StrictExecutionError` from `_validate_builtin_simulate`), which
            # short-circuits initial planning before any repair block runs.
            # For P5 we care about repair-block behavior, so switch to the
            # external runtime (validator only checks TRAINING_EXTERNAL_CMD).
            patches.extend(
                [
                    patch.object(settings, "TRAINING_BACKEND", "external"),
                    patch.object(settings, "TRAINING_EXTERNAL_CMD", "/bin/echo dummy"),
                ]
            )
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            resp = self.client.post(
                f"/api/projects/{project_id}/training/autopilot/v2/orchestrate",
                json=payload,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        return dict(resp.json() or {})

    def _assert_strict_refusal(
        self,
        payload: dict,
        *,
        reason_code: str,
        stage: str,
    ) -> dict:
        guardrails = dict(payload.get("guardrails") or {})
        reason_codes = list(guardrails.get("reason_codes") or [])
        self.assertIn(reason_code, reason_codes, guardrails)
        self.assertFalse(bool(guardrails.get("can_run", False)), guardrails)
        decisions = [
            row
            for row in list(payload.get("decision_log") or [])
            if isinstance(row, dict)
        ]
        matched = [
            row
            for row in decisions
            if row.get("step") == stage and row.get("status") == "blocked"
        ]
        self.assertGreaterEqual(len(matched), 1, decisions)
        metadata = dict(matched[-1].get("metadata") or {})
        self.assertEqual(metadata.get("reason_code"), reason_code, matched[-1])
        return matched[-1]

    # -- intent rewrite ---------------------------------------------------

    def test_strict_mode_refuses_autopilot_intent_rewrite(self):
        project_id = self._create_project("phase55-intent-auto")
        self._seed_prepared_rows(project_id)

        # Deliberately vague intent so autopilot offers a rewrite suggestion.
        payload = self._orchestrate_dry_run(
            project_id,
            strict=True,
            intent="help me",
            auto_apply_rewrite=True,
        )
        repair = dict(dict(payload.get("repairs") or {}).get("intent_rewrite") or {})
        self.assertTrue(bool(repair.get("strict_mode_blocked")), repair)
        self.assertFalse(bool(repair.get("applied")), repair)
        self._assert_strict_refusal(
            payload,
            reason_code=STRICT_MODE_REFUSED_INTENT_REWRITE,
            stage="intent_rewrite",
        )
        # Effective intent did NOT change.
        self.assertEqual(str(payload.get("intent") or ""), "help me")

    def test_strict_mode_allows_user_supplied_intent_rewrite(self):
        project_id = self._create_project("phase55-intent-user")
        self._seed_prepared_rows(project_id)

        payload = self._orchestrate_dry_run(
            project_id,
            strict=True,
            intent="help me",
            intent_rewrite="Train a concise support assistant with JSON output.",
            auto_apply_rewrite=False,
        )
        repair = dict(dict(payload.get("repairs") or {}).get("intent_rewrite") or {})
        self.assertTrue(bool(repair.get("applied")), repair)
        self.assertFalse(bool(repair.get("strict_mode_blocked", False)), repair)
        guardrails = dict(payload.get("guardrails") or {})
        reason_codes = list(guardrails.get("reason_codes") or [])
        self.assertNotIn(STRICT_MODE_REFUSED_INTENT_REWRITE, reason_codes, reason_codes)

    # -- dataset auto-prepare ---------------------------------------------

    def test_strict_mode_refuses_dataset_auto_prepare(self):
        project_id = self._create_project("phase55-data")
        # Upload raw docs (pending) so autopilot would normally auto-prepare.
        raw_content = "\n".join(
            f"Ticket {idx}: customer cannot sign in. Include reset steps."
            for idx in range(36)
        )
        self._upload_raw_text(project_id, "support_pending.txt", raw_content)

        payload = self._orchestrate_dry_run(
            project_id,
            strict=True,
            intent="Train a support assistant",
            auto_prepare_data=True,
        )
        repair = dict(dict(payload.get("repairs") or {}).get("dataset_auto_prepare") or {})
        self.assertTrue(bool(repair.get("strict_mode_blocked")), repair)
        self.assertFalse(bool(repair.get("applied")), repair)
        self._assert_strict_refusal(
            payload,
            reason_code=STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE,
            stage="dataset_auto_prepare",
        )

    def test_non_strict_dataset_auto_prepare_still_runs_normally(self):
        project_id = self._create_project("phase55-data-nonstrict")
        raw_content = "\n".join(
            f"Ticket {idx}: help needed." for idx in range(30)
        )
        self._upload_raw_text(project_id, "support_pending.txt", raw_content)

        # With strict_mode OFF, the repair block should attempt auto-prepare
        # and surface as `applied` (or at least not as a strict-blocked entry).
        payload = self._orchestrate_dry_run(
            project_id,
            strict=False,
            intent="Train a support assistant",
            auto_prepare_data=True,
        )
        guardrails = dict(payload.get("guardrails") or {})
        reason_codes = list(guardrails.get("reason_codes") or [])
        self.assertNotIn(
            STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE, reason_codes, reason_codes
        )
        repair = dict(dict(payload.get("repairs") or {}).get("dataset_auto_prepare") or {})
        self.assertFalse(bool(repair.get("strict_mode_blocked", False)), repair)

    # -- target fallback --------------------------------------------------

    def test_strict_mode_refuses_target_fallback(self):
        project_id = self._create_project("phase55-target")
        self._seed_prepared_rows(project_id)

        # `mobile_cpu` typically triggers target-incompatibility in this harness,
        # producing the precondition for the target_fallback repair.
        payload = self._orchestrate_dry_run(
            project_id,
            strict=True,
            intent="Train a support assistant",
            target_profile_id="mobile_cpu",
            allow_target_fallback=True,
        )
        repair = dict(dict(payload.get("repairs") or {}).get("target_fallback") or {})
        # Either the strict refusal fired (ideal) or target incompat short-circuited
        # earlier; accept both as long as target was NOT silently swapped.
        if repair.get("strict_mode_blocked"):
            self.assertFalse(bool(repair.get("applied")), repair)
            self._assert_strict_refusal(
                payload,
                reason_code=STRICT_MODE_REFUSED_TARGET_FALLBACK,
                stage="target_fallback",
            )
        else:
            self.assertFalse(bool(repair.get("applied", False)), repair)
        guardrails = dict(payload.get("guardrails") or {})
        self.assertFalse(bool(guardrails.get("can_run", False)), guardrails)

    # -- profile autotune -------------------------------------------------

    def test_strict_mode_refuses_profile_autotune(self):
        project_id = self._create_project("phase55-profile")
        self._seed_prepared_rows(project_id)

        # Use a patch to force profile_autotune to fire: mark the requested
        # profile's preflight as failed, with a runnable alternative available.
        from app.api import training as training_module

        original_build = training_module._build_newbie_autopilot_plan_v2

        async def build_with_failed_preflight(*, db, project_id, req):
            resp = await original_build(db=db, project_id=project_id, req=req)
            plans = [dict(p) for p in list(resp.plans or [])]
            if len(plans) >= 2:
                requested = training_module.normalize_training_plan_profile(
                    getattr(req, "plan_profile", "balanced")
                ) or "balanced"
                for idx, plan in enumerate(plans):
                    profile = training_module.normalize_training_plan_profile(
                        plan.get("profile")
                    ) or ""
                    pre = dict(plan.get("preflight") or {})
                    if profile == requested:
                        pre["ok"] = False
                    else:
                        pre["ok"] = True
                    plans[idx]["preflight"] = pre
                resp.plans = plans
            return resp

        with patch.object(
            training_module,
            "_build_newbie_autopilot_plan_v2",
            side_effect=build_with_failed_preflight,
        ):
            payload = self._orchestrate_dry_run(
                project_id,
                strict=True,
                intent="Train a support assistant",
                plan_profile="balanced",
                allow_profile_autotune=True,
            )

        repair = dict(dict(payload.get("repairs") or {}).get("profile_autotune") or {})
        self.assertTrue(bool(repair.get("strict_mode_blocked")), repair)
        self.assertFalse(bool(repair.get("applied")), repair)
        self._assert_strict_refusal(
            payload,
            reason_code=STRICT_MODE_REFUSED_PROFILE_AUTOTUNE,
            stage="profile_autotune",
        )

    # -- no-op + persistence ---------------------------------------------

    def test_strict_mode_is_no_op_when_no_repair_would_fire(self):
        project_id = self._create_project("phase55-clean")
        self._seed_prepared_rows(project_id)

        # Setup so none of the four auto-repairs would fire:
        # - auto_apply_rewrite=False → no autopilot intent rewrite proposed
        # - auto_prepare_data=False → no dataset auto-prepare
        # - allow_target_fallback=False → no target fallback
        # - allow_profile_autotune=False → no profile autotune
        payload = self._orchestrate_dry_run(
            project_id,
            strict=True,
            intent="Train a concise support assistant with JSON output.",
            auto_prepare_data=False,
            auto_apply_rewrite=False,
            allow_target_fallback=False,
            allow_profile_autotune=False,
        )
        guardrails = dict(payload.get("guardrails") or {})
        reason_codes = list(guardrails.get("reason_codes") or [])
        for code in (
            STRICT_MODE_REFUSED_INTENT_REWRITE,
            STRICT_MODE_REFUSED_DATASET_AUTO_PREPARE,
            STRICT_MODE_REFUSED_TARGET_FALLBACK,
            STRICT_MODE_REFUSED_PROFILE_AUTOTUNE,
        ):
            self.assertNotIn(code, reason_codes, reason_codes)

    def test_strict_refusal_persists_to_autopilot_decisions_table(self):
        project_id = self._create_project("phase55-persist")
        self._seed_prepared_rows(project_id)

        payload = self._orchestrate_dry_run(
            project_id,
            strict=True,
            intent="help me",
            auto_apply_rewrite=True,
        )
        run_id = str(payload.get("run_id") or "")
        self.assertTrue(run_id)

        resp = self.client.get(
            "/api/autopilot/decisions",
            params={
                "run_id": run_id,
                "stage": "intent_rewrite",
                "status": "blocked",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        items = list(dict(resp.json() or {}).get("items") or [])
        self.assertGreaterEqual(len(items), 1, items)
        self.assertEqual(
            items[0].get("reason_code"), STRICT_MODE_REFUSED_INTENT_REWRITE
        )
        self.assertEqual(items[0].get("action"), "blocked")


if __name__ == "__main__":
    unittest.main()
