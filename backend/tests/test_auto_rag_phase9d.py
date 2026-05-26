"""Tests for Phase 9d — auto-RAG default-on heuristic + target
profile + comparison endpoint + Coach Mode eval-stage nudge.

Covers:
  * ``_decide_auto_rag_default`` truth table (no recipe / non-RAG /
    qa-sft → on).
  * ``create_experiment`` integration: explicit False respected,
    explicit True respected, unset+qa-sft auto-defaults, unset+
    classification no-default. Mirrors Phase 6d's curriculum-default
    test pattern.
  * ``qa_with_auto_rag`` target profile registered + visible via the
    target_profile_service.
  * ``GET /api/projects/{id}/auto-rag/comparison`` endpoint:
    - 404 unknown project / no cached comparison
    - 400 no recipe / non-RAG-eligible recipe
    - 200 returns the cached payload when present
  * Coach Mode ``_auto_rag_eval_nudge`` truth table — fires only when
    recipe=qa-sft + pass_rate<0.5 + auto_rag.enabled=False; skips
    otherwise. Body cites the +146% Phase 9c lift.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_TEST_DATA_DIR = Path(__file__).resolve().parent / "auto_rag_phase9d_data"
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")
os.environ["DATA_DIR"] = _TEST_DATA_DIR.as_posix()

from app.config import settings  # noqa: E402
from app.services.training_service import _decide_auto_rag_default  # noqa: E402


def _project_stub(recipe_id: str | None):
    p = MagicMock()
    p.selected_recipe = {"recipe_id": recipe_id} if recipe_id else None
    return p


def _clear_tree(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()


# ─────────────────────────────────────────────────────────────────────
# _decide_auto_rag_default
# ─────────────────────────────────────────────────────────────────────


class DecideAutoRagDefaultTests(unittest.TestCase):
    def test_no_recipe_returns_no_default(self):
        decision = _decide_auto_rag_default(project_obj=_project_stub(None))
        self.assertFalse(decision["should_default_on"])
        self.assertEqual(decision["reason"], "no_recipe_selected")

    def test_non_rag_recipe_returns_no_default(self):
        for recipe in ("classification", "span-extraction", "summarization",
                       "code-review", "generic-sft"):
            with self.subTest(recipe=recipe):
                decision = _decide_auto_rag_default(project_obj=_project_stub(recipe))
                self.assertFalse(decision["should_default_on"])
                self.assertIn(f"recipe_has_no_auto_rag:{recipe}", decision["reason"])

    def test_qa_sft_recipe_defaults_on(self):
        decision = _decide_auto_rag_default(project_obj=_project_stub("qa-sft"))
        self.assertTrue(decision["should_default_on"])
        self.assertIn("rag_eligible_recipe:qa-sft", decision["reason"])


# ─────────────────────────────────────────────────────────────────────
# create_experiment integration (mocks DB, no FastAPI)
# ─────────────────────────────────────────────────────────────────────


class CreateExperimentAutoRagDefaultTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        _clear_tree(_TEST_DATA_DIR)
        _TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    async def _create(self, *, project_id: int, recipe_id: str | None, config: dict) -> tuple[dict, bool]:
        from app.models.experiment import TrainingMode
        from app.services import training_service

        project = _project_stub(recipe_id)

        async def _fake_execute(_stmt):
            r = MagicMock()
            r.scalar_one_or_none = MagicMock(return_value=project)
            return r

        async def _flush():
            return None

        async def _refresh(_obj):
            return None

        captured: dict[str, object] = {}

        def _add(exp):
            captured["exp"] = exp
            if exp.id is None:
                exp.id = project_id * 10

        db = MagicMock()
        db.execute = AsyncMock(side_effect=_fake_execute)
        db.add = MagicMock(side_effect=_add)
        db.flush = AsyncMock(side_effect=_flush)
        db.refresh = AsyncMock(side_effect=_refresh)

        with patch.object(
            training_service,
            "evaluate_training_base_model_compatibility",
            return_value={"ok": True, "errors": []},
        ):
            exp = await training_service.create_experiment(
                db=db,
                project_id=project_id,
                name="t",
                base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                config=dict(config),
                training_mode=TrainingMode.SFT,
            )
        return exp.config, "_auto_rag_auto_defaulted" in (exp.config or {})

    async def test_explicit_false_respected_on_qa_sft(self):
        cfg, auto = await self._create(
            project_id=11001, recipe_id="qa-sft",
            config={"auto_rag": {"enabled": False}},
        )
        self.assertEqual(cfg["auto_rag"], {"enabled": False})
        self.assertFalse(auto)

    async def test_explicit_true_respected(self):
        cfg, auto = await self._create(
            project_id=11002, recipe_id="qa-sft",
            config={"auto_rag": {"enabled": True}},
        )
        self.assertEqual(cfg["auto_rag"], {"enabled": True})
        self.assertFalse(auto)  # respected as-is, not "auto"

    async def test_unset_on_qa_sft_auto_defaults_on(self):
        cfg, auto = await self._create(
            project_id=11003, recipe_id="qa-sft", config={},
        )
        self.assertEqual(cfg["auto_rag"], {"enabled": True})
        self.assertTrue(auto)
        self.assertIn("rag_eligible_recipe:qa-sft", cfg["_auto_rag_auto_defaulted"])

    async def test_unset_on_classification_no_default(self):
        cfg, auto = await self._create(
            project_id=11004, recipe_id="classification", config={},
        )
        self.assertNotIn("auto_rag", cfg)
        self.assertFalse(auto)

    async def test_unset_no_recipe_no_default(self):
        cfg, auto = await self._create(
            project_id=11005, recipe_id=None, config={},
        )
        self.assertNotIn("auto_rag", cfg)
        self.assertFalse(auto)


# ─────────────────────────────────────────────────────────────────────
# qa_with_auto_rag target profile
# ─────────────────────────────────────────────────────────────────────


class QaWithAutoRagTargetProfileTests(unittest.TestCase):
    def test_qa_with_auto_rag_profile_registered(self):
        from app.services.target_profile_service import list_targets

        profiles = list_targets()
        # TargetProfile is a pydantic model — attribute access, not dict subscript.
        ids = {p.id for p in profiles}
        self.assertIn("qa_with_auto_rag", ids)

    def test_qa_with_auto_rag_profile_metadata(self):
        from app.services.target_profile_service import list_targets

        profiles = {p.id: p for p in list_targets()}
        prof = profiles["qa_with_auto_rag"]
        self.assertEqual(prof.name, "QA with auto-RAG")
        self.assertEqual(prof.device_class, "server")
        self.assertIn("retrieval at inference", prof.description.lower())
        # vLLM-compatible serve target.
        self.assertEqual(prof.inference_runner_default, "runner.vllm")


# ─────────────────────────────────────────────────────────────────────
# Coach Mode eval-stage nudge
# ─────────────────────────────────────────────────────────────────────


class AutoRagEvalNudgeTests(unittest.TestCase):
    def setUp(self):
        from app.services.coach_service import _auto_rag_eval_nudge
        self.fn = _auto_rag_eval_nudge

    def test_fires_when_qa_sft_struggling_and_auto_rag_off(self):
        nudge = self.fn(
            project_id=12001,
            recipe_id="qa-sft",
            pass_rate=0.42,
            latest_experiment_config={"auto_rag": {"enabled": False}},
        )
        self.assertIsNotNone(nudge)
        self.assertEqual(nudge["severity"], "info")
        self.assertEqual(nudge["action"]["kind"], "navigate")
        self.assertEqual(nudge["action"]["params"]["target"], "training-config")
        # Body cites Phase 9c numbers explicitly so the recommendation
        # is auditable, not handwaving.
        self.assertIn("+146%", nudge["body"])
        self.assertIn("Phase 9c", nudge["body"])
        # Context carries the lift + ab run date for downstream surfaces.
        self.assertEqual(nudge["context"]["phase_9c_lift_pct"], 146.49)

    def test_no_nudge_when_recipe_not_qa_sft(self):
        for recipe in ("classification", "span-extraction", "generic-sft"):
            with self.subTest(recipe=recipe):
                self.assertIsNone(self.fn(
                    project_id=12002, recipe_id=recipe, pass_rate=0.40,
                    latest_experiment_config={"auto_rag": {"enabled": False}},
                ))

    def test_no_nudge_when_pass_rate_above_threshold(self):
        # 0.5+ pass rate → model isn't meaningfully struggling.
        for pr in (0.50, 0.65, 0.90):
            with self.subTest(pass_rate=pr):
                self.assertIsNone(self.fn(
                    project_id=12003, recipe_id="qa-sft", pass_rate=pr,
                    latest_experiment_config={"auto_rag": {"enabled": False}},
                ))

    def test_no_nudge_when_auto_rag_already_on(self):
        """No double-coaching — if the latest experiment already had
        auto_rag enabled, the nudge would be redundant."""
        for cfg in (
            {"auto_rag": {"enabled": True}},
            {"auto_rag": True},  # legacy bool shape, also supported
        ):
            with self.subTest(cfg=cfg):
                self.assertIsNone(self.fn(
                    project_id=12004, recipe_id="qa-sft", pass_rate=0.30,
                    latest_experiment_config=cfg,
                ))

    def test_no_nudge_when_no_recipe(self):
        self.assertIsNone(self.fn(
            project_id=12005, recipe_id=None, pass_rate=0.30,
            latest_experiment_config={},
        ))

    def test_no_nudge_when_pass_rate_none(self):
        self.assertIsNone(self.fn(
            project_id=12006, recipe_id="qa-sft", pass_rate=None,
            latest_experiment_config={"auto_rag": {"enabled": False}},
        ))

    def test_handles_missing_experiment_config_as_auto_rag_off(self):
        """Missing config = treat as auto_rag off (safe default —
        worst case is a false-positive nudge that the user dismisses)."""
        self.assertIsNotNone(self.fn(
            project_id=12007, recipe_id="qa-sft", pass_rate=0.40,
            latest_experiment_config=None,
        ))


# ─────────────────────────────────────────────────────────────────────
# /auto-rag/comparison endpoint
# ─────────────────────────────────────────────────────────────────────


_API_TEST_DB_PATH = Path(__file__).resolve().parent / "auto_rag_phase9d_api.db"


class AutoRagComparisonApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_API_TEST_DB_PATH.as_posix()}"
        os.environ["DEBUG"] = "false"
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if _API_TEST_DB_PATH.exists():
            _API_TEST_DB_PATH.unlink()
        _clear_tree(_TEST_DATA_DIR)
        _TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        from fastapi.testclient import TestClient
        from app.main import app
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if _API_TEST_DB_PATH.exists():
            _API_TEST_DB_PATH.unlink()
        _clear_tree(_TEST_DATA_DIR)

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def _seed_cached_comparison(self, project_id: int, payload: dict) -> Path:
        path = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / "comparison.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_unknown_project_returns_404(self):
        resp = self.client.get("/api/projects/99999/auto-rag/comparison")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_classification_project_returns_400_unsupported_recipe(self):
        project = self._instantiate_template("ticket-router", "AutoRAG Comparison Wrong Recipe")
        resp = self.client.get(f"/api/projects/{project['id']}/auto-rag/comparison")
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("classification", resp.text)

    def test_qa_sft_without_cached_comparison_returns_404_with_command(self):
        project = self._instantiate_template("policy-qa-style", "AutoRAG Comparison Not Yet Run")
        resp = self.client.get(f"/api/projects/{project['id']}/auto-rag/comparison")
        self.assertEqual(resp.status_code, 404, resp.text)
        # Detail includes the exact CLI command — frontend renders it.
        self.assertIn("--project", resp.text)
        self.assertIn(str(project["id"]), resp.text)

    def test_qa_sft_with_cached_comparison_returns_payload(self):
        project = self._instantiate_template("policy-qa-style", "AutoRAG Comparison Cached")
        cached_payload = {
            "project_id": project["id"],
            "cached_at": "2026-05-25T12:00:00+00:00",
            "experiment_id": 42,
            "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "model_dir": "/tmp/model",
            "summary": {
                "off_mean_f1": 0.10,
                "on_mean_f1": 0.30,
                "absolute_lift": 0.20,
                "relative_lift_pct": 200.0,
                "n_val_rows": 28,
                "rag_k": 3,
                "phase_9c_reference_lift_pct": 146.49,
            },
            "rows": [
                {
                    "question": "How many PTO days?",
                    "reference": "Up to 5 days.",
                    "without_rag": {"generated": "weak answer", "f1": 0.1},
                    "with_rag": {"generated": "Up to 5 days.", "f1": 1.0, "retrieved_row_count": 3},
                },
            ],
        }
        self._seed_cached_comparison(project["id"], cached_payload)
        resp = self.client.get(f"/api/projects/{project['id']}/auto-rag/comparison")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["recipe_id"], "qa-sft")
        self.assertEqual(body["summary"]["off_mean_f1"], 0.10)
        self.assertEqual(body["summary"]["on_mean_f1"], 0.30)
        self.assertEqual(len(body["rows"]), 1)
        self.assertEqual(body["rows"][0]["question"], "How many PTO days?")


if __name__ == "__main__":
    unittest.main()
