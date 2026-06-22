"""Pipeline plan refinement — Phase 1 tests.

The load-bearing test is the PRIVACY invariant: the cloud-safe profile must
carry only aggregates — never ingested row text, gold answers, or label names.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(tempfile.gettempdir()) / f"brewslm-refine-{uuid.uuid4().hex[:8]}.db"
TEST_DATA_DIR = Path(tempfile.gettempdir()) / f"brewslm-refine-{uuid.uuid4().hex[:8]}"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

import asyncio  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


# Distinctive strings we'll assert never leak into the cloud-safe profile.
SECRET_TEXT = "ACME_PROPRIETARY_INVOICE_PAYLOAD_zzz"
SECRET_LABELS = ["billing_internal", "technical_internal", "refund_internal"]


class PipelineRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _seed_project(self) -> int:
        resp = self.client.post("/api/projects", json={"name": f"refine-{uuid.uuid4().hex[:6]}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        pid = int(resp.json()["id"])

        gold_path = settings.DATA_DIR / "projects" / str(pid) / "gold_dev.jsonl"
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            [{"text": f"{SECRET_TEXT} {i}", "label": SECRET_LABELS[0]} for i in range(6)]
            + [{"text": f"{SECRET_TEXT} t", "label": SECRET_LABELS[1]}]   # below floor
            + [{"text": f"{SECRET_TEXT} r", "label": SECRET_LABELS[2]}]   # below floor
        )
        gold_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        async def _configure():
            from app.database import async_session_factory
            from app.models.dataset import Dataset, DatasetType
            from app.models.project import Project
            async with async_session_factory() as db:
                project = await db.get(Project, pid)
                project.selected_recipe = {"recipe_id": "classification", "task_profile": "classification"}
                project.base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
                project.target_profile_id = "mobile_cpu"
                db.add(Dataset(
                    project_id=pid, name="Gold Dev", dataset_type=DatasetType.GOLD_DEV,
                    file_path=str(gold_path), record_count=len(rows),
                ))
                await db.commit()

        asyncio.run(_configure())
        return pid

    def test_refine_plan_returns_profile_and_plan_health(self):
        pid = self._seed_project()
        resp = self.client.get(f"/api/projects/{pid}/refine-plan")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()

        # Plan echoes the project config.
        self.assertEqual(body["plan"]["recipe_id"], "classification")
        self.assertEqual(body["plan"]["base_model_name"], "Qwen/Qwen1.5-1.8B-Chat")
        self.assertEqual(body["plan"]["target_profile_id"], "mobile_cpu")

        # Aggregate profile: distribution SHAPE, not names.
        profile = body["cloud_safe_profile"]
        shape = profile["label_distribution_shape"]
        self.assertEqual(shape["num_classes"], 3)
        self.assertEqual(shape["min_class_count"], 1)
        self.assertEqual(shape["max_class_count"], 6)
        self.assertEqual(shape["classes_below_floor"], 2)  # the two singletons
        self.assertGreaterEqual(profile["labelled_row_count"], 1)

        # Plan-fit roll-up flags the below-floor classes (attention, not ready).
        self.assertIn(body["plan_health"]["verdict"], ("attention", "mismatch"))
        sids = {s["id"] for s in body["plan_health"]["signals"]}
        self.assertIn("plan.classes_below_floor", sids)

        # Phase-1 framing + provider support kept in mind.
        self.assertFalse(body["cloud_refinement"]["available"])
        self.assertIn("deepseek", body["cloud_refinement"]["supported_providers"])
        self.assertIn("qwen", body["cloud_refinement"]["supported_providers"])
        self.assertEqual(body["privacy"]["cloud_sharing"], "aggregate_only")

    def test_cloud_safe_profile_never_leaks_raw_data(self):
        # THE invariant: no ingested text, no gold answers, and no label NAMES
        # may appear anywhere in the cloud-safe profile.
        pid = self._seed_project()
        body = self.client.get(f"/api/projects/{pid}/refine-plan").json()
        serialized = json.dumps(body["cloud_safe_profile"])
        self.assertNotIn(SECRET_TEXT, serialized)
        for label in SECRET_LABELS:
            self.assertNotIn(label, serialized)

    def test_no_recipe_project_is_a_mismatch(self):
        resp = self.client.post("/api/projects", json={"name": f"refine-norecipe-{uuid.uuid4().hex[:6]}"})
        pid = int(resp.json()["id"])
        body = self.client.get(f"/api/projects/{pid}/refine-plan").json()
        self.assertEqual(body["plan_health"]["verdict"], "mismatch")
        self.assertIn("plan.no_recipe", {s["id"] for s in body["plan_health"]["signals"]})

    def test_unknown_project_404(self):
        resp = self.client.get("/api/projects/999999/refine-plan")
        self.assertEqual(resp.status_code, 404, resp.text)

    # ── Phase 2: cloud strategy pass ──────────────────────────────────

    def test_validate_strategy_clamps_to_known_menu(self):
        from app.services.pipeline_refinement_service import validate_strategy
        profile = {"recipe_id": "classification", "task_profile": "classification"}
        raw = {
            "plan_delta": {
                "task_profile": "summarization",       # valid + a change → kept
                "base_model_size_class": "mid",        # valid → kept
                "training_mode": "telepathy",          # off-menu → dropped
                "rag_first": True,                     # valid bool → kept
            },
            "directional_config": [
                {"kind": "num_epochs_recommend", "direction": "down", "reason": "memorization risk"},
                {"kind": "set_learning_rate", "direction": "0.0005"},   # off-menu → dropped
            ],
            "rationale": "x" * 5000,                   # truncated
            "confidence": 0.8,
        }
        out = validate_strategy(raw, profile)
        self.assertEqual(out["plan_delta"]["task_profile"], "summarization")
        self.assertEqual(out["plan_delta"]["base_model_size_class"], "mid")
        self.assertTrue(out["plan_delta"]["rag_first"])
        self.assertNotIn("training_mode", out["plan_delta"])          # off-menu dropped
        kinds = {d["kind"] for d in out["directional_config"]}
        self.assertEqual(kinds, {"num_epochs_recommend"})            # off-menu dropped
        self.assertEqual(out["confidence"], 0.8)
        self.assertLessEqual(len(out["rationale"]), 1200)

    def test_validate_strategy_drops_unsupported_and_hallucinated_gaps(self):
        from app.services.pipeline_refinement_service import validate_strategy
        # Profile evidences class imbalance but NOT seq-length or leakage.
        profile = {
            "label_distribution_shape": {"classes_below_floor": 2, "imbalance_ratio": 0.16},
            "truncation_risk": "ok", "tokenizer_oov": "ok", "forecast_verdict": "likely_pass",
        }
        raw = {"data_gaps": [
            {"kind": "class_balance", "detail": "minority classes thin", "suggested_count": 30},
            {"kind": "seq_length", "detail": "rows are long"},     # unsupported → dropped
            {"kind": "leakage", "detail": "train leaks to test"},  # never accepted → dropped
            {"kind": "made_up", "detail": "nonsense"},             # off-menu → dropped
        ]}
        out = validate_strategy(raw, profile)
        self.assertEqual([g["kind"] for g in out["data_gaps"]], ["class_balance"])
        self.assertEqual(out["data_gaps"][0]["suggested_count"], 30)
        self.assertGreaterEqual(out["dropped"]["data_gaps"], 3)

    def test_run_cloud_strategy_pass_injected_fn_validates_and_caches(self):
        pid = self._seed_project()

        async def _run():
            from app.database import async_session_factory
            from app.services.pipeline_refinement_service import run_cloud_strategy_pass

            calls = {"n": 0}

            async def fake_fn(profile):
                calls["n"] += 1
                # Profile is the cloud-safe aggregate — assert no secret leaked.
                self.assertNotIn(SECRET_TEXT, json.dumps(profile))
                return {
                    "plan_delta": {"rag_first": True},
                    "data_gaps": [{"kind": "class_balance", "detail": "balance refund", "suggested_count": 30}],
                    "rationale": "Tiny imbalanced gold → balance + consider retrieval.",
                    "confidence": 0.7,
                }

            async with async_session_factory() as db:
                first = await run_cloud_strategy_pass(db, pid, strategy_fn=fake_fn, model_label="test:model")
                # Second call hits the cache (profile unchanged) — no new fn call.
                second = await run_cloud_strategy_pass(db, pid, strategy_fn=fake_fn, model_label="test:model")
                return first, second, calls["n"]

        first, second, n_calls = asyncio.run(_run())
        self.assertTrue(first["plan_delta"]["rag_first"])
        self.assertEqual([g["kind"] for g in first["data_gaps"]], ["class_balance"])
        self.assertEqual(first["provenance"]["model"], "test:model")
        self.assertEqual(first["provenance"]["shared"], "cloud_safe_profile")
        self.assertFalse(first["from_cache"])
        self.assertTrue(second["from_cache"])
        self.assertEqual(n_calls, 1)  # cache prevented a second call

    def test_run_cloud_strategy_pass_fallback_when_fn_returns_none(self):
        pid = self._seed_project()

        async def _run():
            from app.database import async_session_factory
            from app.services.pipeline_refinement_service import run_cloud_strategy_pass

            async def none_fn(profile):
                return None

            async with async_session_factory() as db:
                return await run_cloud_strategy_pass(db, pid, strategy_fn=none_fn)

        self.assertIsNone(asyncio.run(_run()))

    def test_cloud_endpoint_unavailable_without_provider(self):
        # No PLAN_REFINE_*/ANTHROPIC/OPENAI configured in the test env → fallback.
        pid = self._seed_project()
        resp = self.client.post(f"/api/projects/{pid}/refine-plan/cloud")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["available"])
        self.assertIsNone(body["refinement"])

    # ── Phase 3: accept / apply ───────────────────────────────────────

    def _seed_cached_refinement(self, pid: int, refinement: dict) -> None:
        async def _set():
            from app.database import async_session_factory
            from app.models.project import Project
            async with async_session_factory() as db:
                project = await db.get(Project, pid)
                rc = dict(project.runtime_config or {})
                rc["plan_refinement"] = {"profile_hash": "seed", "refinement": refinement}
                project.runtime_config = rc
                await db.commit()
        asyncio.run(_set())

    def test_apply_plan_delta_through_canonical_paths(self):
        pid = self._seed_project()
        self._seed_cached_refinement(pid, {
            "plan_delta": {"rag_first": True, "base_model_size_class": "large", "task_profile": "rag_qa"},
            "directional_config": [{"kind": "max_seq_length_raise", "reason": "long rows"}],
            "data_gaps": [], "rationale": "x", "confidence": 0.6,
        })
        resp = self.client.post(f"/api/projects/{pid}/refine-plan/apply", json={})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        statuses = {o["field"]: o["status"] for o in body["plan_delta"]}
        self.assertEqual(statuses["rag_first"], "applied")
        self.assertEqual(statuses["base_model_size_class"], "applied")
        self.assertEqual(statuses["task_profile"], "applied")
        # max_seq_length_raise has no one-click patch → surfaced as manual.
        dstatus = {o["kind"]: o["status"] for o in body["directional_config"]}
        self.assertEqual(dstatus["max_seq_length_raise"], "manual")
        self.assertIn("rag_first", body["applied"]["plan_delta"])

        # Canonical mutations actually landed on the project.
        async def _read():
            from app.database import async_session_factory
            from app.models.project import Project
            async with async_session_factory() as db:
                p = await db.get(Project, pid)
                return p.base_model_name, dict(p.runtime_config or {})
        base, rc = asyncio.run(_read())
        self.assertEqual(base, "meta-llama/Meta-Llama-3-8B-Instruct")  # large
        self.assertTrue(rc.get("rag_first"))
        self.assertEqual(rc["plan_refinement"]["applied"]["plan_delta"], ["rag_first", "base_model_size_class", "task_profile"])

    def test_apply_is_selective(self):
        pid = self._seed_project()
        self._seed_cached_refinement(pid, {
            "plan_delta": {"rag_first": True, "base_model_size_class": "large"},
            "directional_config": [], "data_gaps": [], "rationale": "x", "confidence": 0.5,
        })
        # Accept only rag_first — the base-size change must NOT land.
        resp = self.client.post(
            f"/api/projects/{pid}/refine-plan/apply",
            json={"plan_delta_fields": ["rag_first"], "directional_kinds": []},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["applied"]["plan_delta"], ["rag_first"])

        async def _read():
            from app.database import async_session_factory
            from app.models.project import Project
            async with async_session_factory() as db:
                p = await db.get(Project, pid)
                return p.base_model_name
        # Base model stays the seeded Qwen — NOT flipped to the "large" Llama.
        self.assertEqual(asyncio.run(_read()), "Qwen/Qwen1.5-1.8B-Chat")

    def test_apply_without_cached_refinement_404(self):
        pid = self._seed_project()
        resp = self.client.post(f"/api/projects/{pid}/refine-plan/apply", json={})
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_get_surfaces_applied_marker(self):
        pid = self._seed_project()
        # Seed a refinement whose hash matches the live profile so GET surfaces it.
        async def _seed_matching():
            from app.database import async_session_factory
            from app.models.project import Project
            from app.services.pipeline_refinement_service import build_cloud_safe_profile, _profile_hash
            async with async_session_factory() as db:
                profile = await build_cloud_safe_profile(db, pid)
                project = await db.get(Project, pid)
                rc = dict(project.runtime_config or {})
                rc["plan_refinement"] = {
                    "profile_hash": _profile_hash(profile),
                    "refinement": {"plan_delta": {"rag_first": True}, "directional_config": [],
                                   "data_gaps": [], "rationale": "x", "confidence": 0.5},
                    "applied": {"plan_delta": ["rag_first"], "directional": []},
                }
                project.runtime_config = rc
                await db.commit()
        asyncio.run(_seed_matching())
        body = self.client.get(f"/api/projects/{pid}/refine-plan").json()
        self.assertIsNotNone(body["refinement"])
        self.assertEqual(body["refinement"]["applied"]["plan_delta"], ["rag_first"])


if __name__ == "__main__":
    unittest.main()
