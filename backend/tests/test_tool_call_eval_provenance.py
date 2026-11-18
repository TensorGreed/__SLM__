"""Which config a tool_calling eval used: the project's CURRENT tool flags (re-read, so a long-lived
lift-check session never scores with the flags it loaded at start), recorded as ``metrics.tool_flags``
(allowed tools, fingerprint, whether the block equals the one the training rows carry), the rows the
split has vs the rows scored, and the lift check's row cap (per-project ``auto_lift_max_samples``).

pytest-only. Model inference is stubbed; no network."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.services import post_training_eval_service as pte
from app.services import structured_plan_prompt as spp
from app.services import tool_call_scoring as tcs
from app.services import tool_catalog_service as tcat
from app.services.data_adapter_service import map_record_with_adapter

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    from _job_wait import wait_for_jobs_idle

    wait_for_jobs_idle(client, timeout_s=60)
    client.__exit__(None, None, None)


def run_async(fn):
    async def go():
        async with async_session_factory() as db:
            return await fn(db)

    return asyncio.run(go())


def _tool(name: str) -> dict:
    return {"name": name, "description": f"{name} tool",
            "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
            "annotations": {"readOnlyHint": True}, "read_only": True, "destructive": None}


TOOLS = [_tool(n) for n in ("get_item", "list_items", "find_item", "count_items")]


def set_flags(pid: int, flags: dict) -> None:
    async def go(db):
        project = await db.get(Project, pid)
        tcat.set_tool_flags(project, flags)
        await db.commit()

    run_async(go)


def seed(n_rows: int = 6) -> tuple[int, int]:
    pid = client.post("/api/projects", json={"name": f"tool-prov-{time.time_ns()}", "description": "x"}).json()["id"]

    async def cat(db):
        project = await db.get(Project, pid)
        tcat.set_catalog_tools(project, TOOLS, {"kind": "manual"})
        await db.commit()

    run_async(cat)
    rc = run_async(lambda db: db.get(Project, pid)).runtime_config
    block = spp.tool_block_for(rc)  # training block: every tool allowed
    prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "manifest.json").write_text(json.dumps({"task_profile": "tool_calling", "adapter_id": "tool-call-json"}))
    rows = [map_record_with_adapter({"prompt": f"Get item {i}", "tool_name": "get_item", "tool_args": {"id": f"i{i}"},
                                     "result": "ok"}, adapter_id="tool-call-json", adapter_config={"tool_block": block})
            for i in range(n_rows)]
    (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    async def add(db):
        db.add(Dataset(project_id=pid, name="test", dataset_type=DatasetType.TEST,
                       file_path=str(prepared / "test.jsonl"), record_count=len(rows)))
        exp = Experiment(project_id=pid, name="trained", status=ExperimentStatus.COMPLETED, base_model="toy",
                         config={}, output_dir=str(prepared), completed_at=datetime.now(timezone.utc))
        db.add(exp)
        await db.commit()
        return exp.id

    return pid, run_async(add)


def evaluate(pid: int, exp_id: int, max_samples: int, prompts: list | None = None) -> dict:
    def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True,
            decoding="neutral"):
        if prompts is not None:
            prompts.extend(p["prompt"] for p in pairs)
        preds = [{**p, "formatted_prompt": "", "prediction": json.dumps({"tool": "get_item", "args": {"id": f"i{i}"}}),
                  "generated_tokens": 5, "latency_ms": 1.0} for i, p in enumerate(pairs)]
        return preds, {"engine": "stub", "decoding": {"mode": decoding}}

    with patch("app.services.evaluation_service._run_local_inference", run):
        resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
            "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": max_samples})
    assert resp.status_code == 201, resp.text
    return resp.json()


class ToolFlagsProvenanceTests(unittest.TestCase):
    def test_eval_records_the_flags_it_used_and_whether_they_match_training(self):
        pid, exp_id = seed()
        same = evaluate(pid, exp_id, 100)["metrics"]["tool_flags"]
        self.assertEqual((same["tool_count"], same["allowed_count"]), (4, 4))
        self.assertTrue(same["matches_training"])
        self.assertEqual(same["training_allowed_count"], 4)

        # eval-time flags differ from the training flags by design: the eval follows the CURRENT flags
        set_flags(pid, {"find_item": {"allowed": False}, "count_items": {"allowed": False}})
        prompts: list[str] = []
        out = evaluate(pid, exp_id, 100, prompts)
        flags = out["metrics"]["tool_flags"]
        self.assertEqual(flags["allowed_count"], 2)
        self.assertEqual(flags["allowed_tools"], ["get_item", "list_items"])
        self.assertFalse(flags["matches_training"])
        self.assertEqual(flags["training_allowed_count"], 4)
        self.assertNotEqual(flags["catalog_fingerprint"], same["catalog_fingerprint"])
        self.assertNotIn("find_item", prompts[0])  # the prompt used the current block, not the training one
        self.assertTrue(prompts[0].endswith("Get item 0"))

    def test_rows_available_vs_scored(self):
        pid, exp_id = seed(n_rows=6)
        out = evaluate(pid, exp_id, 4)
        self.assertEqual(out["metrics"]["total"], 4)
        self.assertEqual(out["details"]["dataset"]["rows_available"], 6)
        self.assertEqual(out["details"]["dataset"]["max_samples"], 4)

    def test_no_catalog_records_no_flags(self):
        self.assertIsNone(tcs.tool_flags_provenance({}))

    def test_long_lived_session_reads_the_current_flags(self):
        """The lift-check Job keeps one session for both legs; the live config must not come from the
        Project row that session loaded when it started."""
        pid, _ = seed()
        from app.services.evaluation_service import _apply_tool_call_live_config

        async def go():
            async with async_session_factory() as long_lived:
                held = await long_lived.get(Project, pid)  # loaded at job start (kept alive, like the Job runner)
                async with async_session_factory() as other:
                    project = await other.get(Project, pid)
                    tcat.set_tool_flags(project, {"list_items": {"allowed": False}})
                    await other.commit()
                ctx = SimpleNamespace(manifest={})
                await _apply_tool_call_live_config(long_lived, pid, ctx, SimpleNamespace(profile_id="tool_calling"))
                assert held is not None
                return ctx.manifest["tool_block"]

        block = asyncio.run(go())
        self.assertIn("get_item", block)
        self.assertNotIn("list_items", block)


class LiftScopeTests(unittest.TestCase):
    def test_lift_max_samples_default_setting_and_project_override(self):
        with patch.object(settings, "AUTO_LIFT_EVAL_MAX_SAMPLES", 100):
            self.assertEqual(pte.lift_max_samples(None), 100)
            self.assertEqual(pte.lift_max_samples(SimpleNamespace(runtime_config={})), 100)
            self.assertEqual(pte.lift_max_samples(SimpleNamespace(runtime_config={"auto_lift_max_samples": 250})), 250)
            whole = pte.WHOLE_SPLIT_MAX_SAMPLES
            self.assertEqual(pte.lift_max_samples(SimpleNamespace(runtime_config={"auto_lift_max_samples": "all"})), whole)
            self.assertEqual(pte.lift_max_samples(SimpleNamespace(runtime_config={"auto_lift_max_samples": 0})), whole)
            self.assertEqual(pte.lift_max_samples(SimpleNamespace(runtime_config={"auto_lift_max_samples": "x"})), 100)
        with patch.object(settings, "AUTO_LIFT_EVAL_MAX_SAMPLES", 0):
            self.assertEqual(pte.lift_max_samples(None), pte.WHOLE_SPLIT_MAX_SAMPLES)

    def test_lift_scope_reports_rows_and_flag_mismatch(self):
        def result(fp, total=4, available=6):
            return SimpleNamespace(
                metrics={"total": total, "tool_call_fingerprint": fp,
                         "tool_flags": {"catalog_fingerprint": fp, "allowed_count": 2, "tool_count": 4,
                                        "matches_training": False}},
                details={"dataset": {"rows_available": available}})

        same = pte.lift_scope_fields(result("a"), result("a"))
        self.assertNotIn("tool_flags_mismatch", same)
        self.assertEqual(same["scope"]["trained"]["rows_scored"], 4)
        self.assertEqual(same["scope"]["trained"]["rows_available"], 6)
        self.assertEqual(same["scope"]["baseline"]["tool_flags"]["allowed_count"], 2)
        self.assertTrue(pte.lift_scope_fields(result("a"), result("b"))["tool_flags_mismatch"])
        self.assertEqual(pte.lift_scope_fields(None, result("a"))["scope"]["baseline"], None)

    def test_training_tool_block_reads_the_prepared_rows(self):
        pid, _ = seed()
        path = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared" / "test.jsonl"
        block = tcs.training_tool_block(path)
        self.assertTrue(block.startswith("Available tools (call only these):"))
        self.assertIsNone(tcs.training_tool_block(path.with_name("missing.jsonl")))


if __name__ == "__main__":
    unittest.main()
