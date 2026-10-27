"""Acceptance: tool-call scoring end to end. A tool catalog is imported from the fake MCP server over
stdio (allow-listed by realpath); a structured-plan project whose schema embeds the steps and a
tool_calling project are evaluated through the held-out endpoint (model inference stubbed), then read
back through ``/evaluation/summary``. Also: the prompt carries the tool block, the dry run executes
read-only calls only (checked against the fake server's call log), and the fingerprint follows the flags.

pytest-only. No real model, no network."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.services import structured_plan_prompt as spp
from app.services import tool_catalog_service as tcat
from app.services.data_adapter_service import map_record_with_adapter

client: TestClient | None = None
FAKE_SERVER = str(Path(__file__).with_name("fake_mcp_server.py"))
N_ROWS = 8

STEP = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["step", "tool", "args"],
    "properties": {
        "step": {"type": "string"},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "depends_on": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}
PLAN = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["steps"],
    "properties": {"steps": {"type": "array", "minItems": 1, "items": {"$ref": "step.schema.json"}}},
}
FILES = {"plan.schema.json": PLAN, "step.schema.json": STEP}


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def run_async(coro_fn):
    async def go():
        async with async_session_factory() as db:
            return await coro_fn(db)

    return asyncio.run(go())


def gold_plan(i: int) -> dict:
    return {"steps": [
        {"step": "s1", "tool": "get_service", "args": {"name": f"svc-{i}"}, "depends_on": []},
        {"step": "s2", "tool": "create_service", "args": {"name": f"svc-{i}", "tier": "pro"}, "depends_on": ["s1"]},
    ]}


def base_answer(i: int) -> str:
    plan = gold_plan(i)
    if i % 4 == 1:
        plan["steps"][0]["tool"] = "restart_service"  # not in the catalog
    elif i % 4 == 2:
        plan["steps"][1]["args"]["tier"] = "gold"  # outside the enum
    elif i % 4 == 3:
        plan["steps"][0]["depends_on"] = ["s2"]  # cycle
    return json.dumps(plan)


def gold_answer(i: int) -> str:
    return json.dumps(gold_plan(i))


def dry_run_answer(i: int) -> str:
    plan = gold_plan(i)
    plan["steps"].append({"step": "s3", "tool": "delete_service", "args": {"name": f"svc-{i}"}, "depends_on": ["s2"]})
    return json.dumps(plan)


class ToolScoringAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tool-scoring-")
        self.call_log = os.path.join(self.tmp, "calls.log")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(settings, "MCP_STDIO_ALLOWED_BINARIES", os.path.realpath(sys.executable)))

    # ── helpers ──

    def new_project(self, recipe: str | None) -> int:
        pid = client.post("/api/projects", json={"name": f"tool-accept-{time.time_ns()}", "description": "x"}).json()["id"]
        if recipe:
            put = client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": recipe})
            self.assertEqual(put.status_code, 200, put.text)
        return pid

    def import_catalog(self, pid: int) -> dict:
        async def go(db):
            project = await db.get(Project, pid)
            status = await tcat.import_from_mcp(db, project, {
                "kind": "mcp_stdio", "label": "deploy fake", "command": [sys.executable, FAKE_SERVER],
                "env": {"FAKE_MCP_CALL_LOG": self.call_log},
            })
            await db.commit()
            return status

        status = run_async(go)
        self.assertEqual(len(status["catalog"]["tools"]), 4)
        return status

    def mutate_catalog(self, pid: int, fn):
        async def go(db):
            project = await db.get(Project, pid)
            fn(project)
            await db.commit()

        run_async(go)

    def seed_plan_project(self) -> tuple[int, int, int]:
        pid = self.new_project("structured-plan")
        cfg = client.put(f"/api/projects/{pid}/structured-plan-config", json={
            "schema_files": FILES, "root_schema": "plan.schema.json", "judge_enabled": False, "retrieve_passages": False})
        self.assertEqual(cfg.status_code, 200, cfg.text)
        self.import_catalog(pid)
        from app.services.dataset_service import _recipe_output_schema

        schema = run_async(lambda db: _recipe_output_schema(db, pid))
        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({
            "task_profile": "structured_extraction", "adapter_id": "structured-extraction", "output_schema": schema}))
        rc = run_async(lambda db: db.get(Project, pid)).runtime_config
        block = spp.tool_block_for(rc)
        self.assertTrue(block.startswith("Available tools (call only these):"))
        rows = []
        for i in range(N_ROWS):
            row = {"row_kind": "structured_plan", "scenario": f"Stand up svc-{i}.", "plan_json": gold_plan(i)}
            mapped = map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config={"structured_plan": {
                "schema_files": FILES, "root_schema": "plan.schema.json", "project_id": 0, "retrieve_passages": False,
                "tool_block": block}})
            rows.append({**row, **mapped})
        return (pid, *self.finish_seed(pid, prepared, rows))

    def finish_seed(self, pid: int, prepared: Path, rows: list[dict]) -> tuple[int, int]:
        (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        async def add(db):
            db.add(Dataset(project_id=pid, name="test", dataset_type=DatasetType.TEST, file_path=str(prepared / "test.jsonl"),
                           record_count=len(rows)))
            now = datetime.now(timezone.utc)
            base = Experiment(project_id=pid, name="base model", status=ExperimentStatus.COMPLETED, base_model="toy",
                              config={"is_baseline": True}, output_dir=str(prepared), completed_at=now)
            trained = Experiment(project_id=pid, name="trained", status=ExperimentStatus.COMPLETED, base_model="toy",
                                 config={}, output_dir=str(prepared), completed_at=now)
            db.add_all([base, trained])
            await db.commit()
            return base.id, trained.id

        return run_async(add)

    def evaluate(self, pid: int, exp_id: int, answers, prompts: list | None = None) -> dict:
        def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True, decoding="neutral"):
            if prompts is not None:
                prompts.extend(p["prompt"] for p in pairs)
            preds = [{**p, "formatted_prompt": "", "prediction": answers(i), "generated_tokens": 5, "latency_ms": 1.0}
                     for i, p in enumerate(pairs)]
            return preds, {"engine": "stub", "decoding": {"mode": decoding}}

        with patch("app.services.evaluation_service._run_local_inference", run):
            resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 40, "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    # ── structured-plan ──

    def test_structured_plan_tool_scoring_in_the_summary(self):
        pid, base_id, trained_id = self.seed_plan_project()
        prompts: list[str] = []
        base = self.evaluate(pid, base_id, base_answer, prompts)
        trained = self.evaluate(pid, trained_id, gold_answer)
        self.assertTrue(all("Available tools (call only these):" in p and "get_service(name: string)" in p for p in prompts))
        self.assertLess(prompts[0].index("Available tools"), prompts[0].index("JSON Schema the plan"))
        bm, tm = base["metrics"], trained["metrics"]

        self.assertEqual(tm["tool_call_valid_rate"], 1.0)
        self.assertEqual(tm["structured_plan_pass_rate"], 1.0)
        self.assertEqual(bm["schema_valid_rate"], 1.0)  # every base plan is schema-valid; only the tools are wrong
        self.assertEqual(bm["tool_call_valid_rate"], 0.5)
        self.assertEqual(bm["plan_order_valid_rate"], 0.75)
        self.assertEqual(bm["structured_plan_pass_rate"], 0.25)
        self.assertEqual(bm["correct"], 2)
        tools = bm["tools"]
        self.assertEqual(tools["source_kind"], "mcp_stdio")
        self.assertEqual(tools["modes"], ["tool_call_valid", "tool_args_match", "plan_order_valid"])
        self.assertEqual(tools["unknown_tools"], {"restart_service": 2})
        self.assertEqual(tools["per_tool"]["create_service"]["invalid_args"], 2)
        self.assertIsNone(tools["dry_run"])

        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": trained_id})
        self.assertEqual(summary.status_code, 200, summary.text)
        body = summary.json()
        self.assertEqual(body["headline"]["metric_id"], "structured_plan_pass_rate")
        lifts = {row["metric_id"]: row for row in body["metric_lifts"]}
        self.assertIn("tool_call_valid_rate", lifts, sorted(lifts))
        self.assertEqual(lifts["tool_call_valid_rate"]["baseline_value"], 0.5)
        self.assertEqual(lifts["tool_call_valid_rate"]["trained_value"], 1.0)

        previews = base["details"]["predictions_preview"]
        bad = next(p for p in previews if p["row_tools_pass"] is False and p["row_tool_results"]["calls"][0]["known"] is False)
        self.assertEqual(bad["row_tool_results"]["errors"][0], "unknown tool restart_service")
        invalid = next(p for p in previews if any(c["known"] and c["args_valid"] is False for c in p["row_tool_results"]["calls"]))
        self.assertTrue(invalid["row_tool_results"]["calls"][1]["args_errors"])
        ok = trained["details"]["predictions_preview"][0]
        self.assertTrue(ok["row_tools_pass"])
        self.assertTrue(ok["row_structured_plan_pass"])

        dump = os.environ.get("TOOL_SCORING_DUMP")
        if dump:
            Path(dump).write_text(json.dumps({
                "tools": tools, "metrics_top": {k: bm[k] for k in ("tool_call_valid_rate", "tool_args_match_rate", "plan_order_valid_rate", "structured_plan_pass_rate")},
                "plan_preview_row": invalid, "plan_preview_ok": ok}, indent=2, default=str))

    def test_fingerprint_and_scores_follow_the_flags(self):
        pid, base_id, trained_id = self.seed_plan_project()
        first = self.evaluate(pid, trained_id, gold_answer)["metrics"]
        self.assertEqual(first["tool_call_valid_rate"], 1.0)
        self.mutate_catalog(pid, lambda project: tcat.set_tool_flags(project, {"create_service": {"allowed": False}}))
        second = self.evaluate(pid, trained_id, gold_answer)["metrics"]
        self.assertNotEqual(first["structured_plan_fingerprint"], second["structured_plan_fingerprint"])
        self.assertEqual(second["tool_call_valid_rate"], 0.0)
        self.assertEqual(second["structured_plan_pass_rate"], 0.0)
        self.assertEqual(second["tools"]["per_tool"]["create_service"]["not_allowed"], N_ROWS)
        # the prompt no longer offers the disallowed tool
        prompts: list[str] = []
        self.evaluate(pid, trained_id, gold_answer, prompts)
        self.assertNotIn("create_service", prompts[0])
        # disabling tool scoring restores the plain G6 behaviour
        self.assertEqual(client.put(f"/api/projects/{pid}/tool-scoring-config", json={"enabled": False}).status_code, 200)
        third = self.evaluate(pid, trained_id, gold_answer)["metrics"]
        self.assertNotIn("tools", third)
        self.assertEqual(third["structured_plan_pass_rate"], 1.0)

    def test_dry_run_executes_read_only_calls_only(self):
        pid, base_id, trained_id = self.seed_plan_project()
        with patch.object(settings, "MCP_DRY_RUN_ENABLED", True):
            put = client.put(f"/api/projects/{pid}/tool-scoring-config", json={"dry_run": True, "modes": ["tool_call_valid"]})
            self.assertEqual(put.status_code, 200, put.text)
            metrics = self.evaluate(pid, trained_id, dry_run_answer)["metrics"]
        dry = metrics["tools"]["dry_run"]
        self.assertEqual((dry["enabled"], dry["calls"], dry["ok"], dry["errors"]), (True, N_ROWS, N_ROWS, 0))
        self.assertEqual(dry["skipped"], 2 * N_ROWS)  # create_service + delete_service: never executed
        logged = [json.loads(line) for line in Path(self.call_log).read_text().splitlines()]
        self.assertEqual({entry["name"] for entry in logged}, {"get_service"})
        self.assertEqual(len(logged), N_ROWS)
        self.assertEqual(metrics["tool_call_valid_rate"], 1.0)  # informational: the verdict is unchanged

    def test_dry_run_off_means_no_calls(self):
        pid, base_id, trained_id = self.seed_plan_project()
        with patch.object(settings, "MCP_DRY_RUN_ENABLED", True):
            client.put(f"/api/projects/{pid}/tool-scoring-config", json={"dry_run": True})
        # the admin switch is off again at eval time: the summary says so and nothing is executed
        metrics = self.evaluate(pid, trained_id, dry_run_answer)["metrics"]
        self.assertFalse(metrics["tools"]["dry_run"]["enabled"])
        self.assertFalse(os.path.exists(self.call_log))

    # ── tool_calling ──

    def seed_tool_calling_project(self) -> tuple[int, int, int]:
        pid = self.new_project(None)
        self.import_catalog(pid)
        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({"task_profile": "tool_calling", "adapter_id": "tool-call-json"}))
        rc = run_async(lambda db: db.get(Project, pid)).runtime_config
        block = spp.tool_block_for(rc)
        rows = []
        for i in range(N_ROWS):
            record = {"prompt": f"Look up svc-{i}", "tool_name": "get_service", "tool_args": {"name": f"svc-{i}"}, "result": "found"}
            rows.append(map_record_with_adapter(record, adapter_id="tool-call-json", adapter_config={"tool_block": block}))
        return (pid, *self.finish_seed(pid, prepared, rows))

    def test_tool_calling_profile_is_scored(self):
        pid, base_id, trained_id = self.seed_tool_calling_project()

        def trained(i):
            return f'Tool Call: get_service({{"name": "svc-{i}"}})\nAssistant: found'

        def base(i):
            if i % 4 == 0:
                return json.dumps({"tool": "get_service", "args": {"name": f"svc-{i}"}})
            if i % 4 == 1:
                return json.dumps({"tool": "list_services", "args": {}})  # wrong tool
            if i % 4 == 2:
                return json.dumps({"tool": "get_service", "args": {"name": f"other-{i}"}})  # wrong args
            return json.dumps({"tool": "get_service", "args": {}})  # required `name` missing: schema-invalid

        prompts: list[str] = []
        b = self.evaluate(pid, base_id, base, prompts)
        t = self.evaluate(pid, trained_id, trained)
        self.assertTrue(prompts[0].startswith("Available tools (call only these):"))
        self.assertTrue(prompts[0].endswith("Look up svc-0"))
        bm, tm = b["metrics"], t["metrics"]
        self.assertEqual(tm["scoring_mode"], "tool_call")
        self.assertEqual((tm["tool_selection_accuracy"], tm["tool_call_pass_rate"], tm["exact_match"]), (1.0, 1.0, 1.0))
        self.assertEqual(tm["tool_call_valid_rate"], 1.0)
        self.assertEqual(bm["tool_selection_accuracy"], 0.75)
        self.assertEqual(bm["tool_call_valid_rate"], 0.75)  # the missing required argument breaks the schema
        self.assertEqual(bm["tool_call_pass_rate"], 0.25)  # right tool + valid + args match only for i % 4 == 0
        self.assertEqual(bm["tools"]["per_tool"]["get_service"]["invalid_args"], 2)

        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": trained_id})
        self.assertEqual(summary.status_code, 200, summary.text)
        body = summary.json()
        self.assertEqual(body["headline"]["metric_id"], "tool_call_pass_rate", body["headline"])
        self.assertEqual(body["verdict"], "better")
        preview = b["details"]["predictions_preview"][0]
        self.assertTrue(preview["row_tool_call_pass"])
        self.assertIn("calls", preview["row_tool_results"])

        dump = os.environ.get("TOOL_SCORING_DUMP")
        if dump:
            Path(dump + ".call").write_text(json.dumps({"metrics": bm, "preview": b["details"]["predictions_preview"][2]}, indent=2, default=str))

    def test_tool_calling_without_a_catalog_scores_selection_and_args(self):
        pid = self.new_project(None)
        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({"task_profile": "tool_calling", "adapter_id": "tool-call-json"}))
        rows = [map_record_with_adapter({"prompt": f"Look up svc-{i}", "tool_name": "get_service", "tool_args": {"name": f"svc-{i}"}, "result": "ok"},
                                        adapter_id="tool-call-json", adapter_config={}) for i in range(4)]
        base_id, trained_id = self.finish_seed(pid, prepared, rows)
        prompts: list[str] = []
        m = self.evaluate(pid, trained_id, lambda i: json.dumps({"tool": "get_service", "args": {"name": f"svc-{i if i < 2 else 9}"}}), prompts)["metrics"]
        self.assertEqual(prompts[0], "Look up svc-0")  # no catalog: the prompt is untouched
        self.assertNotIn("tools", m)
        self.assertEqual((m["tool_selection_accuracy"], m["tool_call_pass_rate"]), (1.0, 0.5))

    def test_heldout_gold_calls_reach_the_scorer_without_the_generator_copying_pairs(self):
        """Regression: real inference builds fresh ``{prompt, reference, prediction}`` dicts (it does not
        copy the pair), so the gold calls must be carried over by ``copy_row_extras`` -- before, every
        tool-call held-out eval scored 0 whatever the model said."""
        pid = self.new_project(None)
        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({"task_profile": "tool_calling", "adapter_id": "tool-call-json"}))
        rows = [map_record_with_adapter({"prompt": f"Look up svc-{i}", "tool_name": "get_service", "tool_args": {"name": f"svc-{i}"}, "result": "ok"},
                                        adapter_id="tool-call-json", adapter_config={}) for i in range(3)]
        _, trained_id = self.finish_seed(pid, prepared, rows)

        def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True, decoding="neutral"):
            preds = [{"prompt": p["prompt"], "formatted_prompt": "", "reference": p["reference"],
                      "prediction": json.dumps({"tool": "get_service", "args": {"name": f"svc-{i}"}}),
                      "generated_tokens": 5, "latency_ms": 1.0} for i, p in enumerate(pairs)]
            return preds, {"engine": "stub", "decoding": {"mode": decoding}}

        with patch("app.services.evaluation_service._run_local_inference", run):
            resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
                "experiment_id": trained_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 10, "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        m = resp.json()["metrics"]
        self.assertEqual((m["tool_selection_accuracy"], m["tool_args_match_rate"], m["tool_call_pass_rate"]), (1.0, 1.0, 1.0))

    def test_adapter_target_is_the_call_and_the_gold_call_scores_one(self):
        """Raw prompt / tool_name / tool_args / tool_result rows through the real adapter: the training target
        is the call JSON, a model that emits it scores 1.0, and a JSON tool result with ``name`` keys never
        becomes the gold."""
        pid = self.new_project(None)
        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({"task_profile": "tool_calling", "adapter_id": "tool-call-json"}))
        raw = [
            {"prompt": f"Look up svc-{i}", "tool_name": "get_service", "tool_args": {"name": f"svc-{i}"},
             "tool_result": {"name": f"svc-{i}", "tier": "pro"} if i % 2 else "found"}
            for i in range(4)
        ]
        rows = [map_record_with_adapter(r, adapter_id="tool-call-json", adapter_config={}) for r in raw]
        for i, row in enumerate(rows):
            self.assertEqual(json.loads(row["answer"]), {"tool": "get_service", "args": {"name": f"svc-{i}"}})
            self.assertEqual(row["target_text"], row["answer"])
        base_id, trained_id = self.finish_seed(pid, prepared, rows)
        m = self.evaluate(pid, trained_id, lambda i: rows[i]["answer"])["metrics"]
        self.assertEqual((m["tool_selection_accuracy"], m["tool_args_match_rate"], m["tool_call_pass_rate"]), (1.0, 1.0, 1.0))
        # the stamped scoring fingerprint exists without a catalog and follows catalog changes
        self.assertTrue(m["tool_call_fingerprint"].startswith("tool_call/v1+"))
        self.import_catalog(pid)
        m2 = self.evaluate(pid, trained_id, lambda i: rows[i]["answer"])["metrics"]
        self.assertNotEqual(m["tool_call_fingerprint"], m2["tool_call_fingerprint"])
        self.mutate_catalog(pid, lambda project: tcat.set_tool_flags(project, {"list_services": {"allowed": False}}))
        m3 = self.evaluate(pid, trained_id, lambda i: rows[i]["answer"])["metrics"]
        self.assertNotEqual(m2["tool_call_fingerprint"], m3["tool_call_fingerprint"])


if __name__ == "__main__":
    unittest.main()
