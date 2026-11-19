"""Held-out eval with per-row task context, end to end through ``/evaluation/run-heldout`` (inference
stubbed): a tool_calling row's own tool list leads its prompt and is what its call is scored against
(allowed / unknown / argument schema / strict args), the project's flags still apply by name, rows
without a list fall back to the project catalog, and provenance lands in ``metrics.tool_flags.per_row``,
``metrics.tools.per_row`` and ``metrics.row_context``. A row's own system message reaches eval as
``row_system`` for QA and classification too. Every new eval carries ``prompt_format_version``.
Without any row context the metrics keep their old shape and fingerprint.

pytest-only. Model inference is stubbed; no network."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
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
from app.services import row_context_service as rcs
from app.services import tool_call_scoring as tcs
from app.services import tool_catalog_service as tcat

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


def tool(name: str, required: bool = True) -> dict:
    return {"name": name, "description": f"{name} tool",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}},
                            "required": ["id"] if required else []}}


def seed(rows: list[dict], *, profile: str, adapter: str, catalog: list[dict] | None = None,
         flags: dict | None = None, scoring: dict | None = None, config: dict | None = None) -> tuple[int, int]:
    from app.models.dataset import DatasetType as DT
    from app.services.dataset_service import _normalize_rows_for_training

    pid = client.post("/api/projects", json={"name": f"rowctx-{time.time_ns()}", "description": "x"}).json()["id"]

    async def setup(db):
        project = await db.get(Project, pid)
        if catalog:
            tcat.set_catalog_tools(project, catalog, {"kind": "manual"})
        if flags:
            tcat.set_tool_flags(project, flags)
        if scoring:
            project.runtime_config = {**(project.runtime_config or {}), "tool_scoring": scoring}
        await db.commit()
        return project.runtime_config

    rc = run_async(setup)
    cfg = dict(config or {})
    if adapter == "tool-call-json":
        from app.services import structured_plan_prompt as spp

        block = spp.tool_block_for(rc)
        if block:
            cfg["tool_block"] = block
        overrides = rcs.project_tool_overrides(rc)
        if overrides:
            cfg["tool_overrides"] = overrides
    prepared_rows = _normalize_rows_for_training(rows, DT.RAW, "llama3", adapter_id=adapter, adapter_config=cfg,
                                                 task_profile=profile)
    assert len(prepared_rows) == len(rows)
    prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "manifest.json").write_text(json.dumps({"task_profile": profile, "adapter_id": adapter}))
    (prepared / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prepared_rows))

    async def add(db):
        db.add(Dataset(project_id=pid, name="test", dataset_type=DatasetType.TEST,
                       file_path=str(prepared / "test.jsonl"), record_count=len(prepared_rows)))
        exp = Experiment(project_id=pid, name="trained", status=ExperimentStatus.COMPLETED, base_model="toy",
                         config={}, output_dir=str(prepared), completed_at=datetime.now(timezone.utc))
        db.add(exp)
        await db.commit()
        return exp.id

    return pid, run_async(add)


def evaluate(pid: int, exp_id: int, outputs: list[str], pairs_out: list | None = None) -> dict:
    def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True,
            decoding="neutral"):
        if pairs_out is not None:
            pairs_out.extend(dict(p) for p in pairs)
        preds = [{**p, "formatted_prompt": "", "prediction": outputs[i % len(outputs)], "generated_tokens": 5,
                  "latency_ms": 1.0} for i, p in enumerate(pairs)]
        return preds, {"engine": "stub", "decoding": {"mode": decoding}}

    with patch("app.services.evaluation_service._run_local_inference", run):
        resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
            "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 100})
    assert resp.status_code == 201, resp.text
    return resp.json()


def call(name: str, **args) -> str:
    return json.dumps({"tool": name, "args": args})


def tool_row(i: int, *, tools: list[dict] | None = None, system: str | None = None, gold: str = "get_item") -> dict:
    row = {"prompt": f"Get item {i}", "tool_name": gold, "tool_args": {"id": str(i)}, "result": "ok"}
    ctx = {}
    if tools is not None:
        ctx["tools"] = tools
    if system:
        ctx["system"] = system
    if ctx:
        row["row_context"] = ctx
    return row


class PerRowToolListTests(unittest.TestCase):
    def test_row_lists_without_a_catalog_are_prompted_and_scored_per_row(self):
        rows = [
            tool_row(0, tools=[tool("get_item")]),                  # right call, in list
            tool_row(1, tools=[tool("find_item")], gold="get_item"),  # gold not in the list; model calls get_item
            tool_row(2, tools=[tool("get_item")]),                  # args invalid vs the row schema
            tool_row(3),                                            # no own list, no catalog: no catalog check
        ]
        pid, exp = seed(rows, profile="tool_calling", adapter="tool-call-json", scoring={"strict_args": True})
        pairs: list[dict] = []
        outputs = [call("get_item", id="0"), call("get_item", id="1"), json.dumps({"tool": "get_item", "args": {"id": 2}}),
                   call("get_item", id="3", extra="x")]
        out = evaluate(pid, exp, outputs, pairs)
        m = out["metrics"]
        self.assertIn("- get_item(id: string)", pairs[0]["prompt"])
        self.assertIn("- find_item(", pairs[1]["prompt"])
        self.assertNotIn("Available tools", pairs[3]["prompt"])
        self.assertEqual(pairs[3]["prompt"], "Get item 3")
        preview = out["details"]["predictions_preview"]
        self.assertEqual([p.get("row_tool_list_source") for p in preview], ["row", "row", "row", None])
        res = [p.get("row_tool_results") for p in preview]
        self.assertTrue(res[0]["tool_call_valid"])
        self.assertIn("unknown tool get_item", res[1]["errors"])
        self.assertFalse(res[2]["tool_call_valid"])
        self.assertIsNone(res[3])  # not catalog-checked
        self.assertEqual([p["row_tool_call_pass"] for p in preview], [True, False, False, True])
        self.assertEqual(m["tool_call_pass_rate"], 0.5)
        self.assertEqual(m["tool_call_strict_pass_rate"], 0.25)  # row 3 has an extra argument
        self.assertEqual(m["tool_flags"]["per_row"]["rows"], 3)
        self.assertEqual(m["tool_flags"]["per_row"]["distinct_lists"], 2)
        self.assertIsNone(m["tool_flags"]["matches_training"])
        self.assertEqual(m["tool_flags"]["matches_training_reason"], "per_row_lists")
        self.assertEqual(m["tools"]["per_row"]["rows_without_catalog_check"], 1)
        self.assertEqual(m["row_context"]["kinds"]["tools"], {"row": 3, "project": 0, "none": 1})
        self.assertIn("+rowtools:", m["tool_call_fingerprint"])
        self.assertEqual(m["prompt_format_version"], rcs.EVAL_PROMPT_FORMAT_VERSION)
        self.assertEqual(preview[0]["row_context"]["tools"][0]["name"], "get_item")

    def test_row_order_and_older_sorted_splits(self):
        shuffled = [tool("zap"), tool("get_item"), tool("alpha")]
        pid, exp = seed([tool_row(0, tools=shuffled)], profile="tool_calling", adapter="tool-call-json")
        pairs: list[dict] = []
        new = evaluate(pid, exp, [call("get_item", id="0")], pairs)
        names = [ln.split("(")[0][2:] for ln in pairs[0]["prompt"].splitlines() if ln.startswith("- ")]
        self.assertEqual(names, ["zap", "get_item", "alpha"])
        # A split prepared before the stored lists kept their order (per-row lists in train, no manifest
        # stamp): eval uses each row's own training prompt, and the fingerprint marks the new prompts.
        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        (prepared / "train.jsonl").write_text((prepared / "test.jsonl").read_text())
        stored = json.loads((prepared / "test.jsonl").read_text().splitlines()[0])
        pairs = []
        older = evaluate(pid, exp, [call("get_item", id="0")], pairs)
        self.assertEqual(pairs[0]["prompt"], stored["question"])
        from app.services import tool_call_scoring as tcs

        self.assertNotEqual(older["metrics"]["tool_call_fingerprint"], tcs.tool_call_fingerprint({}, row_tool_lists=True))
        self.assertNotEqual(new["metrics"]["tool_call_fingerprint"], older["metrics"]["tool_call_fingerprint"])

    def test_train_and_eval_prompts_are_byte_identical_with_unsorted_parameters(self):
        """Golden parity: the prepared training prompt and the held-out eval prompt of a row whose tool
        parameters are not in name order are the same bytes (P5 gap 6: eval listed them A-Z)."""
        create = {"name": "create_event", "description": "Create an event.", "inputSchema": {
            "type": "object", "required": ["title", "date", "start_time"],
            "properties": {"title": {"type": "string"}, "date": {"type": "string"},
                           "start_time": {"type": "string"}, "duration_minutes": {"type": "integer"},
                           "location": {"type": "string"}}}}
        row = {"prompt": "Lunch with Bo on 2026-12-01 at noon", "tool_name": "create_event",
               "tool_args": {"title": "Lunch with Bo", "date": "2026-12-01", "start_time": "12:00"}, "result": "ok",
               "row_context": {"tools": [create, tool("get_item")]}}
        pid, exp = seed([row], profile="tool_calling", adapter="tool-call-json")
        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        stored = json.loads((prepared / "test.jsonl").read_text().splitlines()[0])
        self.assertIn("create_event(title: string, date: string, start_time: string, duration_minutes?: integer, "
                      "location?: string)", stored["question"])
        # the stored context keeps the schema's key order
        self.assertEqual(list(rcs.decode(stored["row_context"])["tools"][0]["input_schema"]["properties"]),
                         ["title", "date", "start_time", "duration_minutes", "location"])
        pairs: list[dict] = []
        out = evaluate(pid, exp, [call("create_event", title="Lunch with Bo", date="2026-12-01", start_time="12:00")],
                       pairs)
        self.assertEqual(pairs[0]["prompt"], stored["question"])
        self.assertEqual(out["metrics"]["tool_call_pass_rate"], 1.0)
        # a stamped split re-renders from the stored list (live project flags) and still matches
        (prepared / "manifest.json").write_text(json.dumps(
            {"task_profile": "tool_calling", "adapter_id": "tool-call-json", "row_tool_order": "kept"}))
        pairs = []
        evaluate(pid, exp, [call("get_item", id="0")], pairs)
        self.assertEqual(pairs[0]["prompt"], stored["question"])

    def test_project_flags_apply_by_name_and_rows_without_a_list_use_the_catalog(self):
        catalog = [tool("get_item"), tool("drop_item")]
        rows = [tool_row(0, tools=[tool("get_item"), tool("drop_item")], gold="drop_item"), tool_row(1)]
        pid, exp = seed(rows, profile="tool_calling", adapter="tool-call-json", catalog=catalog,
                        flags={"drop_item": {"allowed": False}})
        pairs: list[dict] = []
        out = evaluate(pid, exp, [call("drop_item", id="0"), call("get_item", id="1")], pairs)
        self.assertNotIn("drop_item(", pairs[0]["prompt"])  # the project flag hides it from the row's block
        preview = out["details"]["predictions_preview"]
        self.assertIn("tool drop_item is not allowed", preview[0]["row_tool_results"]["errors"])
        self.assertEqual([p["row_tool_list_source"] for p in preview], ["row", "project"])
        self.assertTrue(preview[1]["row_tool_results"]["tool_call_valid"])
        self.assertEqual(out["metrics"]["row_context"]["kinds"]["tools"], {"row": 1, "project": 1, "none": 0})

    def test_no_context_keeps_the_old_metrics_shape(self):
        catalog = [tool("get_item")]
        pid, exp = seed([tool_row(0), tool_row(1)], profile="tool_calling", adapter="tool-call-json", catalog=catalog)
        out = evaluate(pid, exp, [call("get_item", id="0"), call("get_item", id="1")])
        m = out["metrics"]
        self.assertNotIn("row_context", m)
        self.assertNotIn("per_row", m["tool_flags"])
        self.assertNotIn("per_row", m["tools"])
        self.assertTrue(m["tool_flags"]["matches_training"])
        rc = run_async(lambda db: db.get(Project, pid)).runtime_config
        self.assertEqual(m["tool_call_fingerprint"], tcs.tool_call_fingerprint(rc))
        self.assertNotIn("+rowtools", m["tool_call_fingerprint"])
        for p in out["details"]["predictions_preview"]:
            self.assertNotIn("row_context", p)
            self.assertNotIn("row_tool_list_source", p)


class SystemMessageTests(unittest.TestCase):
    """The row's own system message reaches eval for every task shape (genericity)."""

    def _check(self, rows, profile, adapter, outputs, config=None):
        pid, exp = seed(rows, profile=profile, adapter=adapter, config=config)
        pairs: list[dict] = []
        out = evaluate(pid, exp, outputs, pairs)
        self.assertEqual(pairs[0]["row_system"], "Follow the house style.")
        self.assertNotIn("row_system", pairs[1])
        self.assertEqual(out["metrics"]["row_context"]["kinds"]["system"], {"row": 1, "none": 1})
        self.assertEqual(out["details"]["predictions_preview"][0]["row_context"]["system"], "Follow the house style.")
        self.assertNotIn("row_context", out["details"]["predictions_preview"][1])
        self.assertEqual(out["metrics"]["prompt_format_version"], 2)
        return out

    def test_qa(self):
        self._check([{"question": "Capital of France?", "answer": "Paris",
                      "row_context": {"system": "Follow the house style."}},
                     {"question": "2+2?", "answer": "4"}], "qa", "default-canonical", ["Paris", "4"])

    def test_classification(self):
        self._check([{"text": "great", "label": "pos", "row_context": {"system": "Follow the house style."}},
                     {"text": "awful", "label": "neg"}], "classification", "classification-label", ["pos", "neg"],
                    config={"candidates": ["neg", "pos"]})

    def test_tool_calling(self):
        self._check([tool_row(0, system="Follow the house style."), tool_row(1)], "tool_calling", "tool-call-json",
                    [call("get_item", id="0"), call("get_item", id="1")])

    def test_unused_kinds_are_reported(self):
        out = self._check([{"question": "q?", "answer": "a", "row_context": {"system": "Follow the house style.",
                                                                            "labels": ["a", "b"]}},
                           {"question": "r?", "answer": "b"}], "qa", "default-canonical", ["a", "b"])
        self.assertEqual(out["metrics"]["row_context"]["unused_kinds"], {"labels": 1})


if __name__ == "__main__":
    unittest.main()
