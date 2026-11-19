"""Import keeps ``row_context``: the predefined train / test import, the shared data-prep mapper (the
random split) and the answer-key file import. A row whose context is unusable is rejected with a
``row_context_*`` reason and its line (grouped; the gold import lets the user drop those rows). Data
health treats a row's own tool block as context, not content: it never makes rows near-duplicates, and
the predefined leak key stays the bare user prompt.

pytest-only. No model, no network."""

from __future__ import annotations

import json
import time
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.models.dataset import DatasetType
from app.services.data_health_service import context_block_lines
from app.services import row_context_service as rcs
from app.services.dataset_service import _normalize_rows_for_training, _prep_dir

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    from _job_wait import wait_for_jobs_idle

    wait_for_jobs_idle(client, timeout_s=60)
    client.__exit__(None, None, None)


def tool(name: str) -> dict:
    return {"name": name, "description": f"{name} tool",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}}


TOOLS = [tool(f"tool_{i}") for i in range(8)]


def trow(i: int, **ctx) -> dict:
    row = {"prompt": f"Please fetch record {i} from the archive", "tool_name": "tool_1", "tool_args": {"id": str(i)},
           "result": "ok"}
    if ctx:
        row["row_context"] = ctx
    return row


def jsonl(rows: list) -> bytes:
    return ("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n").encode()


def new_project() -> int:
    return client.post("/api/projects", json={"name": f"rowctx-import-{time.time_ns()}", "description": "x"}).json()["id"]


def predefined(pid: int, train: list, test: list):
    files = {"train": ("train.jsonl", jsonl(train), "application/octet-stream"),
             "test": ("test.jsonl", jsonl(test), "application/octet-stream")}
    return client.post(f"/api/projects/{pid}/dataset/predefined-split", files=files,
                       data={"adapter_id": "tool-call-json", "task_profile": "tool_calling"})


def prepared(pid: int, name: str) -> list[dict]:
    path = _prep_dir(pid) / f"{name}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


class PredefinedImportTests(unittest.TestCase):
    def test_row_context_survives_and_renders_the_rows_own_list(self):
        pid = new_project()
        train = [trow(i, tools=TOOLS[:4], system="Use the archive tools.") for i in range(3)]
        test = [trow(10, tools=TOOLS[4:]), trow(11)]
        resp = predefined(pid, train, test)
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = prepared(pid, "train")
        self.assertIsInstance(rows[0]["row_context"], str)  # prepared files: canonical JSON text
        self.assertEqual(len(rcs.decode(rows[0]["row_context"])["tools"]), 4)
        self.assertEqual(rows[0]["system_prompt"], "Use the archive tools.")
        self.assertIn("- tool_3(", rows[0]["question"])
        self.assertTrue(rows[0]["question"].endswith("\n\n" + rows[0]["user_prompt"]))
        test_rows = prepared(pid, "test")
        self.assertIn("- tool_7(", test_rows[0]["question"])
        self.assertNotIn("row_context", test_rows[1])  # a row without context: unchanged
        self.assertEqual(test_rows[1]["question"], "Please fetch record 11 from the archive")

    def test_tool_list_order_is_the_rows_and_the_manifest_says_so(self):
        pid = new_project()
        shuffled = [TOOLS[3], TOOLS[0], TOOLS[2]]
        resp = predefined(pid, [trow(i, tools=shuffled) for i in range(3)], [trow(10, tools=shuffled)])
        self.assertEqual(resp.status_code, 200, resp.text)
        q = prepared(pid, "train")[0]["question"]
        order = [q.index(f"- {t['name']}(") for t in shuffled]
        self.assertEqual(order, sorted(order))
        manifest = json.loads((_prep_dir(pid) / "manifest.json").read_text())
        self.assertEqual(manifest["row_tool_order"], "row")
        plain = new_project()
        self.assertEqual(predefined(plain, [trow(i) for i in range(3)], [trow(10)]).status_code, 200)
        self.assertNotIn("row_tool_order", json.loads((_prep_dir(plain) / "manifest.json").read_text()))

    def test_bad_row_context_is_rejected_with_its_line(self):
        pid = new_project()
        train = [trow(0), trow(1, tools=[tool("dup"), tool("dup")]), trow(2, sytem="typo")]
        resp = predefined(pid, train, [trow(10)])
        self.assertEqual(resp.status_code, 422, resp.text)
        by_reason = resp.json()["metadata"]["rejected"]["train"]["by_reason"]
        self.assertEqual(by_reason["row_context_tools_invalid"]["sample_rows"], [2])
        self.assertEqual(by_reason["row_context_unknown_kind"]["sample_rows"], [3])

    def test_leak_key_is_the_bare_prompt_not_the_tool_block(self):
        pid = new_project()
        # same request in train and test, different per-row lists: still a leak
        resp = predefined(pid, [trow(0, tools=TOOLS[:2]), trow(1), trow(2)], [trow(0, tools=TOOLS[5:])])
        self.assertEqual(resp.status_code, 409, resp.text)


class MapperTests(unittest.TestCase):
    def test_random_split_mapper_rejects_with_reason(self):
        rejected: list = []
        rows = _normalize_rows_for_training(
            [{"question": "q1", "answer": "a"}, {"question": "q2", "answer": "b", "row_context": {"system": 5}},
             {"question": "q3", "answer": "c", "row_context": {"system": "S"}}],
            DatasetType.RAW, "llama3", adapter_id="default-canonical", rejected=rejected,
        )
        self.assertEqual(rejected, [(1, "row_context_system_invalid")])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["system_prompt"], "S")
        self.assertNotIn("system_prompt", rows[0])

    def test_prepared_files_with_differently_shaped_contexts_load_in_train_py(self):
        # Per-row tool lists whose schemas differ (parameter names, types, nested objects, enums, a type
        # clash on the same key) plus other kinds on other rows. As objects, Arrow schema inference in
        # ``datasets.load_dataset("json")`` (train.py's loader) fails or null-fills them on datasets 3.x;
        # prepared files store ``row_context`` as JSON text, which always loads and round-trips exactly.
        import importlib.util
        import sys
        from pathlib import Path

        from datasets import load_dataset

        def schema_tool(name, props, required=()):
            return {"name": name, "description": f"{name} tool",
                    "inputSchema": {"type": "object", "properties": props, "required": list(required)}}

        lists = [
            [schema_tool("get_a", {"id": {"type": "string", "default": "x"}}, ["id"])],
            [schema_tool("get_b", {"id": {"type": "integer", "default": 3},
                                   "opts": {"type": "object", "properties": {"mode": {"type": "string",
                                                                                     "enum": ["fast", "slow"]}}}})],
            [schema_tool("get_c", {"tags": {"type": "array", "items": {"enum": ["a", 1, True]}},
                                   "n": {"type": ["number", "null"]}}, ["tags"]),
             schema_tool("get_d", {})],
        ]
        train = [trow(i, tools=lst, system="S" if i == 1 else None) for i, lst in enumerate(lists)]
        train.append(trow(3, tools=[]))
        train.append(trow(4))  # no context at all
        pid = new_project()
        resp = predefined(pid, train, [trow(10, tools=lists[1]), trow(11)])
        self.assertEqual(resp.status_code, 200, resp.text)
        written = prepared(pid, "train")
        with_ctx = [r for r in written if "row_context" in r]
        self.assertEqual(len(with_ctx), 4)
        self.assertTrue(all(isinstance(r["row_context"], str) for r in with_ctx))
        # Mixed kinds in one file (labels / output_schema are kept, not consumed) also load.
        extra = _normalize_rows_for_training(
            [{"question": "q", "answer": "a", "row_context": {"labels": ["p", "n"]}},
             {"question": "q2", "answer": "b", "row_context": {"output_schema": {
                 "type": "object", "properties": {"v": {"type": "integer"}}}}}],
            DatasetType.RAW, "llama3")
        path = _prep_dir(pid) / "train.jsonl"
        path.write_text(path.read_text() + "".join(json.dumps(r) + "\n" for r in extra))

        spec = importlib.util.spec_from_file_location(
            "brewslm_train_loader", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
        train_mod = importlib.util.module_from_spec(spec)
        sys.modules["brewslm_train_loader"] = train_mod
        spec.loader.exec_module(train_mod)
        splits = train_mod._load_jsonl_splits(load_dataset, path, _prep_dir(pid) / "test.jsonl")
        ds = splits["train"]
        self.assertEqual(len(ds), len(written) + 2)
        on_disk = prepared(pid, "train")
        for i, row in enumerate(ds):
            self.assertEqual(row.get("row_context"), on_disk[i].get("row_context"), f"row {i}")
            self.assertEqual(rcs.decode(row.get("row_context")), rcs.decode(on_disk[i].get("row_context")))
        self.assertEqual(rcs.row_tools(ds[1])[0]["name"], "get_b")
        self.assertEqual(train_mod._row_context_system(ds[1]), "S")
        self.assertEqual(train_mod._row_context_system(ds[0]), "")
        self.assertEqual(rcs.decode(ds[len(written)]["row_context"]), {"labels": ["p", "n"]})
        self.assertEqual(len(splits["validation"]), 2)


class GoldImportTests(unittest.TestCase):
    def _import(self, pid, rows, **form):
        data = {"dataset_type": "gold_dev", **{k: str(v) for k, v in form.items()}}
        return client.post(f"/api/projects/{pid}/gold/import-file", data=data,
                           files={"file": ("gold.jsonl", jsonl(rows), "application/octet-stream")})

    def test_gold_rows_keep_row_context_and_bad_ones_are_droppable(self):
        pid = new_project()
        rows = [{"question": "q", "answer": "a", "row_context": {"system": "S", "tools": [tool("b"), tool("a")]}},
                {"question": "q2", "answer": "b", "row_context": {"tools": "nope"}}]
        resp = self._import(pid, rows)
        self.assertEqual(resp.status_code, 422, resp.text)
        group = resp.json()["metadata"]["rejected"]["by_reason"]["row_context_tools_invalid"]
        self.assertEqual(group["lines"], [2])
        self.assertIn("row_context.tools", group["label"])
        resp = self._import(pid, rows, drop_rejected="true")
        self.assertIn(resp.status_code, (200, 201), resp.text)
        entries = client.get(f"/api/projects/{pid}/gold/entries?dataset_type=gold_dev").json()
        entries = entries.get("entries", entries) if isinstance(entries, dict) else entries
        self.assertEqual(len(entries), 1)
        self.assertEqual([t["name"] for t in entries[0]["row_context"]["tools"]], ["b", "a"])  # the row's order

    def test_json_endpoints_validate_row_context_like_the_file_import(self):
        pid = new_project()
        bad = {"question": "q2", "answer": "b", "row_context": {"tools": "nope"}}
        good = {"question": "q", "answer": "a", "row_context": {"tools": [tool("b"), tool("a")]}}
        base = f"/api/projects/{pid}/gold"
        for resp in (client.post(f"{base}/import", json={"pairs": [good, bad], "dataset_type": "gold_dev"}),
                     client.post(f"{base}/add", json={**bad, "dataset_type": "gold_dev"}),
                     client.put(f"{base}/gold_dev", json={"rows": [good, bad]})):
            with self.subTest(url=str(resp.request.url)):
                self.assertEqual(resp.status_code, 422, resp.text)
                body = resp.json()
                self.assertEqual(body["error_code"], "GOLD_ROW_CONTEXT_INVALID")
                self.assertEqual(body["metadata"]["reason"], "row_context_tools_invalid")
                self.assertIn("row_context_tools_invalid", body["metadata"]["rejected"]["by_reason"])
        entries = client.get(f"{base}/entries?dataset_type=gold_dev").json()["entries"]
        self.assertEqual(entries, [])  # nothing was written
        resp = client.post(f"{base}/import", json={"pairs": [good], "dataset_type": "gold_dev"})
        self.assertEqual(resp.status_code, 200, resp.text)
        resp = client.post(f"{base}/add", json={**good, "dataset_type": "gold_dev"})
        self.assertEqual(resp.status_code, 201, resp.text)
        entries = client.get(f"{base}/entries?dataset_type=gold_dev").json()["entries"]
        # stored normalised (in the row's order), like the file import
        self.assertEqual([[t["name"] for t in e["row_context"]["tools"]] for e in entries], [["b", "a"], ["b", "a"]])


class DataHealthTests(unittest.TestCase):
    def test_per_row_tool_blocks_are_context_lines(self):
        from app.services.data_adapter_service import map_record_with_adapter

        mapped = [map_record_with_adapter(trow(i, tools=TOOLS[i % 2 * 4:(i % 2 + 1) * 4]), adapter_id="tool-call-json")
                  for i in range(4)]
        mapped = [{**m, "row_context": {"tools": []}} for m in mapped]
        lines = context_block_lines(mapped)
        self.assertIn("Available tools (call only these):", lines)
        self.assertEqual(sum(1 for ln in lines if ln.startswith("- tool_")), 8)  # both lists, unioned

    def test_plain_rows_have_no_context_lines(self):
        self.assertEqual(context_block_lines([{"question": "q", "answer": "a"}]), frozenset())

    def test_non_tool_rows_with_row_context_keep_their_lines_and_a_leak_is_caught(self):
        # A non-tool task whose rows carry row_context (a system message) and a question shaped like
        # "block + blank line + user_prompt": nothing is stripped, so a near-duplicate train/test pair
        # that differs only outside the block is still detected.
        from app.services.data_health_service import _scan_leakage_rows, leakage_boilerplate

        passage = "The Eiffel Tower in Paris was completed in 1889 for the World Fair by Gustave Eiffel's company."
        train = [{"question": f"{passage}\n\nWhen?", "user_prompt": "When?", "answer": "In 1889.",
                  "row_context": {"system": "Answer briefly."}}]
        test = [{"question": f"{passage}\n\nWhen?", "user_prompt": "When?", "answer": "1889",
                 "row_context": {"system": "Answer briefly."}}]
        self.assertEqual(context_block_lines(train + test), frozenset())
        boilerplate = leakage_boilerplate(train + test, {})
        self.assertEqual(boilerplate, frozenset())
        report = _scan_leakage_rows(train, {"test": test}, boilerplate)
        self.assertEqual(report["total_leaked"], 1)
        # What the old rule did: the passage was stripped as "context" and the leak was missed.
        stripped = _scan_leakage_rows(train, {"test": test}, frozenset({passage}))
        self.assertEqual(stripped["total_leaked"], 0)


class EffectiveProfileTests(unittest.TestCase):
    """The answer-key listing reports which row_context kinds the task reads (the UI labels from it)."""

    def _write_manifest(self, pid: int, profile: str) -> None:
        from pathlib import Path

        from app.config import settings

        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({"task_profile": profile}))

    def _reads(self, pid: int) -> dict:
        resp = client.get(f"/api/projects/{pid}/gold/entries", params={"dataset_type": "gold_dev"})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_tool_calling_project_reads_tools(self):
        pid = new_project()
        self._write_manifest(pid, "tool_calling")
        body = self._reads(pid)
        self.assertEqual(body["effective_task_profile"], "tool_calling")
        self.assertIn("tools", body["row_context_reads"])
        self.assertIn("system", body["row_context_reads"])

    def test_classification_project_does_not_read_tools(self):
        pid = new_project()
        self._write_manifest(pid, "classification")
        body = self._reads(pid)
        self.assertEqual(body["effective_task_profile"], "classification")
        self.assertNotIn("tools", body["row_context_reads"])
        self.assertEqual(body["row_context_reads"], ["system"])

    def test_tool_scoring_config_reports_the_effective_profile(self):
        # Before any prepare, the saved dataset preset is what the next prepare (and so eval) uses; the
        # Train page shows its tool-scoring section from this, not from the project response.
        pid = new_project()
        body = client.get(f"/api/projects/{pid}/tool-scoring-config").json()
        self.assertNotEqual(body.get("effective_task_profile"), "tool_calling")
        resp = client.put(f"/api/projects/{pid}/dataset/adapter-preference",
                          json={"adapter_id": "tool-call-json", "task_profile": "tool_calling"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = client.get(f"/api/projects/{pid}/tool-scoring-config").json()
        self.assertEqual(body["effective_task_profile"], "tool_calling")
        self.assertIn("tools", body["row_context_reads"])
        # a prepared manifest wins over the preset (what eval reads first)
        self._write_manifest(pid, "classification")
        body = client.put(f"/api/projects/{pid}/tool-scoring-config", json={"strict_args": True}).json()
        self.assertEqual(body["effective_task_profile"], "classification")
        self.assertTrue(body["config"]["strict_args"])

    def test_unknown_profile_still_reads_system_only(self):
        body = self._reads(new_project())
        self.assertNotIn("tools", body["row_context_reads"])


if __name__ == "__main__":
    unittest.main()
