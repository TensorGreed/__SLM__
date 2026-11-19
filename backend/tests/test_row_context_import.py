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
        self.assertEqual(len(rows[0]["row_context"]["tools"]), 4)
        self.assertEqual(rows[0]["system_prompt"], "Use the archive tools.")
        self.assertIn("- tool_3(", rows[0]["question"])
        self.assertTrue(rows[0]["question"].endswith("\n\n" + rows[0]["user_prompt"]))
        test_rows = prepared(pid, "test")
        self.assertIn("- tool_7(", test_rows[0]["question"])
        self.assertNotIn("row_context", test_rows[1])  # a row without context: unchanged
        self.assertEqual(test_rows[1]["question"], "Please fetch record 11 from the archive")

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
        self.assertEqual([t["name"] for t in entries[0]["row_context"]["tools"]], ["a", "b"])


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


if __name__ == "__main__":
    unittest.main()
