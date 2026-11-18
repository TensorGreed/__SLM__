"""Predefined-split import on a tool-calling project with a tool catalog: imported rows must carry the
project's tool block (allowed tools only) exactly as a random split renders it, without passing the block
in the import settings. The leak / duplicate key stays the bare user prompt (the block is shared
boilerplate). A project without a catalog imports as before.

pytest-only. No model, no network."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.project import Project
from app.services import structured_plan_prompt as spp
from app.services.dataset_service import _prep_dir

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    from _job_wait import wait_for_jobs_idle

    wait_for_jobs_idle(client, timeout_s=60)
    client.__exit__(None, None, None)


def _tool(name: str, props: dict, required=()) -> dict:
    return {"name": name, "description": f"{name} tool",
            "input_schema": {"type": "object", "properties": props, "required": list(required)},
            "annotations": {"readOnlyHint": True}, "read_only": True, "destructive": None}


TOOLS = [
    _tool("get_item", {"id": {"type": "string"}}, ["id"]),
    _tool("list_items", {}),
    _tool("drop_item", {"id": {"type": "string"}}, ["id"]),
]


def run_async(fn):
    async def go():
        async with async_session_factory() as db:
            return await fn(db)

    return asyncio.run(go())


def new_project(with_catalog: bool = True) -> int:
    pid = client.post("/api/projects", json={"name": f"tool-import-{time.time_ns()}", "description": "x"}).json()["id"]
    if with_catalog:
        from app.services import tool_catalog_service as tcat

        async def seed(db):
            project = await db.get(Project, pid)
            tcat.set_catalog_tools(project, TOOLS, {"kind": "manual"})
            tcat.set_tool_flags(project, {"drop_item": {"allowed": False}})
            await db.commit()

        run_async(seed)
    return pid


def row(i: int) -> dict:
    return {"prompt": f"Fetch item number {i} please", "tool_name": "get_item", "tool_args": {"id": f"it-{i}"},
            "result": "ok"}


def jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def predefined(pid: int, train, test, **form) -> dict:
    files = {"train": ("train.jsonl", jsonl(train), "application/octet-stream"),
             "test": ("test.jsonl", jsonl(test), "application/octet-stream")}
    resp = client.post(f"/api/projects/{pid}/dataset/predefined-split", files=files, data=form)
    assert resp.status_code == 200, resp.text
    return resp.json()


def prepared(pid: int) -> dict[str, list[dict]]:
    out = {}
    for name in ("train", "val", "test"):
        path = _prep_dir(pid) / f"{name}.jsonl"
        out[name] = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    return out


def block_of(pid: int) -> str:
    return spp.tool_block_for(run_async(lambda db: db.get(Project, pid)).runtime_config)


class PredefinedToolBlockTests(unittest.TestCase):
    def check_rows(self, pid: int, rows_by_split: dict[str, list[dict]]) -> None:
        block = block_of(pid)
        self.assertIn("get_item", block)
        self.assertNotIn("drop_item", block)  # the allowed flags are respected
        for split in ("train", "test"):
            self.assertEqual(len(rows_by_split[split]), 3 if split == "train" else 2)
            for r in rows_by_split[split]:
                self.assertTrue(r["question"].startswith(block), r["question"][:80])
                self.assertTrue(r["question"].endswith(r["user_prompt"]))
                self.assertTrue(r["user_prompt"].startswith("Fetch item number"))

    def test_adapter_request_renders_the_tool_block(self):
        pid = new_project()
        manifest = predefined(pid, [row(i) for i in range(3)], [row(i) for i in range(10, 12)],
                              adapter_id="tool-call-json", task_profile="tool_calling")
        self.check_rows(pid, prepared(pid))
        # derived from the project, never persisted into the wiring
        self.assertNotIn("tool_block", manifest["adapter_config"])

    def test_auto_adapter_resolved_from_the_task_profile_renders_the_tool_block(self):
        pid = new_project()
        manifest = predefined(pid, [row(i) for i in range(3)], [row(i) for i in range(10, 12)],
                              adapter_id="auto", task_profile="tool_calling")
        self.assertEqual(manifest["adapter_id"], "tool-call-json")
        self.check_rows(pid, prepared(pid))

    def test_saved_adapter_preference_renders_the_tool_block(self):
        pid = new_project()
        resp = client.put(f"/api/projects/{pid}/dataset/adapter-preference",
                          json={"adapter_id": "tool-call-json", "task_profile": "tool_calling"})
        self.assertEqual(resp.status_code, 200, resp.text)
        predefined(pid, [row(i) for i in range(3)], [row(i) for i in range(10, 12)])
        self.check_rows(pid, prepared(pid))

    def test_matches_the_random_split_prompt(self):
        pid = new_project()
        predefined(pid, [row(i) for i in range(3)], [row(i) for i in range(10, 12)],
                   adapter_id="tool-call-json", task_profile="tool_calling")
        imported = {r["user_prompt"]: r for split in prepared(pid).values() for r in split}

        ref = new_project()
        client.put(f"/api/projects/{ref}/dataset/adapter-preference",
                   json={"adapter_id": "tool-call-json", "task_profile": "tool_calling"})
        path = Path(settings.DATA_DIR) / "projects" / str(ref) / f"synthetic_{time.time_ns()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [row(i) for i in range(3)] + [row(i) for i in range(10, 12)]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        async def add(db):
            db.add(Dataset(project_id=ref, name="synthetic", dataset_type=DatasetType.SYNTHETIC,
                           file_path=str(path), record_count=len(rows)))
            await db.commit()

        run_async(add)
        resp = client.post(f"/api/projects/{ref}/dataset/split", json={
            "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "dedup_rows": False,
            "adapter_id": "tool-call-json", "task_profile": "tool_calling"})
        self.assertEqual(resp.status_code, 200, resp.text)
        native = {r["user_prompt"]: r for split in prepared(ref).values() for r in split}
        self.assertEqual(set(native), set(imported))
        for key, r in imported.items():
            for field in ("question", "source_text", "text", "answer"):
                self.assertEqual(r[field], native[key][field], field)

    def test_near_duplicate_leak_key_is_the_user_prompt(self):
        pid = new_project()
        files = {"train": ("train.jsonl", jsonl([row(1), row(2)]), "application/octet-stream"),
                 "test": ("test.jsonl", jsonl([row(1)]), "application/octet-stream")}
        resp = client.post(f"/api/projects/{pid}/dataset/predefined-split", files=files,
                           data={"adapter_id": "tool-call-json", "task_profile": "tool_calling"})
        self.assertEqual(resp.status_code, 409, resp.text)  # the same request in train and test is a leak
        pairs = resp.json()["metadata"]["pairs"]
        self.assertEqual(pairs[0]["counts"], {"train": 1, "test": 1})  # only that row, not every row

    def test_no_catalog_imports_as_before(self):
        pid = new_project(with_catalog=False)
        predefined(pid, [row(i) for i in range(3)], [row(i) for i in range(10, 12)],
                   adapter_id="tool-call-json", task_profile="tool_calling")
        for split in prepared(pid).values():
            for r in split:
                self.assertNotIn("user_prompt", r)
                self.assertTrue(r["question"].startswith("Fetch item number"))


if __name__ == "__main__":
    unittest.main()
