"""Gold file management: import-file (JSONL / JSON / CSV, any task shape), grouped rejects with
select-and-drop, append vs replace, delete rows / clear set, stable row ids, version snapshots
(answer_key_stamp stays resolvable), locked refusal, ``source_row_id``, and the legacy endpoints
unchanged. pytest-only; no Jobs."""

from __future__ import annotations

import hashlib
import json
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def _project() -> int:
    return client.post("/api/projects", json={"name": f"gold-mgmt-{time.time_ns()}", "description": "x"}).json()["id"]


def _jsonl(rows: list) -> bytes:
    return ("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n").encode()


# One row shape per task type: the import must not care which.
SHAPES = {
    "qa": [{"question": "Capital of France?", "answer": "Paris"}, {"question": "2+2?", "answer": "4"}],
    "classification": [{"text": "great product", "label": "positive"}, {"text": "broke fast", "label": "negative"}],
    "span": [{"text": "Ada lives in Paris", "entities": [{"text": "Ada", "type": "PERSON", "start": 0, "end": 3}]},
             {"text": "no entities here", "entities": []}],
    "extraction": [{"text": "Invoice 7 for $5", "structured_output": {"invoice": 7, "total": 5}},
                   {"text": "Invoice 8 for $9", "structured_output": {"invoice": 8, "total": 9}}],
    "tool_call": [{"question": "weather in Oslo", "tool_name": "get_weather", "tool_args": {"city": "Oslo"}},
                  {"question": "time in Rome", "tool_name": "get_time", "tool_args": {"city": "Rome"}}],
}


def _import(pid, rows, filename="gold.jsonl", **form):
    data = {"dataset_type": "gold_dev", **{k: str(v) for k, v in form.items()}}
    return client.post(
        f"/api/projects/{pid}/gold/import-file", data=data,
        files={"file": (filename, rows if isinstance(rows, bytes) else _jsonl(rows), "application/octet-stream")},
    )


def _entries(pid, ds="gold_dev"):
    return client.get(f"/api/projects/{pid}/gold/entries?dataset_type={ds}").json()


class ImportAcrossTaskShapesTests(unittest.TestCase):
    def test_every_task_shape_imports_and_round_trips(self):
        for name, rows in SHAPES.items():
            with self.subTest(task=name):
                pid = _project()
                resp = _import(pid, rows)
                self.assertEqual(resp.status_code, 200, resp.text)
                self.assertEqual(resp.json()["imported"], 2)
                got = _entries(pid)
                self.assertEqual(got["total"], 2)
                for sent, stored in zip(rows, got["entries"]):
                    for key, value in sent.items():
                        self.assertEqual(stored[key], value)
                self.assertEqual([e["id"] for e in got["entries"]], [1, 2])
                self.assertEqual(got["dataset"]["dataset_type"], "gold_dev")
                self.assertEqual(got["dataset"]["record_count"], 2)

    def test_delete_and_replace_work_for_every_task_shape(self):
        for name, rows in SHAPES.items():
            with self.subTest(task=name):
                pid = _project()
                _import(pid, rows)
                gone = client.post(f"/api/projects/{pid}/gold/gold_dev/rows/delete", json={"row_ids": [1]}).json()
                self.assertEqual((gone["deleted"], gone["remaining"]), (1, 1))
                self.assertEqual([e["id"] for e in _entries(pid)["entries"]], [2])
                rep = client.put(f"/api/projects/{pid}/gold/gold_dev", json={"rows": rows[:1]}).json()
                self.assertEqual(rep["total"], 1)

    def test_json_array_and_csv_files(self):
        pid = _project()
        arr = json.dumps(SHAPES["classification"]).encode()
        self.assertEqual(_import(pid, arr, "gold.json").json()["imported"], 2)
        pid2 = _project()
        csv_bytes = b'text,label,entities\nhello,greet,"[{""text"": ""x""}]"\nbye,leave,\n'
        self.assertEqual(_import(pid2, csv_bytes, "gold.csv").json()["imported"], 2)
        rows = _entries(pid2)["entries"]
        self.assertEqual(rows[0]["entities"], [{"text": "x"}])
        self.assertEqual(rows[1]["label"], "leave")

    def test_field_mapping_renames_columns(self):
        pid = _project()
        resp = _import(pid, [{"prompt": "q1", "completion": "a1"}], field_mapping=json.dumps({"prompt": "question", "completion": "answer"}))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(_entries(pid)["entries"][0]["question"], "q1")


class RejectsAndDropTests(unittest.TestCase):
    ROWS = [
        {"question": "ok1", "answer": "a"},
        "{not json",
        {"question": "", "answer": "  "},
        '["a list"]',
        "also broken",
        {"question": "ok2", "answer": "b"},
    ]

    def test_rejects_are_grouped_by_reason_with_lines_and_nothing_is_written(self):
        pid = _project()
        resp = _import(pid, self.ROWS)
        self.assertEqual(resp.status_code, 422)
        body = resp.json()
        self.assertEqual(body["error_code"], "GOLD_IMPORT_REJECTED_ROWS")
        groups = body["metadata"]["rejected"]["by_reason"]
        self.assertEqual(groups["invalid_json"]["lines"], [2, 5])
        self.assertEqual(groups["empty_row"]["lines"], [3])
        self.assertEqual(groups["not_an_object"]["lines"], [4])
        self.assertEqual(body["metadata"]["row_counts"]["usable"], 2)
        self.assertEqual(_entries(pid)["total"], 0)

    def test_selected_rows_dropped_then_rest_still_blocked_until_all_dropped(self):
        pid = _project()
        partial = _import(pid, self.ROWS, drop_lines=json.dumps([2, 5, 3]))
        self.assertEqual(partial.status_code, 422)
        self.assertEqual(list(partial.json()["metadata"]["rejected"]["by_reason"]), ["not_an_object"])
        done = _import(pid, self.ROWS, drop_lines=json.dumps([2, 5, 3, 4]))
        self.assertEqual(done.status_code, 200, done.text)
        self.assertEqual((done.json()["imported"], done.json()["dropped_rows"]), (2, 4))
        self.assertEqual([e["question"] for e in _entries(pid)["entries"]], ["ok1", "ok2"])

    def test_drop_rejected_imports_the_rest(self):
        pid = _project()
        resp = _import(pid, self.ROWS, drop_rejected="true")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["dropped_lines"], [2, 3, 4, 5])

    def test_nothing_usable_is_an_error_even_when_dropping(self):
        pid = _project()
        resp = _import(pid, ["nope", "nope2"], drop_rejected="true")
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error_code"], "GOLD_IMPORT_NO_ROWS")

    def test_unreadable_file_is_400(self):
        pid = _project()
        self.assertEqual(_import(pid, b"[1, 2", "g.json").status_code, 400)
        self.assertEqual(_import(pid, SHAPES["qa"], mode="bogus").status_code, 400)


class ManagementTests(unittest.TestCase):
    def test_append_keeps_ids_stable_after_delete(self):
        pid = _project()
        _import(pid, SHAPES["qa"] + [{"question": "q3", "answer": "a3"}])
        client.post(f"/api/projects/{pid}/gold/gold_dev/rows/delete", json={"row_ids": [2]})
        self.assertEqual([e["id"] for e in _entries(pid)["entries"]], [1, 3])
        _import(pid, [{"question": "q4", "answer": "a4"}])
        self.assertEqual([e["id"] for e in _entries(pid)["entries"]], [1, 3, 4])
        # the legacy single-row path uses the same id rule
        added = client.post(f"/api/projects/{pid}/gold/add", json={"question": "q5", "answer": "a5"}).json()
        self.assertEqual(added["id"], 5)

    def test_delete_reports_missing_and_writes_nothing_when_none_match(self):
        pid = _project()
        _import(pid, SHAPES["qa"])
        out = client.post(f"/api/projects/{pid}/gold/gold_dev/rows/delete", json={"row_ids": [99]}).json()
        self.assertEqual((out["deleted"], out["missing"], out["remaining"]), (0, [99], 2))
        self.assertIsNone(out["snapshot_version"])
        self.assertEqual(client.post(f"/api/projects/{pid}/gold/gold_dev/rows/delete", json={"row_ids": []}).status_code, 400)

    def test_replace_via_file_snapshots_the_old_file_with_its_hash(self):
        pid = _project()
        _import(pid, SHAPES["qa"])
        ds = _entries(pid)["dataset"]
        path = Path(next(iter(_gold_paths(pid))))
        old_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        resp = _import(pid, SHAPES["classification"], mode="replace")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["snapshot_version"], 1)
        self.assertEqual([e["label"] for e in _entries(pid)["entries"]], ["positive", "negative"])
        self.assertEqual(ds["record_count"], 2)
        snap = _snapshot_files(pid)
        self.assertEqual(len(snap), 1)
        self.assertEqual(hashlib.sha256(snap[0].read_bytes()).hexdigest(), old_sha)

    def test_clear_keeps_dataset_and_versions(self):
        pid = _project()
        _import(pid, SHAPES["span"])
        out = client.delete(f"/api/projects/{pid}/gold/gold_dev").json()
        self.assertEqual((out["deleted"], out["remaining"], out["snapshot_version"]), (2, 0, 1))
        self.assertEqual(_entries(pid)["total"], 0)
        # clearing an empty set is a no-op that makes no snapshot
        again = client.delete(f"/api/projects/{pid}/gold/gold_dev").json()
        self.assertIsNone(again["snapshot_version"])

    def test_locked_set_refuses_every_change(self):
        pid = _project()
        _import(pid, SHAPES["qa"], dataset_type="gold_test")
        client.post(f"/api/projects/{pid}/gold/lock?dataset_type=gold_test")
        for resp in (
            _import(pid, SHAPES["qa"], dataset_type="gold_test"),
            _import(pid, SHAPES["qa"], dataset_type="gold_test", mode="replace"),
            client.post(f"/api/projects/{pid}/gold/gold_test/rows/delete", json={"row_ids": [1]}),
            client.put(f"/api/projects/{pid}/gold/gold_test", json={"rows": SHAPES["qa"]}),
            client.delete(f"/api/projects/{pid}/gold/gold_test"),
        ):
            self.assertEqual(resp.status_code, 409, resp.text)
            self.assertEqual(resp.json()["error_code"], "GOLD_LOCKED")
        self.assertEqual(_entries(pid, "gold_test")["total"], 2)
        self.assertTrue(_entries(pid, "gold_test")["dataset"]["is_locked"])

    def test_dev_and_test_are_independent_and_bad_type_404(self):
        pid = _project()
        _import(pid, SHAPES["qa"])
        _import(pid, SHAPES["span"], dataset_type="gold_test")
        client.delete(f"/api/projects/{pid}/gold/gold_dev")
        self.assertEqual(_entries(pid, "gold_test")["total"], 2)
        self.assertEqual(client.delete(f"/api/projects/{pid}/gold/gold_nope").status_code, 404)

    def test_caller_id_kept_as_source_row_id_and_system_id_wins(self):
        pid = _project()
        _import(pid, [{"id": "ext-17", "question": "q", "answer": "a"}, {"id": 900, "question": "q2", "answer": "a2"}])
        rows = _entries(pid)["entries"]
        self.assertEqual([(r["id"], r["source_row_id"]) for r in rows], [(1, "ext-17"), (2, 900)])
        # legacy JSON import: id still system-owned; source_row_id is additive
        client.post(f"/api/projects/{pid}/gold/import", json={"pairs": [{"id": "x", "question": "q3", "answer": "a3"}]})
        self.assertEqual(_entries(pid)["entries"][-1]["id"], 3)

    def test_legacy_endpoints_unchanged(self):
        pid = _project()
        self.assertEqual(client.post(f"/api/projects/{pid}/gold/import", json={"pairs": SHAPES["qa"]}).json(), {"imported": 2, "total": 2})
        self.assertEqual(client.put(f"/api/projects/{pid}/gold/review", json={}).status_code, 422)  # still routed to /review
        got = _entries(pid)
        self.assertEqual((got["total"], len(got["entries"])), (2, 2))


def _gold_paths(pid):
    from app.config import settings

    d = settings.DATA_DIR / "projects" / str(pid) / "gold"
    return [str(p) for p in d.glob("gold_dev.jsonl")]


def _snapshot_files(pid):
    from app.config import settings

    return sorted((settings.DATA_DIR / "projects" / str(pid) / "gold" / "versions").glob("*.jsonl"))


if __name__ == "__main__":
    unittest.main()
