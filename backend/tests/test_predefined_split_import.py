"""POST /dataset/predefined-split: import train / val / test files with exact
membership, through the same row pipeline and prepared artefacts as a random
split. Pytest-only (TestClient + shared engine)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType, DatasetVersion, DocumentStatus, RawDocument
from app.models.project import Project
from app.services.dataset_service import _prep_dir
from sqlalchemy import select

client: TestClient
_cm = None
_prev_auth = None


def setup_module(module):
    global _cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _cm = TestClient(app)
    client = _cm.__enter__()


def teardown_module(module):
    _cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


def _qa(prefix: str, n: int, start: int = 0) -> list[dict]:
    return [
        {"question": f"{prefix} question number {start + i}?", "answer": f"{prefix} answer {start + i}"}
        for i in range(n)
    ]


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def _new_project() -> int:
    resp = client.post("/api/projects", json={"name": f"predef-{uuid4().hex[:6]}", "description": ""})
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _predef(pid: int, train=None, val=None, test=None, **form):
    files = {}
    for name, content, fname in (("train", train, "train.jsonl"), ("val", val, "val.jsonl"), ("test", test, "test.jsonl")):
        if content is not None:
            files[name] = (fname, content, "application/octet-stream")
    return client.post(f"/api/projects/{pid}/dataset/predefined-split", files=files, data=form)


def _prepared(pid: int, split: str) -> list[dict]:
    path = _prep_dir(pid) / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _texts(rows: list[dict]) -> list[str]:
    return [r["question"] for r in rows]


def _run(coro):
    return asyncio.run(coro)


async def _query(fn):
    async with async_session_factory() as db:
        return await fn(db)


def test_exact_membership_and_manifest_and_versions():
    pid = _new_project()
    train, val, test = _qa("tr", 12), _qa("va", 4), _qa("te", 5)
    resp = _predef(pid, _jsonl(train), _jsonl(val), _jsonl(test))
    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    for name, rows in (("train", train), ("val", val), ("test", test)):
        got = _prepared(pid, name)
        assert [r["question"] for r in got] == [r["question"] for r in rows], "exact membership, file order"
        assert hashlib.sha256((_prep_dir(pid) / f"{name}.jsonl").read_bytes()).hexdigest() == manifest["file_hashes"][name]
    assert manifest["split_source"] == "predefined"
    assert manifest["splits"] == {"train": 12, "val": 4, "test": 5}
    assert manifest["source_files"]["test"]["filename"] == "test.jsonl"
    assert manifest["leak_report"] == {"on_leak": "fail", "pairs": [], "dropped_from_train": 0}
    assert manifest["duplicate_report"] == {"train": 0, "val": 0, "test": 0}
    assert manifest["prepared_version"] and set(manifest["dataset_versions"]) == {"train", "val", "test"}
    on_disk = json.loads((_prep_dir(pid) / "manifest.json").read_text())
    assert on_disk["split_source"] == "predefined" and on_disk["file_hashes"] == manifest["file_hashes"]
    assert (_prep_dir(pid) / "versions" / str(manifest["prepared_version"]) / "test.jsonl").exists()

    async def versions(db):
        out = {}
        for ds_type in (DatasetType.TRAIN, DatasetType.VALIDATION, DatasetType.TEST):
            ds = (await db.execute(select(Dataset).where(Dataset.project_id == pid, Dataset.dataset_type == ds_type))).scalar_one()
            vs = (await db.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id))).scalars().all()
            out[ds_type.value] = [v.manifest for v in vs]
        docs = (await db.execute(select(RawDocument))).scalars().all()
        out["docs"] = [(d.source, d.status.value, (d.metadata_ or {}).get("split_role")) for d in docs if d.source.startswith("predefined_split:")]
        return out

    got = _run(_query(versions))
    assert got["test"][0]["split_source"] == "predefined" and got["test"][0]["source_filename"] == "test.jsonl"
    assert got["train"][0]["count"] == 12
    assert ("predefined_split:val", "accepted", "val") in got["docs"]


def _set_pref(pid: int, **body):
    resp = client.put(f"/api/projects/{pid}/dataset/adapter-preference", json=body)
    assert resp.status_code == 200, resp.text


def _clear_split_roles(pid: int) -> None:
    """Let a random split pool the registered files (the test file is normally
    refused: see the keep_test tests)."""

    async def clear(db):
        raw = (await db.execute(select(Dataset).where(Dataset.project_id == pid, Dataset.dataset_type == DatasetType.RAW))).scalar_one()
        for d in (await db.execute(select(RawDocument).where(RawDocument.dataset_id == raw.id))).scalars():
            meta = dict(d.metadata_ or {})
            meta.pop("split_role", None)
            d.metadata_ = meta
        await db.commit()

    _run(_query(clear))
    manifest_path = _prep_dir(pid) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("split_source", None)
    manifest_path.write_text(json.dumps(manifest))


def _chat(user: str, answer: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


def _equiv_cases():
    cls = lambda p, n, labels: [{"text": f"{p} text {i}", "label": labels[i % len(labels)]} for i in range(n)]
    ext = lambda p, n, keys: [{"text": f"{p} doc {i}", "structured_output": {k: i for k in keys}} for i in range(n)]
    ins = lambda p, n: [{"instruction": "Summarize.", "input": f"{p} body {i}", "output": f"{p} sum {i}"} for i in range(n)]
    return {
        "qa": (None, _qa("tr", 14), _qa("va", 3), _qa("te", 3)),
        # the test file has a label the train file lacks: a per-file scan would differ
        "classification": (
            ("classification-label", "classification"),
            cls("tr", 12, ["pos", "neg"]), cls("va", 4, ["pos", "neg"]), cls("te", 6, ["pos", "neg", "neutral"]),
        ),
        "structured-extraction": (
            ("structured-extraction", "structured_extraction"),
            ext("tr", 12, ["a", "b"]), ext("va", 3, ["a", "b"]), ext("te", 4, ["a"]),
        ),
        "instruction": (None, ins("tr", 10), ins("va", 3), ins("te", 3)),
        "chat": (None, [_chat(f"tr q{i}", f"a{i}") for i in range(10)], [_chat(f"va q{i}", "a") for i in range(3)], [_chat(f"te q{i}", "a") for i in range(3)]),
    }


def test_equivalent_to_random_split_of_the_same_rows():
    for name, (adapter, train, val, test) in _equiv_cases().items():
        pid = _new_project()
        if adapter:
            _set_pref(pid, adapter_id=adapter[0], task_profile=adapter[1])
        resp = _predef(pid, _jsonl(train), _jsonl(val), _jsonl(test))
        assert resp.status_code == 200, (name, resp.text)
        n = len(train) + len(val) + len(test)
        predefined = Counter(
            json.dumps(r, sort_keys=True) for s in ("train", "val", "test") for r in _prepared(pid, s)
        )
        # The same source rows, prepared by a RANDOM split of the registered files.
        _clear_split_roles(pid)
        resp = client.post(
            f"/api/projects/{pid}/dataset/split",
            json={"include_types": ["raw"], "dedup_rows": False, "auto_stratify": False},
        )
        assert resp.status_code == 200, (name, resp.text)
        assert resp.json().get("split_source") is None
        randomised = Counter(
            json.dumps(r, sort_keys=True) for s in ("train", "val", "test") for r in _prepared(pid, s)
        )
        assert sum(randomised.values()) == n, name
        assert predefined == randomised, f"{name}: byte-identical prepared rows, only the membership differs"


def test_rejected_rows_fail_422_grouped_with_sample_rows():
    pid = _new_project()
    assert client.put(
        f"/api/projects/{pid}/dataset/adapter-preference",
        json={"adapter_id": "structured-extraction", "task_profile": "structured_extraction"},
    ).status_code == 200
    good = [{"text": f"doc {i}", "structured_output": {"a": i}} for i in range(5)]
    bad_train = good + [{"text": "no target here"}, {"note": "nothing usable"}]
    bad_test = [{"text": "t0", "structured_output": {"a": 0}}, {"text": ""}]
    resp = _predef(pid, _jsonl(bad_train), None, _jsonl(bad_test))
    assert resp.status_code == 422, resp.text
    body = resp.json()
    rejected = body["metadata"]["rejected"]
    assert rejected["train"]["count"] == 2
    reasons = rejected["train"]["by_reason"]["adapter_unmappable_or_empty"]
    assert reasons["sample_rows"] == [6, 7] and reasons["count"] == 2
    assert rejected["test"]["by_reason"]["adapter_unmappable_or_empty"]["sample_rows"] == [2]
    assert not (_prep_dir(pid) / "train.jsonl").exists(), "nothing written on failure"


def test_leak_each_pair_409_and_val_test_always():
    for pair, mk in (
        ("train_val", lambda: (_qa("tr", 6) + [_qa("va", 1)[0]], _qa("va", 3), _qa("te", 3))),
        ("train_test", lambda: (_qa("tr", 6) + [_qa("te", 1)[0]], _qa("va", 3), _qa("te", 3))),
        ("val_test", lambda: (_qa("tr", 6), _qa("va", 3) + [_qa("te", 1)[0]], _qa("te", 3))),
    ):
        pid = _new_project()
        train, val, test = mk()
        resp = _predef(pid, _jsonl(train), _jsonl(val), _jsonl(test))
        assert resp.status_code == 409, (pair, resp.text)
        pairs = {p["pair"]: p for p in resp.json()["metadata"]["pairs"]}
        assert set(pairs) == {pair}
        assert sum(pairs[pair]["counts"].values()) >= 2
        assert pairs[pair]["sample_rows"][pair.split("_")[0]] == [7] or pair == "val_test"
        assert not (_prep_dir(pid) / "manifest.json").exists()
    # val<->test overlap fails even when asked to drop from train.
    pid = _new_project()
    resp = _predef(
        pid, _jsonl(_qa("tr", 5)), _jsonl(_qa("va", 3) + _qa("te", 1)), _jsonl(_qa("te", 3)), on_leak="drop_from_train",
    )
    assert resp.status_code == 409 and resp.json()["error_code"] == "PREDEFINED_SPLIT_VAL_TEST_OVERLAP"


def test_leak_matching_ignores_case_and_spacing():
    pid = _new_project()
    train = _qa("tr", 5) + [{"question": "TE   question number 1?", "answer": "x"}]
    resp = _predef(pid, _jsonl(train), None, _jsonl(_qa("te", 3)))
    assert resp.status_code == 409, resp.text


def test_drop_from_train_never_touches_val_or_test():
    pid = _new_project()
    train = _qa("tr", 8) + [_qa("va", 2)[1], _qa("te", 3)[2]]
    val, test = _qa("va", 3), _qa("te", 4)
    resp = _predef(pid, _jsonl(train), _jsonl(val), _jsonl(test), on_leak="drop_from_train")
    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["splits"] == {"train": 8, "val": 3, "test": 4}
    assert manifest["leak_report"]["on_leak"] == "drop_from_train"
    assert manifest["leak_report"]["dropped_from_train"] == 2
    assert manifest["leak_report"]["dropped_sample_rows"] == [9, 10]
    assert {p["pair"] for p in manifest["leak_report"]["pairs"]} == {"train_val", "train_test"}
    assert _texts(_prepared(pid, "val")) == _texts(_qa("va", 3)) and _texts(_prepared(pid, "test")) == _texts(_qa("te", 4))
    assert _texts(_prepared(pid, "train")) == _texts(_qa("tr", 8))


def test_within_split_duplicates_are_kept_and_counted():
    pid = _new_project()
    train = _qa("tr", 5) + [_qa("tr", 1)[0], _qa("tr", 1)[0]]
    test = _qa("te", 3) + [_qa("te", 1)[0]]
    resp = _predef(pid, _jsonl(train), None, _jsonl(test))
    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["splits"] == {"train": 7, "val": 0, "test": 4}, "nothing removed, no empty-val padding"
    assert manifest["duplicate_report"] == {"train": 2, "val": 0, "test": 1}
    assert len(_prepared(pid, "train")) == 7


def test_csv_files_and_field_mapping():
    pid = _new_project()
    train = b"prompt,reply\nwhat is a?,A\nwhat is b?,B\nwhat is c?,C\n"
    test = b"prompt,reply\nwhat is z?,Z\n"
    resp = client.post(
        f"/api/projects/{pid}/dataset/predefined-split",
        files={"train": ("t.csv", train, "text/csv"), "test": ("e.csv", test, "text/csv")},
        data={"field_mapping": json.dumps({"prompt": "question", "reply": "answer"})},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["splits"] == {"train": 3, "val": 0, "test": 1}
    assert _texts(_prepared(pid, "test")) == ["what is z?"]
    bad = client.post(
        f"/api/projects/{pid}/dataset/predefined-split",
        files={"train": ("t.pdf", b"%PDF", "application/pdf")},
    )
    assert bad.status_code == 400


def test_structured_extraction_rows_carry_the_declared_field_list():
    pid = _new_project()
    fields = ["amount", "date", "vendor"]
    assert client.put(
        f"/api/projects/{pid}/dataset/adapter-preference",
        json={"adapter_id": "structured-extraction", "task_profile": "structured_extraction",
              "adapter_config": {"fields": fields}},
    ).status_code == 200

    async def pick_recipe(db):
        project = await db.get(Project, pid)
        project.selected_recipe = {"recipe_id": "span-extraction"}
        await db.commit()

    _run(_query(pick_recipe))

    def rows(prefix, n):
        # Deliberately partial payloads: the pinned list must win over the scan.
        return [{"text": f"{prefix} invoice {i}", "structured_output": {"amount": i}} for i in range(n)]

    resp = _predef(pid, _jsonl(rows("tr", 6)), _jsonl(rows("va", 2)), _jsonl(rows("te", 2)))
    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["adapter_id"] == "structured-extraction"
    assert manifest["adapter_config"]["fields"] == fields
    assert manifest["output_schema"]["scoring_mode"] == "span_set"
    for split in ("train", "val", "test"):
        for row in _prepared(pid, split):
            assert all(f in row["source_text"] for f in fields), row["source_text"]


def test_later_random_resplit_and_keep_test_keep_the_predefined_test_rows():
    pid = _new_project()
    train, val, test = _qa("tr", 30), _qa("va", 5), _qa("te", 6)
    assert _predef(pid, _jsonl(train), _jsonl(val), _jsonl(test)).status_code == 200
    test_before = _prepared(pid, "test")
    resp = client.post(
        f"/api/projects/{pid}/dataset/split",
        json={"include_types": ["raw"], "keep_test": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["keep_test"]["applied"] is True
    assert _prepared(pid, "test") == test_before
    assert len(_prepared(pid, "train")) + len(_prepared(pid, "val")) == 35
    # a plain random split would replace the user's own test file: refused, not silent
    plain = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"]})
    assert plain.status_code == 409, plain.text
    assert "test split came from your own test file" in plain.json()["message"]
    assert _prepared(pid, "test") == test_before


def test_reimport_supersedes_the_previous_files():
    pid = _new_project()
    assert _predef(pid, _jsonl(_qa("a", 5)), None, _jsonl(_qa("b", 2))).status_code == 200
    assert _predef(pid, _jsonl(_qa("c", 6)), None, _jsonl(_qa("d", 3))).status_code == 200

    async def statuses(db):
        docs = (await db.execute(select(RawDocument))).scalars().all()
        return Counter(
            d.status for d in docs
            if d.source.startswith("predefined_split:") and (d.metadata_ or {}).get("split_role")
        )

    # other tests' docs share the table; the latest two are accepted, so at least 2
    assert _run(_query(statuses))[DocumentStatus.ACCEPTED] >= 2
    resp = client.post(
        f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"], "dedup_rows": False, "keep_test": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["total_entries"] == 6, "only the latest import's train file is pooled (test rows are kept, not pooled)"
    assert len(_prepared(pid, "test")) == 3


def test_training_and_eval_read_the_prepared_files_unchanged():
    from app.services.eval_task_handler_service import read_prepared_manifest

    pid = _new_project()
    assert _predef(pid, _jsonl(_qa("tr", 6)), _jsonl(_qa("va", 2)), _jsonl(_qa("te", 2))).status_code == 200
    manifest = read_prepared_manifest(pid)
    assert manifest["split_source"] == "predefined"
    # Same paths the trainer / eval / lift-eval read (prepared/<split>.jsonl).
    for split, n in (("train", 6), ("val", 2), ("test", 2)):
        assert Path(manifest["file_paths"][split]) == _prep_dir(pid) / f"{split}.jsonl"
        assert len(_prepared(pid, split)) == n


# --- review-2 regressions -----------------------------------------------------


def _own_test_survives(pid: int, test_rows: list[dict], text_key: str = "text") -> None:
    """keep_test re-split keeps every test-file row in test and none in train / val;
    without keep_test the re-split is refused."""
    test_before = _prepared(pid, "test")
    refused = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"]})
    assert refused.status_code == 409, refused.text
    assert _prepared(pid, "test") == test_before
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"], "keep_test": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["keep_test"]["applied"] is True
    assert _prepared(pid, "test") == test_before
    test_texts = {r[text_key] for r in test_rows}
    pooled = _prepared(pid, "train") + _prepared(pid, "val")
    assert not [r for r in pooled if r.get(text_key) in test_texts], "test-file rows leaked into train / val"
    assert not [r for r in pooled if r.get("_source_document") and "test" in str(r["_source_document"])]


def _cls(prefix, n, labels):
    return [{"text": f"{prefix} text {i}", "label": labels[i % len(labels)]} for i in range(n)]


def test_keep_test_resplit_never_pools_the_test_file_classification():
    for tr_labels, te_labels in (
        (["pos", "neg"], ["pos", "neg", "neutral"]),  # test has a label train lacks
        (["pos", "neg", "neutral"], ["pos", "neg"]),  # test lacks a label train has
    ):
        pid = _new_project()
        _set_pref(pid, adapter_id="classification-label", task_profile="classification")
        test = _cls("te", 6, te_labels) + [{"text": "te rare", "label": "mixed"}]
        assert _predef(pid, _jsonl(_cls("tr", 21, tr_labels)), _jsonl(_cls("va", 3, tr_labels)), _jsonl(test)).status_code == 200
        _own_test_survives(pid, test)


def test_keep_test_resplit_never_pools_the_test_file_structured_extraction():
    pid = _new_project()
    _set_pref(pid, adapter_id="structured-extraction", task_profile="structured_extraction")
    tr = [{"text": f"tr doc {i}", "structured_output": {"a": i, "b": "x"}} for i in range(20)]
    va = [{"text": f"va doc {i}", "structured_output": {"a": i, "b": "x"}} for i in range(4)]
    te = [{"text": f"te doc {i}", "structured_output": {"a": i}} for i in range(6)]
    assert _predef(pid, _jsonl(tr), _jsonl(va), _jsonl(te)).status_code == 200
    _own_test_survives(pid, te)


def test_adapter_wiring_is_saved_as_the_project_preference():
    pid = _new_project()
    _set_pref(pid, adapter_id="classification-label", task_profile="classification")
    assert _predef(pid, _jsonl(_cls("tr", 8, ["pos", "neg"])), None, _jsonl(_cls("te", 3, ["neutral"]))).status_code == 200
    pref = client.get(f"/api/projects/{pid}/dataset/adapter-preference").json()
    assert pref["adapter_id"] == "classification-label"
    assert sorted(pref["adapter_config"]["candidates"]) == ["neg", "neutral", "pos"]


def test_test_file_never_pooled_without_keep_test_even_after_a_keep_test_resplit():
    pid = _new_project()
    assert _predef(pid, _jsonl(_qa("tr", 20)), _jsonl(_qa("va", 4)), _jsonl(_qa("te", 5))).status_code == 200
    assert client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"], "keep_test": True}).status_code == 200
    # the manifest no longer says "predefined", but the registered test document still guards the test rows
    assert client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"]}).status_code == 409


def _lines(*parts: str) -> bytes:
    return ("\n".join(parts) + "\n").encode()


def test_malformed_jsonl_lines_are_reported_by_physical_line_not_dropped():
    pid = _new_project()
    good = lambda i: json.dumps({"question": f"tr q{i}?", "answer": "a"})
    # line 1 ok, 2 blank, 3 truncated, 4 ok, 5 a JSON list, 6 ok
    train = _lines(good(0), "", '{"question": "broken?", "answer": ', good(1), "[1, 2]", good(2))
    resp = _predef(pid, train, None, _jsonl(_qa("te", 2)))
    assert resp.status_code == 422, resp.text
    slot = resp.json()["metadata"]["rejected"]["train"]["by_reason"]["unparseable_line"]
    assert slot == {"count": 2, "sample_rows": [3, 5]}
    assert not (_prep_dir(pid) / "train.jsonl").exists()
    # blank lines do not shift the numbering of later (mapping) rejects
    pid = _new_project()
    _set_pref(pid, adapter_id="structured-extraction", task_profile="structured_extraction")
    train = _lines(json.dumps({"text": "a", "structured_output": {"x": 1}}), "", "", json.dumps({"note": "nothing"}))
    resp = _predef(pid, train, None, _jsonl([{"text": "t", "structured_output": {"x": 1}}]))
    assert resp.json()["metadata"]["rejected"]["train"]["by_reason"]["adapter_unmappable_or_empty"]["sample_rows"] == [4]


def test_csv_rows_with_the_wrong_column_count_are_malformed_and_numbered_from_the_header():
    pid = _new_project()
    # header = line 1; line 3 has an extra cell, line 4 is blank, line 5 is short
    train = b"question,answer\nq one?,A\nq two?,B,extra\n\nq three?\nq four?,D\n"
    resp = client.post(
        f"/api/projects/{pid}/dataset/predefined-split",
        files={"train": ("t.csv", train, "text/csv")},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["metadata"]["rejected"]["train"]["by_reason"]["malformed_row"] == {"count": 2, "sample_rows": [3, 5]}


def test_json_array_rows_are_numbered_by_record_index():
    pid = _new_project()
    payload = json.dumps([{"question": "q1?", "answer": "a"}, "not an object", {"question": "q3?", "answer": "a"}]).encode()
    resp = client.post(
        f"/api/projects/{pid}/dataset/predefined-split", files={"train": ("t.json", payload, "application/json")},
    )
    assert resp.status_code == 422
    assert resp.json()["metadata"]["rejected"]["train"]["by_reason"]["unparseable_line"]["sample_rows"] == [2]


def test_review_queue_rows_are_reported_not_dropped_silently():
    pid = _new_project()
    train = _qa("tr", 4)
    train[1]["review_status"] = "pending"
    train[3]["review_status"] = "rejected"
    resp = _predef(pid, _jsonl(train), None, _jsonl(_qa("te", 2)))
    assert resp.status_code == 422, resp.text
    by_reason = resp.json()["metadata"]["rejected"]["train"]["by_reason"]
    assert by_reason["review_status_pending"]["sample_rows"] == [2]
    assert by_reason["review_status_rejected"]["sample_rows"] == [4]
    ok = _qa("tr", 4)
    ok[0]["review_status"] = "approved"
    assert _predef(pid, _jsonl(ok), None, _jsonl(_qa("te", 2))).status_code == 200


def _predef_rows(shape: str, prefix: str, i: int, **kw) -> dict:
    return {
        "instruction": {"instruction": kw.get("instruction", "Summarize."), "input": kw.get("input", f"{prefix} body {i}"), "output": kw.get("output", "s")},
        "chat": _chat(kw.get("user", f"{prefix} q{i}"), kw.get("answer", "a")),
        "extraction": {"text": kw.get("text", f"{prefix} doc {i}"), "structured_output": kw.get("out", {"a": i})},
    }[shape]


def test_leak_key_is_the_input_side_not_the_whole_row():
    # shared instruction, different inputs -> not a leak
    pid = _new_project()
    resp = _predef(
        pid, _jsonl([_predef_rows("instruction", "tr", i) for i in range(5)]), None,
        _jsonl([_predef_rows("instruction", "te", i) for i in range(2)]),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["duplicate_report"] == {"train": 0, "val": 0, "test": 0}
    # same instruction + input, different output -> leak
    pid = _new_project()
    resp = _predef(
        pid, _jsonl([_predef_rows("instruction", "tr", i) for i in range(5)] + [_predef_rows("instruction", "te", 0, output="x", input="te body 0")]),
        None, _jsonl([_predef_rows("instruction", "te", i) for i in range(2)]),
    )
    assert resp.status_code == 409
    # chat: same user turn, different answers -> leak; different user turns -> not
    pid = _new_project()
    tr = [_predef_rows("chat", "tr", i) for i in range(5)] + [_predef_rows("chat", "x", 0, user="shared prompt", answer="answer one")]
    te = [_predef_rows("chat", "te", 0, user="shared prompt", answer="answer two"), _predef_rows("chat", "te", 1)]
    resp = _predef(pid, _jsonl(tr), None, _jsonl(te))
    assert resp.status_code == 409, resp.text
    assert resp.json()["metadata"]["pairs"][0]["sample_rows"] == {"train": [6], "test": [1]}
    pid = _new_project()
    assert _predef(pid, _jsonl(tr[:5]), None, _jsonl(te[1:])).status_code == 200
    # extraction: keyed on the document text, not the output
    pid = _new_project()
    _set_pref(pid, adapter_id="structured-extraction", task_profile="structured_extraction")
    tr = [_predef_rows("extraction", "tr", i) for i in range(5)] + [_predef_rows("extraction", "x", 0, text="shared doc", out={"a": 99})]
    te = [_predef_rows("extraction", "te", 0, text="Shared   DOC", out={"a": 1}), _predef_rows("extraction", "te", 1)]
    assert _predef(pid, _jsonl(tr), None, _jsonl(te)).status_code == 409


def test_input_key_ignores_prompt_template_and_target_fields():
    from app.services.predefined_split_service import _input_side_key

    assert _input_side_key({"question": "Why?", "answer": "A"}) == _input_side_key({"prompt": "why?  ", "response": "B"})
    assert _input_side_key({"instruction": "X", "input": "one"}) != _input_side_key({"instruction": "X", "input": "two"})
    chat = dict(adapter_id="chat-messages")
    assert _input_side_key(_chat("hi", "a"), **chat) == _input_side_key(_chat("hi", "b"), **chat)
    assert _input_side_key({"note": "x", "label": "pos"}) == _input_side_key({"note": "x", "label": "neg"})
    # an id / index column never enters the key
    assert _input_side_key({"sentence": "good film", "label": 1, "idx": 0}) == _input_side_key(
        {"sentence": "good film", "label": 0, "idx": 99}
    )


def test_row_with_no_identifiable_input_is_rejected_422_no_input_field():
    pid = _new_project()
    train = [{"question": f"q {i}?", "answer": f"a {i}"} for i in range(5)] + [{"id": 7, "label": "x", "answer": "y"}]
    resp = _predef(pid, _jsonl(train), None, _jsonl(_qa("te", 2)))
    assert resp.status_code == 422, resp.text
    reasons = resp.json()["metadata"]["rejected"]["train"]["by_reason"]
    assert reasons["no_input_field"]["sample_rows"] == [6]
    assert not (_prep_dir(pid) / "manifest.json").exists()


# --- review-4: leak keys follow the adapter-mapped input ---


def test_sst2_shape_same_sentence_different_idx_or_label_is_a_leak():
    pid = _new_project()
    tr = [{"sentence": f"movie {i} was fine", "label": i % 2, "idx": i} for i in range(20)]
    te = [{"sentence": "movie 3 was fine", "label": 0, "idx": 0}, {"sentence": "brand new", "label": 0, "idx": 1}]
    resp = _predef(pid, _jsonl(tr), None, _jsonl(te))
    assert resp.status_code == 409, resp.text
    assert resp.json()["metadata"]["pairs"][0]["sample_rows"] == {"train": [4], "test": [1]}
    # genuinely different sentences with colliding idx / labels are NOT a leak
    pid = _new_project()
    te = [{"sentence": "something else entirely", "label": 1, "idx": 3}, {"sentence": "brand new", "label": 0, "idx": 1}]
    assert _predef(pid, _jsonl(tr), None, _jsonl(te)).status_code == 200


def test_review_sentiment_same_review_different_label_is_a_leak():
    pid = _new_project()
    tr = [{"review": f"movie {i} was fine", "sentiment": "pos" if i % 2 else "neg"} for i in range(20)]
    te = [{"review": "movie 3 was fine", "sentiment": "neg"}, {"review": "brand new", "sentiment": "neg"}]
    resp = _predef(pid, _jsonl(tr), None, _jsonl(te))
    assert resp.status_code == 409, resp.text


def test_nli_pairs_key_on_both_sentences_not_the_label():
    pid = _new_project()
    tr = [{"premise": f"p {i}", "hypothesis": f"h {i}", "label": i % 3} for i in range(12)]
    same_pair = [{"premise": "p 3", "hypothesis": "h 3", "label": 2}, {"premise": "new", "hypothesis": "pair", "label": 0}]
    assert _predef(pid, _jsonl(tr), None, _jsonl(same_pair)).status_code == 409
    # same premise, different hypothesis: a different example
    pid = _new_project()
    shared_premise = [{"premise": "p 3", "hypothesis": "another hypothesis", "label": 2}, {"premise": "new", "hypothesis": "pair", "label": 0}]
    assert _predef(pid, _jsonl(tr), None, _jsonl(shared_premise)).status_code == 200


def test_token_tagging_rows_key_on_tokens_not_ner_tags():
    pid = _new_project()
    tr = [{"tokens": ["w", str(i), "went"], "ner_tags": [0, 1, 0]} for i in range(10)]
    te = [{"tokens": ["w", "4", "went"], "ner_tags": [0, 0, 0]}, {"tokens": ["fresh", "row"], "ner_tags": [0, 0]}]
    assert _predef(pid, _jsonl(tr), None, _jsonl(te)).status_code == 409
    pid = _new_project()
    te = [{"tokens": ["fresh", "row"], "ner_tags": [0, 1, 0][:2]}, {"tokens": ["another"], "ner_tags": [0]}]
    assert _predef(pid, _jsonl(tr), None, _jsonl(te)).status_code == 200


def test_classification_adapter_keys_on_the_text_not_the_label_or_prompt():
    pid = _new_project()
    _set_pref(pid, adapter_id="classification-label", task_profile="classification")
    tr = [{"text": f"review {i}", "label": "pos" if i % 2 else "neg"} for i in range(10)]
    te = [{"text": "Review   3", "label": "neg"}, {"text": "other thing", "label": "pos"}]
    assert _predef(pid, _jsonl(tr), None, _jsonl(te)).status_code == 409


def test_extraction_key_ignores_the_shared_field_list_prompt():
    pid = _new_project()
    _set_pref(pid, adapter_id="structured-extraction", task_profile="structured_extraction")
    tr = [{"text": f"doc {i}", "structured_output": {"a": i, "b": "x"}} for i in range(6)]
    te = [{"text": "other doc", "structured_output": {"a": 1, "b": "x"}}, {"text": "another doc", "structured_output": {"a": 1, "b": "x"}}]
    resp = _predef(pid, _jsonl(tr), None, _jsonl(te))
    assert resp.status_code == 200, resp.text
    assert resp.json()["duplicate_report"] == {"train": 0, "val": 0, "test": 0}


# --- review-4: dropped overlap rows stay out of every later re-split ---


def _dropped_import(pid: int, test_rows: list[dict]):
    tr = _qa("tr", 20) + [{"question": t["question"], "answer": "DIFFERENT " + t["answer"]} for t in test_rows]
    return _predef(pid, _jsonl(tr), None, _jsonl(test_rows), on_leak="drop_from_train")


def test_drop_from_train_rows_never_return_on_a_later_resplit():
    pid = _new_project()
    test_rows = _qa("te", 5)
    resp = _dropped_import(pid, test_rows)
    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["leak_report"]["dropped_from_train"] == 5
    info = manifest["source_files"]["train"]
    assert info["registered_rows"] == 20 and info["source_rows"] == 25 and info["dropped_row_lines"] == [21, 22, 23, 24, 25]
    assert info["sha256"] != info["registered_sha256"], "original upload hash stays on record"
    (train_doc,) = [d for d in _run(_query(_docs_for(pid))) if d[3] == "train" and d[2] == "accepted"]
    assert train_doc[4]["original_sha256"] == info["sha256"] and train_doc[4]["dropped_row_lines"] == [21, 22, 23, 24, 25]
    test_q = {r["question"] for r in test_rows}
    for body in ({"include_types": ["raw"], "keep_test": True}, {"include_types": None, "keep_test": True}):
        resp = client.post(f"/api/projects/{pid}/dataset/split", json=body)
        assert resp.status_code == 200, (body, resp.text)
        assert resp.json()["predefined_test_active"] is True
        pooled = _prepared(pid, "train") + _prepared(pid, "val")
        assert pooled and not [r for r in pooled if r["question"] in test_q], body
        assert {r["question"] for r in _prepared(pid, "test")} == test_q
    # a plain random re-split (own test file replaced) must not resurrect them either
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"], "discard_predefined_test": True, "dedup_rows": False})
    assert resp.status_code == 200, resp.text
    assert "predefined_test_active" not in resp.json()
    assert not [r for s in ("train", "val", "test") for r in _prepared(pid, s) if r["question"] in test_q]


def test_drop_from_train_registers_a_csv_train_as_csv():
    pid = _new_project()
    header = "question,answer\n"
    train = header + "".join(f"tr q {i}?,a {i}\n" for i in range(12)) + "te question number 0?,other\n"
    resp = client.post(
        f"/api/projects/{pid}/dataset/predefined-split",
        files={"train": ("train.csv", train.encode(), "text/csv"), "test": ("test.jsonl", _jsonl(_qa("te", 2)), "application/octet-stream")},
        data={"on_leak": "drop_from_train"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["source_files"]["train"]["stored_as"].endswith(".csv")
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"], "keep_test": True})
    assert resp.status_code == 200, resp.text
    assert not [r for s in ("train", "val") for r in _prepared(pid, s) if r["question"].startswith("te question")]


def _docs_for(pid: int):
    async def go(db):
        q = await db.execute(
            select(RawDocument).join(Dataset, Dataset.id == RawDocument.dataset_id).where(Dataset.project_id == pid)
        )
        return [(d.id, d.source, d.status.value, (d.metadata_ or {}).get("split_role"), dict(d.metadata_ or {})) for d in q.scalars()]

    return go


# --- review-4: predefined_test_active, process, scoped guard ---


def test_predefined_test_active_only_appears_when_true():
    pid = _new_project()
    resp = _predef(pid, _jsonl(_qa("tr", 20)), _jsonl(_qa("va", 4)), _jsonl(_qa("te", 5)))
    assert resp.json()["predefined_test_active"] is True
    assert json.loads((_prep_dir(pid) / "manifest.json").read_text())["predefined_test_active"] is True
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={"keep_test": True, "include_types": ["raw"]})
    assert resp.json()["predefined_test_active"] is True
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={"discard_predefined_test": True})
    assert resp.status_code == 200 and "predefined_test_active" not in resp.json()
    # no test file imported -> never present
    pid = _new_project()
    assert "predefined_test_active" not in _predef(pid, _jsonl(_qa("tr", 20)), _jsonl(_qa("va", 4))).json()


def test_process_does_not_reaccept_a_rejected_predefined_document():
    pid = _new_project()
    assert _predef(pid, _jsonl(_qa("tr", 20)), None, _jsonl(_qa("te", 5))).status_code == 200
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"], "discard_predefined_test": True})
    assert resp.status_code == 200
    (doc_id,) = _test_doc_ids(pid)
    resp = client.post(f"/api/projects/{pid}/ingestion/documents/{doc_id}/process")
    assert resp.status_code == 409 and resp.json()["error_code"] == "PREDEFINED_SPLIT_DOCUMENT_REJECTED"
    assert _test_doc_ids(pid, [DocumentStatus.ACCEPTED]) == []
    assert client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"]}).status_code == 200


def test_guard_ignores_a_document_id_from_another_project():
    pid, other = _new_project(), _new_project()
    assert _predef(other, _jsonl(_qa("tr", 20)), None, _jsonl(_qa("te", 5))).status_code == 200
    foreign = json.loads((_prep_dir(other) / "manifest.json").read_text())["source_files"]["test"]["document_id"]
    assert _predef(pid, _jsonl(_qa("tr", 20)), None, _jsonl(_qa("te", 5))).status_code == 200
    (own,) = _test_doc_ids(pid)
    _set_doc_status(own, DocumentStatus.REJECTED)
    manifest_path = _prep_dir(pid) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_files"]["test"]["document_id"] = foreign  # accepted, but in another project
    manifest_path.write_text(json.dumps(manifest))
    assert client.post(f"/api/projects/{pid}/dataset/split", json={}).status_code == 200


def test_exits_duplicate_and_delete_keep_working():
    te = _qa("te", 5)
    teq = {t["question"] for t in te}
    pid = _new_project()
    assert _predef(pid, _jsonl(_qa("tr", 20)), _jsonl(_qa("va", 4)), _jsonl(te)).status_code == 200
    assert client.post(f"/api/projects/{pid}/dataset/split", json={"keep_test": True, "include_types": ["raw"]}).status_code == 200
    assert client.post(f"/api/projects/{pid}/dataset/split", json={}).status_code == 409
    assert client.post(f"/api/projects/{pid}/dataset/split", json={"discard_predefined_test": True}).status_code == 200
    assert not [r for s in ("train", "val", "test") for r in _prepared(pid, s) if r["question"] in teq]
    # duplicate of a predefined project: guard fires on the clone; discard clears it
    pid2 = _new_project()
    assert _predef(pid2, _jsonl(_qa("tr", 20)), None, _jsonl(te)).status_code == 200
    resp = client.post(f"/api/projects/{pid2}/duplicate", json={})
    assert resp.status_code < 300, resp.text
    clone = resp.json()["id"]
    assert client.post(f"/api/projects/{clone}/dataset/split", json={"include_types": ["raw"]}).status_code == 409
    resp = client.post(f"/api/projects/{clone}/dataset/split", json={"include_types": ["raw"], "discard_predefined_test": True})
    assert resp.status_code == 200, resp.text
    assert not [r for s in ("train", "val", "test") for r in _prepared(clone, s) if r["question"] in teq]


def test_chat_template_is_resolved_like_split_and_accepts_a_form_field():
    pid = _new_project()
    resp = _predef(pid, _jsonl(_qa("tr", 4)), None, _jsonl(_qa("te", 2)))
    assert resp.json()["chat_template"] == "llama3"
    assert _prepared(pid, "train")[0]["text"].startswith("<|start_header_id|>")
    assert resp.json()["resolved_split_config"]["chat_template"] == "llama3"
    resp = _predef(pid, _jsonl(_qa("tr", 4)), None, _jsonl(_qa("te", 2)), chat_template="chatml")
    assert resp.status_code == 200, resp.text
    assert resp.json()["chat_template"] == "chatml"
    assert resp.json()["resolved_split_config"]["chat_template"] == "chatml"
    assert _prepared(pid, "train")[0]["text"].startswith("<|im_start|>user")
    # the profile's split default applies when the form does not set one
    from unittest import mock

    with mock.patch("app.api.dataset.get_dataset_split_defaults", return_value={"chat_template": "zephyr"}):
        resp = _predef(pid, _jsonl(_qa("tr", 4)), None, _jsonl(_qa("te", 2)))
    assert resp.json()["chat_template"] == "zephyr"
    assert resp.json()["profile_defaults_applied"] == ["chat_template"]


def test_manifest_has_the_same_enrichment_fields_as_split():
    pid = _new_project()
    _set_pref(pid, adapter_id="default-canonical")
    manifest = _predef(pid, _jsonl(_qa("tr", 4)), None, _jsonl(_qa("te", 2))).json()
    random_manifest = client.post(
        f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"], "keep_test": True},
    ).json()
    for key in (
        "domain_pack_applied", "domain_pack_source", "domain_profile_applied", "domain_profile_source",
        "profile_split_defaults", "profile_defaults_applied", "adapter_preference_source", "dedup_inherited_config",
        "resolved_split_config",
    ):
        assert key in manifest, key
        assert key in random_manifest, key
    assert manifest["adapter_preference_source"] == "project"
    assert manifest["dedup_inherited_config"] == []
    # intentional difference: observed (not requested) ratios
    assert manifest["resolved_split_config"]["train_ratio"] == round(4 / 6, 6)


def test_sending_field_mapping_resets_the_adapter_like_split():
    pid = _new_project()
    _set_pref(pid, adapter_id="classification-label", task_profile="classification")
    resp = client.post(
        f"/api/projects/{pid}/dataset/predefined-split",
        files={"train": ("t.jsonl", _jsonl(_qa("tr", 4)), "x"), "test": ("e.jsonl", _jsonl(_qa("te", 2)), "x")},
        data={"field_mapping": json.dumps({"question": "question", "answer": "answer"})},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["adapter_id"] == "default-canonical" and body["adapter_preference_source"] == "request"
    assert body["task_profile_requested"] is None


def test_failure_after_registering_files_removes_the_orphans():
    from unittest import mock

    import pytest

    pid = _new_project()
    assert _predef(pid, _jsonl(_qa("a", 5)), None, _jsonl(_qa("b", 2))).status_code == 200

    async def snapshot(db):
        raw = (await db.execute(select(Dataset).where(Dataset.project_id == pid, Dataset.dataset_type == DatasetType.RAW))).scalar_one()
        docs = (await db.execute(select(RawDocument).where(RawDocument.dataset_id == raw.id))).scalars().all()
        return sorted((d.id, d.status.value, d.file_path) for d in docs)

    before = _run(_query(snapshot))
    assert before and all(Path(p).exists() for _, _, p in before)

    async def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    with mock.patch("app.services.predefined_split_service._write_prepared_splits", boom):
        with pytest.raises(RuntimeError):
            _predef(pid, _jsonl(_qa("c", 5)), None, _jsonl(_qa("d", 2)))
    after = _run(_query(snapshot))
    assert after == before, "no orphan documents; the previous import is still the accepted one"
    stored = list((Path(before[0][2]).parent).glob("*.jsonl"))
    assert {p.name for p in stored} == {Path(p).name for _, _, p in before}, "no orphan files on disk"


def test_upload_size_cap_is_enforced_per_file_with_413():
    from unittest import mock

    pid = _new_project()
    with mock.patch("app.api.dataset_import.MAX_UPLOAD_BYTES", 64):
        resp = _predef(pid, _jsonl(_qa("tr", 10)), None, _jsonl(_qa("te", 1)))
        assert resp.status_code == 413, resp.text
        assert "train" in resp.json()["detail"]
    assert _predef(pid, _jsonl(_qa("tr", 10)), None, _jsonl(_qa("te", 1))).status_code == 200


# --- review-3: the random path is untouched; test files are kept out of every source ---


def _test_doc_ids(pid: int, statuses=None) -> list[int]:
    async def go(db):
        raw = (await db.execute(select(Dataset).where(Dataset.project_id == pid, Dataset.dataset_type == DatasetType.RAW))).scalar_one()
        docs = (await db.execute(select(RawDocument).where(RawDocument.dataset_id == raw.id))).scalars().all()
        return [d.id for d in docs if (d.metadata_ or {}).get("split_role") == "test" and (statuses is None or d.status in statuses)]

    return _run(_query(go))


def _set_doc_status(doc_id: int, status) -> None:
    async def go(db):
        (await db.get(RawDocument, doc_id)).status = status
        await db.commit()

    _run(_query(go))


def _seed_cleaned(pid: int, rows: list[dict]) -> None:
    """A CLEANED dataset + file with ``rows`` (as cleaning would have written)."""
    from app.services.cleaning_service import _cleaned_dir, get_or_create_cleaned_dataset

    path = _cleaned_dir(pid) / "cleaned.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    async def go(db):
        ds = await get_or_create_cleaned_dataset(db, pid)
        ds.file_path = str(path)
        ds.record_count = len(rows)
        await db.commit()

    _run(_query(go))


def test_random_split_rows_carry_no_input_key_and_no_helper_fields():
    pid = _new_project()
    raw = ("\n".join(json.dumps(r) for r in _qa("x", 30)) + "\n").encode()
    assert client.post(f"/api/projects/{pid}/ingestion/upload", files={"file": ("a.jsonl", raw, "application/octet-stream")}).status_code == 201

    async def accept(db):
        for d in (await db.execute(select(RawDocument))).scalars():
            d.status = DocumentStatus.ACCEPTED
        await db.commit()

    _run(_query(accept))
    assert client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"]}).status_code == 200
    for split in ("train", "val", "test"):
        assert all("_input_key" not in r for r in _prepared(pid, split)), split


def test_guard_fires_with_default_include_types_and_names_both_exits():
    pid = _new_project()
    assert _predef(pid, _jsonl(_qa("tr", 20)), None, _jsonl(_qa("te", 5))).status_code == 200
    for body in ({"include_types": None}, {}, {"include_types": ["cleaned"]}, {"include_types": ["raw"]}):
        resp = client.post(f"/api/projects/{pid}/dataset/split", json=body)
        assert resp.status_code == 409, (body, resp.text)
        assert resp.json()["error_code"] == "PREDEFINED_TEST_SPLIT_WOULD_BE_REPLACED"
        message = resp.json()["message"]
        assert "Keep the current test examples" in message and "Replace my own test file" in message


def test_discard_predefined_test_rejects_the_doc_and_runs_a_random_split():
    pid = _new_project()
    train, test = _qa("tr", 30), _qa("te", 6)
    assert _predef(pid, _jsonl(train), None, _jsonl(test)).status_code == 200
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={"discard_predefined_test": True, "dedup_rows": False})
    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest.get("split_source") != "predefined"
    assert manifest["total_entries"] == 30, "the old test rows are NOT pooled into training"
    test_q = {r["question"] for r in test}
    assert not [r for s in ("train", "val", "test") for r in _prepared(pid, s) if r["question"] in test_q]
    assert _test_doc_ids(pid, [DocumentStatus.ACCEPTED]) == []
    # the guard is gone for the next split
    assert client.post(f"/api/projects/{pid}/dataset/split", json={}).status_code == 200
    # opposites are refused
    assert client.post(f"/api/projects/{pid}/dataset/split", json={"keep_test": True, "discard_predefined_test": True}).status_code == 400


def test_rejecting_or_deleting_the_test_doc_clears_the_guard():
    for how in ("reject", "delete"):
        pid = _new_project()
        assert _predef(pid, _jsonl(_qa("tr", 20)), None, _jsonl(_qa("te", 5))).status_code == 200
        (doc_id,) = _test_doc_ids(pid)
        assert client.post(f"/api/projects/{pid}/dataset/split", json={}).status_code == 409
        if how == "reject":
            _set_doc_status(doc_id, DocumentStatus.REJECTED)
        else:
            assert client.delete(f"/api/projects/{pid}/ingestion/documents/{doc_id}").status_code == 204
        resp = client.post(f"/api/projects/{pid}/dataset/split", json={})
        assert resp.status_code == 200, (how, resp.text)


def test_test_role_document_rows_reach_no_training_source():
    """Raw, cleaned (live and from a superseded import) and the synthetic seed pickers."""
    from app.services.cleaning_service import clean_document
    from app.services.split_role_service import is_test_role_document, test_role_document_ids
    from app.services.synthetic_service import _load_project_cleaned_chunks

    pid = _new_project()
    train, test = _qa("tr", 20), _qa("te", 5)
    assert _predef(pid, _jsonl(train), None, _jsonl(test)).status_code == 200
    (first_test,) = _test_doc_ids(pid)
    # a re-import supersedes the first: its (REJECTED) test doc stays a test-role doc
    assert _predef(pid, _jsonl(train), None, _jsonl(test)).status_code == 200
    (second_test,) = _test_doc_ids(pid, [DocumentStatus.ACCEPTED])
    assert first_test != second_test and sorted(_test_doc_ids(pid)) == sorted([first_test, second_test])
    assert _run(_query(lambda db: test_role_document_ids(db, pid))) == {first_test, second_test}

    # cleaned: clean-batch refuses the test doc and writes no rows/chunks for it
    resp = client.post(f"/api/projects/{pid}/cleaning/clean-batch", json={"document_ids": [second_test]})
    assert resp.status_code == 200 and resp.json()["cleaned"] == 0
    assert "your own test file" in resp.json()["errors"][0]["error"]
    cleaned_path = Path(settings.DATA_DIR) / "projects" / str(pid) / "cleaned" / "cleaned.jsonl"
    assert not cleaned_path.exists() or second_test not in {json.loads(l).get("source_document_id") for l in cleaned_path.read_text().splitlines() if l.strip()}

    # cleaned rows that DO carry a test-role source id (older data / a superseded import) are dropped by the pool
    test_rows = [dict(r, source_document_id=i, row_index=n) for n, r in enumerate(_qa("te", 5)) for i in (first_test, second_test)]
    fresh_rows = [dict(r, source_document_id=999999, row_index=n) for n, r in enumerate(_qa("fresh", 4))]
    _seed_cleaned(pid, test_rows + fresh_rows)
    resp = client.post(
        f"/api/projects/{pid}/dataset/split",
        json={"include_types": ["raw", "cleaned"], "discard_predefined_test": True, "dedup_rows": False},
    )
    assert resp.status_code == 200, resp.text
    pooled = [r for s in ("train", "val", "test") for r in _prepared(pid, s)]
    questions = {r["question"] for r in pooled}
    assert not [q for q in questions if q.startswith("te ")], "test-file rows (raw or cleaned) reached a split"
    assert {r["question"] for r in fresh_rows} <= questions, "other cleaned rows still pool"
    assert resp.json()["total_entries"] == 20 + 4

    # synthetic seed pickers: GET /cleaning/chunks and the synth chunk pool skip the test doc / its rows
    chunks = client.get(f"/api/projects/{pid}/cleaning/chunks", params={"limit": 500}).json()["chunks"]
    assert not [c for c in chunks if "te question" in str(c.get("text"))], "test rows offered as synth seeds"
    assert [c for c in chunks if "fresh question" in str(c.get("text"))]
    docs = _run(_query(lambda db: _docs_by_id(db, pid)))
    assert is_test_role_document(docs[second_test])
    # chunk files beside a test-role doc are not read
    chunk_file = Path(docs[second_test].file_path).with_suffix(".chunks.jsonl")
    chunk_file.write_text(json.dumps({"chunk_id": 0, "text": "SECRET TEST CHUNK"}) + "\n")
    assert "SECRET TEST CHUNK" not in _run(_load_project_cleaned_chunks(pid))


async def _docs_by_id(db, pid):
    raw = (await db.execute(select(Dataset).where(Dataset.project_id == pid, Dataset.dataset_type == DatasetType.RAW))).scalar_one()
    return {d.id: d for d in (await db.execute(select(RawDocument).where(RawDocument.dataset_id == raw.id))).scalars()}


def test_adapter_preference_origin_is_predefined_split():
    pid = _new_project()
    _set_pref(pid, adapter_id="classification-label", task_profile="classification")
    assert _predef(pid, _jsonl(_cls("tr", 8, ["pos", "neg"])), None, _jsonl(_cls("te", 3, ["neg"]))).status_code == 200

    async def origin(db):
        return (await db.get(Project, pid)).dataset_adapter_preset.get("origin")

    assert _run(_query(origin)) == "predefined_split"
