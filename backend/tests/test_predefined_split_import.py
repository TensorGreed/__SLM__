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


def test_equivalent_to_random_split_of_the_same_rows():
    pid = _new_project()
    train, val, test = _qa("tr", 14), _qa("va", 3), _qa("te", 3)
    assert _predef(pid, _jsonl(train), _jsonl(val), _jsonl(test)).status_code == 200
    predefined = Counter(
        json.dumps(r, sort_keys=True) for s in ("train", "val", "test") for r in _prepared(pid, s)
    )
    # The same source rows, prepared by a RANDOM split of the registered files.
    resp = client.post(
        f"/api/projects/{pid}/dataset/split",
        json={"include_types": ["raw"], "dedup_rows": False, "auto_stratify": False},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json().get("split_source") is None
    randomised = Counter(
        json.dumps(r, sort_keys=True) for s in ("train", "val", "test") for r in _prepared(pid, s)
    )
    assert sum(randomised.values()) == 20
    assert predefined == randomised, "byte-identical prepared rows, only the membership differs"


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
    # a plain random split afterwards still works
    assert client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"]}).status_code == 200


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
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["raw"], "dedup_rows": False})
    assert resp.status_code == 200
    assert resp.json()["total_entries"] == 9, "only the latest import is pooled"


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
