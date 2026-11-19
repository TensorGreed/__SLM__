"""Prepared-split content hash (answer-key identity that survives a re-stamp).

Pins ``dataset_service.split_content_hash`` / ``eval_input_identity``, the
additive ``content_hashes`` manifest key + ``content_sha256`` on each
DatasetVersion, the held-out stamp (``answer_key_stamp``) agreeing with the
write-time hash, and Compare treating a re-stamped test file as the same
answer key: by content hash (recorded, or looked up among the retained
snapshots for older results), else by identical row keys -- never for
remediation, whose rows pick passages by source document.
Pytest-only (TestClient + shared engine).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType, DatasetVersion
from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode
from app.services import compare_row_evidence_service as cre
from app.services.compare_row_evidence_service import (
    RESTAMPED_NOTE,
    SAME_ROWS_NOTE,
    answer_key_stamp,
    compare_provenance,
    content_hash_for_file_sha,
)
from app.services import dataset_service
from app.services.dataset_service import (
    EVAL_INPUT_MANIFEST_KEYS,
    _prep_dir,
    eval_input_identity,
    split_content_hash,
)
from app.services.experiment_comparison_service import compare_experiments

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


def _write(path: Path, rows: list, raw_lines: list[str] | None = None) -> Path:
    lines = [json.dumps(r) for r in rows] + list(raw_lines or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── the hash itself ─────────────────────────────────────────────────────


def test_hash_ignores_source_stamps_key_order_and_blank_lines(tmp_path):
    a = _write(tmp_path / "a.jsonl", [
        {"question": "q1", "answer": "a1", "_source_document_id": 3, "_source_document": "test.jsonl"},
        {"question": "q2", "answer": "a2", "_source_document_id": 3, "_source_document": "test.jsonl"},
    ])
    b = tmp_path / "b.jsonl"
    b.write_text(
        json.dumps({"_source_document": "new.jsonl", "answer": "a1", "question": "q1", "_source_document_id": 9})
        + "\n\n   \n" + json.dumps({"answer": "a2", "question": "q2"}, indent=None, separators=(", ", ": ")) + "\n",
        encoding="utf-8",
    )
    assert hashlib.sha256(a.read_bytes()).hexdigest() != hashlib.sha256(b.read_bytes()).hexdigest()
    assert split_content_hash(a) == split_content_hash(b)


def test_hash_changes_on_content_order_other_underscore_keys_and_invalid_lines(tmp_path):
    rows = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    base = split_content_hash(_write(tmp_path / "base.jsonl", rows))
    assert split_content_hash(_write(tmp_path / "edit.jsonl", [rows[0], {"question": "q2", "answer": "other"}])) != base
    assert split_content_hash(_write(tmp_path / "order.jsonl", rows[::-1])) != base
    # Only the two source-document stamps are stripped: other _-keys can steer prompting / scoring.
    for key in ("_source_dataset", "_row_index", "_x"):
        marked = [dict(rows[0], **{key: "answer_key"}), rows[1]]
        assert split_content_hash(_write(tmp_path / f"{key}.jsonl", marked)) != base, key
    bad_1 = split_content_hash(_write(tmp_path / "bad1.jsonl", rows, ["{not json"]))
    bad_2 = split_content_hash(_write(tmp_path / "bad2.jsonl", rows, ["{not json either"]))
    assert len({base, bad_1, bad_2}) == 3, "an invalid line still counts, and editing it changes the hash"


def test_remediation_keeps_the_source_document_id(tmp_path):
    rows = [{"question": "q1", "answer": "a1", "_source_document_id": 1, "_source_document": "x"}]
    moved = [{"question": "q1", "answer": "a1", "_source_document_id": 2, "_source_document": "x"}]
    a, b = _write(tmp_path / "a.jsonl", rows), _write(tmp_path / "b.jsonl", moved)
    plain = {"task_profile": "qa"}
    remediation = {"task_profile": "qa", "output_schema": {"scoring_mode": "remediation"}}
    assert eval_input_identity(a, plain) == eval_input_identity(b, plain)
    # Remediation matches passages to a row by _source_document_id: the stamp is input there.
    assert eval_input_identity(a, remediation)["content_sha256"] != eval_input_identity(b, remediation)["content_sha256"]


def test_eval_input_covers_the_manifest_fields_the_eval_reads(tmp_path):
    path = _write(tmp_path / "t.jsonl", [{"question": "q1", "answer": "a1"}])
    base = eval_input_identity(path, {"task_profile": "qa", "created_at": "t1", "file_hashes": {"test": "x"}})
    # Bookkeeping fields (timestamps, file hashes, versions) don't change the eval input...
    assert eval_input_identity(path, {"task_profile": "qa", "created_at": "t2", "file_hashes": {"test": "y"}}) == base
    # ...the fields the eval reads do.
    for changed in ({"task_profile": "classification"}, {"task_profile": "qa", "output_schema": {"scoring_mode": "x"}},
                    {"task_profile": "qa", "labels": ["a", "b"]}, {"task_profile": "qa", "subtask": "ner"},
                    {"task_profile": "qa", "tgt_lang": "de"}):
        other = eval_input_identity(path, changed)
        assert other["content_sha256"] == base["content_sha256"]
        assert other["eval_input_sha256"] != base["eval_input_sha256"], changed
    # Fields the eval never reads (the adapter wiring is already applied to the rows; the chat template
    # comes from the tokenizer) don't make two evals of the same rows different.
    unread = {"task_profile": "qa", "chat_template": "chatml", "adapter_id": "x", "adapter_config": {"a": 1},
              "field_mapping": {"question": "q"}}
    assert eval_input_identity(path, unread)["eval_input_sha256"] == base["eval_input_sha256"]


def test_eval_input_keys_are_exactly_the_manifest_fields_the_eval_reads():
    # Grep the eval's readers of the prepared manifest; runtime-injected keys (tool_block, tool_overrides,
    # tool_project_catalog, tool_scoring_cfg) are set by the eval itself, never read from the file.
    import re

    root = Path(dataset_service.__file__).parent
    read: set[str] = set()
    for name in ("eval_task_handler_service.py", "evaluation_service.py"):
        text = (root / name).read_text(encoding="utf-8")
        read |= set(re.findall(r"manifest(?:\s*or\s*\{\}\))?\.get\(\s*\"([a-z_]+)\"", text))
        read |= set(re.findall(r"manifest\[\"([a-z_]+)\"\]", text))
    # null_as_absent: the opt-in null rule (null_scoring), set on the context by the eval from the project / run.
    read -= {"tool_block", "tool_scoring_cfg", "tool_overrides", "tool_project_catalog", "null_as_absent", "null_as_absent_source"}
    assert read == set(EVAL_INPUT_MANIFEST_KEYS)


def test_lone_surrogates_and_undecodable_bytes_hash_safely_and_distinctly(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text('{"question": "q\\ud83d", "answer": "a"}\n', encoding="utf-8")
    b = tmp_path / "b.jsonl"
    b.write_text('{"question": "q\\ud83e", "answer": "a"}\n', encoding="utf-8")
    assert split_content_hash(a) != split_content_hash(b)  # no UnicodeEncodeError
    assert eval_input_identity(a, {})["eval_input_sha256"]
    ff, fe = tmp_path / "ff.jsonl", tmp_path / "fe.jsonl"
    ff.write_bytes(b'{"question": "q", "answer": "a"}\n\xff\n')
    fe.write_bytes(b'{"question": "q", "answer": "a"}\n\xfe\n')
    assert split_content_hash(ff) != split_content_hash(fe), "files differing only in undecodable bytes differ"


# ── written by the split writer, stamped by the eval ───────────────────


def _qa(prefix: str, n: int) -> list[dict]:
    return [{"question": f"{prefix} question number {i}?", "answer": f"{prefix} answer {i}"} for i in range(n)]


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def _new_project() -> int:
    resp = client.post("/api/projects", json={"name": f"content-hash-{uuid4().hex[:6]}", "description": ""})
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _import(pid: int) -> dict:
    files = {name: (f"{name}.jsonl", _jsonl(_qa(name, n)), "application/octet-stream")
             for name, n in (("train", 12), ("val", 4), ("test", 5))}
    resp = client.post(f"/api/projects/{pid}/dataset/predefined-split", files=files)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _test_dataset(db, pid: int) -> Dataset:
    return (await db.execute(
        select(Dataset).where(Dataset.project_id == pid, Dataset.dataset_type == DatasetType.TEST)
    )).scalar_one()


def test_reimport_restamps_the_file_but_keeps_the_content_hash_and_the_stamp_agrees():
    pid = _new_project()
    first = _import(pid)
    second = _import(pid)  # the same files again: new document ids are stamped on every row
    assert first["file_hashes"]["test"] != second["file_hashes"]["test"], "re-stamped bytes"
    assert first["content_hashes"]["test"] == second["content_hashes"]["test"]
    assert set(second["content_hashes"]) == {"train", "val", "test"}
    test_path = _prep_dir(pid) / "test.jsonl"
    assert second["content_hashes"]["test"] == split_content_hash(test_path)
    on_disk = json.loads((_prep_dir(pid) / "manifest.json").read_text())
    assert on_disk["content_hashes"] == second["content_hashes"]

    async def _go():
        async with async_session_factory() as db:
            ds = await _test_dataset(db, pid)
            versions = (await db.execute(
                select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id).order_by(DatasetVersion.version)
            )).scalars().all()
            stamp = await answer_key_stamp(db, ds, test_path)
            return [v.manifest for v in versions], stamp

    manifests, stamp = asyncio.run(_go())
    assert [m["content_sha256"] for m in manifests] == [first["content_hashes"]["test"]] * 2
    # Write-time hash == stamp-time hash (both re-read the file).
    assert stamp["content_sha256"] == second["content_hashes"]["test"]
    assert stamp["sha256"] == second["file_hashes"]["test"]
    assert stamp["version"] == 2, "file-sha match wins (newest)"
    assert stamp["eval_input_sha256"] == eval_input_identity(test_path, on_disk)["eval_input_sha256"]
    # The writer (stamp) and the lift check's reader agree on the same project.
    from app.services.post_training_eval_service import _current_eval_input_sha256

    assert stamp["eval_input_sha256"] == _current_eval_input_sha256(pid)


def test_prepare_hashes_the_splits_off_the_event_loop(monkeypatch):
    on_loop: list[bool] = []
    real = dataset_service.split_content_hash

    def _spy(path, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real(path, **kwargs)

    monkeypatch.setattr(dataset_service, "split_content_hash", _spy)
    _import(_new_project())
    assert on_loop and not any(on_loop), on_loop


def test_snapshot_lookup_finds_an_old_files_content_and_is_read_only():
    pid = _new_project()
    first = _import(pid)
    _import(pid)
    old_sha = first["file_hashes"]["test"]
    snap = _prep_dir(pid) / "versions" / str(first["prepared_version"]) / "test.jsonl"
    before = snap.stat().st_mtime_ns
    assert content_hash_for_file_sha(pid, "test", old_sha) == first["content_hashes"]["test"]
    assert snap.stat().st_mtime_ns == before
    assert content_hash_for_file_sha(pid, "test", "f" * 64) is None, "no retained file has that hash"
    assert content_hash_for_file_sha(pid, "gold_test", old_sha) is None, "only prepared splits are snapshotted"


def test_snapshot_lookup_hashes_content_only_for_the_matching_file(monkeypatch):
    pid = _new_project()
    _import(pid)
    _import(pid)
    cre._FILE_HASH_MEMO.clear()
    calls: list[str] = []
    real = dataset_service.split_content_hash
    monkeypatch.setattr(dataset_service, "split_content_hash", lambda p, **kw: calls.append(str(p)) or real(p, **kw))
    assert content_hash_for_file_sha(pid, "test", "f" * 64) is None
    assert calls == [], "no file sha matched, so no content hash was computed"


def test_snapshot_identity_uses_the_snapshots_own_manifest():
    pid = _new_project()
    first = _import(pid)
    _import(pid)
    snap = _prep_dir(pid) / "versions" / str(first["prepared_version"])
    manifest = json.loads((snap / "manifest.json").read_text())
    found = cre.identity_for_file_sha(pid, "test", first["file_hashes"]["test"])
    assert found == eval_input_identity(snap / "test.jsonl", manifest)


# ── Compare ─────────────────────────────────────────────────────────────


def _prov(*, sha: str, version: int, content=None, eval_input=None, rows=None, mode=None, inferred_content=None):
    # rows: a row_keys_sha256 value (opaque here); _rows_sha() builds one from real keys.
    return {
        "answer_key": {"dataset_id": 1, "dataset_type": "test", "version": version, "sha256": sha,
                       "version_source": "recorded", "content_sha256": content, "eval_input_sha256": eval_input,
                       "inferred_content_sha256": inferred_content},
        "judge": {"answer": None, "rubric": None, "rubric_sha": None, "code_example": None},
        "scoring": {"eval_type": "exact_match", "scoring_mode": mode, "fingerprints": {}},
        "row_keys_sha256": rows,
    }


def test_compare_same_content_is_the_same_answer_key_with_a_note():
    a = _prov(sha="a" * 64, version=3, content="c" * 64, eval_input="e" * 64)
    b = _prov(sha="b" * 64, version=5, content="c" * 64, eval_input="e" * 64)
    out = compare_provenance(a, b)
    assert out["comparable"] and out["differences"] == []
    assert out["notes"] == [RESTAMPED_NOTE]


def test_compare_same_rows_with_changed_eval_settings_is_a_difference():
    a = _prov(sha="a" * 64, version=3, content="c" * 64, eval_input="e" * 64)
    b = _prov(sha="b" * 64, version=5, content="c" * 64, eval_input="f" * 64)
    out = compare_provenance(a, b)
    assert not out["comparable"]
    assert out["differences"][0]["message"].startswith("different eval input: same test rows")


def test_compare_different_content_keeps_todays_message():
    a = _prov(sha="a" * 64, version=3, content="c" * 64, rows="r" * 64)
    b = _prov(sha="b" * 64, version=5, content="d" * 64, rows="r" * 64)
    out = compare_provenance(a, b)
    assert [d["message"] for d in out["differences"]] == ["different answer key: split v3 vs v5"]


def test_compare_old_results_fall_back_to_identical_row_keys_but_never_for_remediation():
    a = _prov(sha="a" * 64, version=3, rows="r" * 64)
    b = _prov(sha="b" * 64, version=5, rows="r" * 64)
    out = compare_provenance(a, b)
    assert out["comparable"] and out["notes"] == [SAME_ROWS_NOTE]
    b_other = _prov(sha="b" * 64, version=5, rows="s" * 64)
    assert [d["message"] for d in compare_provenance(a, b_other)["differences"]] == ["different answer key: split v3 vs v5"]
    rem_a = _prov(sha="a" * 64, version=3, rows="r" * 64, mode="remediation")
    rem_b = _prov(sha="b" * 64, version=5, rows="r" * 64, mode="remediation")
    assert not compare_provenance(rem_a, rem_b)["comparable"]


def _rows_sha(keys: list[str]) -> str | None:
    er = EvalResult(details={"row_scores": {"keys": keys, "correct": [1] * len(keys)}})
    return cre._row_keys_sha256(er)


def test_row_key_fallback_needs_unique_keys_in_the_same_order():
    # A constant prompt with two labels (image classification): the keys repeat, so they can't tell
    # two different test sets apart -> no fallback, the versions differ.
    dup_a = _rows_sha(["k_cat", "k_dog", "k_cat", "k_dog"])
    dup_b = _rows_sha(["k_dog", "k_cat", "k_dog", "k_cat"])
    assert dup_a is None and dup_b is None
    out = compare_provenance(_prov(sha="a" * 64, version=3, rows=dup_a), _prov(sha="b" * 64, version=5, rows=dup_b))
    assert [d["message"] for d in out["differences"]] == ["different answer key: split v3 vs v5"]
    # Same duplicate multiset in the same order still isn't evidence.
    same_dup = _rows_sha(["k_cat", "k_dog", "k_cat", "k_dog"])
    assert not compare_provenance(_prov(sha="a" * 64, version=3, rows=same_dup),
                                  _prov(sha="b" * 64, version=5, rows=same_dup))["comparable"]
    # Unique keys: hashed in order, so a reordered (re-split) list is not the same rows.
    assert _rows_sha(["k1", "k2", "k3"]) != _rows_sha(["k3", "k2", "k1"])
    uniq = _rows_sha(["k1", "k2", "k3"])
    assert compare_provenance(_prov(sha="a" * 64, version=3, rows=uniq),
                              _prov(sha="b" * 64, version=5, rows=uniq))["notes"] == [SAME_ROWS_NOTE]


def test_same_rows_note_does_not_claim_the_same_answer_key():
    # A max_samples eval may have read only a shared prefix of two different files.
    assert "Same answer key" not in SAME_ROWS_NOTE
    assert "split files differ" in SAME_ROWS_NOTE


def test_same_file_with_changed_eval_settings_is_a_difference():
    a = _prov(sha="a" * 64, version=3, content="c" * 64, eval_input="e" * 64)
    b = _prov(sha="a" * 64, version=3, content="c" * 64, eval_input="f" * 64)
    out = compare_provenance(a, b)
    assert not out["comparable"]
    assert out["differences"][0]["message"].startswith("different eval input: same test rows")
    same = _prov(sha="a" * 64, version=3, content="c" * 64, eval_input="e" * 64)
    assert compare_provenance(a, same)["comparable"]


def test_compare_inferred_content_match_says_so():
    a = _prov(sha="a" * 64, version=3, inferred_content="c" * 64)
    a["answer_key"]["sha256"] = None
    a["answer_key"]["inferred_sha256"] = "a" * 64
    a["answer_key"]["version_source"] = "inferred"
    b = _prov(sha="b" * 64, version=5, content="c" * 64)
    out = compare_provenance(a, b)
    assert out["comparable"]
    assert out["notes"][0].startswith(RESTAMPED_NOTE) and "inferred" in out["notes"][0]


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _rows(keys: list[str], before: bool) -> dict:
    return {"keys": keys, "correct": [int(i % 2 == 0) if before else 1 for i in range(len(keys))]}


def _seed_pair(pid: int, a_dataset: dict, b_dataset: dict, *, a_keys: list[str], b_keys: list[str],
               a_extra: dict | None = None, a_day: int = 20) -> tuple[int, int]:
    async def _go():
        async with async_session_factory() as db:
            ds = await _test_dataset(db, pid)
            exps = []
            for name in ("A", "B"):
                exp = Experiment(project_id=pid, name=name, base_model="base-m", training_mode=TrainingMode.SFT,
                                 config={"base_model": "base-m"}, status=ExperimentStatus.COMPLETED)
                db.add(exp)
                exps.append(exp)
            await db.flush()
            for exp, dataset, keys, before, day in ((exps[0], a_dataset, a_keys, True, a_day),
                                                    (exps[1], b_dataset, b_keys, False, 21)):
                db.add(EvalResult(
                    experiment_id=exp.id, dataset_name="test", eval_type="exact_match",
                    metrics={"exact_match": 0.5, **((a_extra or {}) if before else {})},
                    details={"dataset": {"id": ds.id, "dataset_type": "test", **dataset},
                             "row_scores": _rows(keys, before)},
                    pass_rate=0.5, created_at=T0 + timedelta(days=day),
                ))
            await db.commit()
            return exps[0].id, exps[1].id

    return asyncio.run(_go())


def _compare(pid: int, a: int, b: int) -> dict:
    async def _go():
        async with async_session_factory() as db:
            return await compare_experiments(db, project_id=pid, exp_a_id=a, exp_b_id=b)

    return asyncio.run(_go())


def test_old_result_content_is_looked_up_in_snapshots_then_falls_back_when_pruned():
    pid = _new_project()
    first = _import(pid)
    second = _import(pid)
    keys = [f"row{i}" for i in range(5)]
    # A was stamped before content hashes existed (file sha + version only); B is current.
    a, b = _seed_pair(
        pid, {"sha256": first["file_hashes"]["test"], "version": 1},
        {"sha256": second["file_hashes"]["test"], "version": 2, "content_sha256": second["content_hashes"]["test"]},
        a_keys=keys, b_keys=keys[:-1] + ["other"],
    )
    body = _compare(pid, a, b)
    key_a = body["a"]["provenance"]["answer_key"]
    assert key_a["content_sha256"] == first["content_hashes"]["test"]
    # The eval-input hash comes from the snapshot's own manifest.
    snap = _prep_dir(pid) / "versions" / str(first["prepared_version"])
    assert key_a["eval_input_sha256"] == eval_input_identity(
        snap / "test.jsonl", json.loads((snap / "manifest.json").read_text()))["eval_input_sha256"]
    ev = body["row_evidence"]
    assert ev["comparability"]["comparable"] and ev["comparability"]["notes"] == [RESTAMPED_NOTE]

    # The old snapshot is pruned: no content hash for A -> the file-sha rule (keys differ) applies.
    cre._FILE_HASH_MEMO.clear()
    shutil.rmtree(_prep_dir(pid) / "versions" / str(first["prepared_version"]))
    body = _compare(pid, a, b)
    assert body["a"]["provenance"]["answer_key"]["content_sha256"] is None
    assert body["row_evidence"]["message"] == "different answer key: split v1 vs v2"

    # Same pruned snapshot, but both evals scored exactly the same rows in the same order -> same rows.
    a2, b2 = _seed_pair(
        pid, {"sha256": first["file_hashes"]["test"], "version": 1},
        {"sha256": second["file_hashes"]["test"], "version": 2, "content_sha256": second["content_hashes"]["test"]},
        a_keys=keys, b_keys=list(keys),
    )
    ev = _compare(pid, a2, b2)["row_evidence"]
    assert ev["comparability"]["comparable"] and ev["comparability"]["notes"] == [SAME_ROWS_NOTE]

    # Reordered (a re-split) or repeated keys are no evidence -> the file-sha rule.
    for a_keys, b_keys in ((keys, list(reversed(keys))), (["dup", "dup", "x"], ["dup", "dup", "x"])):
        a3, b3 = _seed_pair(
            pid, {"sha256": first["file_hashes"]["test"], "version": 1},
            {"sha256": second["file_hashes"]["test"], "version": 2,
             "content_sha256": second["content_hashes"]["test"]},
            a_keys=a_keys, b_keys=b_keys,
        )
        assert _compare(pid, a3, b3)["row_evidence"]["message"] == "different answer key: split v1 vs v2"


def test_mixed_old_new_compare_sees_a_changed_manifest_through_the_snapshot():
    pid = _new_project()
    first = _import(pid)
    second = _import(pid)
    test_path = _prep_dir(pid) / "test.jsonl"
    live = json.loads((_prep_dir(pid) / "manifest.json").read_text())
    # B was evaluated after the labels the eval reads changed (same rows); A is an old result
    # (file sha only) whose snapshot manifest has the old settings.
    changed = eval_input_identity(test_path, {**live, "labels": ["yes", "no"]})
    a, b = _seed_pair(
        pid, {"sha256": first["file_hashes"]["test"], "version": 1},
        {"sha256": second["file_hashes"]["test"], "version": 2, **changed},
        a_keys=[f"row{i}" for i in range(5)], b_keys=[f"row{i}" for i in range(5)],
    )
    ev = _compare(pid, a, b)["row_evidence"]
    assert ev["verdict"] == "not_comparable"
    assert ev["message"].startswith("different eval input: same test rows")


def test_inferred_version_path_uses_the_versions_content_hash():
    pid = _new_project()
    first = _import(pid)
    second = _import(pid)
    keys = [f"row{i}" for i in range(5)]

    async def _date_versions():
        # v1 before A's eval (day 20), v2 after it.
        async with async_session_factory() as db:
            ds = await _test_dataset(db, pid)
            for v in (await db.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id))).scalars():
                v.created_at = T0 + timedelta(days=10 if v.version == 1 else 21)
            await db.commit()

    asyncio.run(_date_versions())
    # A recorded nothing (an old result): its version + content are inferred from eval time.
    a, b = _seed_pair(
        pid, {}, {"sha256": second["file_hashes"]["test"], "version": 2,
                  "content_sha256": second["content_hashes"]["test"]},
        a_keys=keys, b_keys=keys[:-1] + ["other"],
    )
    body = _compare(pid, a, b)
    key_a = body["a"]["provenance"]["answer_key"]
    assert key_a["version_source"] == "inferred" and key_a["sha256"] is None
    assert key_a["inferred_content_sha256"] == first["content_hashes"]["test"]
    ev = body["row_evidence"]
    assert ev["comparability"]["comparable"], ev["comparability"]
    assert "inferred" in ev["comparability"]["notes"][0]


def test_remediation_old_result_with_a_moved_source_document_stays_different():
    pid = _new_project()
    first = _import(pid)
    second = _import(pid)
    keys = [f"row{i}" for i in range(5)]
    # Remediation: the snapshot lookup keeps the source stamps, and the two imports stamped
    # different document ids -> different content; identical row keys don't override it.
    a, b = _seed_pair(
        pid, {"sha256": first["file_hashes"]["test"], "version": 1},
        {"sha256": second["file_hashes"]["test"], "version": 2},
        a_keys=keys, b_keys=keys, a_extra={"scoring_mode": "remediation"},
    )
    # Both sides scored in remediation mode so only the answer key can differ.
    async def _mode_b():
        async with async_session_factory() as db:
            er = (await db.execute(select(EvalResult).join(Experiment).where(Experiment.id == b))).scalar_one()
            er.metrics = {**er.metrics, "scoring_mode": "remediation"}
            await db.commit()

    asyncio.run(_mode_b())
    ev = _compare(pid, a, b)["row_evidence"]
    assert ev["verdict"] == "not_comparable"
    assert ev["message"] == "different answer key: split v1 vs v2"
