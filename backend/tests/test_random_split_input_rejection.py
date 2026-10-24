"""Random split: classification reads sentence / premise+hypothesis / comment / review columns,
and rows whose input would be their own label (classification) or gold record (structured
extraction) are rejected and reported in the manifest, never trained on. Pytest-only."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import data_adapter_service as das
from app.services.dataset_service import _normalize_rows_for_training, _prep_dir

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


def _project() -> int:
    resp = client.post("/api/projects", json={"name": f"rsir-{uuid4().hex[:6]}", "description": ""})
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _random_split(pid: int, rows: list[dict], **pref):
    if pref:
        assert client.put(f"/api/projects/{pid}/dataset/adapter-preference", json=pref).status_code == 200
    from app.database import async_session_factory
    from app.services.ingestion_service import ingest_file
    import asyncio

    async def go():
        async with async_session_factory() as db:
            from app.models.dataset import DocumentStatus

            doc = await ingest_file(db, pid, "rows.jsonl", ("\n".join(json.dumps(r) for r in rows) + "\n").encode(),
                                    source="upload")
            doc.status = DocumentStatus.ACCEPTED
            await db.commit()

    asyncio.run(go())
    return client.post(f"/api/projects/{pid}/dataset/split",
                       json={"include_types": ["raw"], "dedup_rows": False, "auto_stratify": False})


def _prepared(pid: int) -> list[dict]:
    out: list[dict] = []
    for name in ("train", "val", "test"):
        path = _prep_dir(pid) / f"{name}.jsonl"
        if path.exists():
            out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


CLS = {"adapter_id": "classification-label", "task_profile": "classification"}


def test_sst2_sentence_column_is_the_input_on_a_random_split():
    pid = _project()
    rows = [{"sentence": f"movie number {i} was fine", "label": i % 2, "idx": i} for i in range(30)]
    resp = _random_split(pid, rows, **CLS)
    assert resp.status_code == 200, resp.text
    got = _prepared(pid)
    assert len(got) == 30
    for r in got:
        assert r["text"].startswith("movie number "), r
        assert r["text"] != str(r["label"])
        assert f"Text: {r['text']}\nLabel:" in r["source_text"]


def test_alias_columns_and_pairs_map_to_text():
    cfg = {}
    for col in ("comment", "review"):
        m = das.map_record_with_adapter({col: "so good", "label": "pos"}, adapter_id="classification-label", adapter_config=cfg)
        assert m["text"] == "so good", col
    m = das.map_record_with_adapter({"premise": "A man walks.", "hypothesis": "Someone moves.", "label": 1},
                                    adapter_id="classification-label")
    assert m["text"] == "Premise: A man walks.\nHypothesis: Someone moves." and m["label"] == "1"
    assert "Text: Premise: A man walks.\nHypothesis: Someone moves.\nLabel:" in m["source_text"]
    m = das.map_record_with_adapter({"sentence1": "a", "sentence2": "b", "label": 0}, adapter_id="classification-label")
    assert m["text"] == "Sentence 1: a\nSentence 2: b"
    # the existing aliases still win over the new ones
    m = das.map_record_with_adapter({"text": "T", "sentence": "S", "label": "x"}, adapter_id="classification-label")
    assert m["text"] == "T"


def test_classification_row_whose_input_is_its_label_is_rejected_not_trained():
    pid = _project()
    good = [{"text": f"good row {i}", "label": ["a", "b"][i % 2]} for i in range(20)]
    bad = [{"blob": "xx", "label": "a"}]  # no input column: the canonical text would be the label
    rows = good + [{"label": "b"}] + [{"idx": 3, "label": "a"}]
    resp = _random_split(pid, rows, **CLS)
    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert len(_prepared(pid)) == 20
    assert manifest["include_types_resolution"]["input_rejected"] == {"classification_input_is_label": 2}
    assert not any(r["text"] == r["label"] for r in _prepared(pid))
    rejected: list = []
    _normalize_rows_for_training(bad, None, "llama3", adapter_id="classification-label", rejected=rejected)
    assert rejected == [(0, "classification_input_is_label")]


def test_everything_rejected_is_an_error_naming_the_cause():
    pid = _project()
    resp = _random_split(pid, [{"label": "a"}, {"label": "b"}], **CLS)
    assert resp.status_code in (400, 422), resp.text
    assert "own label" in resp.text


def test_structured_row_without_a_document_column_does_not_put_gold_in_the_prompt():
    pid = _project()
    ok = [{"text": f"doc {i}", "structured_output": {"a": i}} for i in range(20)]
    bad = {"title": "Report", "structured_output": {"a": 1, "secret": "GOLDVALUE"}}
    resp = _random_split(pid, ok + [bad], adapter_id="structured-extraction", task_profile="structured_extraction")
    assert resp.status_code == 200, resp.text
    assert resp.json()["include_types_resolution"]["input_rejected"] == {"structured_input_contains_gold": 1}
    got = _prepared(pid)
    assert len(got) == 20 and not any("GOLDVALUE" in json.dumps(r) for r in got)


def test_structured_with_recognised_or_mapped_document_is_unchanged():
    row = {"body": "the doc", "structured_output": {"a": 1}}
    m = das.map_record_with_adapter(row, adapter_id="structured-extraction")
    assert m["text"] == "the doc"  # canonical ``body`` alias, not a fallback concatenation
    m = das.map_record_with_adapter({"narr": "N", "structured_output": {"a": 1}}, adapter_id="structured-extraction",
                                    field_mapping={"text": "narr"})
    assert m["text"] == "N"


def test_a_saved_question_mapping_trains_the_mapped_column_not_the_label():
    pid = _project()
    rows = [{"utterance": f"book a flight {i}", "intent": ["book", "cancel"][i % 2]} for i in range(20)]
    resp = _random_split(pid, rows, **CLS, field_mapping={"question": "utterance", "answer": "intent"})
    assert resp.status_code == 200, resp.text
    got = _prepared(pid)
    assert len(got) == 20 and "input_rejected" not in (resp.json()["include_types_resolution"] or {})
    for r in got:
        assert r["text"].startswith("book a flight "), r
        assert f"Text: {r['text']}\nLabel:" in r["source_text"]
        assert r["text"] != r["label"]


def test_an_explicit_input_mapping_keeps_rows_whose_input_equals_the_answer():
    pid = _project()
    rows = [{"utterance": u, "intent": u} for u in ("help", "stop", "yes")] + [
        {"utterance": f"book a flight {i}", "intent": "book"} for i in range(17)]
    resp = _random_split(pid, rows, **CLS, field_mapping={"text": "utterance", "answer": "intent"})
    assert resp.status_code == 200, resp.text
    assert len(_prepared(pid)) == 20 and "input_rejected" not in (resp.json()["include_types_resolution"] or {})
    pid = _project()
    se = [{"report": f"A cat {i} sat", "lab": "cat"} for i in range(20)]
    resp = _random_split(pid, se, adapter_id="structured-extraction", task_profile="structured_extraction",
                         field_mapping={"text": "report", "answer": "lab"})
    assert resp.status_code == 200, resp.text
    got = _prepared(pid)
    assert len(got) == 20 and all(r["text"].startswith("A cat ") for r in got)


def test_unmapped_fallback_rows_stay_rejected_and_structured_mapping_reads_the_document():
    pid = _project()
    rows = [{"utt": f"u {i}", "label": i % 3} for i in range(21)]
    resp = _random_split(pid, rows, **CLS)
    assert resp.status_code in (400, 422) and "own label" in resp.text
    m = das.map_record_with_adapter({"narr": "N", "gold": {"a": 1}, "structured_output": {"a": 1}},
                                    adapter_id="structured-extraction", field_mapping={"question": "narr"})
    assert m["text"] == "N"


def test_a_mapped_input_only_fills_in_when_no_input_alias_matched():
    # main's precedence: the ``text`` alias wins over a (stale) mapping to another column
    m = das.map_record_with_adapter({"text": "T", "label": "L"}, adapter_id="classification-label",
                                    field_mapping={"text": "label"})
    assert m["text"] == "T" and m["label"] == "L"
    m = das.map_record_with_adapter({"text": "T", "structured_output": {"a": 1}, "other": "O"},
                                    adapter_id="structured-extraction",
                                    field_mapping={"text": "structured_output"})
    assert m["text"] == "T"
    # no alias column, mapped input IS the label column -> rejected
    assert das.map_record_with_adapter({"utt": "u", "label": "L"}, adapter_id="classification-label",
                                       field_mapping={"text": "label"}) is None
    assert das.unmappable_reason({"utt": "u", "label": "L"}, adapter_id="classification-label",
                                 field_mapping={"text": "label"}) == "classification_input_is_label"
    # mapped input from a different column that happens to equal the label -> kept
    m = das.map_record_with_adapter({"name": "book", "label": "book"}, adapter_id="classification-label",
                                    field_mapping={"text": "name"})
    assert m["text"] == "book"
    # structured: mapped column holding the gold JSON is rejected
    assert das.unmappable_reason({"doc": '{"a": 1}', "structured_output": {"a": 1}},
                                 adapter_id="structured-extraction",
                                 field_mapping={"text": "doc"}) == "structured_input_contains_gold"
