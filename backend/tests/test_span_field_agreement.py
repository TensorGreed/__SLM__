"""``bio_to_spans`` writes the field the structured-extraction adapter reads
(``entities``), and the adapter still reads rows written with the old
``entities_json`` string.
"""

from __future__ import annotations

import json

from app.services.data_adapter_service import map_record_with_adapter
from app.services.dataset_import import resolve_mapper
from app.services.dataset_import.protocols import ImportContext, TransformedRow

BIO_ROW = {
    "tokens": ["Email", "is", "foo@example.com"],
    "labels": ["O", "O", "B-EMAIL"],
    "trailing_whitespace": [True, True, False],
}


def _ctx() -> ImportContext:
    return ImportContext(
        project_id=0, project_task_profile=None, source_id="jsonl",
        mapper_id="bio_to_spans", locator="jsonl:/tmp/x.jsonl",
    )


def _bio(field_map=None) -> dict:
    (row,) = list(resolve_mapper("bio_to_spans").transform([BIO_ROW], field_map or {}, ctx=_ctx()))
    assert isinstance(row, TransformedRow)
    return row.payload


def _adapt(row: dict) -> dict | None:
    return map_record_with_adapter(row, adapter_id="structured-extraction", task_profile="structured_extraction")


def test_bio_to_spans_writes_the_entity_list_the_adapter_reads():
    payload = _bio({"entity_type_map": {"EMAIL": "email"}})
    assert payload["entities"] == [{"type": "email", "start": 9, "end": 24, "text": "foo@example.com"}]
    assert "entities_json" not in payload
    mapped = _adapt(payload)
    assert mapped is not None
    assert mapped["structured_output"] == payload["entities"]
    assert json.loads(mapped["target_text"]) == payload["entities"]


def test_explicit_legacy_output_field_still_writes_the_json_string():
    payload = _bio({"output_field": "entities_json"})
    assert json.loads(payload["entities_json"])["entities"][0]["text"] == "foo@example.com"


def test_adapter_reads_the_legacy_entities_json_string():
    ents = [{"type": "email", "start": 9, "end": 24, "text": "foo@example.com"}]
    legacy = {"text": "Email is foo@example.com", "entities_json": json.dumps({"entities": ents})}
    mapped = _adapt(legacy)
    assert mapped is not None
    assert mapped["structured_output"] == {"entities": ents}
    assert json.loads(mapped["target_text"]) == {"entities": ents}


def test_entities_wins_over_entities_json():
    row = {
        "text": "Email is foo@example.com",
        "entities": [{"type": "email", "start": 9, "end": 24}],
        "entities_json": json.dumps({"entities": []}),
    }
    assert _adapt(row)["structured_output"] == row["entities"]


def test_training_field_scan_matches_the_adapter_on_legacy_rows():
    from app.services.dataset_service import _scan_structured_extraction_fields

    legacy = {"text": "Email is foo@example.com", "entities_json": json.dumps({"entities": []})}
    assert _scan_structured_extraction_fields([legacy], adapter_config=None) == ["entities"]
    # An explicit output_fields list turns the fallback off, as in the adapter.
    assert _scan_structured_extraction_fields([legacy], adapter_config={"output_fields": ["answer"]}) is None
