"""Span-offset JSONL import: ``{text, entities: [{type, start, end, text}]}``
rows go through the import wizard / CLI as span rows with the offsets kept
exactly (``span_offsets_passthrough`` + the introspector rule that proposes it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.dataset_import import resolve_mapper
from app.services.dataset_import.introspector import (
    ENTITY_LIST_JSON,
    detect_shape,
    propose_mapping,
    sniff_columns,
)
from app.services.dataset_import.protocols import ImportContext, RejectedRow, TransformedRow
from app.services.dataset_import.service import preview_import


TEXT_A = "Write to jane@example.com before Friday, please."
ROW_A = {
    "id": "a1",
    "text": TEXT_A,
    "entities": [{"type": "email", "start": 9, "end": 25, "text": "jane@example.com"}],
}
TEXT_B = "Order 4471 shipped from the north warehouse today."
ROW_B = {
    "id": "b2",
    "text": TEXT_B,
    "entities": [
        {"type": "order_id", "start": 6, "end": 10, "text": "4471"},
        {"type": "place", "start": 28, "end": 43},
    ],
}
ROW_NEG = {"id": "n3", "text": "Nothing to tag in this ordinary sentence at all.", "entities": []}


def _ctx() -> ImportContext:
    return ImportContext(
        project_id=0, project_task_profile=None, source_id="jsonl",
        mapper_id="span_offsets_passthrough", locator="jsonl:/tmp/x.jsonl",
    )


def _run(rows, field_map=None):
    mapper = resolve_mapper("span_offsets_passthrough")
    out = list(mapper.transform(rows, field_map or {}, ctx=_ctx()))
    accepted = [o for o in out if isinstance(o, TransformedRow)]
    rejected = [o for o in out if isinstance(o, RejectedRow)]
    return accepted, rejected


def test_offsets_are_kept_exactly():
    accepted, rejected = _run([ROW_A, ROW_B])
    assert rejected == []
    a, b = (r.payload for r in accepted)
    assert a["text"] == TEXT_A
    assert a["entities"] == [{"type": "email", "start": 9, "end": 25, "text": "jane@example.com"}]
    # A span without its text gets text[start:end]; offsets untouched.
    assert b["entities"][1] == {"type": "place", "start": 28, "end": 43, "text": TEXT_B[28:43]}
    for payload in (a, b):
        for ent in payload["entities"]:
            assert payload["text"][ent["start"]:ent["end"]] == ent["text"]
    assert accepted[0].row_key == "jsonl-a1"


def test_zero_entity_row_is_accepted_as_a_negative():
    accepted, rejected = _run([ROW_NEG])
    assert rejected == []
    assert accepted[0].payload == {"text": ROW_NEG["text"], "entities": []}


def test_json_string_entities_and_type_map():
    row = {"body": TEXT_A, "spans": json.dumps({"entities": ROW_A["entities"]})}
    accepted, rejected = _run(
        [row], {"text_field": "body", "entities_field": "spans", "entity_type_map": {"email": "contact"}}
    )
    assert rejected == []
    assert accepted[0].payload["entities"][0]["type"] == "contact"
    assert accepted[0].payload["entities"][0]["start"] == 9


@pytest.mark.parametrize(
    "row, reason",
    [
        ({"text": "", "entities": []}, "empty_text"),
        ({"text": "Some text here"}, "missing_entities"),
        ({"text": "Some text here", "entities": "not json"}, "missing_entities"),
        ({"text": "Some text", "entities": [{"type": "x", "start": 2, "end": 99}]}, "span_out_of_range"),
        ({"text": "Some text", "entities": [{"type": "x", "start": 0, "end": 4, "text": "Sxme"}]}, "span_text_mismatch"),
        ({"text": "Some text", "entities": [{"start": 0, "end": 4}]}, "bad_span"),
        ({"text": "  Some text", "entities": [{"type": "x", "start": 2, "end": 6}]}, "text_edge_whitespace"),
        ({"__parse_error__": "bad line"}, "parse_error"),
    ],
)
def test_bad_rows_are_rejected_with_a_reason(row, reason):
    accepted, rejected = _run([row])
    assert accepted == []
    assert [r.reason for r in rejected] == [reason]


def test_introspector_proposes_span_offsets_mapper_even_with_negatives():
    sample = [ROW_A, ROW_NEG, ROW_B]
    sigs = sniff_columns(sample)
    assert sigs["entities"].column_type == ENTITY_LIST_JSON
    proposal = propose_mapping(sample)
    assert proposal is not None
    assert proposal.mapper_id == "span_offsets_passthrough"
    assert proposal.target_task_profile == "structured_extraction"
    assert proposal.field_map["text_field"] == "text"
    assert proposal.field_map["entities_field"] == "entities"
    assert proposal.confidence >= 0.8


def _span_hyps(sample):
    return [h for h in detect_shape(sniff_columns(sample), sample) if h.mapper_id == "span_offsets_passthrough"]


def test_offsets_that_mostly_do_not_fit_the_text_are_not_proposed():
    bad = {"text": "Short text here.", "entities": [{"type": "x", "start": 3, "end": 400}]}
    assert _span_hyps([ROW_A, bad]) == []


def test_one_misfit_among_many_spans_proposes_below_the_gate():
    bad = {"text": "Short text here.", "entities": [{"type": "x", "start": 3, "end": 400}]}
    sample = [ROW_A, ROW_B, ROW_A, ROW_B, bad]  # 1 misfit of 7 spans
    hyps = _span_hyps(sample)
    assert hyps and hyps[0].confidence < 0.8 and hyps[0].warnings
    # Below the gate, but still ranked above text_only (which would drop every label).
    sample = _unique_ids(sample)
    proposal = propose_mapping(sample)
    assert proposal.mapper_id == "span_offsets_passthrough" and proposal.confidence < 0.8


def _unique_ids(rows):
    return [{**row, "id": f"r{i}"} for i, row in enumerate(rows)]


def test_one_bad_span_in_many_stays_auto_runnable():
    bad = {"text": "Short text here.", "entities": [{"type": "x", "start": 3, "end": 400}]}
    sample = _unique_ids([ROW_A, ROW_B] * 12 + [bad])  # 1 misfit of 37 spans (>= 95% fit)
    proposal = propose_mapping(sample)
    assert proposal.mapper_id == "span_offsets_passthrough"
    assert proposal.confidence >= 0.8
    assert any("rejects those rows" in w for w in proposal.warnings)


def test_text_only_ranks_below_spans_and_warns_it_drops_labels():
    bad = {"text": "Short text here.", "entities": [{"type": "x", "start": 3, "end": 400}]}
    sample = [{k: v for k, v in r.items() if k != "id"} for r in (ROW_A, ROW_B, ROW_A, ROW_B, bad)]
    hyps = detect_shape(sniff_columns(sample), sample)
    ids = [h.mapper_id for h in hyps]
    assert ids[0] == "span_offsets_passthrough" and "text_only" in ids
    text_only = hyps[ids.index("text_only")]
    assert text_only.confidence < hyps[0].confidence
    assert any("every label would be dropped" in w for w in text_only.warnings)


def test_cli_auto_prints_proposal_warnings(tmp_path, capsys, monkeypatch):
    from app.cli import dataset_import as cli

    async def fake_introspect(locator, **_):
        return {"proposal": {"mapper_id": "span_offsets_passthrough", "field_map": {}, "rationale": "r",
                             "confidence": 0.85, "warnings": ["1 of 37 sampled spans don't fit 'text'"]}}

    monkeypatch.setattr("app.services.dataset_import.service.introspect_locator", fake_introspect)
    assert cli._resolve_via_auto("jsonl:/x", False)[0] == "span_offsets_passthrough"
    assert "--auto warning: 1 of 37 sampled spans" in capsys.readouterr().err

    async def low(locator, **_):
        return {"proposal": {"mapper_id": "span_offsets_passthrough", "field_map": {}, "rationale": "r",
                             "confidence": 0.7, "warnings": ["3 of 20 sampled spans don't fit 'text'"]}}

    monkeypatch.setattr("app.services.dataset_import.service.introspect_locator", low)
    with pytest.raises(SystemExit, match="3 of 20 sampled spans"):
        cli._resolve_via_auto("jsonl:/x", False)


SPEECH = "Well, I think we should start the meeting now. Agreed, let's begin with the budget."


def test_float_time_segments_are_not_proposed():
    rows = [{"text": SPEECH, "segments": [{"label": "spk1", "start": 0.5, "end": 2.25}, {"label": "spk2", "start": 2.3, "end": 4.0}]}] * 3
    assert sniff_columns(rows)["segments"].column_type == ENTITY_LIST_JSON
    assert _span_hyps(rows) == []
    assert propose_mapping(rows).mapper_id == "text_only"


def test_integer_time_segments_without_span_text_are_not_proposed():
    rows = [{"text": SPEECH, "segments": [{"label": "spk1", "start": 0, "end": 12}, {"label": "spk2", "start": 12, "end": 30}]}] * 3
    assert _span_hyps(rows) == []
    assert propose_mapping(rows).mapper_id == "text_only"


def test_type_keyed_event_lists_without_span_text_are_not_proposed():
    rows = [{"text": SPEECH, "events": [{"type": "pause", "start": 3, "end": 9}]}] * 3
    assert _span_hyps(rows) == []


def test_type_keyed_entities_without_span_text_need_force():
    rows = [{"text": TEXT_A, "entities": [{"type": "email", "start": 9, "end": 25}]}] * 3
    hyps = _span_hyps(rows)
    assert hyps and hyps[0].confidence < 0.8
    assert "can't be checked" in hyps[0].warnings[0]


@pytest.mark.parametrize("bad", ["--3", "\u00b2", "1.5", 1.5, True, None, "x"])
def test_odd_offsets_are_rejected_not_raised(bad):
    accepted, rejected = _run([{"text": "Some text", "entities": [{"type": "x", "start": bad, "end": 4}]}])
    assert accepted == []
    assert [r.reason for r in rejected] == ["bad_span"]


def test_string_and_integral_float_offsets_are_read():
    accepted, rejected = _run([{"text": "Some text", "entities": [{"type": "x", "start": "0", "end": 4.0}]}])
    assert rejected == []
    assert accepted[0].payload["entities"][0] == {"type": "x", "start": 0, "end": 4, "text": "Some"}


EMOJI_TEXT = "Hi \U0001F600 mail jane@example.com now"


def test_code_point_offsets_after_an_emoji_are_kept():
    start = EMOJI_TEXT.index("jane")
    row = {"text": EMOJI_TEXT, "entities": [{"type": "email", "start": start, "end": start + 16, "text": "jane@example.com"}]}
    accepted, rejected = _run([row])
    assert rejected == []
    assert accepted[0].payload["entities"][0]["start"] == start


def test_utf16_offsets_after_an_emoji_are_caught_by_the_span_text():
    start = EMOJI_TEXT.index("jane") + 1  # UTF-16 counts the emoji as two units
    row = {"text": EMOJI_TEXT, "entities": [{"type": "email", "start": start, "end": start + 16, "text": "jane@example.com"}]}
    accepted, rejected = _run([row])
    assert [r.reason for r in rejected] == ["span_text_mismatch"]


def test_preview_end_to_end_from_jsonl(tmp_path: Path):
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (ROW_A, ROW_B, ROW_NEG)) + "\n")
    result = preview_import(
        project_id=0, project_task_profile=None, locator=f"jsonl:{path}",
        mapper_id="span_offsets_passthrough", field_map={},
    )
    assert result.rejected_count == 0
    assert result.target_task_profile == "structured_extraction"
    assert [r.payload["entities"] for r in result.accepted_rows][0] == ROW_A["entities"]


def test_label_and_value_keys_are_read_in_place_of_type_and_text():
    row = {"source_text": TEXT_A, "mask": [{"label": "email", "start": 9, "end": 25, "value": "jane@example.com"}]}
    proposal = propose_mapping([row, {"source_text": ROW_NEG["text"], "mask": []}])
    assert proposal.mapper_id == "span_offsets_passthrough"
    assert proposal.field_map == {"text_field": "source_text", "entities_field": "mask"}
    accepted, rejected = _run([row], proposal.field_map)
    assert rejected == []
    assert accepted[0].payload["entities"] == [{"type": "email", "start": 9, "end": 25, "text": "jane@example.com"}]
