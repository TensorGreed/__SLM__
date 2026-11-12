"""Zero-span rows are valid "nothing to tag" examples: the structured-extraction
adapter keeps them (``entities: []`` and the span playbooks' ``spans: []``),
the token-classification trainer reads them as all-O, and the span_set scorer
scores them (bare-list references / predictions included).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.services.data_adapter_service import map_record_with_adapter
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler
from app.services.synth_playbooks.span_extraction_hard_negatives import SpanExtractionHardNegativesPlaybook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_token_classification as tc  # noqa: E402

TEXT = "Write to jane@example.com before Friday, please."
EMAIL = {"type": "email", "start": 9, "end": 25, "text": "jane@example.com"}
NEG_TEXT = "Nothing to tag in this ordinary sentence at all."


def _adapt(row: dict, config: dict | None = None) -> dict | None:
    return map_record_with_adapter(
        row, adapter_id="structured-extraction", adapter_config=config, task_profile="structured_extraction"
    )


def _prepared(row: dict) -> dict:
    """What dataset_service writes to prepared/*.jsonl: the source row + the adapter's keys."""
    mapped = _adapt(row)
    assert mapped is not None
    return {**row, **mapped}


# ── adapter ──────────────────────────────────────────────────────────


def test_empty_entity_list_is_kept_as_a_negative():
    mapped = _adapt({"text": NEG_TEXT, "entities": []})
    assert mapped is not None
    assert mapped["structured_output"] == []
    assert mapped["target_text"] == " []"
    assert mapped["answer"] == "[]"


def test_span_playbook_rows_map_positive_and_negative():
    assert _adapt({"text": TEXT, "spans": [EMAIL]})["structured_output"] == [EMAIL]
    assert _adapt({"text": NEG_TEXT, "spans": []})["target_text"] == " []"


def test_hard_negatives_playbook_output_reaches_training():
    rows = SpanExtractionHardNegativesPlaybook().validate([{"text": NEG_TEXT, "spans": []}], ctx=None)
    assert rows
    assert _adapt(rows[0]["payload"]) is not None


def test_legacy_empty_entities_json_is_kept():
    assert _adapt({"text": NEG_TEXT, "entities_json": "[]"})["target_text"] == " []"
    assert _adapt({"text": NEG_TEXT, "entities_json": '{"entities": []}'}) is not None


def test_rows_without_any_target_are_still_dropped():
    assert _adapt({"text": NEG_TEXT}) is None
    assert _adapt({"text": NEG_TEXT, "answer": ""}) is None
    assert _adapt({"text": NEG_TEXT, "answer": {}}) is None


def test_a_present_target_wins_over_an_empty_list():
    mapped = _adapt({"text": TEXT, "labels": [], "entities": [EMAIL]})
    assert mapped["structured_output"] == [EMAIL]


def test_explicit_output_fields_keep_empty_lists_only_on_opt_in():
    row = {"text": NEG_TEXT, "found": []}
    assert _adapt(row, {"output_fields": ["found"]}) is None
    mapped = _adapt(row, {"output_fields": ["found"], "keep_empty_targets": True})
    assert mapped is not None and mapped["answer"] == "[]"


# ── empty targets are a span-shape thing (other structured projects unchanged) ──


import pytest  # noqa: E402


@pytest.mark.parametrize("key", ["output", "json", "labels", "structured_output", "answer", "target"])
def test_empty_lists_under_non_span_fields_still_drop_without_a_recipe(key):
    assert _adapt({"text": NEG_TEXT, key: []}) is None


@pytest.mark.parametrize("key", ["entities", "output", "spans"])
def test_a_field_match_recipe_drops_empty_lists_as_before(key):
    assert _adapt({"text": NEG_TEXT, key: []}, {"span_set": False}) is None


def test_a_span_set_recipe_keeps_empty_lists_under_any_output_field():
    assert _adapt({"text": NEG_TEXT, "output": []}, {"span_set": True})["answer"] == "[]"


def test_opt_in_keeps_empty_lists_for_any_structured_project():
    assert _adapt({"text": NEG_TEXT, "output": []}, {"span_set": False, "keep_empty_targets": True})["answer"] == "[]"


def test_kv_to_structured_rows_stay_unmapped():
    """``{"entities": [{field, value}]}`` is a field list, not spans: dropped as before the span fixes."""
    row = {"text": "Invoice total 5 USD", "entities_json": json.dumps({"entities": [{"field": "total", "value": "5"}]})}
    assert _adapt(row) is None


def test_recipe_scoring_mode_is_stamped_on_the_adapter_config():
    import asyncio
    from unittest import mock

    from app.services import dataset_service as ds

    async def schema(mode):
        async def _fake(*_a, **_k):
            return {"scoring_mode": mode}
        with mock.patch.object(ds, "_recipe_output_schema", _fake):
            return await ds._apply_recipe_adapter_config(None, 1, "structured-extraction", None)

    assert asyncio.run(schema("span_set"))["span_set"] is True
    assert asyncio.run(schema("field_match"))["span_set"] is False

    async def none(*_a, **_k):
        return None

    with mock.patch.object(ds, "_recipe_output_schema", none):
        assert asyncio.run(ds._apply_recipe_adapter_config(None, 1, "structured-extraction", None)) is None


# ── token-classification training ────────────────────────────────────


def test_token_classifier_reads_prepared_negative_rows_as_all_outside():
    for row in ({"text": NEG_TEXT, "entities": []}, {"text": NEG_TEXT, "spans": []}):
        assert tc.row_to_example(_prepared(row)) == (NEG_TEXT, [])


def test_token_classifier_reads_playbook_spans():
    text, spans = tc.row_to_example({"text": TEXT, "spans": [EMAIL]})
    assert spans == [EMAIL]


# ── eval + span_set scoring ──────────────────────────────────────────


def _score(pairs):
    handler = StructuredExtractionHandler()
    preds = [{"prediction": p, "reference": r} for p, r in pairs]
    return handler._score_span_set(preds, {"required": []}), preds


def test_negative_row_with_empty_prediction_is_an_exact_match():
    metrics, preds = _score([("[]", "[]")])
    assert metrics["exact_match"] == 1.0
    assert metrics["json_validity_rate"] == 1.0
    assert (metrics["precision"], metrics["recall"], metrics["f1"]) == (1.0, 1.0, 1.0)
    assert preds[0]["row_hallucinated_entities"] == []


def test_span_on_a_negative_row_is_a_false_positive():
    metrics, preds = _score([(json.dumps([EMAIL]), "[]"), (json.dumps({"entities": [EMAIL]}), json.dumps([EMAIL]))])
    assert metrics["per_class"]["email"]["fp"] == 1
    assert metrics["per_class"]["email"]["tp"] == 1
    assert metrics["exact_match"] == 0.5
    assert len(preds[0]["row_hallucinated_entities"]) == 1


def test_bare_list_reference_is_read_as_the_gold_spans():
    metrics, _ = _score([(json.dumps({"entities": [EMAIL]}), json.dumps([EMAIL]))])
    assert metrics["f1"] == 1.0


def test_eval_prompt_for_list_targets_matches_the_training_prompt():
    """Bare-list targets carry no field names: eval builds the same no-list
    prompt the adapter trained on (it used to list the first span's keys)."""
    rows = [_prepared({"text": TEXT, "entities": [EMAIL]}), _prepared({"text": NEG_TEXT, "entities": []})]
    handler = StructuredExtractionHandler()
    ctx = EvalContext(
        project_id=0, experiment_id=0, eval_type="f1", task_profile="structured_extraction",
        handler_id="structured_extraction", prepared_dir=Path("/nonexistent"), dataset_name="test",
        manifest={"output_schema": {"scoring_mode": "span_set"}},
    )
    built = handler.build_prompts(rows, ctx)
    assert built[0].prompt == rows[0]["source_text"]
    assert built[1].prompt == rows[1]["source_text"]
    assert built[1].reference == "[]"


def test_raw_playbook_row_reference_is_read():
    handler = StructuredExtractionHandler()
    assert handler._reference_as_string({"text": NEG_TEXT, "spans": []}) == json.dumps({"entities": []})
