"""Per-slice breakdown on an eval result (P5b gap 4): row tags, project slices, Wilson intervals."""

from app.services.slice_breakdown_service import build_slice_breakdown, row_slice_tags, wilson


def _failed(p):
    return not p.get("ok")


def _unscored(p):
    return p.get("ok") is None


def test_row_slice_tags_reads_top_level_and_metadata():
    assert row_slice_tags({"q": 1}) is None
    assert row_slice_tags({"slice": "a", "tags": "b, c", "metadata": {"slices": ["d", 3]}}) == [
        ["slice", "a"], ["tags", "b"], ["tags", "c"], ["metadata.slices", "d"], ["metadata.slices", "3"],
    ]
    assert row_slice_tags({"slice": "x, y"}) == [["slice", "x, y"]]  # only tags split on commas


def test_wilson_matches_hand_value():
    lo, hi = wilson(174, 200)
    assert abs(lo - 0.8158) < 1e-3 and abs(hi - 0.9093) < 1e-3
    assert wilson(0, 0) is None


def test_breakdown_from_tags_and_definitions():
    preds = [
        {"ok": True, "row_slice_tags": [["slice", "a"]], "row_tool_call_strict_pass": False, "prompt": "p"},
        {"ok": False, "row_slice_tags": [["slice", "a"]], "row_tool_call_strict_pass": False, "prompt": "p"},
        {"ok": None, "row_slice_tags": [["slice", "b"]], "prompt": "long prompt"},
        {"ok": True, "row_skipped": True, "row_slice_tags": [["slice", "b"]]},
    ]
    defs = [{"slice_id": "long", "display_name": "Long prompts", "where": [{"field": "input_length", "op": "gt", "value": 5}]}]
    block = build_slice_breakdown(preds, failed=_failed, unscored=_unscored, slice_definitions=defs)
    by = {s["id"]: s for s in block["slices"]}
    assert (by["slice=a"]["rows"], by["slice=a"]["scored"], by["slice=a"]["pass"]) == (2, 2, 1)
    assert by["slice=a"]["strict"] == {"scored": 2, "pass": 0, "rate": 0.0, "ci": wilson(0, 2)}
    assert (by["slice=b"]["rows"], by["slice=b"]["scored"], by["slice=b"]["pass_rate"]) == (1, 0, None)
    assert "strict" not in by["slice=b"]
    assert by["long"]["label"] == "Long prompts" and by["long"]["rows"] == 1
    assert block["overall"]["scored"] == 2


def test_no_slices_no_block():
    assert build_slice_breakdown([{"ok": True}], failed=_failed, unscored=_unscored) is None
    assert build_slice_breakdown([], failed=_failed, unscored=_unscored) is None
