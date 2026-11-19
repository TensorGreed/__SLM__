"""span_set results carry ``span_set_scoring_version``; a base-model result
cached under an older span_set scorer / eval prompt is never reused as the
bar for a new tuned eval — the lift check re-runs it."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from app.services import post_training_eval_service as pte
from app.services.eval_task_handler_service import SPAN_SET_SCORING_VERSION, StructuredExtractionHandler

EMAIL = {"type": "email", "start": 0, "end": 5, "text": "a@b.c"}


def test_span_set_metrics_are_stamped_with_the_version():
    handler = StructuredExtractionHandler()
    metrics = handler._score_span_set([{"prediction": json.dumps([EMAIL]), "reference": json.dumps([EMAIL])}], {})
    assert metrics["span_set_scoring_version"] == SPAN_SET_SCORING_VERSION
    assert handler._score_span_set([], {})["span_set_scoring_version"] == SPAN_SET_SCORING_VERSION


def test_scoring_match_requires_the_current_span_set_version():
    current = SimpleNamespace(metrics={"scoring_mode": "span_set", "span_set_scoring_version": SPAN_SET_SCORING_VERSION, "prompt_format_version": 2})
    old = SimpleNamespace(metrics={"scoring_mode": "span_set"})  # scored before the version existed
    older = SimpleNamespace(metrics={"scoring_mode": "span_set", "span_set_scoring_version": 1})
    assert pte._baseline_scoring_matches(current, None)
    assert not pte._baseline_scoring_matches(old, None)
    assert not pte._baseline_scoring_matches(older, None)
    # Other scoring modes are unaffected.
    assert pte._baseline_scoring_matches(SimpleNamespace(metrics={"scoring_mode": "field_match", "prompt_format_version": 2}), None)


def _reuse(candidates):
    db = mock.Mock()
    scalars = mock.Mock()
    scalars.scalars.return_value.all.return_value = candidates

    async def execute(*_a, **_k):
        return scalars

    db.execute = execute

    async def mtime(_pid):
        return None

    with mock.patch.object(pte, "_test_split_mtime", mtime), \
            mock.patch.object(pte, "_baseline_judge_matches", return_value=True), \
            mock.patch.object(pte, "_baseline_decoding_matches", return_value=True):
        return asyncio.run(pte._reusable_baseline_result(db, mock.Mock(id=1), 1))


def _cached(**metrics):
    return mock.Mock(
        created_at=datetime.now(timezone.utc), details={"row_scores": {}},
        metrics={"scoring_mode": "span_set", "inference": {}, "prompt_format_version": 2, **metrics},
    )


def test_a_baseline_cached_before_the_scorer_change_is_re_run():
    """Same test split, same decoding: only the scorer changed. The pre-change
    base result (no version) is skipped, so the lift check scores the base model again."""
    assert _reuse([_cached(f1=1.0)]) is None


def test_a_baseline_cached_after_the_change_is_reused():
    fresh = _cached(f1=0.4, span_set_scoring_version=SPAN_SET_SCORING_VERSION)
    assert _reuse([fresh]) is fresh
    # The newest reusable one wins even when an older-scorer result sits behind it.
    assert _reuse([fresh, _cached(f1=1.0)]) is fresh
    assert _reuse([_cached(f1=1.0), fresh]) is fresh
