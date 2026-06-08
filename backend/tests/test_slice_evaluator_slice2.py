"""Quality-Lift phase 2, slice 2 — slice predicate evaluator + handler wrapper.

Pins (slice 2: predicate engine + score_with_slices + run_evaluation
wiring; gates land in slice 3):

  Pure predicate engine (no DB, no handlers):
    * Platform fields injected on every row (input_length,
      prediction_length, reference_length, _dataset_index, etc.).
    * Each closed-set op behaves correctly under presence/absence/
      type mismatch (fails closed on mismatched types, not raises).
    * AND across clauses; OR is achieved by defining multiple slices.
    * `exists` honors presence + non-null (None ≠ missing).
    * Empty / None slice_definitions short-circuits to {}.
    * Original prediction row is never mutated.

  score_with_slices wrapper:
    * No slice_definitions → byte-identical to handler.score().
    * Empty subset emits {"support": 0} rather than skipping the
      slice — silent skipping would mask "0 rows match" bugs.
    * per_slice block carries every defined slice; subsets reuse the
      handler's own score path so metric shape matches the overall
      shape (e.g. classification per_class nests inside per_slice).
    * `support` key normalized across handler quirks so slice 3's
      worst-slice gate can read it uniformly.

  End-to-end through run_evaluation (with DB):
    * Project with no slice_definitions → metrics.per_slice absent.
    * Project with slice_definitions → metrics.per_slice present
      with one entry per slice id, each carrying scored metrics.
    * The aggregator-friendly nested shape composes with phase 1 —
      a per_slice entry can itself carry per_class entries that
      future multi-seed runs aggregate via compute_variance_stats.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import EvalResult, Experiment, TrainingMode  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.services.slice_evaluator_service import (  # noqa: E402
    PLATFORM_FIELDS,
    apply_slices,
    inject_platform_fields,
    score_with_slices,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-sliceeval-{uuid.uuid4().hex[:8]}"
)


def setUpModule() -> None:
    settings.AUTH_ENABLED = False
    settings.DEBUG = False
    settings.DATA_DIR = TEST_DATA_DIR.resolve()
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    global _CLIENT_CM, CLIENT
    _CLIENT_CM = TestClient(app)
    CLIENT = _CLIENT_CM.__enter__()


def tearDownModule() -> None:
    _CLIENT_CM.__exit__(None, None, None)


def _create_project() -> int:
    resp = CLIENT.post(
        "/api/projects",
        json={"name": f"sliceeval-{uuid.uuid4().hex[:6]}"},
    )
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


# ────────────────────────────────────────────────────────────────────────
# Platform-field injection
# ────────────────────────────────────────────────────────────────────────


class InjectPlatformFieldsTests(unittest.TestCase):

    def test_input_length_derived_from_prompt(self):
        row = {"prompt": "hello world", "prediction": "hi", "reference": "hi there"}
        enriched = inject_platform_fields(row, index=0)
        self.assertEqual(enriched["input_length"], len("hello world"))
        self.assertEqual(enriched["prediction_length"], len("hi"))
        self.assertEqual(enriched["reference_length"], len("hi there"))
        self.assertEqual(enriched["_dataset_index"], 0)

    def test_input_length_falls_back_through_aliases(self):
        # Legacy data paths write ``input`` or ``question`` instead of
        # ``prompt``; slicing must work uniformly.
        for key in ("input", "question"):
            with self.subTest(key=key):
                row = {key: "a longer string", "prediction": "out"}
                enriched = inject_platform_fields(row, index=1)
                self.assertEqual(enriched["input_length"], len("a longer string"))

    def test_word_count_fallback_when_no_tokenizer(self):
        row = {"prompt": "a b c d e"}
        enriched = inject_platform_fields(row, index=0, tokenizer=None)
        # Whitespace word count — predictable + ordering-correct.
        self.assertEqual(enriched["input_token_count"], 5)

    def test_tokenizer_used_when_available(self):
        class FakeTokenizer:
            def encode(self, text, **_):
                # 2 tokens per word, just to verify the path is hit.
                return ["t"] * (2 * len(text.split()))

        row = {"prompt": "a b c"}
        enriched = inject_platform_fields(row, index=0, tokenizer=FakeTokenizer())
        self.assertEqual(enriched["input_token_count"], 6)

    def test_does_not_mutate_input_row(self):
        # Predictions list flows through multiple downstream summarizers
        # (details / preview / lift); mutating it would surprise them.
        row = {"prompt": "x", "prediction": "y"}
        before = dict(row)
        inject_platform_fields(row, index=0)
        self.assertEqual(row, before)

    def test_platform_fields_table_is_documented(self):
        # The slice editor (slice 3) reads PLATFORM_FIELDS to render
        # the field-picker; if a field is added to the injector
        # without being added here, the editor silently hides it.
        names = {name for name, _ in PLATFORM_FIELDS}
        self.assertIn("input_length", names)
        self.assertIn("input_token_count", names)
        self.assertIn("prediction_length", names)
        self.assertIn("reference_length", names)
        self.assertIn("_dataset_index", names)


# ────────────────────────────────────────────────────────────────────────
# apply_slices — predicate matching across the closed op set
# ────────────────────────────────────────────────────────────────────────


def _slice(slice_id: str, where: list[dict]) -> dict:
    return {
        "slice_id": slice_id,
        "display_name": slice_id,
        "where": where,
    }


class ApplySlicesTests(unittest.TestCase):

    def test_empty_slice_definitions_returns_empty_dict(self):
        # Fast-path the no-slices configured case so single-handler
        # eval pays no cost.
        self.assertEqual(apply_slices([{"prompt": "x"}], None), {})
        self.assertEqual(apply_slices([{"prompt": "x"}], []), {})

    def test_eq_neq_match(self):
        preds = [
            {"prompt": "x", "source": "synth"},
            {"prompt": "x", "source": "human"},
        ]
        buckets = apply_slices(preds, [
            _slice("synth", [{"field": "source", "op": "eq", "value": "synth"}]),
            _slice("not_synth", [{"field": "source", "op": "neq", "value": "synth"}]),
        ])
        self.assertEqual(len(buckets["synth"]), 1)
        self.assertEqual(buckets["synth"][0]["source"], "synth")
        self.assertEqual(len(buckets["not_synth"]), 1)
        self.assertEqual(buckets["not_synth"][0]["source"], "human")

    def test_numeric_ops_match_on_platform_field(self):
        # input_length is platform-computed — users can slice without
        # the dataset carrying any length column.
        preds = [
            {"prompt": "a"},        # length 1
            {"prompt": "ab"},       # length 2
            {"prompt": "abcde"},    # length 5
        ]
        buckets = apply_slices(preds, [
            _slice("short", [{"field": "input_length", "op": "lt", "value": 3}]),
            _slice("long", [{"field": "input_length", "op": "gte", "value": 3}]),
        ])
        self.assertEqual(len(buckets["short"]), 2)
        self.assertEqual(len(buckets["long"]), 1)
        self.assertEqual(buckets["long"][0]["prompt"], "abcde")

    def test_in_not_in_match(self):
        preds = [
            {"prompt": "p", "language": "en"},
            {"prompt": "p", "language": "hi"},
            {"prompt": "p", "language": "fr"},
        ]
        buckets = apply_slices(preds, [
            _slice("indic", [{"field": "language", "op": "in", "value": ["hi", "te"]}]),
            _slice("non_european", [{"field": "language", "op": "not_in", "value": ["en", "fr"]}]),
        ])
        self.assertEqual([r["language"] for r in buckets["indic"]], ["hi"])
        self.assertEqual([r["language"] for r in buckets["non_european"]], ["hi"])

    def test_contains_is_case_insensitive(self):
        preds = [
            {"prompt": "Hello WORLD"},
            {"prompt": "goodbye"},
        ]
        buckets = apply_slices(preds, [
            _slice("greets", [{"field": "prompt", "op": "contains", "value": "hello"}]),
        ])
        self.assertEqual(len(buckets["greets"]), 1)

    def test_regex_match(self):
        preds = [
            {"prompt": "abc-123-xyz"},
            {"prompt": "no-digits"},
            {"prompt": "456-only-digits"},
        ]
        buckets = apply_slices(preds, [
            _slice("has_number", [
                {"field": "prompt", "op": "regex", "value": r"\d+"},
            ]),
        ])
        self.assertEqual(len(buckets["has_number"]), 2)

    def test_exists_true_requires_field_and_non_null(self):
        preds = [
            {"prompt": "x", "language": "en"},
            {"prompt": "x", "language": None},
            {"prompt": "x"},
        ]
        buckets = apply_slices(preds, [
            _slice("has_lang", [{"field": "language", "op": "exists", "value": True}]),
            _slice("no_lang", [{"field": "language", "op": "exists", "value": False}]),
        ])
        # Only the en row counts — None is treated as missing per design.
        self.assertEqual(len(buckets["has_lang"]), 1)
        self.assertEqual(len(buckets["no_lang"]), 2)

    def test_dot_path_resolves_nested_metadata(self):
        preds = [
            {"prompt": "x", "metadata": {"source": "synth"}},
            {"prompt": "x", "metadata": {"source": "human"}},
            {"prompt": "x", "metadata": {}},
        ]
        buckets = apply_slices(preds, [
            _slice("synth", [{"field": "metadata.source", "op": "eq", "value": "synth"}]),
        ])
        self.assertEqual(len(buckets["synth"]), 1)

    def test_missing_field_fails_closed(self):
        # If the field is absent, every op (except `exists: false`) is
        # falsy. The clue an eval-time bug is masking data.
        preds = [{"prompt": "x"}]  # no `source` field at all
        buckets = apply_slices(preds, [
            _slice("synth", [{"field": "source", "op": "eq", "value": "synth"}]),
            _slice("long", [{"field": "source", "op": "gte", "value": 5}]),
        ])
        self.assertEqual(buckets["synth"], [])
        self.assertEqual(buckets["long"], [])

    def test_type_mismatch_fails_closed(self):
        # Numeric op against a string row value — predicate fails
        # quietly instead of raising "<" not supported between str
        # and int. That's important: a malformed row shouldn't break
        # eval for the whole dataset.
        preds = [{"prompt": "x", "n": "not-a-number"}]
        buckets = apply_slices(preds, [
            _slice("big_n", [{"field": "n", "op": "gte", "value": 5}]),
        ])
        self.assertEqual(buckets["big_n"], [])

    def test_and_across_clauses(self):
        preds = [
            {"prompt": "long input text here please", "language": "hi"},
            {"prompt": "short", "language": "hi"},
            {"prompt": "long input text here please", "language": "en"},
        ]
        buckets = apply_slices(preds, [
            _slice("hindi_long", [
                {"field": "language", "op": "eq", "value": "hi"},
                {"field": "input_length", "op": "gte", "value": 10},
            ]),
        ])
        # Only the first row clears both clauses.
        self.assertEqual(len(buckets["hindi_long"]), 1)


# ────────────────────────────────────────────────────────────────────────
# score_with_slices — wrapper integrates with any TaskHandler
# ────────────────────────────────────────────────────────────────────────


class _FakeAccuracyHandler:
    """Minimal handler — counts predictions that match their reference.
    Mirrors the per-row score-then-aggregate shape of the real handlers
    so the wrapper integration test isn't faking the contract.
    """

    profile_id = "fake"

    def score(self, predictions: list[dict], ctx: Any) -> dict[str, Any]:
        if not predictions:
            return {"accuracy": 0.0, "total": 0}
        correct = sum(
            1 for p in predictions
            if str(p.get("prediction") or "") == str(p.get("reference") or "")
        )
        return {
            "accuracy": round(correct / len(predictions), 4),
            "total": len(predictions),
            "correct": correct,
        }


class ScoreWithSlicesTests(unittest.TestCase):

    def setUp(self):
        self.handler = _FakeAccuracyHandler()
        self.preds = [
            {"prompt": "a", "prediction": "yes", "reference": "yes", "language": "en"},
            {"prompt": "ab", "prediction": "no",  "reference": "yes", "language": "en"},
            {"prompt": "long input here", "prediction": "yes", "reference": "yes", "language": "hi"},
            {"prompt": "xyz", "prediction": "yes", "reference": "yes", "language": "hi"},
        ]

    def test_no_slice_definitions_is_byte_identical_to_handler(self):
        # Critical: projects with no slicing configured must see EXACTLY
        # the same metric dict the handler would return on its own. Any
        # incidental key the wrapper adds (per_slice, etc.) would break
        # existing scorecard renderers downstream.
        bare = self.handler.score(self.preds, ctx=None)
        wrapped = score_with_slices(
            self.handler, self.preds, ctx=None, slice_definitions=None,
        )
        self.assertEqual(bare, wrapped)
        self.assertNotIn("per_slice", wrapped)

    def test_per_slice_block_added_when_slices_configured(self):
        result = score_with_slices(
            self.handler, self.preds, ctx=None,
            slice_definitions=[
                _slice("hindi", [{"field": "language", "op": "eq", "value": "hi"}]),
                _slice("english", [{"field": "language", "op": "eq", "value": "en"}]),
            ],
        )
        self.assertIn("per_slice", result)
        # Overall metric still computed across all 4 predictions.
        self.assertEqual(result["total"], 4)
        # Hindi subset: 2 rows, both correct → 1.0.
        self.assertEqual(result["per_slice"]["hindi"]["accuracy"], 1.0)
        self.assertEqual(result["per_slice"]["hindi"]["support"], 2)
        # English subset: 2 rows, 1 correct → 0.5.
        self.assertEqual(result["per_slice"]["english"]["accuracy"], 0.5)
        self.assertEqual(result["per_slice"]["english"]["support"], 2)

    def test_empty_slice_yields_support_zero(self):
        # The slice predicate matches nothing — emit {"support": 0}
        # rather than skipping the slice entry. The UI / gate evaluator
        # depend on every defined slice appearing in per_slice.
        result = score_with_slices(
            self.handler, self.preds, ctx=None,
            slice_definitions=[
                _slice("french", [{"field": "language", "op": "eq", "value": "fr"}]),
            ],
        )
        self.assertEqual(result["per_slice"]["french"], {"support": 0})

    def test_support_key_canonicalized_when_handler_returns_total(self):
        # Handler emits ``total`` not ``support``; wrapper still
        # backfills ``support`` so the slice 3 worst-slice gate has
        # one key to look at across all handler types.
        result = score_with_slices(
            self.handler, self.preds, ctx=None,
            slice_definitions=[
                _slice("hindi", [{"field": "language", "op": "eq", "value": "hi"}]),
            ],
        )
        self.assertIn("support", result["per_slice"]["hindi"])
        self.assertEqual(result["per_slice"]["hindi"]["support"], 2)


# ────────────────────────────────────────────────────────────────────────
# End-to-end through run_evaluation
# ────────────────────────────────────────────────────────────────────────


class EndToEndRunEvaluationSliceTests(unittest.TestCase):
    """Drive run_evaluation against a real Project + Experiment to verify
    the metric snapshot lands ``per_slice`` on the EvalResult and the
    no-slices case round-trips unchanged."""

    def _seed_experiment(self, project_id: int) -> int:
        async def _go() -> int:
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=project_id,
                    name="slice-e2e",
                    status="completed",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    training_mode=TrainingMode.SFT,
                    config={"task_type": "causal_lm"},
                )
                session.add(exp)
                await session.commit()
                return int(exp.id)

        return asyncio.run(_go())

    def _set_slices(self, project_id: int, slices: list[dict]) -> None:
        resp = CLIENT.put(
            f"/api/projects/{project_id}/slice-definitions",
            json={"slices": slices},
        )
        assert resp.status_code == 200, resp.text

    def _run_eval(self, project_id: int, experiment_id: int, predictions: list[dict]):
        """Drive run_evaluation directly; we don't need real inference
        for this contract test — predictions are already shaped."""
        from app.services.evaluation_service import run_evaluation

        async def _go() -> EvalResult:
            async with async_session_factory() as session:
                return await run_evaluation(
                    db=session,
                    project_id=project_id,
                    experiment_id=experiment_id,
                    dataset_name="held_out",
                    eval_type="exact_match",
                    predictions=predictions,
                )

        return asyncio.run(_go())

    def test_no_slice_definitions_no_per_slice_key(self):
        # Projects without slice_definitions must round-trip byte-identical
        # — the existing scorecard/eval surfaces shouldn't see ``per_slice``
        # appear out of nowhere.
        pid = _create_project()
        eid = self._seed_experiment(pid)
        preds = [
            {"prompt": "x", "prediction": "yes", "reference": "yes"},
            {"prompt": "x", "prediction": "no", "reference": "yes"},
        ]
        eval_result = self._run_eval(pid, eid, preds)
        self.assertNotIn("per_slice", eval_result.metrics)
        # Overall metric still computed.
        self.assertIn("exact_match", eval_result.metrics)

    def test_per_slice_lands_in_eval_result_metrics(self):
        pid = _create_project()
        eid = self._seed_experiment(pid)
        self._set_slices(pid, [
            {
                "slice_id": "long_input",
                "display_name": "Long inputs",
                "where": [{"field": "input_length", "op": "gte", "value": 5}],
            },
            {
                "slice_id": "short_input",
                "display_name": "Short inputs",
                "where": [{"field": "input_length", "op": "lt", "value": 5}],
            },
        ])

        preds = [
            # Short — gets it wrong.
            {"prompt": "x", "prediction": "no", "reference": "yes"},
            # Long — gets it right.
            {"prompt": "long input here", "prediction": "yes", "reference": "yes"},
            # Long — gets it right.
            {"prompt": "another long one", "prediction": "yes", "reference": "yes"},
        ]
        eval_result = self._run_eval(pid, eid, preds)

        metrics = eval_result.metrics
        self.assertIn("per_slice", metrics)
        # Long inputs: 2 correct out of 2.
        long_slice = metrics["per_slice"]["long_input"]
        self.assertEqual(long_slice["support"], 2)
        self.assertAlmostEqual(long_slice["exact_match"], 1.0, places=4)
        # Short inputs: 0 correct out of 1.
        short_slice = metrics["per_slice"]["short_input"]
        self.assertEqual(short_slice["support"], 1)
        self.assertAlmostEqual(short_slice["exact_match"], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
