"""Tests for the per-class metric flattener + endpoint (Gap #6 slice 1).

The classification handler emits per-class precision / recall / f1 /
support nested under ``metrics["per_class"][label]``. Before slice 1
those values never reached the gate evaluator because the snapshot
builder dropped non-numeric payload entries. Slice 1 flattens them
into three gateable id-shapes per class, exposes the discovered
labels through ``GET /evaluation/per-class-metric-options``, and
extends the validator's known-id check to accept the new patterns.

Tests cover:
  * Pure flattener: per_class dict → short, dot, and eval-type-scoped
    keys, with missing / non-numeric / non-dict cases handled.
  * End-to-end via TestClient: seed Project + Experiment + EvalResult
    with a per_class payload, hit the new endpoint, assert the
    discovered class labels + metric IDs come back.
"""

from __future__ import annotations

import asyncio
import os
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.evaluation_pack_service import (  # noqa: E402
    _build_metric_snapshot,
    _flatten_per_class_metrics,
)


_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802 — unittest convention
    _MODULE_CLIENT_CM.__exit__(None, None, None)


# ─────────────────────────────────────────────────────────────────────
# Pure flattener — no DB
# ─────────────────────────────────────────────────────────────────────


class _StubEvalResult:
    """Minimal stand-in for ``EvalResult`` matching the fields the
    snapshot helpers read."""

    def __init__(self, *, eval_type: str = "classification", row_id: int = 1):
        self.id = row_id
        self.eval_type = eval_type
        self.dataset_name = "test-set"


class FlattenPerClassMetricsTests(unittest.TestCase):

    def _flatten(self, payload_value):
        values: dict[str, float] = {}
        sources: dict[str, dict] = {}
        _flatten_per_class_metrics(
            values, sources,
            payload_key="per_class",
            payload_value=payload_value,
            row=_StubEvalResult(),  # type: ignore[arg-type]
            eval_type="classification",
        )
        return values

    def test_emits_short_dot_and_scoped_keys_per_class(self):
        # The flattener should produce three id-shapes per class so
        # users can write the gate in whichever form feels natural.
        values = self._flatten({
            "benign": {"precision": 0.95, "recall": 0.88, "f1": 0.91, "support": 800},
            "attack": {"precision": 0.72, "recall": 0.85, "f1": 0.78, "support": 200},
        })
        # Short form — used by the editor dropdown.
        self.assertAlmostEqual(values["precision_benign"], 0.95, places=4)
        self.assertAlmostEqual(values["recall_attack"], 0.85, places=4)
        # Dot-path form — power users + the resolver's suffix matcher.
        self.assertAlmostEqual(values["per_class.benign.f1"], 0.91, places=4)
        # Eval-type-scoped — mirrors how other metrics get an
        # ``<eval_type>.<metric>`` alias above.
        self.assertAlmostEqual(values["classification.per_class.attack.recall"], 0.85, places=4)
        # Support flattens too (integer count).
        self.assertEqual(values["support_benign"], 800.0)

    def test_non_dict_class_entry_skipped_silently(self):
        # If a label's value isn't a dict (corrupted row, plugin
        # mishap), skip it instead of raising — eval flow continues.
        values = self._flatten({
            "benign": {"precision": 0.95, "recall": 0.88, "f1": 0.91, "support": 800},
            "weird": "not a dict",
        })
        self.assertIn("precision_benign", values)
        # The weird label produced no keys.
        self.assertFalse(any(k.endswith("weird") for k in values))

    def test_non_numeric_metric_silently_dropped(self):
        # Per-spec, the flattener only emits keys where the value
        # parses as a float. A class entry missing one of the four
        # standard metrics yields the others without raising.
        values = self._flatten({
            "benign": {"precision": 0.95, "f1": 0.91},
            "attack": {"recall": "n/a", "support": 200},
        })
        self.assertIn("precision_benign", values)
        self.assertIn("f1_benign", values)
        # recall missing from benign → no key.
        self.assertNotIn("recall_benign", values)
        # support is numeric → key present even when recall isn't.
        self.assertIn("support_attack", values)
        # recall = "n/a" → not numeric → no key.
        self.assertNotIn("recall_attack", values)

    def test_non_per_class_payload_key_is_ignored(self):
        # The helper is called for every dict-valued payload entry the
        # snapshot loop hits; it must only act when the key is
        # ``per_class`` so confusion_matrix (also a nested dict) stays
        # untouched.
        values: dict[str, float] = {}
        sources: dict[str, dict] = {}
        _flatten_per_class_metrics(
            values, sources,
            payload_key="confusion_matrix",
            payload_value={"benign": {"benign": 800, "attack": 50}},
            row=_StubEvalResult(),  # type: ignore[arg-type]
            eval_type="classification",
        )
        self.assertEqual(values, {})


class BuildMetricSnapshotIntegrationTests(unittest.TestCase):

    def test_per_class_flattened_into_snapshot_values(self):
        # End-to-end through _build_metric_snapshot: the per_class
        # payload that was previously dropped should now show up as
        # gateable short-form keys.
        row = _StubEvalResult(eval_type="classification", row_id=101)
        # Build the payload the classification handler emits.
        row.metrics = {
            "accuracy": 0.90,
            "macro_f1": 0.85,
            "exact_match": 0.90,
            "f1": 0.85,
            "per_class": {
                "benign": {"precision": 0.95, "recall": 0.88, "f1": 0.91, "support": 800},
                "attack": {"precision": 0.72, "recall": 0.85, "f1": 0.78, "support": 200},
            },
            "confusion_matrix": {
                "benign": {"benign": 700, "attack": 100},
                "attack": {"benign": 30, "attack": 170},
            },
        }
        row.pass_rate = 0.90

        values, _sources, _variance = _build_metric_snapshot({"classification": row})  # type: ignore[arg-type]

        # The canonical aliases still resolve.
        self.assertAlmostEqual(values["macro_f1"], 0.85, places=4)
        # Per-class short-form keys are now present.
        self.assertAlmostEqual(values["precision_benign"], 0.95, places=4)
        self.assertAlmostEqual(values["recall_attack"], 0.85, places=4)
        # Confusion matrix is NOT flattened (only per_class is).
        self.assertNotIn("benign_benign", values)


# ─────────────────────────────────────────────────────────────────────
# End-to-end: seed Project + Experiment + EvalResult, hit the endpoint
# ─────────────────────────────────────────────────────────────────────


_PROJECT_NAME_COUNTER = 0


def _next_project_name() -> str:
    global _PROJECT_NAME_COUNTER
    _PROJECT_NAME_COUNTER += 1
    return f"per-class-test-{_PROJECT_NAME_COUNTER}"


class PerClassMetricOptionsEndpointTests(unittest.TestCase):

    def _seed_project_with_eval_result(self, per_class: dict | None) -> int:
        from app.database import async_session_factory
        from app.models.experiment import EvalResult, Experiment, ExperimentStatus
        from app.models.project import Project

        async def _seed() -> int:
            async with async_session_factory() as db:
                project = Project(
                    name=_next_project_name(),
                    description="per-class metric options test",
                    selected_recipe={"recipe_id": "classification"},
                )
                db.add(project)
                await db.flush()
                if per_class is not None:
                    exp = Experiment(
                        project_id=project.id,
                        name="seed-exp",
                        base_model="stub",
                        status=ExperimentStatus.COMPLETED,
                    )
                    db.add(exp)
                    await db.flush()
                    result = EvalResult(
                        experiment_id=exp.id,
                        eval_type="classification",
                        dataset_name="seed",
                        metrics={
                            "accuracy": 0.90,
                            "macro_f1": 0.85,
                            "per_class": per_class,
                        },
                        pass_rate=0.85,
                    )
                    db.add(result)
                    await db.flush()
                await db.commit()
                return int(project.id)

        return asyncio.run(_seed())

    def test_endpoint_returns_discovered_classes_with_three_metrics_each(self):
        project_id = self._seed_project_with_eval_result({
            "benign": {"precision": 0.95, "recall": 0.88, "f1": 0.91, "support": 800},
            "attack": {"precision": 0.72, "recall": 0.85, "f1": 0.78, "support": 200},
        })
        client = TestClient(app)
        resp = client.get(
            f"/api/projects/{project_id}/evaluation/per-class-metric-options"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Two classes discovered; sorted for stable rendering.
        self.assertEqual(body["classes"], ["attack", "benign"])
        # Three metric IDs per class (precision / recall / f1) →
        # 6 entries total.
        metric_ids = {m["metric_id"] for m in body["metrics"]}
        self.assertIn("precision_benign", metric_ids)
        self.assertIn("recall_benign", metric_ids)
        self.assertIn("f1_benign", metric_ids)
        self.assertIn("precision_attack", metric_ids)
        # source_eval_result_id points the FE at the row used for
        # discovery — useful for "metrics last seen in eval #N"
        # framing.
        self.assertIsInstance(body["source_eval_result_id"], int)

    def test_endpoint_returns_empty_when_project_has_no_eval_yet(self):
        # Fresh project with no eval results — endpoint must NOT
        # crash; just return empty so the FE renders "Run an eval
        # first to discover classes".
        project_id = self._seed_project_with_eval_result(None)
        client = TestClient(app)
        resp = client.get(
            f"/api/projects/{project_id}/evaluation/per-class-metric-options"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["classes"], [])
        self.assertEqual(body["metrics"], [])
        self.assertIsNone(body["source_eval_result_id"])

    def test_endpoint_404s_on_unknown_project(self):
        client = TestClient(app)
        resp = client.get(
            "/api/projects/999999/evaluation/per-class-metric-options"
        )
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["detail"], "project_not_found")


if __name__ == "__main__":
    unittest.main()
