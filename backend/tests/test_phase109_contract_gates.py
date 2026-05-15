"""Story 1.5 — training-eval contract gates.

Pins the three task-agnostic gates that close the diagnostic gaps the
training-data gate (commit 222bc5d) couldn't reach:

- **Gate 1** — model recommender refuses with a banner shape when the
  prepared train.jsonl has zero target fields (so a user can't pick
  a model for data that no model could succeed on).
- **Gate 2** — eval-time schema-mismatch detector populates a top-level
  metric when ≥80% of prediction/gold pairs have disjoint top-level
  JSON keys. Works whether the schema is ``{entities}``, ``{summary}``,
  ``{intent}``, ``{chosen, rejected}``, etc.
- **Gate 3** — reconciler flips ``status=RUNNING`` rows to the right
  terminal status when ``training_report.json`` shows ``finished_at``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import (  # noqa: E402
    Experiment,
    ExperimentStatus,
    TrainingMode,
)
from app.services.evaluation_service import (  # noqa: E402
    detect_schema_mismatch,
)
from app.services.model_selection_service import (  # noqa: E402
    recommend_training_base_models,
)
from app.services.training_service import (  # noqa: E402
    reconcile_experiment_if_stale,
    reconcile_stale_running_experiments,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir()) / f"brewslm-phase109-{uuid.uuid4().hex[:8]}"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Gate 1: model-recommender data-shape gate
# ─────────────────────────────────────────────────────────────────────


class Gate1RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()

    def _prepared_path(self, project_id: int) -> Path:
        return (
            TEST_DATA_DIR
            / "projects"
            / str(project_id)
            / "prepared"
            / "train.jsonl"
        )

    def test_no_project_id_returns_unblocked(self):
        """Backwards compat: callers that don't pass project_id keep
        their previous behavior."""
        payload = recommend_training_base_models(
            target_device="server",
            primary_language="english",
            available_vram_gb=24.0,
            task_profile="instruction_sft",
            top_k=3,
        )
        self.assertEqual(payload["blocked_by_data_shape"], False)
        self.assertGreater(len(payload["recommendations"]), 0)

    def test_text_only_train_jsonl_blocks_recommendations(self):
        """Reproduces the Qwen-PII-V2 failure: train.jsonl has only
        ``text`` field, no target. The recommender must refuse rather
        than pick a model for data nothing can train on."""
        pid = 88888
        path = self._prepared_path(pid)
        _write_jsonl(
            path,
            [{"text": "raw chunk", "_source_dataset": "cleaned"}] * 20,
        )
        try:
            payload = recommend_training_base_models(
                target_device="server",
                primary_language="english",
                available_vram_gb=24.0,
                task_profile="instruction_sft",
                top_k=3,
                project_id=pid,
            )
            self.assertTrue(payload["blocked_by_data_shape"])
            self.assertEqual(payload["recommendation_count"], 0)
            self.assertEqual(payload["recommendations"], [])
            self.assertIn("answer", payload["data_shape_message"])
        finally:
            path.unlink(missing_ok=True)

    def test_well_shaped_train_jsonl_returns_recommendations(self):
        """When prepared data carries a target field, the gate is a
        no-op and the recommender returns its usual model list."""
        pid = 88889
        path = self._prepared_path(pid)
        _write_jsonl(
            path,
            [
                {"question": "what is 2+2?", "answer": "4"},
                {"prompt": "summarize", "completion": "ok"},
            ] * 10,
        )
        try:
            payload = recommend_training_base_models(
                target_device="server",
                primary_language="english",
                available_vram_gb=24.0,
                task_profile="instruction_sft",
                top_k=3,
                project_id=pid,
            )
            self.assertFalse(payload["blocked_by_data_shape"])
            self.assertGreater(len(payload["recommendations"]), 0)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_train_jsonl_returns_recommendations(self):
        """Fresh project with no prepared data yet — the recommender
        should still surface model options so the user can plan."""
        payload = recommend_training_base_models(
            target_device="server",
            primary_language="english",
            available_vram_gb=24.0,
            task_profile="instruction_sft",
            top_k=3,
            project_id=99999,  # no prepared/train.jsonl exists
        )
        self.assertFalse(payload["blocked_by_data_shape"])
        self.assertGreater(len(payload["recommendations"]), 0)


# ─────────────────────────────────────────────────────────────────────
# Gate 2: eval-time schema-mismatch detector
# ─────────────────────────────────────────────────────────────────────


class Gate2SchemaMismatchTests(unittest.TestCase):
    """Detector must fire across every task profile that has a
    structured output — parameterized fixtures below cover the four
    common shapes."""

    def _wrap_pairs(self, pred_gold_pairs):
        return [
            {"prediction": pred, "reference": gold}
            for pred, gold in pred_gold_pairs
        ]

    def test_fires_for_pii_span_extraction_mismatch(self):
        """The Qwen-PII-V2 incident exactly: model emits
        {value, label, start, end} when gold is {entities: [...]}."""
        preds = self._wrap_pairs(
            [
                (
                    '{"value": "Qwen", "label": "PERSONNAME"}',
                    '{"entities": [{"type": "person_name"}]}',
                )
                for _ in range(20)
            ]
        )
        report = detect_schema_mismatch(preds)
        self.assertIsNotNone(report)
        self.assertEqual(report["ratio"], 1.0)
        self.assertEqual(report["sample_size"], 20)
        observed = report["observed_top_keys"][0]["keys"]
        expected = report["expected_top_keys"][0]["keys"]
        self.assertIn("value", observed)
        self.assertIn("entities", expected)

    def test_fires_for_summary_vs_text_mismatch(self):
        preds = self._wrap_pairs(
            [
                ('{"text": "the model said this"}', '{"summary": "the gist"}')
                for _ in range(15)
            ]
        )
        report = detect_schema_mismatch(preds)
        self.assertIsNotNone(report)
        self.assertEqual(report["ratio"], 1.0)

    def test_fires_for_alignment_pair_mismatch(self):
        preds = self._wrap_pairs(
            [
                ('{"answer": "yes"}', '{"chosen": "A", "rejected": "B"}')
                for _ in range(12)
            ]
        )
        report = detect_schema_mismatch(preds)
        self.assertIsNotNone(report)

    def test_no_op_when_keys_match(self):
        """Gold and prediction both use the same schema — no banner."""
        preds = self._wrap_pairs(
            [
                ('{"entities": []}', '{"entities": [{"type": "x"}]}')
                for _ in range(10)
            ]
        )
        self.assertIsNone(detect_schema_mismatch(preds))

    def test_no_op_for_free_form_text_tasks(self):
        """Free-form QA / summarization where neither side is JSON.
        Nothing to compare keys on — must return None."""
        preds = self._wrap_pairs(
            [
                ("Paris is the capital of France.", "The capital of France is Paris.")
                for _ in range(20)
            ]
        )
        self.assertIsNone(detect_schema_mismatch(preds))

    def test_no_op_for_partial_json_signal(self):
        """Below the min-JSON-pairs threshold — not enough signal."""
        # 2 JSON pairs + 30 free-form rows.
        preds = (
            self._wrap_pairs(
                [
                    ('{"value": "x"}', '{"entities": []}'),
                    ('{"value": "y"}', '{"entities": []}'),
                ]
            )
            + self._wrap_pairs(
                [("free text", "free gold") for _ in range(30)]
            )
        )
        self.assertIsNone(detect_schema_mismatch(preds))

    def test_no_op_when_below_ratio_threshold(self):
        """Some predictions match gold keys — most do — banner stays
        hidden because the per-row diagnostics are enough signal."""
        matching = self._wrap_pairs(
            [
                ('{"entities": [{}]}', '{"entities": [{"type": "x"}]}')
                for _ in range(15)
            ]
        )
        mismatching = self._wrap_pairs(
            [
                ('{"value": "x"}', '{"entities": [{"type": "x"}]}')
                for _ in range(3)
            ]
        )
        # 3/18 mismatches = ~17%, below 80% threshold.
        self.assertIsNone(detect_schema_mismatch(matching + mismatching))

    def test_tolerates_json_in_code_fences(self):
        """Some teacher models wrap JSON in ```json fences — the
        detector should parse them anyway, so the mismatch ratio
        reflects real schema disagreement, not parse-failure noise."""
        preds = self._wrap_pairs(
            [
                ('```json\n{"value": "a"}\n```', '{"entities": []}')
                for _ in range(10)
            ]
        )
        report = detect_schema_mismatch(preds)
        self.assertIsNotNone(report)
        self.assertEqual(report["ratio"], 1.0)


# ─────────────────────────────────────────────────────────────────────
# Gate 3: stuck-RUNNING reconciliation
# ─────────────────────────────────────────────────────────────────────


class Gate3StatusReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _seed_running_experiment(
        self,
        *,
        finished_at_minutes_ago: int | None,
        status_in_report: str | None,
        started_minutes_ago: int = 90,
    ) -> int:
        """Insert an Experiment row in RUNNING state with a paired
        training_report.json on disk. Returns the experiment_id."""

        async def _go():
            async with async_session_factory() as session:
                resp = self.client.post(
                    "/api/projects",
                    json={"name": f"phase109-{uuid.uuid4().hex[:6]}"},
                )
                project_id = int(resp.json()["id"])

                output_dir = (
                    TEST_DATA_DIR / "projects" / str(project_id) / "exp"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                started_at = datetime.now(timezone.utc) - timedelta(
                    minutes=started_minutes_ago
                )
                exp = Experiment(
                    project_id=project_id,
                    name="phase109-stuck-running",
                    status=ExperimentStatus.RUNNING,
                    training_mode=TrainingMode.SFT,
                    base_model="qwen2.5:7b",
                    output_dir=str(output_dir),
                    started_at=started_at,
                )
                session.add(exp)
                await session.commit()
                await session.refresh(exp)

                report: dict = {}
                if finished_at_minutes_ago is not None:
                    finished_at = datetime.now(timezone.utc) - timedelta(
                        minutes=finished_at_minutes_ago
                    )
                    report["finished_at"] = finished_at.isoformat()
                if status_in_report is not None:
                    report["status"] = status_in_report
                (output_dir / "training_report.json").write_text(
                    json.dumps(report)
                )
                return int(exp.id)

        return asyncio.run(_go())

    def _experiment_status(self, exp_id: int) -> str:
        async def _go():
            async with async_session_factory() as session:
                exp = await session.get(Experiment, exp_id)
                return exp.status.value

        return asyncio.run(_go())

    def _reconcile_one(self, exp_id: int):
        async def _go():
            async with async_session_factory() as session:
                exp = await session.get(Experiment, exp_id)
                report = await reconcile_experiment_if_stale(session, exp)
                await session.commit()
                return report

        return asyncio.run(_go())

    def _run_reaper(self, max_age_minutes: int = 60):
        async def _go():
            async with async_session_factory() as session:
                return await reconcile_stale_running_experiments(
                    session, max_age_minutes=max_age_minutes
                )

        return asyncio.run(_go())

    def test_on_read_reconciler_flips_running_to_completed(self):
        exp_id = self._seed_running_experiment(
            finished_at_minutes_ago=30,
            status_in_report="completed",
        )
        self.assertEqual(self._experiment_status(exp_id), "running")

        report = self._reconcile_one(exp_id)
        self.assertIsNotNone(report)
        self.assertEqual(report["to_status"], "completed")
        self.assertEqual(self._experiment_status(exp_id), "completed")

    def test_on_read_reconciler_flips_running_to_failed(self):
        exp_id = self._seed_running_experiment(
            finished_at_minutes_ago=20,
            status_in_report="failed",
        )
        report = self._reconcile_one(exp_id)
        self.assertIsNotNone(report)
        self.assertEqual(report["to_status"], "failed")
        self.assertEqual(self._experiment_status(exp_id), "failed")

    def test_on_read_reconciler_no_op_when_no_finished_at(self):
        exp_id = self._seed_running_experiment(
            finished_at_minutes_ago=None,
            status_in_report=None,
        )
        report = self._reconcile_one(exp_id)
        self.assertIsNone(report)
        self.assertEqual(self._experiment_status(exp_id), "running")

    def test_startup_reaper_sweeps_stuck_rows(self):
        exp_id = self._seed_running_experiment(
            finished_at_minutes_ago=60,
            status_in_report="completed",
            started_minutes_ago=120,
        )
        reports = self._run_reaper(max_age_minutes=15)
        self.assertGreaterEqual(len(reports), 1)
        ids = {r["experiment_id"] for r in reports}
        self.assertIn(exp_id, ids)
        self.assertEqual(self._experiment_status(exp_id), "completed")

    def test_startup_reaper_skips_recent_running_rows(self):
        """Don't reconcile rows that have only been RUNNING for a few
        minutes — they might still be live training."""
        exp_id = self._seed_running_experiment(
            finished_at_minutes_ago=2,
            status_in_report="completed",
            started_minutes_ago=5,
        )
        reports = self._run_reaper(max_age_minutes=30)
        ids = {r["experiment_id"] for r in reports}
        self.assertNotIn(exp_id, ids)
        self.assertEqual(self._experiment_status(exp_id), "running")

    def test_on_read_reconciler_idempotent(self):
        """Calling the reconciler a second time on an already-fixed
        row is a no-op."""
        exp_id = self._seed_running_experiment(
            finished_at_minutes_ago=30,
            status_in_report="completed",
        )
        self._reconcile_one(exp_id)
        second = self._reconcile_one(exp_id)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
