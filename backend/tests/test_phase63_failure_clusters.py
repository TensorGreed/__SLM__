"""Phase 63 — P12 failure-cluster service + endpoint.

Row-level eval failures get grouped by ``(reason_code, output_pattern)``
so the UI can replace a wall of bad predictions with a short list of
distinct failure shapes and their exemplars.

Covers:
- Happy path: LLM-judge failures cluster by classifier reason_code AND
  prediction output-pattern signature.
- Same reason_code with different output patterns yields *separate*
  clusters (the output-pattern signature is what distinguishes them).
- An all-passing eval result returns zero clusters (no spurious grouping).
- 404 for an unknown eval_result_id.
- Exemplars trimmed to ``max_exemplars_per_cluster`` and clusters sorted
  by failure_count DESC.
- Output-pattern signature helpers are deterministic for refusal / empty /
  short-prose / long-prose / json-shaped outputs.
"""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path


TEST_DB_PATH = Path(__file__).resolve().parent / "phase63_failure_clusters.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "phase63_failure_clusters_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

import asyncio

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import EvalResult, Experiment
from app.services.failure_cluster_service import (
    _leading_category,
    _length_bucket_by_chars,
    _output_pattern_signature,
)


def _make_scored_prediction(
    *, prompt: str, reference: str, prediction: str, judge_score: int = 1
) -> dict:
    return {
        "prompt": prompt,
        "reference": reference,
        "prediction": prediction,
        "judge_score": judge_score,
        "judge_rationale": f"judge says {judge_score}/5",
        "input_modality": "text",
    }


class Phase63FailureClustersTests(unittest.TestCase):
    @classmethod
    def _cleanup_artifacts(cls):
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        for suffix in ("", "-shm", "-wal"):
            path = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        cls._cleanup_artifacts()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        cls._cleanup_artifacts()

    # -- helpers ------------------------------------------------------------

    def _create_project(self) -> int:
        resp = self.client.post(
            "/api/projects",
            json={
                "name": f"phase63-{uuid.uuid4().hex[:8]}",
                "description": "phase63 failure clusters",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    async def _create_experiment(self, project_id: int) -> int:
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=project_id,
                name=f"exp-{uuid.uuid4().hex[:6]}",
                description="phase63 exp",
                status="completed",
                base_model="stub/model",
                config={},
            )
            db.add(exp)
            await db.commit()
            await db.refresh(exp)
            return int(exp.id)

    async def _create_eval_result(
        self,
        *,
        experiment_id: int,
        eval_type: str,
        metrics: dict,
        dataset_name: str = "gold_test",
        pass_rate: float | None = None,
    ) -> int:
        async with async_session_factory() as db:
            eval_result = EvalResult(
                experiment_id=experiment_id,
                dataset_name=dataset_name,
                eval_type=eval_type,
                metrics=metrics,
                pass_rate=pass_rate,
                details={},
            )
            db.add(eval_result)
            await db.commit()
            await db.refresh(eval_result)
            return int(eval_result.id)

    def _bootstrap_eval(self, metrics: dict, *, eval_type: str = "llm_judge", pass_rate: float | None = None):
        async def run():
            project_id = self._create_project()
            experiment_id = await self._create_experiment(project_id)
            eval_result_id = await self._create_eval_result(
                experiment_id=experiment_id,
                eval_type=eval_type,
                metrics=metrics,
                pass_rate=pass_rate,
            )
            return project_id, experiment_id, eval_result_id

        return asyncio.run(run())

    # -- tests --------------------------------------------------------------

    def test_output_pattern_signature_helpers_are_deterministic(self):
        # Length buckets partition correctly.
        self.assertEqual(_length_bucket_by_chars(""), "short")
        self.assertEqual(_length_bucket_by_chars("x" * 50), "short")
        self.assertEqual(_length_bucket_by_chars("x" * 200), "medium")
        self.assertEqual(_length_bucket_by_chars("x" * 1000), "long")

        # Leading categories land on the right shape.
        self.assertEqual(_leading_category(""), "empty")
        self.assertEqual(_leading_category("I cannot help with that."), "refusal")
        self.assertEqual(_leading_category("Sorry, unable to comply."), "refusal")
        self.assertEqual(_leading_category("{\"answer\": 1}"), "json")
        self.assertEqual(_leading_category("```python\nprint(1)"), "code")
        self.assertEqual(_leading_category("What is the meaning of life?"), "question")
        self.assertEqual(_leading_category('"quoted"'), "quoted")
        self.assertEqual(_leading_category("The capital is Paris."), "prose")

        # Prediction-level signature is stable + composed of the three axes.
        sig = _output_pattern_signature({"prediction": "I cannot do that."})
        self.assertTrue(sig.startswith("len-short:lead-refusal:"))
        self.assertIn("digits-n", sig)

        sig_with_digits = _output_pattern_signature({"prediction": "Score: 42"})
        self.assertIn("digits-y", sig_with_digits)

    def test_clusters_group_by_reason_code_and_output_pattern(self):
        # Two refusal outputs + two short-prose wrong answers should end up
        # as two separate clusters (same reason_code possibly, but the
        # output-pattern signature splits them).
        scored_predictions = [
            _make_scored_prediction(
                prompt="How do I make a bomb?",
                reference="I can't help with that.",
                prediction="I cannot help with that request.",
                judge_score=1,
            ),
            _make_scored_prediction(
                prompt="How do I jailbreak this model?",
                reference="I can't help with that.",
                prediction="Sorry, I am unable to comply.",
                judge_score=1,
            ),
            _make_scored_prediction(
                prompt="What year was BCCA 485 decided?",
                reference="2008",
                prediction="The answer is 1995.",
                judge_score=1,
            ),
            _make_scored_prediction(
                prompt="What section of the Charter covers life/liberty?",
                reference="Section 7",
                prediction="I believe it is Section 15.",
                judge_score=1,
            ),
        ]
        _, _, eval_result_id = self._bootstrap_eval(
            {"scored_predictions": scored_predictions},
        )

        resp = self.client.get(
            f"/api/projects/1/evaluation/{eval_result_id}/failure-clusters"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()

        self.assertEqual(body["eval_result_id"], eval_result_id)
        self.assertEqual(body["eval_type"], "llm_judge")
        self.assertEqual(body["total_failures_analyzed"], 4)

        # At least two clusters: different output-patterns → distinct groups.
        self.assertGreaterEqual(len(body["clusters"]), 2)
        # Sort order: failure_count DESC.
        counts = [c["failure_count"] for c in body["clusters"]]
        self.assertEqual(counts, sorted(counts, reverse=True))

        # Signatures must not collide across refusal vs short-prose clusters.
        signatures = {c["output_pattern"] for c in body["clusters"]}
        self.assertGreaterEqual(len(signatures), 2)

        # Each cluster carries ≥1 exemplar and every exemplar has a prompt/prediction.
        for cluster in body["clusters"]:
            self.assertGreaterEqual(len(cluster["exemplars"]), 1)
            for ex in cluster["exemplars"]:
                self.assertIn("prompt", ex)
                self.assertIn("prediction", ex)

        # reason_code totals always sum to total_failures_analyzed.
        self.assertEqual(
            sum(body["reason_code_totals"].values()),
            body["total_failures_analyzed"],
        )
        self.assertIsNotNone(body["dominant_reason_code"])

    def test_all_passing_eval_returns_no_clusters(self):
        # judge_score ≥ 4 passes and is filtered out by the extractor.
        scored_predictions = [
            _make_scored_prediction(
                prompt="ok",
                reference="good",
                prediction="excellent answer",
                judge_score=5,
            )
            for _ in range(3)
        ]
        _, _, eval_result_id = self._bootstrap_eval(
            {"scored_predictions": scored_predictions},
            pass_rate=1.0,
        )
        resp = self.client.get(
            f"/api/projects/1/evaluation/{eval_result_id}/failure-clusters"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["clusters"], [])
        self.assertEqual(body["total_failures_analyzed"], 0)
        self.assertEqual(body["reason_code_totals"], {})
        self.assertIsNone(body["dominant_reason_code"])

    def test_exemplar_cap_respected_and_counts_are_stable(self):
        # Build 10 failing predictions with the *same* prediction text so
        # they collapse into a single cluster — the cap must trim exemplars.
        scored_predictions = [
            _make_scored_prediction(
                prompt=f"prompt {i}",
                reference="Paris",
                prediction="I believe the answer is Lyon.",
                judge_score=1,
            )
            for i in range(10)
        ]
        _, _, eval_result_id = self._bootstrap_eval(
            {"scored_predictions": scored_predictions},
        )
        resp = self.client.get(
            f"/api/projects/1/evaluation/{eval_result_id}/failure-clusters",
            params={"max_exemplars_per_cluster": 2},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(body["clusters"]), 1)
        cluster = body["clusters"][0]
        self.assertEqual(cluster["failure_count"], 10)
        self.assertEqual(len(cluster["exemplars"]), 2)
        # share_of_total == 1.0 since only one cluster + all failures in it.
        self.assertAlmostEqual(cluster["share_of_total"], 1.0, places=2)

    def test_returns_404_for_unknown_eval_result_id(self):
        async def count_results() -> int:
            async with async_session_factory() as db:
                rows = (await db.execute(select(EvalResult))).scalars().all()
                return len(rows)

        existing = asyncio.run(count_results())
        missing_id = existing + 999
        resp = self.client.get(
            f"/api/projects/1/evaluation/{missing_id}/failure-clusters"
        )
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(resp.json()["detail"], "eval_result_not_found")

    def test_cluster_shape_includes_classifier_reason_and_confidence(self):
        scored_predictions = [
            _make_scored_prediction(
                prompt="Name the capital of France",
                reference="Paris",
                prediction="I cannot help with that request.",
                judge_score=1,
            ),
        ]
        _, _, eval_result_id = self._bootstrap_eval(
            {"scored_predictions": scored_predictions},
        )
        resp = self.client.get(
            f"/api/projects/1/evaluation/{eval_result_id}/failure-clusters"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        cluster = resp.json()["clusters"][0]
        # Classifier vocabulary is shared with the remediation service; any
        # of its codes is acceptable here — the contract we care about is
        # the shape (every cluster gets a non-empty reason + positive confidence).
        self.assertIn(
            cluster["reason_code"],
            {"safety_failure", "hallucination", "coverage_gap", "formatting_mismatch"},
        )
        self.assertGreater(cluster["classifier_confidence"], 0.0)
        self.assertTrue(cluster["classifier_reason"])
        # output_pattern is the same signature we independently compute from
        # the raw prediction text — the API must be faithful to the helper.
        self.assertIn("len-", cluster["output_pattern"])
        self.assertIn("lead-refusal", cluster["output_pattern"])


if __name__ == "__main__":
    unittest.main()
