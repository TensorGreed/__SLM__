"""Student-vs-teacher distillation comparison — Track 1, Epic A, slice 3.

Pure math (quality_retained, per-metric, per-slice) + endpoint integration with
seeded experiments/eval results. No model calls — comparison reads stored rows.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.student_teacher_comparison_service import (  # noqa: E402
    _compute_metric_comparisons,
    _compute_slice_comparisons,
    _direction,
    _quality_retained,
)

_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__exit__(None, None, None)


class QualityRetainedMathTests(unittest.TestCase):
    def test_ratio_when_teacher_positive(self):
        self.assertAlmostEqual(_quality_retained(0.65, 0.80), 0.8125, places=4)
        self.assertAlmostEqual(_quality_retained(0.90, 0.80), 1.125, places=4)

    def test_zero_teacher_nonzero_student_is_none(self):
        self.assertIsNone(_quality_retained(0.5, 0.0))

    def test_both_zero_is_full_retention(self):
        self.assertEqual(_quality_retained(0.0, 0.0), 1.0)

    def test_direction(self):
        self.assertEqual(_direction(0.9, 0.8, 1.125), "retained_or_better")
        self.assertEqual(_direction(0.6, 0.8, 0.75), "regressed")
        self.assertEqual(_direction(0.5, 0.0, None), "exceeds")

    def test_metric_comparisons_shared_only_headline_first(self):
        student = {"f1": 0.65, "accuracy": 0.7, "only_student": 0.1}
        teacher = {"f1": 0.80, "accuracy": 0.9, "only_teacher": 0.2}
        rows = _compute_metric_comparisons(student, teacher)
        ids = [r["metric_id"] for r in rows]
        # shared keys only; f1 (headline) before accuracy.
        self.assertEqual(ids, ["f1", "accuracy"])
        self.assertTrue(rows[0]["is_headline"])
        self.assertAlmostEqual(rows[0]["quality_retained"], 0.8125, places=4)

    def test_metric_comparisons_empty_when_no_overlap(self):
        self.assertEqual(_compute_metric_comparisons({"a": 1.0}, {"b": 2.0}), [])

    def test_slice_comparisons(self):
        student_details = {"slice_metrics": {"short": {"f1": 0.5}, "long": {"f1": 0.4}}}
        teacher_details = {"slice_metrics": {"short": {"f1": 0.8}, "other": {"f1": 0.9}}}
        rows = _compute_slice_comparisons(student_details, teacher_details)
        # Only the shared slice "short" × shared metric "f1".
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["slice"], "short")
        self.assertAlmostEqual(rows[0]["quality_retained"], 0.625, places=4)

    def test_slice_comparisons_empty_without_slice_metrics(self):
        self.assertEqual(_compute_slice_comparisons({}, {}), [])
        self.assertEqual(_compute_slice_comparisons({"metrics": {}}, None), [])


class ComparisonEndpointTests(unittest.TestCase):
    client: TestClient

    @classmethod
    def setUpClass(cls):
        cls.client = _MODULE_CLIENT_CM

    def _create_project(self, name: str) -> int:
        resp = self.client.post("/api/projects", json={"name": f"{name}-{uuid.uuid4().hex[:6]}"})
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_experiment(
        self,
        project_id: int,
        name: str,
        *,
        metrics: dict | None = None,
        config: dict | None = None,
        details: dict | None = None,
        with_eval: bool = True,
    ) -> int:
        from app.database import async_session_factory
        from app.models.experiment import EvalResult, Experiment

        async def _create() -> int:
            async with async_session_factory() as db:
                exp = Experiment(
                    project_id=project_id,
                    name=name,
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config=config or {},
                )
                db.add(exp)
                await db.flush()
                exp_id = exp.id
                if with_eval:
                    db.add(
                        EvalResult(
                            experiment_id=exp_id,
                            dataset_name="test",
                            eval_type="f1",
                            metrics=metrics or {},
                            details=details or {},
                        )
                    )
                await db.commit()
                return exp_id

        return asyncio.run(_create())

    def _get(self, pid: int, exp_id: int, teacher_run_id: int | None = None):
        url = f"/api/projects/{pid}/evaluation/student-teacher-comparison/{exp_id}"
        params = {} if teacher_run_id is None else {"teacher_run_id": teacher_run_id}
        return self.client.get(url, params=params)

    def test_ok_with_config_teacher_and_quality_retained(self):
        pid = self._create_project("st-ok")
        teacher = self._seed_experiment(pid, "teacher", metrics={"f1": 0.80, "accuracy": 0.9})
        student = self._seed_experiment(
            pid, "student", metrics={"f1": 0.65, "accuracy": 0.75},
            config={"teacher_baseline_run_id": teacher},
        )
        resp = self._get(pid, student)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["teacher_baseline_run_id"], teacher)
        self.assertAlmostEqual(body["headline_quality_retained"], 0.8125, places=4)
        f1_row = next(r for r in body["metric_comparisons"] if r["metric_id"] == "f1")
        self.assertAlmostEqual(f1_row["quality_retained"], 0.8125, places=4)
        self.assertEqual(f1_row["direction"], "regressed")

    def test_explicit_teacher_run_id_overrides(self):
        pid = self._create_project("st-explicit")
        teacher = self._seed_experiment(pid, "teacher", metrics={"f1": 0.5})
        student = self._seed_experiment(pid, "student", metrics={"f1": 0.5})
        resp = self._get(pid, student, teacher_run_id=teacher)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertAlmostEqual(body["headline_quality_retained"], 1.0, places=4)

    def test_no_teacher_baseline_when_unresolvable(self):
        pid = self._create_project("st-noteacher")
        student = self._seed_experiment(pid, "student", metrics={"f1": 0.5})
        body = self._get(pid, student).json()
        self.assertEqual(body["status"], "no_teacher_baseline")
        self.assertIsNone(body["teacher_baseline_run_id"])

    def test_no_student_eval(self):
        pid = self._create_project("st-nostudent")
        teacher = self._seed_experiment(pid, "teacher", metrics={"f1": 0.8})
        student = self._seed_experiment(pid, "student", with_eval=False)
        body = self._get(pid, student, teacher_run_id=teacher).json()
        self.assertEqual(body["status"], "no_student_eval")

    def test_no_teacher_eval_when_teacher_missing(self):
        pid = self._create_project("st-noteachereval")
        student = self._seed_experiment(pid, "student", metrics={"f1": 0.5})
        body = self._get(pid, student, teacher_run_id=999_999).json()
        self.assertEqual(body["status"], "no_teacher_eval")

    def test_no_overlap(self):
        pid = self._create_project("st-nooverlap")
        teacher = self._seed_experiment(pid, "teacher", metrics={"f1": 0.8})
        student = self._seed_experiment(pid, "student", metrics={"accuracy": 0.7})
        body = self._get(pid, student, teacher_run_id=teacher).json()
        self.assertEqual(body["status"], "no_overlap")

    def test_per_slice_comparison(self):
        pid = self._create_project("st-slice")
        teacher = self._seed_experiment(
            pid, "teacher", metrics={"f1": 0.8},
            details={"slice_metrics": {"short": {"f1": 0.8}}},
        )
        student = self._seed_experiment(
            pid, "student", metrics={"f1": 0.6},
            details={"slice_metrics": {"short": {"f1": 0.6}}},
            config={"teacher_baseline_run_id": teacher},
        )
        body = self._get(pid, student).json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(len(body["slice_comparisons"]), 1)
        self.assertEqual(body["slice_comparisons"][0]["slice"], "short")
        self.assertAlmostEqual(
            body["slice_comparisons"][0]["quality_retained"], 0.75, places=4
        )

    def test_missing_experiment_404(self):
        pid = self._create_project("st-404")
        resp = self._get(pid, 999_999, teacher_run_id=1)
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
