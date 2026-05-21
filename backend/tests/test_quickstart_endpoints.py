"""Tests for the project-guide quickstart endpoints (Theme 1 Epic 4 +
Theme 8 Epic 1): import-sample, train-default, evaluate-latest,
baseline-eval."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_DB_PATH = Path(__file__).resolve().parent / "quickstart_endpoints_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "quickstart_endpoints_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class QuickstartImportSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "quickstart tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_import_sample_falls_back_to_support_faq_without_recipe(self):
        project_id = self._create_project("qs-import-default-slug")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()
        self.assertEqual(payload["status"], "ok")
        summary = payload["summary"]
        self.assertEqual(summary["slug"], "support-faq")
        self.assertGreater(summary["source_row_count"], 0)
        self.assertGreater(summary["gold_row_count"], 0)
        self.assertGreater(summary["prepared_train_rows"], 0)

    def test_import_sample_advances_pipeline_stage_so_ingest_step_completes(self):
        project_id = self._create_project("qs-import-advances-stage")
        self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        # The 'ingest' guide step's predicate is `stageIndex >= 1`.
        # Confirm the project moved off the INGESTION default.
        reread = self.client.get(f"/api/projects/{project_id}")
        self.assertEqual(reread.status_code, 200, reread.text)
        stage = reread.json()["pipeline_stage"]
        self.assertNotEqual(stage, "ingestion")

    def test_import_sample_derives_slug_from_selected_recipe(self):
        project_id = self._create_project("qs-import-from-recipe")
        # Apply the classification recipe — the slug derivation should
        # pick sentiment-classifier as the matching demo bundle.
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["summary"]["slug"], "sentiment-classifier")

    def test_import_sample_explicit_slug_overrides_recipe_derivation(self):
        project_id = self._create_project("qs-import-explicit-slug")
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "classification"},
        )
        # User explicitly passes pii-detector — should win over the
        # classification → sentiment-classifier auto-derivation.
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={"slug": "pii-detector"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["summary"]["slug"], "pii-detector")

    def test_import_sample_unknown_slug_returns_404(self):
        project_id = self._create_project("qs-import-unknown-slug")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={"slug": "not-a-real-bundle"},
        )
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_import_sample_missing_project_returns_404(self):
        resp = self.client.post(
            "/api/projects/99999/quickstart/import-sample",
            json={},
        )
        self.assertEqual(resp.status_code, 404, resp.text)


class QuickstartTrainDefaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "qs-train tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_train_default_rejects_project_without_base_model(self):
        # Fresh project, no recipe applied, no base_model_name set.
        project_id = self._create_project("qs-train-no-base-model")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/train-default",
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("base_model_name", resp.text)

    def test_train_default_creates_and_starts_after_recipe_apply(self):
        # Apply recipe → propagates suggested_base_model onto project,
        # then quickstart should accept the call. With TRAINING_BACKEND=
        # simulate the simulator runs without a real GPU.
        project_id = self._create_project("qs-train-with-recipe")
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "qa-sft"},
        )
        # Also materialize sample data so the training-data gate has
        # something to read.
        self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/train-default",
        )
        # We accept either: (a) 201 with experiment_id, or (b) 400 if
        # the simulated training backend rejects something
        # environment-specific. The contract under test is that the
        # endpoint REACHES start_training; it shouldn't fail on the
        # "no base_model" guard.
        self.assertIn(resp.status_code, (201, 400, 409), resp.text)
        if resp.status_code == 201:
            payload = resp.json()
            self.assertEqual(payload["status"], "training_started")
            self.assertIn("experiment_id", payload)
            self.assertTrue(payload["base_model"])
            self.assertEqual(payload["recipe_id"], "qa-sft")
        else:
            # Even when something downstream errors, it must NOT be
            # the missing-base-model guard.
            self.assertNotIn("base_model_name", resp.text)


class QuickstartTourStateRoundTripTests(unittest.TestCase):
    """The new `projects.quickstart_tour_state` JSON column (Theme 1 Epic 2)
    holds dismissed nudge ids. The frontend writes via PUT /projects/{id}
    and reads via GET. Confirm the round-trip preserves the shape and
    doesn't bleed into other fields."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "quickstart tour tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_default_tour_state_is_null_on_a_fresh_project(self):
        project_id = self._create_project("qs-tour-default")
        resp = self.client.get(f"/api/projects/{project_id}")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIsNone(resp.json().get("quickstart_tour_state"))

    def test_dismissed_nudges_survive_a_round_trip(self):
        project_id = self._create_project("qs-tour-roundtrip")
        # Frontend dismiss flow: PUT the project with the new
        # tour-state payload (mirrors what QuickstartCard does).
        put_resp = self.client.put(
            f"/api/projects/{project_id}",
            json={
                "quickstart_tour_state": {
                    "dismissed_nudges": ["import_to_train"],
                },
            },
        )
        self.assertEqual(put_resp.status_code, 200, put_resp.text)
        self.assertEqual(
            put_resp.json()["quickstart_tour_state"],
            {"dismissed_nudges": ["import_to_train"]},
        )

        # Re-read confirms the JSON column round-tripped.
        get_resp = self.client.get(f"/api/projects/{project_id}")
        self.assertEqual(
            get_resp.json()["quickstart_tour_state"],
            {"dismissed_nudges": ["import_to_train"]},
        )

    def test_appending_a_second_nudge_does_not_clobber_the_first(self):
        project_id = self._create_project("qs-tour-append")
        self.client.put(
            f"/api/projects/{project_id}",
            json={"quickstart_tour_state": {"dismissed_nudges": ["import_to_train"]}},
        )
        # Frontend always sends the full set, not a delta — confirm
        # that contract works.
        resp = self.client.put(
            f"/api/projects/{project_id}",
            json={
                "quickstart_tour_state": {
                    "dismissed_nudges": ["import_to_train", "train_to_eval"],
                },
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(
            resp.json()["quickstart_tour_state"]["dismissed_nudges"],
            ["import_to_train", "train_to_eval"],
        )


class QuickstartEvaluateLatestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "qs-eval tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def test_evaluate_latest_returns_409_when_no_experiments_exist(self):
        project_id = self._create_project("qs-eval-no-exps")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/evaluate-latest",
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertIn("training", resp.text.lower())

    def test_evaluate_latest_missing_project_returns_404(self):
        resp = self.client.post(
            "/api/projects/99999/quickstart/evaluate-latest",
        )
        self.assertEqual(resp.status_code, 404, resp.text)


class QuickstartBaselineEvalTests(unittest.TestCase):
    """Theme 8 Epic 1 — baseline-eval. Creates a synthetic Experiment
    keyed on (project, base_model) and runs `run_heldout_evaluation`
    against the un-fine-tuned base model. The eval handler's local
    inference is mocked so the suite stays hermetic."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled

    def _create_project(self, name: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name, "description": "qs-baseline tests"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _seed_project_with_recipe_and_data(self, project_name: str) -> int:
        """Apply qa-sft recipe (sets base_model_name) + import the
        support-faq demo so the project has a prepared test split."""
        project_id = self._create_project(project_name)
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "qa-sft"},
        )
        self.client.post(
            f"/api/projects/{project_id}/quickstart/import-sample",
            json={},
        )
        return project_id

    def test_baseline_rejects_project_without_base_model(self):
        project_id = self._create_project("qs-baseline-no-model")
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/baseline-eval",
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("base_model_name", resp.text)

    def test_baseline_returns_409_when_no_test_split_yet(self):
        project_id = self._create_project("qs-baseline-no-data")
        self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": "qa-sft"},
        )
        # Recipe applied (base_model set) but data not imported yet.
        resp = self.client.post(
            f"/api/projects/{project_id}/quickstart/baseline-eval",
        )
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertIn("Import sample CSV", resp.text)

    def test_baseline_eval_creates_synthetic_experiment_and_returns_metrics(self):
        project_id = self._seed_project_with_recipe_and_data("qs-baseline-happy")

        mock_predictions = [
            {"prediction": "answer 1", "reference": "answer 1"},
            {"prediction": "answer 2", "reference": "different"},
            {"prediction": "answer 3", "reference": "answer 3"},
        ]
        mock_runtime = {
            "engine": "transformers",
            "device": "cpu",
            "latency_ms_per_sample": 12.3,
        }
        with patch(
            "app.services.evaluation_service._run_local_inference",
            return_value=(mock_predictions, mock_runtime),
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/quickstart/baseline-eval",
            )

        self.assertEqual(resp.status_code, 201, resp.text)
        payload = resp.json()
        self.assertEqual(payload["status"], "baseline_complete")
        self.assertTrue(payload["base_model"])
        self.assertIn("Baseline", payload["experiment_name"])
        self.assertIn("experiment_id", payload)
        self.assertIn("result", payload)

        # The synthetic experiment is visible via the normal
        # experiments list — that's how the Compare page can render
        # baseline vs. trained side-by-side for free.
        listing = self.client.get(f"/api/projects/{project_id}/training/experiments")
        self.assertEqual(listing.status_code, 200, listing.text)
        body = listing.json()
        rows = body if isinstance(body, list) else (body.get("experiments") or [])
        names = [exp.get("name") for exp in rows if isinstance(exp, dict)]
        self.assertTrue(
            any(n and n.startswith("Baseline") for n in names),
            f"expected a Baseline experiment in {names!r}",
        )

    def test_baseline_eval_is_idempotent_for_same_base_model(self):
        project_id = self._seed_project_with_recipe_and_data("qs-baseline-idempotent")

        mock_predictions = [{"prediction": "x", "reference": "x"}]
        mock_runtime = {"engine": "transformers", "device": "cpu"}
        with patch(
            "app.services.evaluation_service._run_local_inference",
            return_value=(mock_predictions, mock_runtime),
        ):
            first = self.client.post(
                f"/api/projects/{project_id}/quickstart/baseline-eval",
            )
            second = self.client.post(
                f"/api/projects/{project_id}/quickstart/baseline-eval",
            )

        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 201, second.text)
        # Same synthetic experiment reused — we don't want a flood of
        # "Baseline 1 / Baseline 2 / ..." rows from accidental re-clicks.
        self.assertEqual(
            first.json()["experiment_id"],
            second.json()["experiment_id"],
        )

    def test_baseline_eval_supplies_model_path_override(self):
        """The whole point of the baseline endpoint is to bypass the
        artifact resolver — the synthetic experiment has no output_dir,
        so `run_heldout_evaluation` must be called with the
        `model_path=project.base_model_name` override. This test
        verifies that contract via a capturing mock."""
        project_id = self._seed_project_with_recipe_and_data("qs-baseline-model-path")

        captured_calls: list[dict] = []

        async def _fake_run_heldout(**kwargs):
            captured_calls.append(kwargs)
            return {
                "experiment_id": kwargs.get("experiment_id"),
                "dataset_name": kwargs.get("dataset_name"),
                "eval_type": kwargs.get("eval_type"),
                "metrics": {"exact_match": 0.42},
                "pass_rate": 0.42,
                "details": {},
            }

        with patch(
            "app.api.quickstart.run_heldout_evaluation",
            side_effect=_fake_run_heldout,
        ):
            resp = self.client.post(
                f"/api/projects/{project_id}/quickstart/baseline-eval",
            )

        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(len(captured_calls), 1)
        call = captured_calls[0]
        self.assertEqual(
            call["model_path"],
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        self.assertEqual(call["dataset_name"], "test")
        self.assertEqual(call["temperature"], 0.0)

    def test_baseline_eval_missing_project_returns_404(self):
        resp = self.client.post(
            "/api/projects/99999/quickstart/baseline-eval",
        )
        self.assertEqual(resp.status_code, 404, resp.text)


if __name__ == "__main__":
    unittest.main()
