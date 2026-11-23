"""Simulate-run UI fixes: preflight bnb warning, per-step metrics history."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.services import training_preflight_service as preflight
from app.services import training_service


def _run(runtime_id: str) -> dict:
    deps = {"torch": True, "transformers": True, "peft": True, "trl": True, "bitsandbytes": False}
    with mock.patch.object(preflight, "_dependency_status_snapshot", return_value=deps):
        return preflight.run_training_preflight(
            project_id=1,
            config={"optimizer": "paged_adamw_8bit", "training_runtime_id": runtime_id},
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        )


class BitsandbytesWarningTests(unittest.TestCase):
    def test_hidden_for_simulate_runtime(self):
        out = _run("builtin.simulate")
        self.assertFalse(any("bitsandbytes" in w for w in out["warnings"]))

    def test_kept_for_real_runtime(self):
        out = _run("builtin.external_celery")
        self.assertTrue(any("bitsandbytes" in w for w in out["warnings"]))


class SimulatedHistoryTests(unittest.TestCase):
    def test_round_trip_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(training_service.read_simulated_metrics_history(tmp), [])
            self.assertEqual(training_service.read_simulated_metrics_history(None), [])
            rows = [{"step": i, "train_loss": 1.0, "eval_loss": None} for i in range(1, 151)]
            training_service._write_simulated_metrics_history(Path(tmp), rows)
            self.assertEqual(len(training_service.read_simulated_metrics_history(tmp)), 150)
            (Path(tmp) / training_service.SIMULATED_METRICS_HISTORY_FILE).write_text("{bad", encoding="utf-8")
            self.assertEqual(training_service.read_simulated_metrics_history(tmp), [])
