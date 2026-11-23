"""Phase 7i: a real run's epoch total and its final loss curve.

- The Train page showed "8.00 / 3" on a 3-epoch form because ``auto_epochs`` made the trainer run 8 epochs
  (108 rows, 216 steps). ``train.py`` now emits an ``epoch_plan`` event at train begin; the worker saves it
  on the run (``total_epochs`` / ``total_steps``) and relays it to the page.
- A finished real run's chart fell back to its 3 checkpoint rows. ``read_run_metrics_history`` reads the
  trainer's ``metrics.jsonl`` (HF log history) so the curve keeps every logged point.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.services import training_service
from app.services.training_epoch_plan_service import epoch_plan_from_event, epoch_plan_log_line
from scripts.train import _epoch_plan_event


# The video-01 run's metrics.jsonl shape (SmolLM2-135M, 108 rows, 8 auto epochs, 216 steps), shortened.
_LOG_HISTORY = [
    {"loss": 3.83, "grad_norm": 1.07, "learning_rate": 0.0002, "epoch": 0.7407407407407407, "step": 20},
    {"loss": 3.19, "grad_norm": 1.43, "learning_rate": 0.00019, "epoch": 1.4814814814814814, "step": 40},
    {"loss": 2.16, "grad_norm": 1.95, "learning_rate": 0.00012, "epoch": 3.7037037037037037, "step": 100},
    {"eval_loss": 2.83, "eval_runtime": 0.26, "epoch": 3.7037037037037037, "step": 100},
    {"loss": 1.50, "grad_norm": 1.90, "learning_rate": 3.2e-06, "epoch": 7.407407407407407, "step": 200},
    {"eval_loss": 2.80, "eval_runtime": 0.25, "epoch": 7.407407407407407, "step": 200},
    {"train_runtime": 29.5, "train_loss": 2.205, "epoch": 8.0, "step": 216},
    {"eval_loss": 2.80, "eval_runtime": 0.25, "epoch": 8.0, "step": 216},
]


def _write_metrics(tmp: str, rows: list[dict], extra_lines: tuple[str, ...] = ()) -> None:
    lines = [json.dumps(r) for r in rows] + list(extra_lines)
    (Path(tmp) / training_service.RUN_METRICS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


class RunMetricsHistoryTests(unittest.TestCase):
    def test_real_run_history_keeps_every_logged_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_metrics(tmp, _LOG_HISTORY, ("not json", "", '{"loss": NaN, "step": 220}'))
            rows = training_service.read_run_metrics_history(tmp)
        self.assertEqual([r["step"] for r in rows if "train_loss" in r], [20, 40, 100, 200])
        self.assertEqual([r["step"] for r in rows if "eval_loss" in r], [100, 200, 216])
        # The run-average summary row is not a step: no 2.205 "train loss" at the end of the curve.
        self.assertFalse(any(r.get("train_loss") == 2.205 for r in rows))
        self.assertEqual(rows[0], {"step": 20, "epoch": 0.7407, "train_loss": 3.83})
        json.dumps(rows, allow_nan=False)  # the NaN row was dropped: the browser can parse the response

    def test_simulated_history_wins_and_missing_files_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(training_service.read_run_metrics_history(tmp), [])
            self.assertEqual(training_service.read_run_metrics_history(None), [])
            _write_metrics(tmp, _LOG_HISTORY)
            sim = [{"step": i, "train_loss": 1.0} for i in range(1, 4)]
            training_service._write_simulated_metrics_history(Path(tmp), sim)
            self.assertEqual(training_service.read_run_metrics_history(tmp), sim)

    def test_long_logs_are_thinned_but_keep_eval_points_and_the_last_train_point(self):
        rows_in = [{"loss": 1.0 + i / 10000, "epoch": i / 100, "step": i} for i in range(1, 5001)]
        rows_in += [{"eval_loss": 2.0, "epoch": i / 100, "step": i} for i in range(500, 5001, 500)]
        with tempfile.TemporaryDirectory() as tmp:
            _write_metrics(tmp, rows_in)
            rows = training_service.read_run_metrics_history(tmp)
        self.assertLessEqual(len(rows), 2001)
        self.assertEqual(len([r for r in rows if "eval_loss" in r]), 10)
        self.assertEqual(max(r["step"] for r in rows if "train_loss" in r), 5000)


class EpochPlanTests(unittest.TestCase):
    def test_trainer_plan_read_at_train_begin(self):
        args = SimpleNamespace(num_train_epochs=8.0, max_steps=-1)
        state = SimpleNamespace(max_steps=216, num_train_epochs=8)
        self.assertEqual(_epoch_plan_event(args, state), {"num_epochs": 8, "total_steps": 216})
        # A step cap decides the length: the trainer's own epoch count, not the form's.
        capped = SimpleNamespace(num_train_epochs=3.0, max_steps=50)
        self.assertEqual(_epoch_plan_event(capped, SimpleNamespace(max_steps=50, num_train_epochs=2)),
                         {"num_epochs": 2, "total_steps": 50})
        self.assertEqual(_epoch_plan_event(SimpleNamespace(), SimpleNamespace()), {})

    def test_event_is_sanitised_and_logged(self):
        self.assertEqual(epoch_plan_from_event({"event": "epoch_plan", "num_epochs": 8, "total_steps": 216}),
                         {"num_epochs": 8, "total_steps": 216})
        self.assertEqual(epoch_plan_from_event({"num_epochs": 2.5}), {"num_epochs": 2.5})
        self.assertEqual(epoch_plan_from_event({"num_epochs": 0, "total_steps": "x"}), {})
        self.assertEqual(epoch_plan_from_event({"num_epochs": True}), {})
        self.assertEqual(epoch_plan_from_event(None), {})
        self.assertEqual(epoch_plan_log_line({"num_epochs": 8, "total_steps": 216}),
                         "[event] training plan: 8 epochs, 216 optimizer steps")
        self.assertEqual(epoch_plan_log_line({}), "")


if __name__ == "__main__":
    unittest.main()
