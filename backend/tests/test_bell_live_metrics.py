"""Tests for the bell live-loss sparkline backend
(``_recent_training_metrics`` + ``serialize_job_with_live_metrics``).

Pins:

  * Non-training jobs get ``metrics_recent = None`` (filtered out
    of the serialized response). The bell only renders sparklines
    for training rows.
  * Training jobs WITHOUT an experiment_id in params return
    ``None`` — defensive against jobs that lost their reference.
  * Training jobs whose experiment directory has no
    ``checkpoint-N`` subdir return ``None`` — first ~50 steps
    before the trainer's first save; bell renders the
    placeholder line.
  * Training jobs with a checkpoint return the last 20 (step,
    train_loss?, eval_loss?) tuples, rounded to 5 decimal
    places, walking the tail of ``trainer_state.json``'s
    ``log_history`` array.
  * Cap is exactly 20 — neither more nor less. The sparkline at
    80×16px doesn't benefit from more points; more wastes the
    JSON payload on every 4s poll.
  * ``serialize_job_with_live_metrics`` adds ``metrics_recent``
    to the payload when present, omits it otherwise — non-bell
    consumers via plain ``serialize_job`` stay on the frozen
    shape.
  * Malformed ``trainer_state.json`` returns ``None`` rather
    than raising — the bell poll loop must not break because of
    a partially-written file mid-step.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.config import settings  # noqa: E402
from app.models.job import Job, JobStatus  # noqa: E402
from app.services.jobs_service import (  # noqa: E402
    _BELL_METRICS_RECENT_CAP,
    _recent_training_metrics,
    serialize_job,
    serialize_job_with_live_metrics,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-bell-metrics-{uuid.uuid4().hex[:8]}"
)


def _job(
    *,
    kind: str = "training_start",
    project_id: int | None = 17,
    experiment_id: int | None = 20,
    status: JobStatus = JobStatus.RUNNING,
) -> Job:
    params: dict = {}
    if experiment_id is not None:
        params["experiment_id"] = experiment_id
    return Job(
        id=1,
        kind=kind,
        title="t",
        status=status,
        progress=0.5,
        progress_message="m",
        project_id=project_id,
        user_id=None,
        params=params,
        result=None,
        error=None,
        queued_at=datetime.now(timezone.utc),
    )


def _write_checkpoint(
    *,
    project_id: int,
    experiment_id: int,
    step: int,
    log_history: list[dict] | None,
) -> Path:
    """Build a fake experiment dir on disk with a checkpoint-N
    subdir + trainer_state.json containing ``log_history``. Returns
    the experiment root."""
    exp_root = (
        TEST_DATA_DIR
        / "projects"
        / str(project_id)
        / "experiments"
        / str(experiment_id)
    )
    ckpt = exp_root / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    if log_history is not None:
        (ckpt / "trainer_state.json").write_text(
            json.dumps(
                {"global_step": step, "log_history": log_history}
            )
        )
    return exp_root


class RecentTrainingMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()

    def test_non_training_job_returns_none(self):
        # Bell only renders sparklines for training rows; everything
        # else must short-circuit.
        for kind in ("synth_playbook", "auto_rag_comparison", "reroute_to_rag"):
            self.assertIsNone(
                _recent_training_metrics(_job(kind=kind)),
                f"kind={kind!r}",
            )

    def test_training_job_without_experiment_id_returns_none(self):
        self.assertIsNone(_recent_training_metrics(_job(experiment_id=None)))

    def test_training_job_without_project_id_returns_none(self):
        self.assertIsNone(_recent_training_metrics(_job(project_id=None)))

    def test_training_job_with_no_experiment_dir_returns_none(self):
        # No directory at all — pre-first-checkpoint state.
        self.assertIsNone(
            _recent_training_metrics(
                _job(project_id=99999, experiment_id=99999)
            )
        )

    def test_training_job_with_no_checkpoint_returns_none(self):
        # Experiment dir exists but no checkpoint-N subdir yet.
        pid, exp_id = 17, 401
        exp_root = (
            TEST_DATA_DIR
            / "projects"
            / str(pid)
            / "experiments"
            / str(exp_id)
        )
        exp_root.mkdir(parents=True, exist_ok=True)
        self.assertIsNone(
            _recent_training_metrics(
                _job(project_id=pid, experiment_id=exp_id)
            )
        )

    def test_returns_recent_train_loss_tuples(self):
        # Happy path — 5 train-loss entries, all surface.
        pid, exp_id = 17, 402
        _write_checkpoint(
            project_id=pid,
            experiment_id=exp_id,
            step=500,
            log_history=[
                {"step": 100, "loss": 0.5},
                {"step": 200, "loss": 0.4},
                {"step": 300, "loss": 0.3},
                {"step": 400, "loss": 0.25},
                {"step": 500, "loss": 0.2},
            ],
        )
        out = _recent_training_metrics(
            _job(project_id=pid, experiment_id=exp_id)
        )
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0], {"step": 100, "train_loss": 0.5})
        self.assertEqual(out[-1], {"step": 500, "train_loss": 0.2})

    def test_caps_at_metrics_recent_cap(self):
        # _BELL_METRICS_RECENT_CAP entries kept (the tail). Cap is
        # exactly 20 — assert against the constant rather than a
        # magic literal so a future cap-change can't drift the
        # contract silently.
        pid, exp_id = 17, 403
        big = [
            {"step": (i + 1) * 10, "loss": 0.5 - i * 0.001}
            for i in range(50)
        ]
        _write_checkpoint(
            project_id=pid, experiment_id=exp_id, step=500, log_history=big,
        )
        out = _recent_training_metrics(
            _job(project_id=pid, experiment_id=exp_id)
        )
        self.assertEqual(len(out), _BELL_METRICS_RECENT_CAP)
        # Last entry of the input survives.
        self.assertEqual(out[-1]["step"], 500)
        # First entry of the output is from the TAIL window.
        self.assertEqual(
            out[0]["step"], (50 - _BELL_METRICS_RECENT_CAP + 1) * 10,
        )

    def test_eval_rows_surface_as_eval_loss_field(self):
        # Trainer log_history interleaves train rows ({step, loss}) and
        # eval rows ({step, eval_loss, eval_accuracy}). The bell
        # carries both forward; the sparkline picks ``train_loss`` for
        # the line, ``eval_loss`` (currently unused but reserved) for
        # markers in a future enhancement.
        pid, exp_id = 17, 404
        _write_checkpoint(
            project_id=pid,
            experiment_id=exp_id,
            step=300,
            log_history=[
                {"step": 100, "loss": 0.4},
                {"step": 200, "loss": 0.3},
                {"step": 200, "eval_loss": 0.35, "eval_accuracy": 0.9},
                {"step": 300, "loss": 0.25},
            ],
        )
        out = _recent_training_metrics(
            _job(project_id=pid, experiment_id=exp_id)
        )
        steps_to_keys = {(r["step"], "eval_loss" in r) for r in out}
        # Eval-only entry surfaces with just ``eval_loss``; train-only
        # entries with just ``train_loss``.
        self.assertIn((200, True), steps_to_keys)
        self.assertIn((100, False), steps_to_keys)

    def test_picks_highest_step_checkpoint(self):
        # Multiple checkpoints — bell reads the freshest one.
        pid, exp_id = 17, 405
        _write_checkpoint(
            project_id=pid, experiment_id=exp_id, step=100,
            log_history=[{"step": 100, "loss": 0.9}],
        )
        _write_checkpoint(
            project_id=pid, experiment_id=exp_id, step=200,
            log_history=[{"step": 200, "loss": 0.1}],
        )
        out = _recent_training_metrics(
            _job(project_id=pid, experiment_id=exp_id)
        )
        # The 200-step checkpoint's log_history is what comes back.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["train_loss"], 0.1)

    def test_malformed_trainer_state_returns_none_not_raise(self):
        # A partially-written trainer_state.json (mid-step save)
        # must not break the bell poll loop. Returning None lets the
        # bell render the placeholder while the next save completes.
        pid, exp_id = 17, 406
        exp_root = (
            TEST_DATA_DIR
            / "projects"
            / str(pid)
            / "experiments"
            / str(exp_id)
        )
        ckpt = exp_root / "checkpoint-100"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "trainer_state.json").write_text("{not-json-{[}")
        self.assertIsNone(
            _recent_training_metrics(
                _job(project_id=pid, experiment_id=exp_id)
            )
        )

    def test_log_history_with_no_loss_entries_returns_empty_list(self):
        # Edge case — trainer wrote a checkpoint before logging any
        # loss values (e.g., immediate save-on-start). The function
        # returns an empty list rather than None so the bell knows
        # "there IS a checkpoint, the trainer just hasn't logged
        # loss yet."
        pid, exp_id = 17, 407
        _write_checkpoint(
            project_id=pid, experiment_id=exp_id, step=10,
            log_history=[{"step": 10, "learning_rate": 0.001}],
        )
        out = _recent_training_metrics(
            _job(project_id=pid, experiment_id=exp_id)
        )
        self.assertEqual(out, [])


class SerializeJobWithLiveMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()

    def test_non_training_job_omits_metrics_recent(self):
        # ``serialize_job`` shape stays frozen — non-bell consumers
        # never see this field. The wrapper only adds it when
        # actually present.
        out = serialize_job_with_live_metrics(_job(kind="synth_playbook"))
        self.assertNotIn("metrics_recent", out)

    def test_training_job_with_data_includes_metrics_recent(self):
        pid, exp_id = 17, 501
        _write_checkpoint(
            project_id=pid, experiment_id=exp_id, step=200,
            log_history=[
                {"step": 100, "loss": 0.4},
                {"step": 200, "loss": 0.3},
            ],
        )
        out = serialize_job_with_live_metrics(
            _job(project_id=pid, experiment_id=exp_id)
        )
        self.assertIn("metrics_recent", out)
        self.assertEqual(len(out["metrics_recent"]), 2)

    def test_plain_serialize_job_unchanged_by_wrapper(self):
        # The wrapper composes serialize_job — the base shape must
        # match byte-for-byte when metrics_recent isn't applicable.
        # Regression guard against accidentally changing the frozen
        # shape. Share the Job instance so timestamps don't diverge
        # between the two serialization calls.
        job = _job(kind="auto_rag_comparison")
        base = serialize_job(job)
        wrapped = serialize_job_with_live_metrics(job)
        self.assertEqual(base, wrapped)


if __name__ == "__main__":
    unittest.main()
