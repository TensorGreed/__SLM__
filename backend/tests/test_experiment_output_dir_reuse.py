"""T1 regression: a new run never adopts an output dir it did not write.

With a fresh DB the experiment ids restart at 1 while DATA_DIR may still hold
``projects/<pid>/experiments/<id>`` from an earlier stack. Before the fix, the
new run reused that dir as is: the status reconcile read the old
``training_report.json`` and marked the run COMPLETED at once with the old
losses, and eval / export read the old weights while train.py wrote new ones.

Now ``create_experiment`` moves a non-empty pre-existing dir aside to
``<id>.stale.<stamp>`` (nothing is deleted), records it in the run config
(``_output_dir_archived``), and the run starts in an empty dir. Starting /
restarting an existing run keeps its own dir.
"""

from __future__ import annotations

import json
import os
import unittest

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from _job_wait import wait_for_jobs_idle  # noqa: E402

_CM = TestClient(app)

OLD_WEIGHTS = b"OLD-ADAPTER-WEIGHTS"
OLD_REPORT = {
    "status": "completed",
    "finished_at": "2026-01-01T00:00:00Z",
    "final_train_loss": 0.123,
    "final_eval_loss": 0.456,
}


def setUpModule() -> None:  # noqa: N802
    _CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _CM.__exit__(None, None, None)


class NewRunNeverAdoptsAnOldDir(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (settings.AUTH_ENABLED, settings.TRAINING_BACKEND, settings.ALLOW_SIMULATED_TRAINING)
        settings.AUTH_ENABLED = False
        settings.TRAINING_BACKEND = "simulate"
        settings.ALLOW_SIMULATED_TRAINING = True

    def tearDown(self) -> None:
        wait_for_jobs_idle(_CM)
        settings.AUTH_ENABLED, settings.TRAINING_BACKEND, settings.ALLOW_SIMULATED_TRAINING = self._saved

    def _project_with_split(self, name: str) -> int:
        r = _CM.post("/api/projects", json={"name": name, "description": "t1 regression"})
        self.assertEqual(r.status_code, 201, r.text)
        pid = r.json()["id"]
        prepared = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        rows = "".join(json.dumps({"question": f"q{i}", "answer": f"a{i}"}) + "\n" for i in range(8))
        (prepared / "train.jsonl").write_text(rows)
        (prepared / "val.jsonl").write_text(rows)
        return pid

    def _plant_old_dirs(self, pid: int, ids: range) -> None:
        for eid in ids:
            d = settings.DATA_DIR / "projects" / str(pid) / "experiments" / str(eid)
            (d / "model").mkdir(parents=True, exist_ok=True)
            (d / "model" / "adapter_model.safetensors").write_bytes(OLD_WEIGHTS)
            (d / "training_report.json").write_text(json.dumps(OLD_REPORT))

    def _next_id(self, pid: int) -> int:
        """The id the next experiment row will get: create a probe run via the API and add one.

        Experiment ids are global (SQLite rowid), so this keeps the test independent of how many
        runs earlier tests in the same process created.
        """
        return self._create(pid, "id-probe")["id"] + 1

    def _create(self, pid: int, name: str) -> dict:
        r = _CM.post(
            f"/api/projects/{pid}/training/experiments",
            json={"name": name, "config": {"base_model": "HuggingFaceTB/SmolLM2-135M-Instruct", "num_epochs": 1}},
        )
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def test_old_dir_with_weights_is_moved_aside_and_the_run_does_not_complete_on_it(self) -> None:
        pid = self._project_with_split("t1-old-dir")
        # Plant for the id the DB will hand out next (probe run + 1), right before creating.
        nxt = self._next_id(pid)
        self._plant_old_dirs(pid, range(nxt, nxt + 1))
        exp = self._create(pid, "new-run")
        eid = exp["id"]
        exp_dir = settings.DATA_DIR / "projects" / str(pid) / "experiments" / str(eid)

        # The new dir holds only what this run wrote.
        self.assertEqual(exp["output_dir"], str(exp_dir))
        self.assertEqual(sorted(p.name for p in exp_dir.iterdir()), ["training_config.json"])
        # The old files are kept, unchanged, in a sibling, and the run says where.
        stale = [p for p in exp_dir.parent.iterdir() if p.name.startswith(f"{eid}.stale.")]
        self.assertEqual(len(stale), 1, [p.name for p in exp_dir.parent.iterdir()])
        self.assertEqual((stale[0] / "model" / "adapter_model.safetensors").read_bytes(), OLD_WEIGHTS)
        self.assertEqual(json.loads((stale[0] / "training_report.json").read_text()), OLD_REPORT)
        archived = exp["config"]["_output_dir_archived"]
        self.assertEqual(archived["from"], str(exp_dir))
        self.assertEqual(archived["to"], str(stale[0]))
        self.assertEqual(archived["reason"], "pre_existing_dir")
        saved_config = json.loads((exp_dir / "training_config.json").read_text())
        self.assertEqual(saved_config["_output_dir_archived"]["to"], str(stale[0]))

        # Start it: the status never reports the old run's report or losses.
        r = _CM.post(f"/api/projects/{pid}/training/experiments/{eid}/start")
        self.assertEqual(r.status_code, 200, r.text)
        status = _CM.get(f"/api/projects/{pid}/training/experiments/{eid}/status").json()
        self.assertNotEqual(status.get("final_train_loss"), OLD_REPORT["final_train_loss"])
        self.assertNotEqual(status.get("final_eval_loss"), OLD_REPORT["final_eval_loss"])
        self.assertFalse((exp_dir / "model" / "adapter_model.safetensors").exists()
                         and (exp_dir / "model" / "adapter_model.safetensors").read_bytes() == OLD_WEIGHTS)

    def test_no_old_dir_changes_nothing(self) -> None:
        pid = self._project_with_split("t1-clean")
        exp = self._create(pid, "clean-run")
        exp_dir = settings.DATA_DIR / "projects" / str(pid) / "experiments" / str(exp["id"])
        self.assertNotIn("_output_dir_archived", exp["config"])
        self.assertEqual([p.name for p in exp_dir.parent.iterdir() if ".stale." in p.name], [])

    def test_an_empty_old_dir_is_reused_without_a_move(self) -> None:
        pid = self._project_with_split("t1-empty")
        nxt = self._next_id(pid)
        for eid in range(nxt, nxt + 1):
            (settings.DATA_DIR / "projects" / str(pid) / "experiments" / str(eid)).mkdir(parents=True, exist_ok=True)
        exp = self._create(pid, "empty-dir-run")
        exp_dir = settings.DATA_DIR / "projects" / str(pid) / "experiments" / str(exp["id"])
        self.assertNotIn("_output_dir_archived", exp["config"])
        self.assertEqual([p.name for p in exp_dir.parent.iterdir() if ".stale." in p.name], [])

    def test_starting_an_existing_run_keeps_its_own_dir(self) -> None:
        pid = self._project_with_split("t1-restart")
        exp = self._create(pid, "own-dir-run")
        eid = exp["id"]
        exp_dir = settings.DATA_DIR / "projects" / str(pid) / "experiments" / str(eid)
        marker = exp_dir / "checkpoint-1" / "marker.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("mine")
        r = _CM.post(f"/api/projects/{pid}/training/experiments/{eid}/start")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([p.name for p in exp_dir.parent.iterdir() if ".stale." in p.name], [])
        self.assertEqual(exp_dir.exists(), True)

    def test_a_failed_move_aside_returns_a_structured_error(self) -> None:
        from unittest import mock

        pid = self._project_with_split("t1-move-fails")
        nxt = self._next_id(pid)
        self._plant_old_dirs(pid, range(nxt, nxt + 1))
        with mock.patch("app.services.training_service.shutil.move", side_effect=PermissionError("denied")):
            r = _CM.post(
                f"/api/projects/{pid}/training/experiments",
                json={"name": "blocked", "config": {"base_model": "HuggingFaceTB/SmolLM2-135M-Instruct", "num_epochs": 1}},
            )
        self.assertEqual(r.status_code, 500, r.text)
        body = r.json()
        self.assertEqual(body["error_code"], "OUTPUT_DIR_MOVE_FAILED")
        self.assertIn("could not be moved aside", body["detail"])
        self.assertEqual(body["metadata"]["from"].rsplit("/", 1)[-1], str(nxt))
        # The old files are untouched.
        old = settings.DATA_DIR / "projects" / str(pid) / "experiments" / str(nxt)
        self.assertEqual((old / "model" / "adapter_model.safetensors").read_bytes(), OLD_WEIGHTS)
