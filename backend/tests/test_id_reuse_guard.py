"""T3 regression: a new row never inherits a deleted row's files through a reused id.

SQLite hands the highest id out again after a delete. Before the fix:
  (a) deleting the newest project and creating a new one gave the new project the old
      ``DATA_DIR/projects/<pid>/`` (prepared split + manifest): ``delete_project`` never touched the dir;
  (b) deleting the newest run and creating the next one let a GGUF export package the deleted run's
      ``compressed/exp-<id>/quantized_*.gguf``.

Now (``app/services/id_reuse_guard.py``): deleting a project / run moves its id-keyed files aside
(``<name>.deleted.<stamp>``), a NEW project / run / export row moves any id-keyed files it finds aside
(``<name>.stale.<stamp>``, SQLAlchemy ``after_insert``), and a GGUF / ONNX export moves aside any converted
file written before the run's row existed. Nothing is erased.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from _job_wait import wait_for_jobs_idle  # noqa: E402

_CM = TestClient(app)

OLD_ROWS = "".join(json.dumps({"question": f"old q{i}", "answer": f"old a{i}"}) + "\n" for i in range(5))
OLD_MANIFEST = {"task_type": "qa", "source": "deleted project"}
STALE_GGUF = b"GGUF-STALE-FROM-A-DELETED-RUN" * 256


def setUpModule() -> None:  # noqa: N802
    _CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _CM.__exit__(None, None, None)


def _siblings(path: Path, marker: str) -> list[Path]:
    if not path.parent.is_dir():
        return []
    return [p for p in path.parent.iterdir() if p.name.startswith(f"{path.name}.{marker}.")]


class _ApiBase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (settings.AUTH_ENABLED, settings.TRAINING_BACKEND, settings.ALLOW_SIMULATED_TRAINING)
        settings.AUTH_ENABLED = False
        settings.TRAINING_BACKEND = "simulate"
        settings.ALLOW_SIMULATED_TRAINING = True

    def tearDown(self) -> None:
        wait_for_jobs_idle(_CM)
        settings.AUTH_ENABLED, settings.TRAINING_BACKEND, settings.ALLOW_SIMULATED_TRAINING = self._saved

    def _project(self, name: str) -> int:
        r = _CM.post("/api/projects", json={"name": f"{name}-{uuid4().hex[:6]}", "description": "t3"})
        self.assertEqual(r.status_code, 201, r.text)
        return int(r.json()["id"])

    def _plant_prepared(self, pid: int) -> Path:
        prepared = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "train.jsonl").write_text(OLD_ROWS)
        (prepared / "test.jsonl").write_text(OLD_ROWS)
        (prepared / "manifest.json").write_text(json.dumps(OLD_MANIFEST))
        return prepared

    def _create_run(self, pid: int, name: str) -> dict:
        r = _CM.post(
            f"/api/projects/{pid}/training/experiments",
            json={"name": name, "config": {"base_model": "HuggingFaceTB/SmolLM2-135M-Instruct", "num_epochs": 1}},
        )
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()


class ProjectIdReuse(_ApiBase):
    def test_deleting_the_newest_project_does_not_hand_its_split_to_the_next_one(self) -> None:
        pid = self._project("t3-old")
        self._plant_prepared(pid)
        exports = settings.DATA_DIR / "exports" / str(pid) / "999"
        exports.mkdir(parents=True, exist_ok=True)
        (exports / "LATEST_RUN").write_text("old")
        project_dir = settings.DATA_DIR / "projects" / str(pid)

        r = _CM.delete(f"/api/projects/{pid}")
        self.assertEqual(r.status_code, 204, r.text)
        # Moved aside, not erased.
        self.assertFalse(project_dir.exists())
        deleted = _siblings(project_dir, "deleted")
        self.assertEqual(len(deleted), 1)
        self.assertEqual((deleted[0] / "prepared" / "train.jsonl").read_text(), OLD_ROWS)
        self.assertEqual(len(_siblings(settings.DATA_DIR / "exports" / str(pid), "deleted")), 1)

        new_pid = self._project("t3-new")
        self.assertEqual(new_pid, pid, "SQLite should reuse the deleted project's id (the hazard under test)")
        self.assertFalse((project_dir / "prepared" / "train.jsonl").exists())
        self.assertFalse((settings.DATA_DIR / "exports" / str(pid) / "999").exists())
        manifest = _CM.get(f"/api/projects/{new_pid}/prepared-manifest")
        if manifest.status_code == 200:
            self.assertNotEqual(manifest.json(), OLD_MANIFEST)

    def test_a_new_project_moves_a_leftover_dir_aside_instead_of_adopting_it(self) -> None:
        # e.g. a fresh DB over an old DATA_DIR, or a delete from before this fix: the dir is already there.
        probe = self._project("t3-probe")
        self.assertEqual(_CM.delete(f"/api/projects/{probe}").status_code, 204)
        project_dir = settings.DATA_DIR / "projects" / str(probe)
        self._plant_prepared(probe)

        new_pid = self._project("t3-adopter")
        self.assertEqual(new_pid, probe)
        self.assertFalse((project_dir / "prepared" / "train.jsonl").exists())
        stale = _siblings(project_dir, "stale")
        self.assertEqual(len(stale), 1)
        self.assertEqual(json.loads((stale[0] / "prepared" / "manifest.json").read_text()), OLD_MANIFEST)


    def test_a_failed_move_fails_the_create_with_the_envelope_and_erases_nothing(self) -> None:
        probe = self._project("t3-probe-fail")
        self.assertEqual(_CM.delete(f"/api/projects/{probe}").status_code, 204)
        project_dir = settings.DATA_DIR / "projects" / str(probe)
        self._plant_prepared(probe)
        name = f"t3-blocked-{uuid4().hex[:6]}"

        with mock.patch("app.services.id_reuse_guard.shutil.move", side_effect=PermissionError("read-only")):
            r = _CM.post("/api/projects", json={"name": name, "description": "t3"})
        self.assertEqual(r.status_code, 500, r.text)
        body = r.json()
        self.assertEqual(body.get("error_code"), "STALE_DIR_MOVE_FAILED", body)
        self.assertIn("Nothing was deleted", json.dumps(body))
        self.assertEqual((project_dir / "prepared" / "train.jsonl").read_text(), OLD_ROWS)
        self.assertEqual(_siblings(project_dir, "stale"), [])
        listed = _CM.get("/api/projects").json()
        rows = listed if isinstance(listed, list) else listed.get("projects", listed.get("items", []))
        self.assertNotIn(name, [p.get("name") for p in rows])

        # Once the folder can move, the same id is created clean.
        new_pid = self._project("t3-unblocked")
        self.assertEqual(new_pid, probe)
        self.assertFalse((project_dir / "prepared").exists())
        self.assertEqual(len(_siblings(project_dir, "stale")), 1)


class RunIdReuseThroughApi(_ApiBase):
    def _plant_gguf(self, pid: int, eid: int) -> Path:
        d = settings.DATA_DIR / "projects" / str(pid) / "compressed" / f"exp-{eid}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "quantized_4bit.gguf").write_bytes(STALE_GGUF)
        return d

    def test_deleting_a_run_moves_its_converted_files_and_a_reused_id_starts_clean(self) -> None:
        pid = self._project("t3-run")
        self._plant_prepared(pid)
        old = self._create_run(pid, "old-run")
        eid = int(old["id"])
        compressed = self._plant_gguf(pid, eid)
        reader = settings.DATA_DIR / "projects" / str(pid) / "auto_rag" / "passage_reader" / f"{eid}.json"
        reader.parent.mkdir(parents=True, exist_ok=True)
        reader.write_text(json.dumps({"summary": {"old": True}}))

        r = _CM.delete(f"/api/projects/{pid}/training/experiments/{eid}")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(compressed.exists())
        self.assertFalse(reader.exists())
        self.assertEqual(len(_siblings(compressed, "deleted")), 1)
        self.assertEqual(len(r.json()["side_artifacts_archived"]), 2)

        # A dir left behind anyway (older delete): the new run with the same id moves it aside and says so.
        self._plant_gguf(pid, eid)
        new = self._create_run(pid, "new-run")
        self.assertEqual(int(new["id"]), eid, "SQLite should reuse the deleted run's id (the hazard under test)")
        self.assertFalse((compressed / "quantized_4bit.gguf").exists())
        self.assertEqual(len(_siblings(compressed, "stale")), 1)
        archived = new["config"]["_stale_artifacts_archived"]
        self.assertEqual([a["from"] for a in archived], [str(compressed)])


class GgufExportProvenance(unittest.TestCase):
    """The export itself never packages a converted file written before the run's row existed (a run created
    before the create-time guard may already sit on a deleted run's ``compressed/exp-<id>/``)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._patches = [
            mock.patch.object(settings, "DATA_DIR", self.root / "data"),
            mock.patch.object(settings, "COMPRESSION_BACKEND", "stub"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def _run(self, plant_after_create: bool, backdate: bool, move_fails: bool = False, plant_lock: bool = False):
        from app.database import Base
        from app.models.experiment import Experiment, ExperimentStatus
        from app.models.export import ExportFormat
        from app.models.project import Project
        from app.services import compression_service
        from app.services.export_service import create_export, run_export

        async def go():
            engine = create_async_engine(f"sqlite+aiosqlite:///{self.root / 'x.db'}", future=True)
            sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            try:
                async with sf() as db:
                    project = Project(name=f"t3-{uuid4().hex[:6]}", description="", base_model_name="toy")
                    db.add(project)
                    await db.flush()
                    out = self.root / "run"
                    (out / "model").mkdir(parents=True, exist_ok=True)
                    (out / "model" / "config.json").write_text("{}")
                    exp = Experiment(project_id=project.id, name="r", status=ExperimentStatus.COMPLETED,
                                     base_model="toy", output_dir=str(out), config={})
                    db.add(exp)
                    await db.flush()
                    gguf = compression_service.experiment_compression_dir(project.id, exp.id) / "quantized_4bit.gguf"
                    if plant_after_create:
                        gguf.write_bytes(STALE_GGUF)
                        compression_service.conversion_marker(gguf).write_text(json.dumps({"size": len(STALE_GGUF)}))
                        if plant_lock:
                            (gguf.parent / compression_service.CONVERT_LOCK_NAME).write_text("")
                        if backdate:
                            for p in gguf.parent.iterdir():
                                os.utime(p, (1_600_000_000, 1_600_000_000))
                    export = await create_export(db, project.id, exp.id, ExportFormat.GGUF, "4-bit")
                    if move_fails:
                        with mock.patch("app.services.id_reuse_guard.shutil.move", side_effect=PermissionError("ro")):
                            export = await run_export(db, project.id, export.id, run_smoke_tests=False)
                    else:
                        export = await run_export(db, project.id, export.id, run_smoke_tests=False)
                    await db.commit()
                    return export, gguf
            finally:
                await engine.dispose()

        return asyncio.run(go())

    def test_a_gguf_older_than_the_run_is_moved_aside_not_packaged(self) -> None:
        from app.models.export import ExportStatus

        export, gguf = self._run(plant_after_create=True, backdate=True)
        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.FAILED, manifest)
        self.assertIn("Open Compression", manifest.get("error", ""))
        self.assertFalse(gguf.exists())
        stale = _siblings(gguf.parent, "stale")
        self.assertEqual(len(stale), 1)
        self.assertEqual((stale[0] / "quantized_4bit.gguf").read_bytes(), STALE_GGUF)
        # Fix 1: the failed export says why the GGUF in the folder was not used, and where it went.
        moved = manifest.get("stale_artifacts_moved") or {}
        self.assertEqual(moved.get("to"), str(stale[0]))
        self.assertIn("quantized_4bit.gguf", moved.get("files") or [])
        self.assertIn(f"moved to {stale[0].name}/", manifest["error"])
        self.assertIn("earlier run with this id", manifest["error"])

    def test_the_conversion_lock_file_is_never_moved(self) -> None:
        from app.services.compression_service import CONVERT_LOCK_NAME

        export, gguf = self._run(plant_after_create=True, backdate=True, plant_lock=True)
        self.assertTrue((gguf.parent / CONVERT_LOCK_NAME).is_file())
        stale = _siblings(gguf.parent, "stale")
        self.assertEqual(len(stale), 1)
        self.assertFalse((stale[0] / CONVERT_LOCK_NAME).exists())
        self.assertNotIn(CONVERT_LOCK_NAME, (export.manifest or {}).get("stale_artifacts_moved", {}).get("files", []))

    def test_a_failed_move_is_a_clean_message_and_erases_nothing(self) -> None:
        from app.models.export import ExportStatus

        export, gguf = self._run(plant_after_create=True, backdate=True, move_fails=True)
        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.FAILED)
        error = manifest.get("error", "")
        self.assertFalse(error.startswith("500"), error)
        self.assertNotIn("error_code", error)
        self.assertIn("could not be moved aside", error)
        self.assertIn("Nothing was deleted", error)
        self.assertEqual(gguf.read_bytes(), STALE_GGUF)

    def test_the_check_waits_for_the_conversion_lock(self) -> None:
        import fcntl

        from app.services import compression_service
        from app.services.export_service import _archive_compression_files_predating_run

        d = settings.DATA_DIR / "projects" / "1" / "compressed" / "exp-1"
        d.mkdir(parents=True)
        (d / "quantized_4bit.gguf").write_bytes(b"x")
        os.utime(d / "quantized_4bit.gguf", (1_600_000_000, 1_600_000_000))
        exp = mock.Mock(id=1, created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
        fd = os.open(d / compression_service.CONVERT_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # a conversion of this run is in progress
            with self.assertRaises(asyncio.TimeoutError):
                asyncio.run(asyncio.wait_for(_archive_compression_files_predating_run(1, exp), 0.3))
            self.assertTrue((d / "quantized_4bit.gguf").exists(), "nothing moves while a conversion holds the lock")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        rec = asyncio.run(_archive_compression_files_predating_run(1, exp))
        self.assertEqual(rec["files"], ["quantized_4bit.gguf"])
        # No compressed folder: no lock taken, no folder created.
        self.assertIsNone(asyncio.run(_archive_compression_files_predating_run(2, exp)))
        self.assertFalse((settings.DATA_DIR / "projects" / "2").exists())

    def test_the_runs_own_conversion_still_ships(self) -> None:
        from app.models.export import ExportStatus

        export, gguf = self._run(plant_after_create=True, backdate=False)
        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.COMPLETED, manifest.get("error"))
        self.assertIsNone(manifest.get("stale_artifacts_moved"))
        run_dir = Path(export.output_path) / manifest["run_dir"]
        self.assertTrue((run_dir / "model" / "quantized_4bit.gguf").is_file())
        self.assertTrue(gguf.exists())


class _IsolatedDb(unittest.TestCase):
    """A private DB + DATA_DIR per test (no shared engine, so rollbacks and id reuse are deterministic)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._patch = mock.patch.object(settings, "DATA_DIR", self.root / "data")
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def _with_db(self, body):
        from app.database import Base

        async def go():
            engine = create_async_engine(f"sqlite+aiosqlite:///{self.root / 'x.db'}", future=True)
            sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            try:
                async with sf() as db:
                    return await body(db)
            finally:
                await engine.dispose()

        return asyncio.run(go())


class HooksAndRollback(_IsolatedDb):
    def test_a_new_export_row_moves_a_leftover_export_dir_aside(self) -> None:
        from app.models.experiment import Experiment, ExperimentStatus
        from app.models.export import Export, ExportFormat

        async def body(db):
            from app.models.project import Project

            project = Project(name="p", description="", base_model_name="toy")
            db.add(project)
            await db.flush()
            exp = Experiment(project_id=project.id, name="r", status=ExperimentStatus.COMPLETED, base_model="toy", config={})
            db.add(exp)
            await db.flush()
            leftover = settings.DATA_DIR / "exports" / str(project.id) / "1"
            leftover.mkdir(parents=True)
            (leftover / "LATEST_RUN").write_text("run-from-a-deleted-export")
            row = Export(project_id=project.id, experiment_id=exp.id, export_format=ExportFormat.GGUF)
            db.add(row)
            await db.flush()
            return row.id, leftover

        export_id, leftover = self._with_db(body)
        self.assertEqual(export_id, 1)
        self.assertFalse((leftover / "LATEST_RUN").exists())
        stale = _siblings(leftover, "stale")
        self.assertEqual(len(stale), 1)
        self.assertEqual((stale[0] / "LATEST_RUN").read_text(), "run-from-a-deleted-export")

    def test_a_rollback_after_the_hook_moved_a_folder_loses_nothing(self) -> None:
        from sqlalchemy import func, select

        from app.models.project import Project

        project_dir = settings.DATA_DIR / "projects" / "1"
        (project_dir / "prepared").mkdir(parents=True)
        (project_dir / "prepared" / "train.jsonl").write_text(OLD_ROWS)

        async def body(db):
            db.add(Project(name="p", description="", base_model_name="toy"))
            await db.flush()  # the hook moves projects/1/ aside here
            await db.rollback()
            count = (await db.execute(select(func.count(Project.id)))).scalar_one()
            again = Project(name="p2", description="", base_model_name="toy")
            db.add(again)
            await db.flush()
            await db.commit()
            return count, again.id

        count, new_id = self._with_db(body)
        self.assertEqual(count, 0, "the rolled-back row is absent")
        self.assertEqual(new_id, 1)
        stale = _siblings(project_dir, "stale")
        self.assertEqual(len(stale), 1, "renamed once, and the next create found nothing to move")
        self.assertEqual((stale[0] / "prepared" / "train.jsonl").read_text(), OLD_ROWS)

    def test_delete_experiment_leaves_side_artifacts_until_the_caller_commits(self) -> None:
        from sqlalchemy import select

        from app.models.experiment import Experiment, ExperimentStatus
        from app.models.project import Project
        from app.services.experiment_recovery_service import delete_experiment

        async def body(db):
            project = Project(name="p", description="", base_model_name="toy")
            db.add(project)
            await db.flush()
            exp = Experiment(project_id=project.id, name="r", status=ExperimentStatus.COMPLETED, base_model="toy", config={})
            db.add(exp)
            await db.flush()
            await db.commit()
            pid, eid = int(project.id), int(exp.id)
            gguf = settings.DATA_DIR / "projects" / str(pid) / "compressed" / f"exp-{eid}" / "quantized_4bit.gguf"
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(STALE_GGUF)
            report = await delete_experiment(db, project_id=pid, experiment_id=eid, purge_output_dir=False)
            await db.rollback()  # the commit failed / the request was cancelled
            still = (await db.execute(select(Experiment.id).where(Experiment.id == eid))).scalar_one_or_none()
            return report, gguf, still

        report, gguf, still = self._with_db(body)
        self.assertIsNotNone(still, "the row survives the rollback")
        self.assertNotIn("side_artifacts_archived", report)
        self.assertEqual(gguf.read_bytes(), STALE_GGUF, "and its converted file stays where it reads it")
        self.assertEqual(_siblings(gguf.parent, "deleted"), [])

    def test_a_reused_run_id_does_not_show_the_deleted_runs_events(self) -> None:
        from sqlalchemy import func, select

        from app.models.experiment import Experiment, ExperimentStatus
        from app.models.project import Project
        from app.models.run_event import RunEvent
        from app.services.run_event_service import emit_event, list_run_events, list_run_events_for_run
        from app.services.timeline_service import build_timeline

        async def body(db):
            project = Project(name="p", description="", base_model_name="toy")
            db.add(project)
            await db.flush()
            old = Experiment(project_id=project.id, name="old", status=ExperimentStatus.COMPLETED, base_model="toy", config={})
            db.add(old)
            await db.flush()
            eid = old.id
            await emit_event(db, project_id=project.id, run_id=f"exp-{eid}", stage="training", summary="old run",
                             ts="2020-01-01T00:00:00Z")
            await db.delete(old)
            await db.flush()
            new = Experiment(project_id=project.id, name="new", status=ExperimentStatus.PENDING, base_model="toy", config={})
            db.add(new)
            await db.flush()
            await emit_event(db, project_id=project.id, run_id=f"exp-{new.id}", stage="training", summary="new run")
            await db.commit()
            per_run = await list_run_events_for_run(db, run_id=f"exp-{eid}")
            listed = await list_run_events(db, project_id=project.id, run_id=f"exp-{eid}")
            timeline = await build_timeline(db, project_id=project.id)
            total = (await db.execute(select(func.count(RunEvent.id)))).scalar_one()
            return new.id, eid, per_run, listed, timeline, total

        new_id, eid, per_run, listed, timeline, total = self._with_db(body)
        self.assertEqual(new_id, eid, "SQLite should reuse the run id (the hazard under test)")
        self.assertEqual([e["summary"] for e in per_run["events"]], ["new run"])
        self.assertEqual([e["summary"] for e in listed["events"]], ["new run"])
        self.assertEqual(timeline["total_events"], 1)
        self.assertEqual(total, 2, "the audit log keeps the deleted run's event (not purged)")

    def test_project_wide_scans_skip_archived_folders(self) -> None:
        from app.api.pipeline import _ensure_stage_requirements
        from app.models.project import PipelineStage, Project
        from app.services.export_service import _discover_optimizer_candidates
        from app.services.workflow_graph_service import collect_available_artifacts
        from fastapi import HTTPException

        async def body(db):
            project = Project(name="p", description="", base_model_name="toy")
            db.add(project)
            await db.flush()
            root = settings.DATA_DIR / "projects" / str(project.id)
            for tag in ("stale", "deleted", "bak"):
                d = root / "compressed" / f"exp-3.{tag}.20260101T000000Z"
                d.mkdir(parents=True)
                (d / "quantize_report.json").write_text("{}")
                (d / "quantized_4bit.gguf").write_bytes(b"x")
                m = root / "experiments" / f"3.{tag}.20260101T000000Z" / "model"
                m.mkdir(parents=True)
                (m / "config.json").write_text("{}")
                (m / "model.safetensors").write_bytes(b"x")
            artifacts = await collect_available_artifacts(db, project.id)
            project.pipeline_stage = PipelineStage.COMPRESSION
            gate_error = None
            try:
                await _ensure_stage_requirements(db, project)
            except HTTPException as exc:
                gate_error = exc
            return artifacts, gate_error, _discover_optimizer_candidates(project.id)

        artifacts, gate_error, candidates = self._with_db(body)
        self.assertNotIn("report.compression", artifacts)
        self.assertNotIn("model.compressed", artifacts)
        self.assertIsNotNone(gate_error, "an archived report does not satisfy the Compression gate")
        self.assertEqual(candidates, [])


class MoveAsideUnit(unittest.TestCase):
    def test_archived_names(self) -> None:
        from app.services.id_reuse_guard import in_archived_folder, is_archived_name

        for name in ("exp-3.stale.20260101T000000Z", "7.deleted.20260101T000000Z-1", "12.bak.20260101T000000Z"):
            self.assertTrue(is_archived_name(name), name)
        for name in ("exp-3", "7", "quantize_report.json", "model.safetensors"):
            self.assertFalse(is_archived_name(name), name)
        root = Path("/r")
        self.assertTrue(in_archived_folder(root / "exp-3.stale.x" / "a.gguf", root))
        self.assertFalse(in_archived_folder(root / "exp-3" / "a.stale.gguf", root))

    def test_empty_dir_is_left_and_names_never_parse_as_ids(self) -> None:
        from app.services.compression_service import experiment_id_for_model_path
        from app.services.id_reuse_guard import move_aside

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(settings, "DATA_DIR", Path(tmp)):
            empty = Path(tmp) / "projects" / "3" / "compressed" / "exp-7"
            empty.mkdir(parents=True)
            self.assertIsNone(move_aside(empty, reason="x"))
            self.assertTrue(empty.is_dir())
            (empty / "f.gguf").write_bytes(b"x")
            rec = move_aside(empty, reason="x")
            self.assertTrue(rec and rec["to"])
            self.assertIsNone(experiment_id_for_model_path(3, str(Path(rec["to"]) / "f.gguf")))
            self.assertIsNone(move_aside(empty, reason="x"))  # absent now


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
