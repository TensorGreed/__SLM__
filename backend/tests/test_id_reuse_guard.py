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

    def _run(self, plant_after_create: bool, backdate: bool):
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
                        if backdate:
                            for p in (gguf, compression_service.conversion_marker(gguf)):
                                os.utime(p, (1_600_000_000, 1_600_000_000))
                    export = await create_export(db, project.id, exp.id, ExportFormat.GGUF, "4-bit")
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

    def test_the_runs_own_conversion_still_ships(self) -> None:
        from app.models.export import ExportStatus

        export, gguf = self._run(plant_after_create=True, backdate=False)
        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.COMPLETED, manifest.get("error"))
        self.assertIsNone(manifest.get("stale_artifacts_moved"))
        run_dir = Path(export.output_path) / manifest["run_dir"]
        self.assertTrue((run_dir / "model" / "quantized_4bit.gguf").is_file())
        self.assertTrue(gguf.exists())


class MoveAsideUnit(unittest.TestCase):
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
