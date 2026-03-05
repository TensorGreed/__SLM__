"""Phase 8 tests: export artifacts + held-out end-to-end evaluation."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401
from app.database import Base
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.models.export import ExportFormat, ExportStatus
from app.models.project import Project
from app.services.evaluation_service import run_heldout_evaluation
from app.services.export_service import create_export, run_export


class Phase8ExportAndHeldoutEvalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmp.name)
        db_path = self.tmp_root / "phase8.db"
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", future=True)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def _seed_project_and_experiment(self, db: AsyncSession, output_dir: Path) -> tuple[Project, Experiment]:
        project = Project(
            name=f"phase8-{uuid4().hex[:10]}",
            description="phase8 project",
            base_model_name="microsoft/phi-2",
        )
        db.add(project)
        await db.flush()

        exp = Experiment(
            project_id=project.id,
            name=f"phase8-exp-{uuid4().hex[:8]}",
            description="phase8 experiment",
            status=ExperimentStatus.COMPLETED,
            base_model="microsoft/phi-2",
            output_dir=str(output_dir),
        )
        db.add(exp)
        await db.flush()
        return project, exp

    async def test_export_run_includes_model_artifacts(self):
        model_dir = self.tmp_root / "experiments" / "101" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        (model_dir / "tokenizer.json").write_text('{"version": 1}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"model-bytes")

        async with self.session_factory() as db:
            project, exp = await self._seed_project_and_experiment(db, output_dir=model_dir.parent)
            export = await create_export(db, project.id, exp.id, ExportFormat.HUGGINGFACE)
            export = await run_export(db, project.id, export.id)
            await db.commit()

            self.assertEqual(export.status, ExportStatus.COMPLETED)
            self.assertIsNotNone(export.manifest)
            self.assertGreater(export.file_size_bytes or 0, 0)

            manifest = export.manifest or {}
            model_info = manifest.get("model_artifacts") or {}
            self.assertGreaterEqual(model_info.get("count", 0), 3)
            self.assertEqual(model_info.get("source"), "experiment_model_dir")

            run_dir = Path(manifest["run_dir"])
            self.assertTrue((run_dir / "model" / "config.json").exists())
            self.assertTrue((run_dir / "model" / "tokenizer.json").exists())
            self.assertTrue((run_dir / "model" / "weights.safetensors").exists())

    async def test_run_heldout_evaluation_generates_predictions(self):
        dataset_dir = self.tmp_root / "prepared"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        heldout_file = dataset_dir / "test.jsonl"
        rows = [
            {"question": "What is 2+2?", "answer": "4"},
            {"prompt": "Capital of France?", "reference": "Paris"},
        ]
        with open(heldout_file, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        model_dir = self.tmp_root / "experiments" / "102" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")

        async with self.session_factory() as db:
            project, exp = await self._seed_project_and_experiment(db, output_dir=model_dir.parent)
            dataset = Dataset(
                project_id=project.id,
                name="Heldout Test",
                dataset_type=DatasetType.TEST,
                file_path=str(heldout_file),
                record_count=2,
            )
            db.add(dataset)
            await db.flush()

            mock_predictions = [
                {"prompt": "What is 2+2?", "reference": "4", "prediction": "4", "latency_ms": 5.0},
                {
                    "prompt": "Capital of France?",
                    "reference": "Paris",
                    "prediction": "Paris",
                    "latency_ms": 7.5,
                },
            ]
            mock_runtime = {
                "engine": "transformers",
                "device": "cpu",
                "average_latency_ms": 6.25,
                "token_throughput_tps": 120.0,
            }

            with patch(
                "app.services.evaluation_service._run_local_inference",
                return_value=(mock_predictions, mock_runtime),
            ):
                result = await run_heldout_evaluation(
                    db=db,
                    project_id=project.id,
                    experiment_id=exp.id,
                    dataset_name="test",
                    eval_type="exact_match",
                    max_samples=2,
                    max_new_tokens=32,
                )
                await db.commit()

            self.assertEqual(result.eval_type, "exact_match")
            self.assertAlmostEqual(float(result.metrics["exact_match"]), 1.0)
            self.assertEqual(result.metrics["evaluated_samples"], 2)
            self.assertEqual(result.details["dataset"]["dataset_type"], DatasetType.TEST.value)
            self.assertEqual(result.details["inference"]["samples"], 2)


if __name__ == "__main__":
    unittest.main()
