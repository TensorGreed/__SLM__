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
from app.services.deployment_target_service import (
    default_deployment_targets_for_format,
    run_deployment_target_suite,
)
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

    async def test_export_run_attaches_deployment_validation_report(self):
        model_dir = self.tmp_root / "experiments" / "103" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        (model_dir / "tokenizer.json").write_text('{"version": 1}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"model-bytes")

        async with self.session_factory() as db:
            project, exp = await self._seed_project_and_experiment(db, output_dir=model_dir.parent)
            export = await create_export(db, project.id, exp.id, ExportFormat.HUGGINGFACE)
            export = await run_export(
                db,
                project.id,
                export.id,
                deployment_targets=["exporter.huggingface"],
                run_smoke_tests=False,
            )
            await db.commit()

            self.assertEqual(export.status, ExportStatus.COMPLETED)
            manifest = export.manifest or {}
            deployment = manifest.get("deployment") or {}
            summary = deployment.get("summary") or {}
            self.assertTrue(bool(summary.get("deployable_artifact")))
            reports = [item for item in deployment.get("target_reports", []) if isinstance(item, dict)]
            exporter_report = next((item for item in reports if item.get("target_id") == "exporter.huggingface"), None)
            self.assertIsNotNone(exporter_report)
            self.assertTrue(bool(exporter_report.get("passed")))

    async def test_export_run_fails_when_format_artifacts_are_missing(self):
        model_dir = self.tmp_root / "experiments" / "104" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        # Intentionally only HF files; no .gguf artifact.
        (model_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"model-bytes")

        async with self.session_factory() as db:
            project, exp = await self._seed_project_and_experiment(db, output_dir=model_dir.parent)
            export = await create_export(db, project.id, exp.id, ExportFormat.GGUF)
            export = await run_export(
                db,
                project.id,
                export.id,
                deployment_targets=["exporter.gguf"],
                run_smoke_tests=False,
            )
            await db.commit()

            self.assertEqual(export.status, ExportStatus.FAILED)
            manifest = export.manifest or {}
            self.assertIn("Deployment validation failed", str(manifest.get("error", "")))
            self.assertIsInstance(manifest.get("deployment"), dict)

    def test_deployment_target_defaults_for_common_formats(self):
        hf = default_deployment_targets_for_format(ExportFormat.HUGGINGFACE)
        gguf = default_deployment_targets_for_format(ExportFormat.GGUF)
        onnx = default_deployment_targets_for_format(ExportFormat.ONNX)

        self.assertIn("exporter.huggingface", hf)
        self.assertIn("runner.vllm", hf)
        self.assertIn("runner.tgi", hf)

        self.assertIn("exporter.gguf", gguf)
        self.assertIn("runner.ollama", gguf)

        self.assertIn("exporter.onnx", onnx)

    def test_deployment_suite_runner_checks_do_not_block_exporter_pass(self):
        run_dir = self.tmp_root / "exports" / "run-hf"
        model_dir = run_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
        (model_dir / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
        (model_dir / "weights.safetensors").write_bytes(b"model-bytes")

        report = run_deployment_target_suite(
            run_dir=run_dir,
            export_format=ExportFormat.HUGGINGFACE,
            deployment_targets=None,
            run_smoke_tests=False,
        )

        summary = report.get("summary") or {}
        self.assertTrue(bool(summary.get("deployable_artifact")))
        reports = [item for item in report.get("target_reports", []) if isinstance(item, dict)]
        exporter = next((item for item in reports if item.get("target_id") == "exporter.huggingface"), None)
        self.assertIsNotNone(exporter)
        self.assertTrue(bool((exporter or {}).get("passed")))

        runner_reports = [item for item in reports if str(item.get("kind")) == "runner"]
        self.assertGreaterEqual(len(runner_reports), 1)
        self.assertTrue(all(not bool(item.get("smoke_executed")) for item in runner_reports))

    def test_deployment_suite_supports_tensorrt_export_profile(self):
        run_dir = self.tmp_root / "exports" / "run-trt"
        model_dir = run_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.engine").write_bytes(b"0" * 4096)

        report = run_deployment_target_suite(
            run_dir=run_dir,
            export_format=ExportFormat.TENSORRT,
            deployment_targets=["exporter.tensorrt"],
            run_smoke_tests=False,
        )
        summary = report.get("summary") or {}
        self.assertTrue(bool(summary.get("deployable_artifact")))

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

    async def test_run_heldout_evaluation_supports_multimodal_rows(self):
        dataset_dir = self.tmp_root / "prepared"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        heldout_file = dataset_dir / "multimodal_test.jsonl"
        rows = [
            {"image_path": "assets/cat.png", "caption": "A cat sitting on a chair."},
            {"audio_path": "assets/hello.wav", "transcript": "hello world"},
        ]
        with open(heldout_file, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        model_dir = self.tmp_root / "experiments" / "105" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")

        captured_pairs: dict[str, list[dict]] = {"rows": []}

        def _mock_local_inference(_model_ref, pairs, _max_new_tokens, _temperature):
            captured_pairs["rows"] = [dict(item) for item in pairs]
            predictions = [
                {
                    "prompt": str(item.get("prompt") or ""),
                    "reference": str(item.get("reference") or ""),
                    "prediction": str(item.get("reference") or ""),
                    "latency_ms": 5.0,
                }
                for item in pairs
            ]
            runtime = {
                "engine": "transformers",
                "device": "cpu",
                "average_latency_ms": 5.0,
                "token_throughput_tps": 100.0,
            }
            return predictions, runtime

        async with self.session_factory() as db:
            project, exp = await self._seed_project_and_experiment(db, output_dir=model_dir.parent)
            dataset = Dataset(
                project_id=project.id,
                name="Heldout Multimodal Test",
                dataset_type=DatasetType.TEST,
                file_path=str(heldout_file),
                record_count=2,
            )
            db.add(dataset)
            await db.flush()

            with patch(
                "app.services.evaluation_service._run_local_inference",
                side_effect=_mock_local_inference,
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

            pairs = list(captured_pairs.get("rows") or [])
            self.assertEqual(len(pairs), 2)
            self.assertTrue(str(pairs[0].get("prompt") or "").startswith("<image:assets/cat.png>"))
            self.assertTrue(str(pairs[1].get("prompt") or "").startswith("<audio:assets/hello.wav>"))
            self.assertEqual(str(pairs[0].get("input_modality") or ""), "vision_language")
            self.assertEqual(str(pairs[1].get("input_modality") or ""), "audio_text")

            modality_breakdown = dict(result.metrics.get("modality_breakdown") or {})
            self.assertEqual(int(modality_breakdown.get("vision_language") or 0), 1)
            self.assertEqual(int(modality_breakdown.get("audio_text") or 0), 1)
            self.assertEqual(result.details["dataset"]["modality_breakdown"], modality_breakdown)
            self.assertIn("vision_language", list(result.details["dataset"].get("modalities_observed") or []))
            self.assertIn("audio_text", list(result.details["dataset"].get("modalities_observed") or []))


if __name__ == "__main__":
    unittest.main()
