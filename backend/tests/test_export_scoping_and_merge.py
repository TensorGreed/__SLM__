"""Export correctness (Wave 1b).

Pins:
  * GGUF/ONNX export only packages the exported run's own compressed
    artifact (``compressed/exp-<id>/``) — never another run's file from the
    project-wide ``compressed/`` root, and never the raw HF weights
    relabelled as GGUF/ONNX;
  * compression writes outputs made from a run's weights under that run;
  * a LoRA run exports as a merged, standalone model (real SmolLM2 adapter;
    skipped without torch/peft or the cached model).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["DEBUG"] = "false"

import app.models  # noqa: F401
from app.config import settings
from app.database import Base
from app.models.experiment import Experiment, ExperimentStatus
from app.models.export import ExportFormat, ExportStatus
from app.models.project import Project
from app.services import compression_service
from app.services.adapter_merge_service import adapter_base_model, read_adapter_config
from app.services.export_service import (
    _resolve_source_model_files,
    create_export,
    run_export,
)

REAL_MODEL = os.environ.get("BREWSLM_REAL_TRAIN_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")


class _Exp:
    def __init__(self, exp_id: int, output_dir: Path):
        self.id = exp_id
        self.output_dir = str(output_dir)


class CompressedScopingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self._patch = mock.patch.object(settings, "DATA_DIR", self.data_dir)
        self._patch.start()
        self.project_id = 4242
        self.project_dir = self.data_dir / "projects" / str(self.project_id)
        for exp_id in (1, 2):
            model_dir = self.project_dir / "experiments" / str(exp_id) / "model"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}")
            (model_dir / "model.safetensors").write_bytes(b"w")

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _exp(self, exp_id: int) -> _Exp:
        return _Exp(exp_id, self.project_dir / "experiments" / str(exp_id))

    def _gguf(self, relative: str) -> Path:
        path = self.project_dir / "compressed" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")
        return path

    def test_gguf_export_uses_only_this_runs_artifact(self):
        self._gguf("quantized_4bit.gguf")  # legacy, project-wide
        other = self._gguf("exp-2/quantized_4bit.gguf")
        mine = self._gguf("exp-1/quantized_4bit.gguf")
        source, files, _ = _resolve_source_model_files(
            self.project_id, self._exp(1), ExportFormat.GGUF, "4-bit"
        )
        self.assertEqual(source, "compressed")
        self.assertEqual(files, [mine])
        self.assertNotIn(other, files)

    def test_gguf_export_never_falls_back_to_raw_weights(self):
        self._gguf("quantized_4bit.gguf")
        self._gguf("exp-2/quantized_4bit.gguf")
        source, files, root = _resolve_source_model_files(
            self.project_id, self._exp(1), ExportFormat.GGUF, "4-bit"
        )
        self.assertEqual((source, files, root), ("none", [], None))

    def test_huggingface_export_ignores_compressed_files(self):
        self._gguf("exp-1/quantized_4bit.gguf")
        source, _, root = _resolve_source_model_files(
            self.project_id, self._exp(1), ExportFormat.HUGGINGFACE, "4-bit"
        )
        self.assertEqual(source, "experiment_model_dir")
        self.assertEqual(root, self.project_dir / "experiments" / "1" / "model")

    def test_compression_output_dir_is_scoped_to_source_run(self):
        run_model = self.project_dir / "experiments" / "7" / "model"
        run_model.mkdir(parents=True)
        self.assertEqual(
            compression_service.experiment_id_for_model_path(self.project_id, str(run_model)), 7
        )
        self.assertEqual(
            compression_service._output_dir_for_source(self.project_id, str(run_model)),
            self.project_dir / "compressed" / "exp-7",
        )
        self.assertIsNone(
            compression_service.experiment_id_for_model_path(self.project_id, "Qwen/Qwen2.5-1.5B")
        )
        self.assertEqual(
            compression_service._output_dir_for_source(self.project_id, "/elsewhere/model"),
            self.project_dir / "compressed",
        )


class AdapterDetectionTests(unittest.TestCase):
    def test_adapter_config_and_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_adapter_config(tmp))
            Path(tmp, "adapter_config.json").write_text('{"base_model_name_or_path": "org/base"}')
            self.assertEqual(adapter_base_model(tmp), "org/base")
            Path(tmp, "adapter_config.json").write_text("{}")
            self.assertEqual(adapter_base_model(tmp, fallback="org/fallback"), "org/fallback")


def _real_merge_skip_reason() -> str | None:
    if os.environ.get("BREWSLM_SKIP_REAL_TRAINING"):
        return "BREWSLM_SKIP_REAL_TRAINING set"
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return f"ML deps unavailable: {exc}"
    try:
        AutoTokenizer.from_pretrained(REAL_MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        return f"{REAL_MODEL} not in the local HF cache"
    return None


_SKIP = _real_merge_skip_reason()


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class LoraExportMergeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._patch = mock.patch.object(settings, "DATA_DIR", self.root / "data")
        self._patch.start()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.root / 'x.db'}", future=True)
        self.sf = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._patch.stop()
        self._tmp.cleanup()

    def _make_adapter(self, out: Path) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(REAL_MODEL, dtype=torch.float32)
        peft_model = get_peft_model(
            model,
            LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        )
        # Non-zero B so the merge visibly changes the weights.
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.02)
        peft_model.save_pretrained(str(out))
        AutoTokenizer.from_pretrained(REAL_MODEL).save_pretrained(str(out))
        self.reference = peft_model

    async def test_lora_run_exports_as_merged_standalone_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_dir = self.root / "run"
        self._make_adapter(output_dir / "model")
        async with self.sf() as db:
            project = Project(name=f"merge-{uuid4().hex[:6]}", description="", base_model_name=REAL_MODEL)
            db.add(project)
            await db.flush()
            exp = Experiment(
                project_id=project.id,
                name="lora-run",
                status=ExperimentStatus.COMPLETED,
                base_model=REAL_MODEL,
                output_dir=str(output_dir),
            )
            db.add(exp)
            await db.flush()
            export = await create_export(db, project.id, exp.id, ExportFormat.HUGGINGFACE)
            export = await run_export(db, project.id, export.id, run_smoke_tests=False)
            await db.commit()

        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.COMPLETED, manifest.get("error"))
        artifacts = manifest["model_artifacts"]
        self.assertEqual(artifacts["source"], "merged_lora")
        self.assertEqual(artifacts["merged_from_adapter"]["base_model"], REAL_MODEL)

        model_dir = Path(manifest["run_dir"]) / "model"
        self.assertFalse((model_dir / "adapter_config.json").exists())
        # Saved in the base checkpoint's own dtype (bf16), not fp32.
        import json as _json
        saved_cfg = _json.loads((model_dir / "config.json").read_text())
        self.assertIn(str(saved_cfg.get("dtype") or saved_cfg.get("torch_dtype")), {"bfloat16"})
        self.assertTrue((model_dir / "config.json").exists())

        merged = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32).eval()
        base = AutoModelForCausalLM.from_pretrained(REAL_MODEL, dtype=torch.float32).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        inputs = tokenizer("Route this ticket to the right team:", return_tensors="pt")
        with torch.no_grad():
            expected = self.reference.eval()(**inputs).logits
            actual = merged(**inputs).logits
            base_logits = base(**inputs).logits
        # The merged export reproduces base+adapter (within the bf16
        # storage rounding) and is measurably not just the base model.
        merge_error = (expected - actual).abs().mean().item()
        adapter_effect = (expected - base_logits).abs().mean().item()
        self.assertLess(merge_error, adapter_effect / 5, (merge_error, adapter_effect))
        self.assertTrue(torch.equal(expected[0, -1].argmax(), actual[0, -1].argmax()))


if __name__ == "__main__":
    unittest.main()
