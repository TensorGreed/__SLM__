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

import json
import os
import sys
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


def _mark_complete(path: Path) -> None:
    """The success marker a finished conversion writes (``compression_service.finish_conversion``)."""
    import json

    compression_service.conversion_marker(path).write_text(json.dumps({"size": path.stat().st_size}))


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
        _mark_complete(path)
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

    def test_quantizing_a_merged_model_stays_under_its_run(self):
        # merge-lora writes compressed/exp-<id>/merged; quantizing that merged
        # model used to drop the GGUF in the project-wide compressed/ root,
        # where export (rightly) never looks — the export then failed.
        merged = self.project_dir / "compressed" / "exp-7" / "merged"
        merged.mkdir(parents=True)
        self.assertEqual(
            compression_service.experiment_id_for_model_path(self.project_id, str(merged)), 7
        )
        self.assertEqual(
            compression_service._output_dir_for_source(self.project_id, str(merged)),
            self.project_dir / "compressed" / "exp-7",
        )
        loose = self.project_dir / "compressed" / "merged"
        loose.mkdir(parents=True)
        self.assertIsNone(
            compression_service.experiment_id_for_model_path(self.project_id, str(loose))
        )

    def test_merged_then_quantized_gguf_is_what_export_packages(self):
        merged = self.project_dir / "compressed" / "exp-1" / "merged"
        merged.mkdir(parents=True)
        out_dir = compression_service._output_dir_for_source(self.project_id, str(merged))
        gguf = out_dir / "quantized_4bit.gguf"
        gguf.write_bytes(b"GGUF")
        _mark_complete(gguf)
        source, files, _ = _resolve_source_model_files(
            self.project_id, self._exp(1), ExportFormat.GGUF, "Q4_K_M"
        )
        self.assertEqual(source, "compressed")
        self.assertEqual(files, [gguf])


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
    if not os.environ.get("BREWSLM_REAL_TRAINING"):
        try:
            import torch

            has_gpu = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            has_gpu = False
        if not has_gpu:
            # Real fine-tunes on CPU (+ a model download) take many minutes —
            # opt in with BREWSLM_REAL_TRAINING=1; GPU boxes run them by default.
            return "needs a GPU or BREWSLM_REAL_TRAINING=1"
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

        model_dir = Path(export.output_path) / manifest["run_dir"] / "model"
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


# A stand-in for scripts/quantize.py: writes a file at --out (a partial one, then exit 3, when told to fail;
# after a short pause, so a second export of the same run overlaps it).
_FAKE_QUANTIZE = (
    'import sys, time, pathlib; a = sys.argv; out = pathlib.Path(a[a.index("--out") + 1]); '
    'out.write_bytes(b"GG"); time.sleep(0.3); '
    'sys.exit(3) if "FAIL" in a[a.index("--model") + 1] else out.write_bytes(b"GGUF" * 1024)'
)


QWEN_LIKE_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{%- endif %}"
    "{%- for message in messages %}{%- if message.role != 'system' %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


class GgufChatTemplateTests(unittest.TestCase):
    def test_known_formats_convert_and_others_say_why(self):
        from app.services.gguf_chat_template import ollama_template

        chatml = ollama_template(QWEN_LIKE_TEMPLATE)
        self.assertEqual((chatml["family"], chatml["system"]), ("chatml", "You are a helpful assistant."))
        llama = ollama_template(
            "{% for m in messages %}{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\\n\\n' + m['content'] + '<|eot_id|>' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
        )
        self.assertEqual(llama["family"], "llama3")
        dated = ollama_template("<|start_header_id|><|eot_id|> Cutting Knowledge Date")
        self.assertIn("date / tool preamble", dated["reason"])
        self.assertIn("does not convert", ollama_template("{{ messages }} [INST]")["reason"])

    def test_num_ctx_and_triple_quotes(self):
        from pathlib import Path as _P

        from app.services.export_service import gguf_num_ctx
        from app.services.gguf_chat_template import modelfile_template

        self.assertEqual(gguf_num_ctx(2048, 64), 2304)
        self.assertEqual(gguf_num_ctx(512, None), 2048)
        self.assertIsNone(gguf_num_ctx(None, 64))
        d = _P(tempfile.mkdtemp())
        (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": QWEN_LIKE_TEMPLATE.replace(
            "You are a helpful assistant.", 'Say \\"\\"\\" never.')}))
        block = modelfile_template([d])
        self.assertFalse(block["written"])
        self.assertIn("triple quote", block["reason"])

    def test_off_tokenizer_template_is_not_claimed(self):
        from app.services.gguf_chat_template import modelfile_template

        block = modelfile_template([], use_tokenizer_chat_template=False)
        self.assertFalse(block["written"])
        self.assertIn("use_tokenizer_chat_template off", block["note"])


class GgufExportConvertsAndCarriesDecodingTests(unittest.TestCase):
    """P5 gaps 10 + 11: the default GGUF export converts the run itself when no converted file exists, and
    the package decodes as evaluated (Ollama Modelfile + serve.py defaults)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._patches = [
            mock.patch.object(settings, "DATA_DIR", self.root / "data"),
            mock.patch.object(settings, "COMPRESSION_BACKEND", "external"),
            mock.patch.object(
                settings, "QUANTIZE_EXTERNAL_CMD",
                f"{sys.executable} -c '{_FAKE_QUANTIZE}' --model \"{{model_path}}\" --bits {{bits}} --out {{output_model_path}}",
            ),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def _export(self, model_dir_name: str = "run", quantization: str = "4-bit", eval_decoding: dict | None = None,
                chat_template: str | None = None, config: dict | None = None, chat_wrap: str | None = None):
        import asyncio

        from app.models.experiment import EvalResult

        async def go():
            engine = create_async_engine(f"sqlite+aiosqlite:///{self.root / 'x.db'}", future=True)
            sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            try:
                async with sf() as db:
                    project = Project(name=f"gguf-{uuid4().hex[:6]}", description="", base_model_name="toy")
                    db.add(project)
                    await db.flush()
                    out = self.root / model_dir_name
                    (out / "model").mkdir(parents=True, exist_ok=True)
                    (out / "model" / "config.json").write_text("{}")
                    if chat_template is not None:
                        (out / "model" / "tokenizer_config.json").write_text(json.dumps({"chat_template": chat_template}))
                    if chat_wrap is not None:
                        (out / "training_report.json").write_text(json.dumps(
                            {"runtime_environment": {"chat_template_rewrap": chat_wrap}}))
                    exp = Experiment(project_id=project.id, name="r", status=ExperimentStatus.COMPLETED,
                                     base_model="toy", output_dir=str(out), config=config or {})
                    db.add(exp)
                    await db.flush()
                    if eval_decoding is not None:
                        db.add(EvalResult(
                            experiment_id=exp.id, dataset_name="test", eval_type="exact_match", pass_rate=0.5,
                            metrics={"inference": {"decoding": eval_decoding}},
                            details={"inference": {"max_new_tokens": 64}},
                        ))
                        await db.flush()
                    export = await create_export(db, project.id, exp.id, ExportFormat.GGUF, quantization)
                    export = await run_export(db, project.id, export.id, run_smoke_tests=False)
                    await db.commit()
                    return export, exp.id
            finally:
                await engine.dispose()

        return asyncio.run(go())

    def test_default_gguf_export_converts_then_packages_with_greedy_decoding(self):
        export, exp_id = self._export()
        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.COMPLETED, manifest.get("error"))
        self.assertEqual(manifest["auto_conversion"]["bits"], 4)
        self.assertEqual(manifest["model_artifacts"]["source"], "compressed")
        run_dir = Path(export.output_path) / manifest["run_dir"]
        self.assertTrue((run_dir / "model" / "quantized_4bit.gguf").is_file())
        modelfile = (run_dir / "Modelfile").read_text()
        self.assertIn('FROM "./model/quantized_4bit.gguf"', modelfile)
        for line in ("PARAMETER temperature 0", "PARAMETER repeat_penalty 1.0", "PARAMETER top_k 0",
                     "PARAMETER top_p 1.0", "PARAMETER min_p 0.0"):
            self.assertIn(line, modelfile)
        self.assertEqual(manifest["decoding"]["modelfile"], "Modelfile")
        serve = (run_dir / "serve.py").read_text()
        self.assertIn('"./model/quantized_4bit.gguf"', serve)
        self.assertIn("temperature: float = 0.0", serve)
        self.assertIn("repeat_penalty=1.0", serve)
        notes = (run_dir / "RELEASE_NOTES.md").read_text()
        self.assertIn("converted to GGUF Q4_K_M (4-bit) by this export", notes)
        self.assertIn("- Quantization: Q4_K_M", notes)
        self.assertEqual(manifest["quantization_label"], "Q4_K_M")
        self.assertIn("Modelfile (Ollama)", notes)
        # the conversion lands where Compression puts it (reused by the next export)
        project_dir = self.root / "data" / "projects" / str(export.project_id)
        self.assertTrue((project_dir / "compressed" / f"exp-{exp_id}" / "quantize_report.json").is_file())

    def test_the_modelfile_carries_the_chat_template_the_run_was_trained_with(self):
        """P5b gap 6: TEMPLATE (+ the template's default SYSTEM and stop) from the tokenizer's chat template, and
        num_ctx from max_seq_length; without a convertible template the Modelfile and notes say so."""
        export, _ = self._export(chat_template=QWEN_LIKE_TEMPLATE, config={"max_seq_length": 2048},
                                 chat_wrap="tokenizer_chat_template")
        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.COMPLETED, manifest.get("error"))
        run_dir = Path(export.output_path) / manifest["run_dir"]
        modelfile = (run_dir / "Modelfile").read_text()
        self.assertIn('TEMPLATE """{{- if .System }}<|im_start|>system', modelfile)
        self.assertIn('SYSTEM """You are a helpful assistant."""', modelfile)
        self.assertIn('PARAMETER stop "<|im_end|>"', modelfile)
        self.assertIn("PARAMETER num_ctx 2304", modelfile)  # 2048 window + 256 generation tokens
        self.assertEqual(manifest["decoding"]["template"]["family"], "chatml")
        self.assertIn("Chat template: Modelfile TEMPLATE written", (run_dir / "RELEASE_NOTES.md").read_text())

        export, _ = self._export(model_dir_name="run2")
        manifest = export.manifest or {}
        run_dir = Path(export.output_path) / manifest["run_dir"]
        modelfile = (run_dir / "Modelfile").read_text()
        self.assertFalse(any(line.startswith(("TEMPLATE", "SYSTEM")) for line in modelfile.splitlines()))
        self.assertNotIn("num_ctx", modelfile)  # no max_seq_length recorded: nothing guessed
        self.assertIn("# No TEMPLATE written: the training report does not record how the rows were rendered.", modelfile)
        self.assertIn("default template for the base model family", (run_dir / "RELEASE_NOTES.md").read_text())

        # a template on disk, but training rendered the rows another way: no TEMPLATE, a note instead
        export, _ = self._export(model_dir_name="run3", chat_template=QWEN_LIKE_TEMPLATE, chat_wrap="preset:llama3")
        run_dir = Path(export.output_path) / export.manifest["run_dir"]
        modelfile = (run_dir / "Modelfile").read_text()
        self.assertFalse(any(line.startswith("TEMPLATE") for line in modelfile.splitlines()))
        self.assertIn("recorded: preset:llama3", modelfile)

    def test_failed_conversion_is_explained_with_the_compression_step(self):
        export, _ = self._export(model_dir_name="FAIL-run")
        self.assertEqual(export.status, ExportStatus.FAILED)
        self.assertIn("Converting it here failed: the conversion exited with code 3", export.manifest["error"])
        self.assertIn("Open Compression", export.manifest["error"])
        # The command wrote a partial file before failing: nothing of it is left to be packaged later.
        compressed = self.root / "data" / "projects" / str(export.project_id) / "compressed"
        self.assertEqual([p.name for p in compressed.rglob("*.gguf")], [])
        self.assertEqual([p.name for p in compressed.rglob(".convert-*")], [])

    def test_an_unsupported_quantization_label_fails_clearly_instead_of_converting_to_another(self):
        export, _ = self._export(quantization="Q4_0")
        self.assertEqual(export.status, ExportStatus.FAILED)
        self.assertIn("Quantization 'Q4_0' is not one BrewSLM converts GGUF to", export.manifest["error"])
        self.assertIn("4-bit (Q4_K_M)", export.manifest["error"])
        compressed = self.root / "data" / "projects" / str(export.project_id) / "compressed"
        self.assertEqual(list(compressed.rglob("*.gguf")), [])

    def test_the_package_decodes_with_the_runs_evaluated_decoding(self):
        export, _ = self._export(eval_decoding={"do_sample": False, "repetition_penalty": 1.1})
        manifest = export.manifest or {}
        self.assertEqual(export.status, ExportStatus.COMPLETED, manifest.get("error"))
        run_dir = Path(export.output_path) / manifest["run_dir"]
        modelfile = (run_dir / "Modelfile").read_text()
        self.assertIn("PARAMETER temperature 0", modelfile)
        self.assertIn("PARAMETER repeat_penalty 1.1", modelfile)
        self.assertIn("PARAMETER num_predict 64", modelfile)
        self.assertTrue(manifest["decoding"]["source"].startswith("eval #"))
        serve = (run_dir / "serve.py").read_text()
        self.assertIn("max_tokens: int = 64", serve)
        self.assertIn("repeat_penalty=1.1", serve)

    def test_a_second_export_of_the_same_run_waits_and_reuses_the_conversion(self):
        import asyncio
        from types import SimpleNamespace

        from app.services.export_service import _auto_convert_for_export

        out = self.root / "run"
        (out / "model").mkdir(parents=True)
        (out / "model" / "config.json").write_text("{}")
        exp = SimpleNamespace(id=7, output_dir=str(out))
        export = SimpleNamespace(export_format=ExportFormat.GGUF, quantization="4-bit")

        async def both():
            return await asyncio.gather(
                _auto_convert_for_export(1, exp, export), _auto_convert_for_export(1, exp, export),
            )

        first, second = asyncio.run(both())
        self.assertTrue(first["ok"] and second["ok"], (first, second))
        self.assertEqual(sorted(bool(r.get("reused")) for r in (first, second)), [False, True])
        compressed = self.root / "data" / "projects" / "1" / "compressed" / "exp-7"
        self.assertEqual(sorted(p.name for p in compressed.rglob("*.gguf")), ["quantized_4bit.gguf"])
        self.assertEqual((compressed / "quantized_4bit.gguf").stat().st_size, 4096)

    def test_without_an_external_backend_the_message_is_unchanged(self):
        with mock.patch.object(settings, "COMPRESSION_BACKEND", "stub"):
            export, _ = self._export()
        self.assertEqual(export.status, ExportStatus.FAILED)
        self.assertNotIn("Converting it here", export.manifest["error"])
        self.assertIn("Open Compression", export.manifest["error"])

    def test_quantization_bits(self):
        bits = compression_service.quantization_bits
        self.assertEqual([bits("4-bit", "gguf"), bits("Q8_0", "gguf"), bits(None, "gguf"), bits("int8", "onnx")],
                         [4, 8, 16, 8])
        self.assertIsNone(bits("7-bit", "gguf"))
        self.assertIsNone(bits("4-bit", "onnx"))
        self.assertIsNone(bits("Q4_0", "gguf"))  # never silently Q4_K_M
        plan = compression_service.quantization_plan
        self.assertEqual(plan(None, "onnx")["label"], "int8 (dynamic)")
        self.assertEqual(plan("8-bit", "gguf")["label"], "Q8_0")

    def test_queued_quantize_is_atomic_and_an_unmarked_file_is_never_packaged(self):
        """P5 re-review: Compression -> Quantize (the Celery job) writes into a temp folder under the run
        folder's lock and publishes with os.replace + a success marker; a failed job leaves no file, and a
        truncated file without its marker is never collected."""
        import asyncio

        from app import worker
        from app.services.export_service import _collect_compressed_files

        sent: dict = {}

        class _Task:
            id = "t1"

        def send_task(name, kwargs):
            sent.update(kwargs)
            return _Task()

        class _Redis:
            async def publish(self, *_a):
                return 0

            async def aclose(self):
                return None

        out_dir = self.root / "data" / "projects" / "1" / "compressed" / "exp-7"
        final = out_dir / "quantized_4bit.gguf"
        with mock.patch("app.services.compression_service.experiment_id_for_model_path", return_value=7), \
                mock.patch("app.worker.celery_app.send_task", side_effect=send_task):
            queued = asyncio.run(compression_service.quantize_model(1, str(self.root / "FAIL-run"), 4, "gguf"))
        self.assertEqual(queued["output_model_path"], str(final))
        self.assertNotIn(str(final), sent["command"])  # the command writes into the temp folder
        self.assertEqual(sent["finalize"]["final_path"], str(final))

        with mock.patch("app.worker.aioredis.from_url", return_value=_Redis()):
            with self.assertRaises(Exception):
                worker.run_quantization_job.run(**sent)  # the fake writes a partial file, then exits 3
        self.assertFalse(final.exists())
        self.assertEqual(list(out_dir.glob(".convert-*")), [])

        ok_sent = {**sent, "command": sent["command"].replace("FAIL-run", "run")}
        with mock.patch("app.worker.aioredis.from_url", return_value=_Redis()):
            worker.run_quantization_job.run(**ok_sent)
        self.assertEqual(final.stat().st_size, 4096)
        self.assertEqual(_collect_compressed_files(1, ExportFormat.GGUF, "4-bit", 7), [final])
        # A file truncated in place (e.g. written by an older, non-atomic job that was killed) is not packaged.
        final.write_bytes(b"GG")
        self.assertEqual(_collect_compressed_files(1, ExportFormat.GGUF, "4-bit", 7), [])
        self.assertFalse(compression_service.is_complete_conversion(final))

    def test_reuse_after_the_lock_matches_the_requested_bit_width(self):
        """An F16 (no label) export never reuses a Q4 file another export left."""
        import asyncio
        from types import SimpleNamespace

        from app.services.export_service import _auto_convert_for_export

        out = self.root / "run"
        (out / "model").mkdir(parents=True)
        (out / "model" / "config.json").write_text("{}")
        exp = SimpleNamespace(id=7, output_dir=str(out))
        q4 = asyncio.run(_auto_convert_for_export(1, exp, SimpleNamespace(export_format=ExportFormat.GGUF, quantization="4-bit")))
        self.assertTrue(q4["ok"] and not q4.get("reused"))
        f16 = asyncio.run(_auto_convert_for_export(1, exp, SimpleNamespace(export_format=ExportFormat.GGUF, quantization=None)))
        self.assertTrue(f16["ok"], f16)
        self.assertFalse(f16.get("reused"))
        self.assertEqual(f16["bits"], 16)
        compressed = self.root / "data" / "projects" / "1" / "compressed" / "exp-7"
        self.assertEqual(sorted(p.name for p in compressed.glob("*.gguf")), ["quantized_16bit.gguf", "quantized_4bit.gguf"])

    def test_packaged_labels_and_the_requested_gguf(self):
        from types import SimpleNamespace

        from app.services.deployment_target_service import _packaged_launch_example
        from app.services.export_service import packaged_quantization_label, pick_gguf

        onnx = SimpleNamespace(export_format=ExportFormat.ONNX, quantization=None)
        self.assertEqual(packaged_quantization_label(onnx, [Path("quantized_8bit.onnx")]), "int8 (dynamic)")
        files = [Path("m/quantized_8bit.gguf"), Path("m/quantized_4bit.gguf")]
        self.assertEqual(pick_gguf(files, "4-bit").name, "quantized_4bit.gguf")
        self.assertEqual(pick_gguf(files, None).name, "quantized_8bit.gguf")
        run_dir = self.root / "pkg"
        (run_dir / "model").mkdir(parents=True)
        (run_dir / "model" / "quantized_4bit.gguf").write_bytes(b"GGUF")
        example = _packaged_launch_example(run_dir, {"launch_example": "llama-cli -m ./model/<file>.gguf -p 'Hello'"})
        self.assertEqual(example, "llama-cli -m ./model/quantized_4bit.gguf -p 'Hello'")


if __name__ == "__main__":
    unittest.main()
