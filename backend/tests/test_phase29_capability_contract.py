"""Phase 29 tests: shared training capability contract checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.config import settings
from app.services.capability_contract_service import (
    build_training_capability_contract,
    evaluate_training_capability_contract,
    resolve_training_adapter_context,
)
from app.services.data_adapter_service import resolve_data_adapter_contract
from app.services.training_runtime_service import (
    TrainingRuntimeStartResult,
    clear_runtime_plugins,
    get_runtime_spec,
    list_runtime_catalog,
    register_training_runtime_plugin,
)


class Phase29CapabilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_root = Path(cls._tmp.name)
        cls._prev_data_dir = settings.DATA_DIR
        settings.DATA_DIR = cls.tmp_root

    @classmethod
    def tearDownClass(cls):
        settings.DATA_DIR = cls._prev_data_dir
        cls._tmp.cleanup()

    def test_build_training_capability_contract_shape(self):
        payload = build_training_capability_contract()
        self.assertEqual(payload.get("contract_version"), "slm.training-capability/v1")
        self.assertIn("causal_lm", payload.get("supported_task_types", []))
        self.assertIn("hf_trainer", payload.get("supported_trainer_backends", []))
        runtime_map = dict(payload.get("runtime_modality_support") or {})
        self.assertIn("builtin.external_celery", runtime_map)
        self.assertEqual(
            runtime_map.get("builtin.external_celery"),
            ["audio_text", "multimodal", "text", "vision_language"],
        )
        declared_map = dict(payload.get("runtime_modality_declared") or {})
        self.assertTrue(bool(declared_map.get("builtin.external_celery")))

    def test_resolve_training_adapter_context_reads_prepared_manifest(self):
        project_id = 9029
        prepared_dir = self.tmp_root / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = prepared_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "adapter_id": "vision-language-pair",
                    "task_profile": "instruction_sft",
                }
            ),
            encoding="utf-8",
        )

        context = resolve_training_adapter_context(project_id=project_id, config={})
        self.assertEqual(context.get("adapter_id"), "vision-language-pair")
        self.assertEqual(context.get("task_profile"), "instruction_sft")
        self.assertEqual(context.get("adapter_modality"), "vision_language")
        self.assertTrue(bool(context.get("prepared_manifest_found")))

    def test_evaluate_capability_allows_builtin_external_runtime_for_vision_adapter(self):
        contract = resolve_data_adapter_contract("vision-language-pair")
        report = evaluate_training_capability_contract(
            task_type="seq2seq",
            training_mode="sft",
            trainer_backend_requested="hf_trainer",
            runtime_id="builtin.external_celery",
            runtime_backend="celery",
            adapter_id="vision-language-pair",
            adapter_contract=contract,
            adapter_task_profile="instruction_sft",
        )
        self.assertTrue(bool(report.get("ok")))
        runtime_spec = get_runtime_spec("builtin.external_celery")
        self.assertIn("vision_language", list(runtime_spec.supported_modalities))
        summary = dict(report.get("summary") or {})
        self.assertEqual(summary.get("adapter_modality"), "vision_language")

    def test_evaluate_capability_blocks_text_only_runtime_for_vision_adapter(self):
        clear_runtime_plugins()
        try:
            async def _start(_ctx):
                return TrainingRuntimeStartResult(message="ok")

            register_training_runtime_plugin(
                runtime_id="custom.text_only",
                label="Text Only Runtime",
                description="Custom runtime supporting text only.",
                execution_backend="external",
                validate=lambda: [],
                start=_start,
                required_dependencies=[],
                supported_modalities=["text"],
                supports_task_tracking=False,
                supports_cancellation=True,
                is_builtin=False,
                source_module="tests.phase29",
            )

            contract = resolve_data_adapter_contract("vision-language-pair")
            report = evaluate_training_capability_contract(
                task_type="seq2seq",
                training_mode="sft",
                trainer_backend_requested="hf_trainer",
                runtime_id="custom.text_only",
                runtime_backend="external",
                adapter_id="vision-language-pair",
                adapter_contract=contract,
                adapter_task_profile="instruction_sft",
            )
            self.assertFalse(bool(report.get("ok")))
            errors = [str(item) for item in list(report.get("errors") or [])]
            self.assertTrue(any("supports modalities" in item for item in errors))
        finally:
            clear_runtime_plugins()
            list_runtime_catalog()

    def test_custom_runtime_declared_modalities_enforced(self):
        clear_runtime_plugins()
        try:
            async def _start(_ctx):
                return TrainingRuntimeStartResult(message="ok")

            register_training_runtime_plugin(
                runtime_id="custom.vision_only",
                label="Vision Runtime",
                description="Custom runtime for vision-language data.",
                execution_backend="external",
                validate=lambda: [],
                start=_start,
                required_dependencies=[],
                supported_modalities=["vision_language"],
                supports_task_tracking=False,
                supports_cancellation=True,
                is_builtin=False,
                source_module="tests.phase29",
            )

            contract = resolve_data_adapter_contract("default-canonical")
            report = evaluate_training_capability_contract(
                task_type="causal_lm",
                training_mode="sft",
                trainer_backend_requested="hf_trainer",
                runtime_id="custom.vision_only",
                runtime_backend="external",
                adapter_id="default-canonical",
                adapter_contract=contract,
                adapter_task_profile="instruction_sft",
            )
            self.assertFalse(bool(report.get("ok")))
            errors = [str(item) for item in list(report.get("errors") or [])]
            self.assertTrue(any("supports modalities" in item for item in errors), errors)
        finally:
            clear_runtime_plugins()
            list_runtime_catalog()
