"""Phase 20 tests: conformance + scale matrix for general SLM workflows.

This suite validates that core adapters, runtime plugins, model families, and
task-types compose correctly, and that mapping quality remains stable at larger
sample sizes.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from itertools import cycle
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ["DEBUG"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from app.config import settings
from app.services.data_adapter_service import (
    map_record_with_adapter,
    preview_data_adapter,
    resolve_task_profile_for_adapter,
)
from app.services.dataset_contract_service import analyze_prepared_dataset_contract
from app.services.training_preflight_service import run_training_preflight
from app.services.training_runtime_service import list_runtime_catalog


class Phase20ConformanceScaleMatrixTests(unittest.TestCase):
    ADAPTER_IDS = (
        "default-canonical",
        "qa-pair",
        "rag-grounded",
        "seq2seq-pair",
        "structured-extraction",
        "classification-label",
        "chat-messages",
        "tool-call-json",
    )

    MODEL_FAMILY_CASES = (
        ("llama", "meta-llama/Llama-3.2-1B-Instruct"),
        ("mistral", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("qwen", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("phi", "microsoft/phi-3-mini-4k-instruct"),
        ("gemma", "google/gemma-2-2b-it"),
        ("encoder_decoder", "google/flan-t5-base"),
        ("encoder_only", "bert-base-uncased"),
    )

    TASK_TYPES = ("causal_lm", "seq2seq", "classification")
    SCALE_SIZES = (8, 256)

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_root = Path(cls._tmp.name)

        cls._prev_data_dir = settings.DATA_DIR
        cls._prev_training_external_cmd = settings.TRAINING_EXTERNAL_CMD
        cls._prev_allow_simulated_training = settings.ALLOW_SIMULATED_TRAINING

        settings.DATA_DIR = cls.tmp_root
        settings.TRAINING_EXTERNAL_CMD = "python -c \"print('ok')\""
        settings.ALLOW_SIMULATED_TRAINING = True

    @classmethod
    def tearDownClass(cls):
        settings.DATA_DIR = cls._prev_data_dir
        settings.TRAINING_EXTERNAL_CMD = cls._prev_training_external_cmd
        settings.ALLOW_SIMULATED_TRAINING = cls._prev_allow_simulated_training
        cls._tmp.cleanup()

    @staticmethod
    def _full_dependency_snapshot() -> dict[str, bool]:
        return {
            "torch": True,
            "transformers": True,
            "datasets": True,
            "accelerate": True,
            "trl": True,
            "peft": True,
            "bitsandbytes": True,
        }

    @staticmethod
    def _adapter_templates(adapter_id: str) -> list[dict[str, Any]]:
        if adapter_id == "default-canonical":
            return [
                {"text": "General text sample", "question": "What is this?", "answer": "A sample."},
                {"question": "Define contract", "answer": "Agreement enforceable by law."},
            ]
        if adapter_id == "qa-pair":
            return [
                {"question": "What is negligence?", "answer": "Failure to exercise due care."},
                {"prompt": "What is tort law?", "response": "Civil wrong law."},
            ]
        if adapter_id == "rag-grounded":
            return [
                {
                    "question": "What elements define negligence?",
                    "context": "Negligence requires duty, breach, causation, and damages.",
                    "answer": "Duty, breach, causation, and damages.",
                },
                {
                    "prompt": "What is consideration in contract law?",
                    "passage": "A promise must include consideration to be enforceable.",
                    "response": "An exchange of value between parties.",
                },
            ]
        if adapter_id == "seq2seq-pair":
            return [
                {"source": "Summarize the clause", "target": "Clause summary"},
                {"input": "Translate: bonjour", "output": "hello"},
            ]
        if adapter_id == "structured-extraction":
            return [
                {"text": "Invoice 42 total $99", "structured_output": {"invoice_id": "42", "total": "99"}},
                {"content": "PO 123 approved by Alice", "json": {"po_id": "123", "approver": "Alice"}},
            ]
        if adapter_id == "classification-label":
            return [
                {"text": "The contract is enforceable.", "label": "legal"},
                {"content": "2 + 2 equals 4", "category": "math"},
            ]
        if adapter_id == "chat-messages":
            return [
                {
                    "messages": [
                        {"role": "user", "content": "Explain estoppel briefly."},
                        {"role": "assistant", "content": "It prevents inconsistent legal claims."},
                    ]
                },
                {
                    "conversations": [
                        {"from": "human", "value": "What is 7 * 6?"},
                        {"from": "gpt", "value": "42."},
                    ]
                },
            ]
        if adapter_id == "tool-call-json":
            return [
                {
                    "prompt": "What is weather in NYC?",
                    "tool_name": "get_weather",
                    "tool_args": {"city": "NYC"},
                    "tool_result": "Sunny and 21C.",
                },
                {
                    "instruction": "Find train fare BOS -> NYC",
                    "function_name": "quote_fare",
                    "arguments": {"origin": "BOS", "destination": "NYC"},
                    "result": "USD 49.",
                },
            ]
        raise ValueError(f"Unsupported adapter_id for matrix test: {adapter_id}")

    @classmethod
    def _build_raw_rows(cls, adapter_id: str, size: int) -> list[dict[str, Any]]:
        templates = cls._adapter_templates(adapter_id)
        rows: list[dict[str, Any]] = []
        for idx, template in zip(range(size), cycle(templates)):
            row = dict(template)
            # Inject a tiny per-row variation so larger scale tests are not
            # trivially identical records.
            row["_row_id"] = f"{adapter_id}-{idx}"
            rows.append(row)
        return rows

    @classmethod
    def _map_rows_for_adapter(cls, adapter_id: str, size: int) -> tuple[str, list[dict[str, Any]]]:
        task_profile = resolve_task_profile_for_adapter(
            adapter_id,
            requested_task_profile="auto",
        )
        raw_rows = cls._build_raw_rows(adapter_id, size)
        mapped_rows: list[dict[str, Any]] = []
        for row in raw_rows:
            mapped = map_record_with_adapter(
                row,
                adapter_id=adapter_id,
                task_profile=task_profile,
            )
            if mapped is None:
                raise AssertionError(f"Adapter '{adapter_id}' failed to map seed row: {row}")
            mapped_rows.append(mapped)
        return task_profile, mapped_rows

    @classmethod
    def _write_prepared_split(
        cls,
        *,
        project_id: int,
        adapter_id: str,
        task_profile: str,
        mapped_rows: list[dict[str, Any]],
    ) -> None:
        prepared_dir = cls.tmp_root / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        train_file = prepared_dir / "train.jsonl"
        val_file = prepared_dir / "val.jsonl"
        manifest_file = prepared_dir / "manifest.json"

        with open(train_file, "w", encoding="utf-8") as f:
            for row in mapped_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(val_file, "w", encoding="utf-8") as f:
            for row in mapped_rows[: max(1, min(16, len(mapped_rows)))]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest_file.write_text(
            json.dumps(
                {
                    "adapter_id": adapter_id,
                    "field_mapping": {},
                    "task_profile": task_profile,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_adapter_scale_conformance_matrix(self):
        checks = 0
        for adapter_id in self.ADAPTER_IDS:
            task_profile = resolve_task_profile_for_adapter(adapter_id, requested_task_profile="auto")
            for size in self.SCALE_SIZES:
                rows = self._build_raw_rows(adapter_id, size)
                result = preview_data_adapter(
                    rows,
                    adapter_id=adapter_id,
                    task_profile=task_profile,
                    preview_limit=2,
                )
                checks += 1
                with self.subTest(adapter_id=adapter_id, size=size):
                    self.assertEqual(result.get("resolved_adapter_id"), adapter_id)
                    self.assertEqual(result.get("resolved_task_profile"), task_profile)
                    self.assertGreaterEqual(int(result.get("mapped_records") or 0), int(size * 0.95))
                    conformance = result.get("conformance_report") or {}
                    self.assertTrue(bool(conformance.get("contract_pass")), conformance)
                    self.assertGreaterEqual(float(conformance.get("mapping_success_rate") or 0.0), 0.95)
        self.assertGreater(checks, 0)

    def test_training_preflight_conformance_matrix(self):
        runtime_catalog = list_runtime_catalog()
        runtimes = [
            item for item in list(runtime_catalog.get("runtimes") or [])
            if str(item.get("runtime_id")) in {"builtin.simulate", "builtin.external_celery"}
        ]
        runtime_ids = [str(item.get("runtime_id")) for item in runtimes]
        runtime_backend_by_id = {
            str(item.get("runtime_id")): str(item.get("execution_backend"))
            for item in runtimes
        }
        self.assertIn("builtin.simulate", runtime_ids)
        self.assertIn("builtin.external_celery", runtime_ids)

        matrix_checks = 0
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            return_value=self._full_dependency_snapshot(),
        ):
            for adapter_index, adapter_id in enumerate(self.ADAPTER_IDS, start=1):
                task_profile, mapped_rows = self._map_rows_for_adapter(adapter_id, size=96)
                project_id = 2000 + adapter_index
                self._write_prepared_split(
                    project_id=project_id,
                    adapter_id=adapter_id,
                    task_profile=task_profile,
                    mapped_rows=mapped_rows,
                )

                dataset_ok_by_task: dict[str, bool] = {}
                for task_type in self.TASK_TYPES:
                    report = analyze_prepared_dataset_contract(
                        project_id=project_id,
                        task_type=task_type,
                        sample_size=256,
                        min_coverage=0.9,
                    )
                    dataset_ok_by_task[task_type] = bool(report.get("ok"))

                for runtime_id in runtime_ids:
                    for family, model_id in self.MODEL_FAMILY_CASES:
                        for task_type in self.TASK_TYPES:
                            config = {
                                "base_model": model_id,
                                "task_type": task_type,
                                "trainer_backend": "hf_trainer",
                                "training_runtime_id": runtime_id,
                                "max_seq_length": 1024,
                                "batch_size": 2,
                                "num_epochs": 1,
                            }
                            preflight = run_training_preflight(
                                project_id=project_id,
                                config=config,
                            )
                            matrix_checks += 1

                            capability = dict(preflight.get("capability_summary") or {})
                            model_summary = dict(capability.get("model") or {})
                            runtime_summary = dict(capability.get("runtime") or {})
                            supported_task_types = {
                                str(item) for item in list(model_summary.get("supported_task_types") or [])
                            }

                            expected_model_ok = task_type in supported_task_types
                            expected_dataset_ok = bool(dataset_ok_by_task.get(task_type))
                            expected_ok = expected_model_ok and expected_dataset_ok

                            errors = [str(item) for item in list(preflight.get("errors") or [])]
                            with self.subTest(
                                adapter_id=adapter_id,
                                runtime_id=runtime_id,
                                model_family=family,
                                model_id=model_id,
                                task_type=task_type,
                            ):
                                self.assertEqual(runtime_summary.get("resolved_runtime_id"), runtime_id)
                                self.assertEqual(capability.get("runtime_backend"), runtime_backend_by_id[runtime_id])
                                self.assertEqual(bool(preflight.get("ok")), expected_ok)

                                if not expected_model_ok:
                                    self.assertTrue(
                                        any("task_type=" in err and "incompatible with base_model" in err for err in errors),
                                        errors,
                                    )
                                if expected_model_ok and not expected_dataset_ok:
                                    self.assertTrue(
                                        any("Dataset contract mismatch for task_type" in err for err in errors),
                                        errors,
                                    )

        self.assertGreater(matrix_checks, 0)


if __name__ == "__main__":
    unittest.main()
