"""Phase 34 tests: preflight model-vs-dataset modality contract gates."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["DEBUG"] = "false"
os.environ["AUTH_ENABLED"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from app.config import settings
from app.services.training_preflight_service import run_training_preflight


class Phase34PreflightModelModalityContractTests(unittest.TestCase):
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

    @staticmethod
    def _dependency_snapshot() -> dict[str, bool]:
        return {
            "torch": True,
            "transformers": True,
            "datasets": True,
            "accelerate": True,
            "trl": True,
            "peft": True,
            "bitsandbytes": True,
        }

    @classmethod
    def _write_prepared_split(cls, project_id: int) -> None:
        prepared_dir = cls.tmp_root / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "question": "What is shown in the image?",
                "answer": "An invoice with line items.",
                "image_path": "images/invoice.png",
            },
            {
                "question": "What scene is visible?",
                "answer": "A storefront at night.",
                "image_path": "images/store.png",
            },
        ]
        with open(prepared_dir / "train.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_preflight_blocks_encoder_architecture_with_vision_adapter(self):
        project_id = 4401
        self._write_prepared_split(project_id)
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            self._dependency_snapshot,
        ):
            preflight = run_training_preflight(
                project_id=project_id,
                config={
                    "base_model": "bert-base-uncased",
                    "task_type": "classification",
                    "trainer_backend": "hf_trainer",
                    "adapter_id": "vision-language-pair",
                    "training_runtime_id": "auto",
                },
                base_model="bert-base-uncased",
            )

        self.assertFalse(bool(preflight.get("ok")), preflight)
        errors = [str(item) for item in list(preflight.get("errors") or [])]
        self.assertTrue(
            any("supports modalities text" in item and "vision_language" in item for item in errors),
            errors,
        )
        summary = dict(preflight.get("capability_summary") or {})
        modality_contract = dict(summary.get("model_modality_contract") or {})
        self.assertEqual(str(modality_contract.get("architecture")), "encoder")
        self.assertEqual(str(modality_contract.get("adapter_modality")), "vision_language")
        self.assertEqual(list(modality_contract.get("supported_modalities") or []), ["text"])

    def test_preflight_warns_for_causal_lm_with_vision_adapter(self):
        project_id = 4402
        self._write_prepared_split(project_id)
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            self._dependency_snapshot,
        ):
            preflight = run_training_preflight(
                project_id=project_id,
                config={
                    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "adapter_id": "vision-language-pair",
                    "training_runtime_id": "auto",
                },
                base_model="meta-llama/Llama-3.2-1B-Instruct",
            )

        self.assertTrue(bool(preflight.get("ok")), preflight)
        warnings = [str(item) for item in list(preflight.get("warnings") or [])]
        self.assertTrue(
            any("multimodal forward inputs" in item for item in warnings),
            warnings,
        )
        summary = dict(preflight.get("capability_summary") or {})
        modality_contract = dict(summary.get("model_modality_contract") or {})
        self.assertEqual(str(modality_contract.get("architecture")), "causal_lm")
        self.assertEqual(str(modality_contract.get("adapter_modality")), "vision_language")
        supported = list(modality_contract.get("supported_modalities") or [])
        self.assertIn("vision_language", supported)
        self.assertIn("text", supported)


if __name__ == "__main__":
    unittest.main()
