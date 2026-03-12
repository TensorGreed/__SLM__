"""Phase 31 tests: model abstraction v2 preflight gates."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["DEBUG"] = "false"
os.environ["TRAINING_BACKEND"] = "simulate"
os.environ["ALLOW_SIMULATED_TRAINING"] = "true"

from app.config import settings
from app.services.training_preflight_service import run_training_preflight


class Phase31ModelAbstractionV2Tests(unittest.TestCase):
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
            {"prompt": "What is due process?", "response": "A legal fairness doctrine."},
            {"prompt": "Summarize negligence.", "response": "Duty, breach, causation, damages."},
        ]
        with open(prepared_dir / "train.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(prepared_dir / "val.jsonl", "w", encoding="utf-8") as f:
            for row in rows[:1]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_preflight_accepts_arbitrary_model_id_when_introspection_resolves_architecture(self):
        project_id = 4101
        self._write_prepared_split(project_id)
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            self._dependency_snapshot,
        ), patch(
            "app.services.training_preflight_service.introspect_hf_model",
            return_value={
                "model_id": "acme/custom-causal-2b",
                "resolved": True,
                "source": "hf_config",
                "model_type": "customlm",
                "architecture": "causal_lm",
                "context_length": 8192,
                "license": "apache-2.0",
                "warnings": [],
            },
        ):
            preflight = run_training_preflight(
                project_id=project_id,
                config={
                    "base_model": "acme/custom-causal-2b",
                    "task_type": "causal_lm",
                    "trainer_backend": "auto",
                    "chat_template": "llama3",
                    "training_runtime_id": "auto",
                },
                base_model="acme/custom-causal-2b",
            )
        self.assertTrue(bool(preflight.get("ok")), preflight)
        capability = dict(preflight.get("capability_summary") or {})
        model = dict(capability.get("model") or {})
        self.assertEqual(str(model.get("architecture")), "causal_lm")

    def test_preflight_blocks_unresolved_architecture_with_hard_gate(self):
        project_id = 4102
        self._write_prepared_split(project_id)
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            self._dependency_snapshot,
        ), patch(
            "app.services.training_preflight_service.introspect_hf_model",
            return_value={
                "model_id": "acme/unknown-arch",
                "resolved": False,
                "source": "none",
                "model_type": None,
                "architecture": "unknown",
                "context_length": None,
                "license": None,
                "warnings": ["network introspection disabled"],
            },
        ):
            preflight = run_training_preflight(
                project_id=project_id,
                config={
                    "base_model": "acme/unknown-arch",
                    "task_type": "causal_lm",
                    "trainer_backend": "auto",
                    "chat_template": "llama3",
                    "training_runtime_id": "auto",
                },
                base_model="acme/unknown-arch",
            )
        self.assertFalse(bool(preflight.get("ok")), preflight)
        errors = [str(item) for item in list(preflight.get("errors") or [])]
        self.assertTrue(
            any("unsupported or unresolved architecture" in item for item in errors),
            errors,
        )
        hints = [str(item) for item in list(preflight.get("hints") or [])]
        self.assertTrue(any("Introspect Model" in item for item in hints), hints)


if __name__ == "__main__":
    unittest.main()

