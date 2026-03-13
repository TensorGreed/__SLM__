"""Phase 35 tests: multimodal media asset contract diagnostics in preflight."""

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


class Phase35PreflightMediaContractTests(unittest.TestCase):
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
    def _prepared_dir(cls, project_id: int) -> Path:
        prepared_dir = cls.tmp_root / "projects" / str(project_id) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        return prepared_dir

    @classmethod
    def _write_train_rows(cls, project_id: int, rows: list[dict]) -> None:
        prepared_dir = cls._prepared_dir(project_id)
        train_file = prepared_dir / "train.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def _touch_media_file(cls, project_id: int, relative_path: str) -> None:
        prepared_dir = cls._prepared_dir(project_id)
        target = prepared_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake")

    def test_preflight_blocks_when_local_media_refs_are_missing(self):
        project_id = 4501
        self._write_train_rows(
            project_id,
            [
                {"question": "q1", "answer": "a1", "image_path": "images/missing-1.png"},
                {"question": "q2", "answer": "a2", "image_path": "images/missing-2.png"},
                {"question": "q3", "answer": "a3", "image_path": "images/missing-3.png"},
            ],
        )
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            self._dependency_snapshot,
        ):
            preflight = run_training_preflight(
                project_id=project_id,
                config={
                    "base_model": "microsoft/phi-2",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "adapter_id": "vision-language-pair",
                    "training_runtime_id": "auto",
                },
                base_model="microsoft/phi-2",
            )

        self.assertFalse(bool(preflight.get("ok")), preflight)
        errors = [str(item) for item in list(preflight.get("errors") or [])]
        self.assertTrue(any("Missing local media assets" in item for item in errors), errors)

        summary = dict(preflight.get("capability_summary") or {})
        dataset = dict(summary.get("dataset") or {})
        media_contract = dict(dataset.get("media_contract") or {})
        self.assertEqual(str(media_contract.get("expected_modality")), "vision_language")
        self.assertEqual(int(media_contract.get("image_rows") or 0), 3)
        self.assertEqual(int(media_contract.get("missing_local_images") or 0), 3)
        self.assertEqual(int(media_contract.get("remote_image_refs") or 0), 0)

    def test_preflight_warns_when_rows_use_remote_media_urls(self):
        project_id = 4502
        self._write_train_rows(
            project_id,
            [
                {
                    "question": "What is shown?",
                    "answer": "A red stop sign.",
                    "image_path": "https://example.com/media/stop-sign.png",
                }
            ],
        )
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            self._dependency_snapshot,
        ):
            preflight = run_training_preflight(
                project_id=project_id,
                config={
                    "base_model": "microsoft/phi-2",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "adapter_id": "vision-language-pair",
                    "training_runtime_id": "auto",
                },
                base_model="microsoft/phi-2",
            )

        self.assertTrue(bool(preflight.get("ok")), preflight)
        warnings = [str(item) for item in list(preflight.get("warnings") or [])]
        self.assertTrue(any("remote media URL reference" in item for item in warnings), warnings)

        summary = dict(preflight.get("capability_summary") or {})
        dataset = dict(summary.get("dataset") or {})
        media_contract = dict(dataset.get("media_contract") or {})
        self.assertEqual(int(media_contract.get("remote_image_refs") or 0), 1)
        self.assertEqual(int(media_contract.get("missing_local_images") or 0), 0)

    def test_preflight_blocks_remote_media_when_require_media_enabled(self):
        project_id = 4504
        self._write_train_rows(
            project_id,
            [
                {
                    "question": "What is shown?",
                    "answer": "A red stop sign.",
                    "image_path": "https://example.com/media/stop-sign.png",
                }
            ],
        )
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            self._dependency_snapshot,
        ):
            preflight = run_training_preflight(
                project_id=project_id,
                config={
                    "base_model": "microsoft/phi-2",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "adapter_id": "vision-language-pair",
                    "training_runtime_id": "auto",
                    "multimodal_require_media": True,
                },
                base_model="microsoft/phi-2",
            )

        self.assertFalse(bool(preflight.get("ok")), preflight)
        errors = [str(item) for item in list(preflight.get("errors") or [])]
        self.assertTrue(any("remote media URL reference" in item for item in errors), errors)

        summary = dict(preflight.get("capability_summary") or {})
        dataset = dict(summary.get("dataset") or {})
        media_contract = dict(dataset.get("media_contract") or {})
        self.assertTrue(bool(media_contract.get("require_media")))
        self.assertEqual(int(media_contract.get("remote_image_refs") or 0), 1)

    def test_preflight_passes_media_contract_when_local_media_is_resolved(self):
        project_id = 4503
        self._touch_media_file(project_id, "images/local-sample.png")
        self._write_train_rows(
            project_id,
            [
                {
                    "question": "Describe the image",
                    "answer": "A small yellow flower.",
                    "image_path": "images/local-sample.png",
                }
            ],
        )
        with patch(
            "app.services.training_preflight_service._dependency_status_snapshot",
            self._dependency_snapshot,
        ):
            preflight = run_training_preflight(
                project_id=project_id,
                config={
                    "base_model": "microsoft/phi-2",
                    "task_type": "causal_lm",
                    "trainer_backend": "hf_trainer",
                    "adapter_id": "vision-language-pair",
                    "training_runtime_id": "auto",
                },
                base_model="microsoft/phi-2",
            )

        summary = dict(preflight.get("capability_summary") or {})
        dataset = dict(summary.get("dataset") or {})
        media_contract = dict(dataset.get("media_contract") or {})
        self.assertTrue(bool(media_contract.get("ok")), media_contract)
        self.assertEqual(int(media_contract.get("resolved_local_images") or 0), 1)
        self.assertEqual(int(media_contract.get("missing_local_images") or 0), 0)
        self.assertEqual(int(media_contract.get("remote_image_refs") or 0), 0)


if __name__ == "__main__":
    unittest.main()
