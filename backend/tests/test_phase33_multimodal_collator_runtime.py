"""Phase 33 tests: mocked multimodal runtime path through train.py collator."""

from __future__ import annotations

import argparse
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from scripts import train as train_script


class _FakeDataset:
    def __init__(self, rows: list[dict]):
        self._rows = [dict(item) for item in list(rows or [])]
        self.column_names = sorted({key for row in self._rows for key in row.keys()})

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [row.get(item) for row in self._rows]
        return dict(self._rows[item])

    def select(self, indices) -> "_FakeDataset":
        return _FakeDataset([self._rows[int(i)] for i in list(indices)])

    def map(self, fn, batched: bool = False, remove_columns=None):  # noqa: ANN001
        if batched:
            payload: dict[str, list] = {}
            for key in self.column_names:
                payload[key] = [row.get(key) for row in self._rows]
            mapped = fn(payload) or {}
            if not isinstance(mapped, dict):
                return _FakeDataset([])
            lengths = [len(value) for value in mapped.values() if isinstance(value, list)]
            total = max(lengths) if lengths else 0
            rows: list[dict] = []
            for idx in range(total):
                row: dict[str, object] = {}
                for key, values in mapped.items():
                    if isinstance(values, list) and idx < len(values):
                        row[key] = values[idx]
                rows.append(row)
            return _FakeDataset(rows)
        rows = []
        for row in self._rows:
            mapped = fn(dict(row)) or {}
            if isinstance(mapped, dict):
                rows.append(dict(mapped))
        return _FakeDataset(rows)

    def filter(self, fn) -> "_FakeDataset":  # noqa: ANN001
        rows = []
        for row in self._rows:
            keep = bool(fn(dict(row)))
            if keep:
                rows.append(dict(row))
        return _FakeDataset(rows)


class _FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.pad_token_id = 0

    def __len__(self) -> int:
        return 64

    def add_special_tokens(self, payload: dict[str, str]) -> int:
        token = str(payload.get("pad_token") or "").strip()
        if token:
            self.pad_token = token
            self.pad_token_id = 0
            return 1
        return 0

    def __call__(self, text, *args, **kwargs):  # noqa: ANN001
        return_tensors = kwargs.get("return_tensors")
        text_target = kwargs.get("text_target")
        if isinstance(text, str):
            batch = [text]
        else:
            batch = [str(item) for item in list(text or [])]
        batch_size = max(1, len(batch))
        seq_len = 8
        ids = torch.ones((batch_size, seq_len), dtype=torch.long)
        attention = torch.ones((batch_size, seq_len), dtype=torch.long)
        payload = {
            "input_ids": ids if return_tensors == "pt" else ids.tolist(),
            "attention_mask": attention if return_tensors == "pt" else attention.tolist(),
        }
        if text_target is not None:
            labels = torch.full((batch_size, seq_len), 2, dtype=torch.long)
            payload["labels"] = labels if return_tensors == "pt" else labels.tolist()
        return payload

    class _TargetCtx:
        def __init__(self, tokenizer: "_FakeTokenizer") -> None:
            self.tokenizer = tokenizer

        def __enter__(self) -> "_FakeTokenizer":
            return self.tokenizer

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            _ = exc_type, exc, tb
            return False

    def as_target_tokenizer(self) -> "_FakeTokenizer._TargetCtx":
        return _FakeTokenizer._TargetCtx(self)

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:  # noqa: ANN001
        _ = token_ids, skip_special_tokens
        return "tok"

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ANN001
        _ = args, kwargs
        return cls()

    def __call__(self, *args, **kwargs):  # noqa: ANN001
        _ = args
        images = kwargs.get("images")
        if images is not None:
            batch_size = len(list(images))
            return {
                "pixel_values": torch.ones((batch_size, 3, 2, 2), dtype=torch.float32),
            }
        audios = kwargs.get("audios") or kwargs.get("audio") or kwargs.get("raw_speech") or kwargs.get("speech")
        if audios is not None:
            batch_size = len(list(audios))
            return {
                "input_features": torch.ones((batch_size, 80, 4), dtype=torch.float32),
            }
        return {}


class _FakeEmbeddings:
    num_embeddings = 32


class _FakeModelOutput:
    def __init__(self, loss: torch.Tensor, logits: torch.Tensor, attentions=None):
        self.loss = loss
        self.logits = logits
        self.attentions = attentions

    def get(self, key: str, default=None):  # noqa: ANN001
        return {
            "loss": self.loss,
            "logits": self.logits,
            "attentions": self.attentions,
        }.get(key, default)


class _FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(pad_token_id=None, use_cache=True)
        self._emb = _FakeEmbeddings()
        self._param = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32), requires_grad=True)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ANN001
        _ = args, kwargs
        return cls()

    def get_input_embeddings(self):
        return self._emb

    def resize_token_embeddings(self, size: int) -> None:
        self._emb.num_embeddings = int(size)

    def gradient_checkpointing_enable(self) -> None:
        return None

    def parameters(self):
        return iter([self._param])

    def named_parameters(self):
        return [("transformer.layers.0.weight", self._param)]

    def forward(
        self,
        input_ids=None,  # noqa: ANN001
        attention_mask=None,  # noqa: ANN001
        labels=None,  # noqa: ANN001
        pixel_values=None,  # noqa: ANN001
        input_features=None,  # noqa: ANN001
        input_values=None,  # noqa: ANN001
        output_attentions: bool = False,
        **kwargs,
    ):
        _ = attention_mask, labels, pixel_values, input_features, input_values, kwargs
        batch = int(getattr(input_ids, "shape", [1, 8])[0]) if input_ids is not None else 1
        seq_len = int(getattr(input_ids, "shape", [1, 8])[1]) if input_ids is not None else 8
        logits = torch.zeros((batch, seq_len, 16), dtype=torch.float32)
        attentions = (torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32),) if output_attentions else None
        return _FakeModelOutput(loss=torch.tensor(0.5), logits=logits, attentions=attentions)

    __call__ = forward


class _FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ANN001
        _ = args, kwargs
        return _FakeTokenizer()


class _FakeTrainerCallback:
    pass


class _FakeTrainingArguments:
    def __init__(self, output_dir: str, overwrite_output_dir: bool = True, **kwargs):
        self.output_dir = output_dir
        self.overwrite_output_dir = bool(overwrite_output_dir)
        for key, value in kwargs.items():
            setattr(self, str(key), value)


class _FakeCollator:
    def __init__(self, *args, **kwargs):  # noqa: ANN001
        _ = args, kwargs

    def __call__(self, features):  # noqa: ANN001
        return dict(features[0]) if features else {}


class _FakeTrainer:
    observed_batch_keys: list[str] = []

    def __init__(
        self,
        model=None,  # noqa: ANN001
        args=None,  # noqa: ANN001
        tokenizer=None,  # noqa: ANN001
        train_dataset=None,  # noqa: ANN001
        eval_dataset=None,  # noqa: ANN001
        data_collator=None,  # noqa: ANN001
        compute_metrics=None,  # noqa: ANN001
        **kwargs,
    ):
        _ = model, args, tokenizer, eval_dataset, compute_metrics, kwargs
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.state = types.SimpleNamespace(
            log_history=[{"loss": 0.42, "step": 1, "epoch": 1.0}],
            global_step=1,
            epoch=1.0,
        )

    def add_callback(self, callback) -> None:  # noqa: ANN001
        _ = callback

    def train(self, resume_from_checkpoint: str | None = None):
        _ = resume_from_checkpoint
        if self.data_collator is not None and self.train_dataset is not None and len(self.train_dataset) > 0:
            batch = self.data_collator([self.train_dataset[0]])
            _FakeTrainer.observed_batch_keys = sorted(str(key) for key in batch.keys())
        return types.SimpleNamespace(metrics={"train_runtime": 0.01})

    def evaluate(self) -> dict[str, float]:
        return {"eval_loss": 0.4}

    def save_model(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "pytorch_model.bin").write_text("fake", encoding="utf-8")


def _fake_load_dataset(_format: str, data_files: dict[str, str]):
    payload = {}
    for split, path in dict(data_files or {}).items():
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if not token:
                    continue
                rows.append(json.loads(token))
        payload[str(split)] = _FakeDataset(rows)
    return payload


class Phase33MultimodalCollatorRuntimeTests(unittest.TestCase):
    def test_run_training_attempt_uses_multimodal_collator_and_injects_pixel_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "prepared"
            image_dir = train_dir / "images"
            train_dir.mkdir(parents=True, exist_ok=True)
            image_dir.mkdir(parents=True, exist_ok=True)

            image_path = image_dir / "sample.png"
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)
            train_file = train_dir / "train.jsonl"
            row = {"text": "Describe the image", "image_path": "images/sample.png"}
            train_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            output_dir = root / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_dir = output_dir / "model"
            config_path = output_dir / "training_config.json"
            config_path.write_text("{}", encoding="utf-8")

            args = argparse.Namespace(
                project=1,
                experiment=1,
                output=str(output_dir),
                base_model="acme/fake-vlm",
                config=str(config_path),
                train_file=str(train_file),
                val_file="",
                data_dir=str(root),
                max_train_samples=0,
                max_eval_samples=0,
                seed=42,
            )

            config = {
                "task_type": "causal_lm",
                "training_mode": "sft",
                "trainer_backend": "hf_trainer",
                "num_epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "save_steps": 1,
                "eval_steps": 1,
                "multimodal_native_beta": True,
                "multimodal_media_loading": True,
            }

            _FakeTrainer.observed_batch_keys = []
            deps = {
                "torch": torch,
                "transformers_module": types.SimpleNamespace(
                    __version__="fake-transformers",
                    AutoProcessor=_FakeProcessor,
                    AutoModelForVision2Seq=None,
                    AutoModelForSpeechSeq2Seq=None,
                ),
                "load_dataset": _fake_load_dataset,
                "AutoModelForCausalLM": _FakeModel,
                "AutoModelForSeq2SeqLM": _FakeModel,
                "AutoModelForSequenceClassification": _FakeModel,
                "AutoTokenizer": _FakeAutoTokenizer,
                "DataCollatorForLanguageModeling": _FakeCollator,
                "DataCollatorForSeq2Seq": _FakeCollator,
                "DataCollatorWithPadding": _FakeCollator,
                "Trainer": _FakeTrainer,
                "TrainerCallback": _FakeTrainerCallback,
                "TrainingArguments": _FakeTrainingArguments,
                "set_seed": lambda *_args, **_kwargs: None,
            }

            with patch.object(
                train_script,
                "_load_training_runtime_dependencies",
                return_value=deps,
            ), patch.object(
                train_script,
                "_collect_runtime_environment",
                return_value={
                    "cuda_available": False,
                    "bf16_supported": False,
                    "device_name": None,
                    "device_capability": None,
                },
            ):
                report = train_script._run_training_attempt(
                    args,
                    config=config,
                    output_dir=output_dir,
                    model_dir=model_dir,
                    config_path=config_path,
                    train_file=train_file,
                    val_file=None,
                    started_at=train_script.utcnow(),
                    attempt_index=0,
                    total_attempts=1,
                    retry_history=[],
                )

        runtime = dict(report.get("runtime_environment") or {})
        self.assertTrue(bool(runtime.get("multimodal_adapter_collator")))
        self.assertTrue(bool(runtime.get("multimodal_processor_loaded")))
        self.assertTrue(bool(runtime.get("multimodal_accepts_pixel_values")))
        self.assertEqual(str(runtime.get("train_modality")), "vision_language")
        stats = dict(runtime.get("multimodal_collator_stats") or {})
        self.assertGreaterEqual(int(stats.get("total_batches") or 0), 1)
        self.assertGreaterEqual(int(stats.get("vision_batches") or 0), 1)
        self.assertEqual(int(stats.get("media_fallback_batches") or 0), 0)
        self.assertIn("pixel_values", _FakeTrainer.observed_batch_keys)

    def test_run_training_attempt_fails_when_require_media_and_asset_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "prepared"
            train_dir.mkdir(parents=True, exist_ok=True)

            train_file = train_dir / "train.jsonl"
            row = {"text": "Describe the image", "image_path": "images/missing.png"}
            train_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            output_dir = root / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_dir = output_dir / "model"
            config_path = output_dir / "training_config.json"
            config_path.write_text("{}", encoding="utf-8")

            args = argparse.Namespace(
                project=1,
                experiment=1,
                output=str(output_dir),
                base_model="acme/fake-vlm",
                config=str(config_path),
                train_file=str(train_file),
                val_file="",
                data_dir=str(root),
                max_train_samples=0,
                max_eval_samples=0,
                seed=42,
            )

            config = {
                "task_type": "causal_lm",
                "training_mode": "sft",
                "trainer_backend": "hf_trainer",
                "num_epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "save_steps": 1,
                "eval_steps": 1,
                "multimodal_native_beta": True,
                "multimodal_media_loading": True,
                "multimodal_require_media": True,
            }

            deps = {
                "torch": torch,
                "transformers_module": types.SimpleNamespace(
                    __version__="fake-transformers",
                    AutoProcessor=_FakeProcessor,
                    AutoModelForVision2Seq=None,
                    AutoModelForSpeechSeq2Seq=None,
                ),
                "load_dataset": _fake_load_dataset,
                "AutoModelForCausalLM": _FakeModel,
                "AutoModelForSeq2SeqLM": _FakeModel,
                "AutoModelForSequenceClassification": _FakeModel,
                "AutoTokenizer": _FakeAutoTokenizer,
                "DataCollatorForLanguageModeling": _FakeCollator,
                "DataCollatorForSeq2Seq": _FakeCollator,
                "DataCollatorWithPadding": _FakeCollator,
                "Trainer": _FakeTrainer,
                "TrainerCallback": _FakeTrainerCallback,
                "TrainingArguments": _FakeTrainingArguments,
                "set_seed": lambda *_args, **_kwargs: None,
            }

            with patch.object(
                train_script,
                "_load_training_runtime_dependencies",
                return_value=deps,
            ), patch.object(
                train_script,
                "_collect_runtime_environment",
                return_value={
                    "cuda_available": False,
                    "bf16_supported": False,
                    "device_name": None,
                    "device_capability": None,
                },
            ):
                with self.assertRaisesRegex(ValueError, "multimodal_require_media=true"):
                    train_script._run_training_attempt(
                        args,
                        config=config,
                        output_dir=output_dir,
                        model_dir=model_dir,
                        config_path=config_path,
                        train_file=train_file,
                        val_file=None,
                        started_at=train_script.utcnow(),
                        attempt_index=0,
                        total_attempts=1,
                        retry_history=[],
                    )


if __name__ == "__main__":
    unittest.main()
