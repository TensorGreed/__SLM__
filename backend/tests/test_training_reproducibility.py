"""Training reproducibility: one config + one dataset → the same weights.

``set_seed`` alone left one GPU source of run-to-run drift: the SDPA
attention backward (memory-efficient / flash) sums with atomics by default.
``train.py`` pins deterministic kernels (``enable_deterministic_training``,
config ``deterministic``, default on) and writes a ``reproducibility`` block
(seed, input-file hashes, ``weights_sha256``) into every training report, so
two runs can be compared without reloading either model.

The pure tests run everywhere with a fake torch. ``RealTrainingReproducibilityTests``
fine-tunes SmolLM2-135M twice through ``run_training`` and asserts
byte-identical weights; it needs a GPU or ``BREWSLM_REAL_TRAINING=1`` and the
cached model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.schemas.training import TrainingConfig
from scripts.train import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    _is_determinism_error,
    enable_deterministic_training,
    reproducibility_block,
    weights_sha256,
)

REAL_MODEL = os.environ.get("BREWSLM_REAL_TRAIN_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")


class _FakeTorch:
    def __init__(self):
        self.calls: list[bool] = []
        self.backends = mock.Mock()
        self.backends.cudnn.deterministic = False
        self.backends.cudnn.benchmark = True

    def use_deterministic_algorithms(self, mode: bool) -> None:
        self.calls.append(mode)


class DeterminismSwitchTests(unittest.TestCase):
    def test_on_pins_kernels_and_cublas_workspace(self):
        torch = _FakeTorch()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            block = enable_deterministic_training(torch, True)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], CUBLAS_DETERMINISTIC_WORKSPACE)
        self.assertEqual(torch.calls, [True])
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertEqual(block["deterministic"], True)
        self.assertEqual(block["cublas_workspace_config"], CUBLAS_DETERMINISTIC_WORKSPACE)

    def test_existing_workspace_setting_is_kept(self):
        with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            block = enable_deterministic_training(_FakeTorch(), True)
        self.assertEqual(block["cublas_workspace_config"], ":16:8")

    def test_off_switches_the_global_back_off(self):
        # use_deterministic_algorithms is process-wide; the unpinned retry
        # after a refusal must clear it.
        torch = _FakeTorch()
        self.assertEqual(enable_deterministic_training(torch, False), {"deterministic": False})
        self.assertEqual(torch.calls, [False])

    def test_schema_default_is_on(self):
        self.assertTrue(TrainingConfig(base_model="x").deterministic)
        self.assertFalse(TrainingConfig(base_model="x", deterministic=False).deterministic)

    def test_determinism_refusal_is_recognised(self):
        self.assertTrue(_is_determinism_error(RuntimeError(
            "scatter_add_cuda_kernel does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True)'."
        )))
        self.assertFalse(_is_determinism_error(RuntimeError("CUDA out of memory")))
        self.assertFalse(_is_determinism_error(ValueError("deterministic")))


class DeterminismFallbackTests(unittest.TestCase):
    def test_refused_op_retries_once_unpinned_and_says_so(self):
        from scripts import train

        seen: list[object] = []

        def attempt(args, *, config, retry_history, **kwargs):  # noqa: ANN001
            seen.append(config.get("deterministic", True))
            if config.get("deterministic", True):
                raise RuntimeError("index_add_cuda_ does not have a deterministic implementation")
            return {"ok": True, "retry_history": list(retry_history)}

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "training_config.json"
            cfg.write_text(json.dumps({"auto_oom_retry": False}))
            train_file = Path(tmp) / "train.jsonl"
            train_file.write_text("{}\n")
            args = argparse.Namespace(output=str(Path(tmp) / "out"), config=str(cfg),
                                      train_file=str(train_file), val_file=None, data_dir=tmp,
                                      project=0, experiment=0)
            with mock.patch.object(train, "_run_training_attempt", side_effect=attempt):
                report = train.run_training(args)
        self.assertEqual(seen, [True, False])
        self.assertTrue(report["ok"])
        (item,) = report["retry_history"]
        self.assertEqual(item["changes"], [{"field": "deterministic", "from": True, "to": False}])
        self.assertIn("deterministic kernels unavailable", item["reason"])


class WeightsFingerprintTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _model(self, name: str, weights: bytes) -> Path:
        model = self.root / name
        model.mkdir()
        (model / "adapter_model.safetensors").write_bytes(weights)
        (model / "adapter_config.json").write_text("{}")
        return model

    def test_same_bytes_same_hash_different_bytes_different_hash(self):
        a = self._model("a", b"\x00\x01")
        b = self._model("b", b"\x00\x01")
        c = self._model("c", b"\x00\x02")
        self.assertIsNotNone(weights_sha256(a))
        self.assertEqual(weights_sha256(a), weights_sha256(b))
        self.assertNotEqual(weights_sha256(a), weights_sha256(c))

    def test_ignores_non_weight_files_and_training_args(self):
        a = self._model("a", b"w")
        b = self._model("b", b"w")
        (b / "training_args.bin").write_bytes(b"pickled args with a timestamp")
        (b / "README.md").write_text("card")
        self.assertEqual(weights_sha256(a), weights_sha256(b))

    def test_no_weights_is_none(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.assertIsNone(weights_sha256(empty))
        self.assertIsNone(weights_sha256(self.root / "missing"))

    def test_block_carries_inputs_and_weights(self):
        model = self._model("m", b"w")
        train = self.root / "train.jsonl"
        train.write_text('{"q": 1}\n')
        block = reproducibility_block(
            seed=42, determinism={"deterministic": True}, train_file=train, val_file=None, model_dir=model,
        )
        self.assertEqual(block["seed"], 42)
        self.assertTrue(block["deterministic"])
        self.assertEqual(len(block["train_file_sha256"]), 64)
        self.assertIsNone(block["val_file_sha256"])
        self.assertEqual(block["weights_sha256"], weights_sha256(model))


def _real_training_skip_reason() -> str | None:
    if os.environ.get("BREWSLM_SKIP_REAL_TRAINING"):
        return "BREWSLM_SKIP_REAL_TRAINING set"
    if not os.environ.get("BREWSLM_REAL_TRAINING"):
        try:
            import torch

            has_gpu = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            has_gpu = False
        if not has_gpu:
            return "needs a GPU or BREWSLM_REAL_TRAINING=1"
    try:
        import peft  # noqa: F401
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return f"ML deps unavailable: {exc}"
    try:
        AutoTokenizer.from_pretrained(REAL_MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        return f"{REAL_MODEL} not in the local HF cache"
    return None


_SKIP_REASON = _real_training_skip_reason()


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class RealTrainingReproducibilityTests(unittest.TestCase):
    """Two LoRA fine-tunes of SmolLM2 on the same rows and config must save
    byte-identical weights. Each run is a fresh subprocess-free call into
    ``run_training`` with its own output dir."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        rng = random.Random(11)
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        teams = ["Billing", "Shipping", "Accounts", "Returns"]
        for name, n in (("train", 48), ("val", 6)):
            with open(root / f"{name}.jsonl", "w", encoding="utf-8") as fh:
                for _ in range(n):
                    order = rng.randint(1000, 9999)
                    team = rng.choice(teams)
                    fh.write(json.dumps({
                        "question": f"Order {order} needs help from {team.lower()}.",
                        "answer": f"Ticket ORD-{order}: route to {team}.",
                    }) + "\n")
        config = {
            "training_mode": "sft",
            "task_type": "causal_lm",
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 5e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "max_seq_length": 128,
            "flash_attention": False,
            "observability_enabled": False,
            "save_steps": 10_000,
            "eval_steps": 10_000,
            "auto_oom_retry": False,
        }
        (root / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
        from scripts.train import run_training

        cls.reports = []
        for run in ("a", "b"):
            args = argparse.Namespace(
                project=0,
                experiment=0,
                output=str(root / run),
                base_model=REAL_MODEL,
                config=str(root / "training_config.json"),
                train_file=str(root / "train.jsonl"),
                val_file=str(root / "val.jsonl"),
                data_dir=str(root),
                max_train_samples=0,
                max_eval_samples=0,
                seed=42,
            )
            cls.reports.append(run_training(args))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_two_runs_save_identical_weights(self):
        first, second = (r["reproducibility"] for r in self.reports)
        self.assertTrue(first["deterministic"])
        self.assertIsNotNone(first["weights_sha256"])
        self.assertEqual(first["train_file_sha256"], second["train_file_sha256"])
        self.assertEqual(first["weights_sha256"], second["weights_sha256"])
        self.assertEqual(self.reports[0]["final_train_loss"], self.reports[1]["final_train_loss"])
        self.assertEqual(self.reports[0]["retry_history"], [])


if __name__ == "__main__":
    unittest.main()
