"""Phase 30 tests: model introspection service foundations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.services.model_introspection_service import introspect_hf_model


class Phase30ModelIntrospectionTests(unittest.TestCase):
    def test_local_config_introspection_detects_architecture_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model-a"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text(
                """{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "vocab_size": 32000,
  "max_position_embeddings": 8192
}
""",
                encoding="utf-8",
            )
            payload = introspect_hf_model(
                model_id=model_dir.as_posix(),
                allow_network=False,
            )

        self.assertTrue(bool(payload.get("resolved")))
        self.assertEqual(payload.get("source"), "local_config")
        self.assertEqual(payload.get("architecture"), "causal_lm")
        self.assertEqual(int(payload.get("context_length") or 0), 8192)
        self.assertTrue(float(payload.get("params_estimate_b") or 0.0) > 0.0)
        memory = dict(payload.get("memory_profile") or {})
        self.assertTrue(float(memory.get("estimated_min_vram_gb") or 0.0) > 0.0)

    def test_encoder_decoder_config_maps_to_seq2seq(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model-b"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text(
                """{
  "model_type": "t5",
  "is_encoder_decoder": true,
  "d_model": 1024,
  "hidden_size": 1024,
  "num_hidden_layers": 12,
  "vocab_size": 32128,
  "n_positions": 2048
}
""",
                encoding="utf-8",
            )
            payload = introspect_hf_model(
                model_id=model_dir.as_posix(),
                allow_network=False,
            )

        self.assertTrue(bool(payload.get("resolved")))
        self.assertEqual(payload.get("architecture"), "seq2seq")
        self.assertEqual(int(payload.get("context_length") or 0), 2048)


if __name__ == "__main__":
    unittest.main()

