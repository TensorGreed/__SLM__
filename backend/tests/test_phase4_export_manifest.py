"""Phase 4 tests: export artifact manifest helpers."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.models.export import ExportFormat
from app.services.export_service import (
    _artifact_manifest_entry,
    _generate_dockerfile,
    _matches_quantization,
    _resolve_source_model_files,
    _sha256_file,
)


class ExportManifestTests(unittest.TestCase):
    def test_sha256_file_and_artifact_manifest_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "serve.py"
            payload = b"print('hello')\n"
            artifact.write_bytes(payload)

            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(_sha256_file(artifact), expected)

            manifest_entry = _artifact_manifest_entry(artifact, root)
            self.assertEqual(manifest_entry["path"], "serve.py")
            self.assertEqual(manifest_entry["size_bytes"], len(payload))
            self.assertEqual(manifest_entry["sha256"], expected)

    def test_generate_dockerfile_has_runtime_command(self):
        gguf = _generate_dockerfile(ExportFormat.GGUF)
        hf = _generate_dockerfile(ExportFormat.HUGGINGFACE)
        self.assertIn("CMD", gguf)
        self.assertIn("CMD", hf)

    def test_matches_quantization_token_variants(self):
        self.assertTrue(_matches_quantization("model_q4_k_m.gguf", "4bit"))
        self.assertTrue(_matches_quantization("quantized-int8.onnx", "int8"))
        self.assertFalse(_matches_quantization("model_q4.gguf", "8bit"))

    def test_resolve_source_model_files_prefers_experiment_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "experiments" / "exp1" / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "weights.safetensors").write_bytes(b"abc")

            class DummyExperiment:
                output_dir = str(model_dir.parent)

            source, files, source_root = _resolve_source_model_files(
                project_id=9999,
                experiment=DummyExperiment(),
                export_format=ExportFormat.HUGGINGFACE,
                quantization=None,
            )
            self.assertEqual(source, "experiment_model_dir")
            self.assertEqual(source_root, model_dir)
            self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
