"""Phase 4 tests: export artifact manifest helpers."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.models.export import ExportFormat
from app.services.export_service import (
    _artifact_manifest_entry,
    _generate_dockerfile,
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


if __name__ == "__main__":
    unittest.main()
