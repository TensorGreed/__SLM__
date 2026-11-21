"""Optimizer candidate scan ignores run dirs moved aside (`.stale.*`) or backed up (`.bak.*`)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from app.services import export_service  # noqa: E402


def _make_model(root: Path, name: str) -> None:
    model = root / "experiments" / name / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"w")


class OptimizerCandidateScanTests(unittest.TestCase):
    def test_stale_and_bak_dirs_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_model(root, "7")
            _make_model(root, "7.stale.20260101T000000Z")
            _make_model(root, "8.bak.20260101T000000Z")
            with mock.patch.object(export_service, "_project_dir", return_value=root):
                found = export_service._discover_optimizer_candidates(project_id=1)
            joined = " ".join(str(c.get("model_path") or c.get("path") or c) for c in found)
            self.assertNotIn(".stale.", joined)
            self.assertNotIn(".bak.", joined)
            self.assertEqual(sum(1 for c in found if "/experiments/7/" in str(c)), 1, found)


if __name__ == "__main__":
    unittest.main()
