"""``scripts/train.py`` is launched as a standalone script by the training
runtime (``python scripts/train.py ...``), not imported from ``backend/``.

Its lazy ``from app.services...`` imports (auto-epochs policy, distillation,
eval handlers) must resolve in that mode. They didn't: only ``scripts/`` was on
sys.path, so every real training run died with "No module named 'app'" while
the test-suite (which imports ``scripts.train`` from ``backend/``) stayed green.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"


class TrainScriptStandaloneTests(unittest.TestCase):
    def test_app_package_is_importable_when_run_as_a_script(self):
        code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('train_standalone', {str(TRAIN_SCRIPT)!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "from app.services.training_epoch_policy import resolve_auto_epochs\n"
            "print('ok')\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        with tempfile.TemporaryDirectory() as cwd:  # not backend/, like the runtime's view of sys.path[0]
            proc = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, cwd=cwd, env=env, timeout=120
            )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("ok", proc.stdout)
