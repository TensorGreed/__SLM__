import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Set up PYTHONPATH for testing
import sys
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.config import settings
from app.services.training_runtime_service import _validate_builtin_simulate
from app.services.readiness_service import get_project_readiness
from app.exceptions import StrictExecutionError

class TestSprint1(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.prev_strict = settings.STRICT_EXECUTION_MODE
        self.prev_teacher_key = settings.TEACHER_MODEL_API_KEY
        settings.TEACHER_MODEL_API_KEY = "test-key"
        
    def tearDown(self):
        settings.STRICT_EXECUTION_MODE = self.prev_strict
        settings.TEACHER_MODEL_API_KEY = self.prev_teacher_key

    def test_strict_execution_blocks_training_simulate(self):
        settings.STRICT_EXECUTION_MODE = True
        with self.assertRaises(StrictExecutionError) as cm:
            _validate_builtin_simulate()
        
        self.assertEqual(cm.exception.detail["error_code"], "STRICT_EXECUTION_VIOLATION")
        self.assertEqual(cm.exception.detail["stage"], "training")
        self.assertIn("STRICT_EXECUTION_MODE is enabled", cm.exception.detail["message"])

    @patch("app.database.async_session_factory")
    @patch("shutil.which")
    @patch("subprocess.run")
    async def test_readiness_api(self, mock_run, mock_which, mock_db):
        # Mock project exists
        mock_session = MagicMock()
        mock_db.return_value.__aenter__.return_value = mock_session
        mock_project = MagicMock()
        mock_project.id = 1
        mock_session.get.return_value = mock_project
        
        # Mock GPU check
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "NVIDIA GeForce RTX 4090, 24576"
        
        settings.STRICT_EXECUTION_MODE = False
        readiness = await get_project_readiness(1)
        
        self.assertEqual(readiness["project_id"], 1)
        self.assertEqual(readiness["status"], "pass")
        self.assertEqual(readiness["strict_mode"], False)
        
        # Ensure GPU check passed
        gpu_check = next(c for c in readiness["checks"] if c["id"] == "gpu")
        self.assertEqual(gpu_check["status"], "pass")
        self.assertIn("RTX 4090", gpu_check["message"])

    @patch("app.database.async_session_factory")
    @patch("shutil.which")
    @patch("torch.cuda.is_available")
    async def test_readiness_strict_mode_fail(self, mock_torch, mock_which, mock_db):
        # Mock project exists
        mock_session = MagicMock()
        mock_db.return_value.__aenter__.return_value = mock_session
        mock_project = MagicMock()
        mock_project.id = 1
        mock_session.get.return_value = mock_project
        
        # Mock NO GPU
        mock_which.return_value = None
        mock_torch.return_value = False
        
        # In strict mode, if GPU is missing it should fail
        settings.STRICT_EXECUTION_MODE = True
        readiness = await get_project_readiness(1)
        
        self.assertEqual(readiness["status"], "fail")
        gpu_check = next(c for c in readiness["checks"] if c["id"] == "gpu")
        self.assertEqual(gpu_check["status"], "fail")
        self.assertIn("Blocked by STRICT_EXECUTION_MODE", gpu_check["message"])

if __name__ == "__main__":
    unittest.main()
