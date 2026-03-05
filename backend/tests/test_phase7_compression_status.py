"""Phase 7 tests: compression job report status and path safety."""

import os
import json
import tempfile
import unittest
from pathlib import Path

os.environ["DEBUG"] = "false"

from app.config import settings
from app.services.compression_service import get_compression_job_status


class CompressionStatusTests(unittest.TestCase):
    def test_job_status_running_when_report_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_data_dir = settings.DATA_DIR
            try:
                settings.DATA_DIR = Path(tmp)
                status = get_compression_job_status(12, "quantize_report.json")
                self.assertEqual(status["status"], "running")
            finally:
                settings.DATA_DIR = previous_data_dir

    def test_job_status_completed_when_report_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_data_dir = settings.DATA_DIR
            try:
                settings.DATA_DIR = Path(tmp)
                report_path = (
                    Path(tmp) / "projects" / "21" / "compressed" / "quantize_report.json"
                )
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(
                        {
                            "command": "echo ok",
                            "returncode": 0,
                            "duration_seconds": 0.1,
                            "stdout": "ok\n",
                            "stderr": "",
                        }
                    ),
                    encoding="utf-8",
                )
                status = get_compression_job_status(21, str(report_path))
                self.assertEqual(status["status"], "completed")
                self.assertEqual(status["returncode"], 0)
            finally:
                settings.DATA_DIR = previous_data_dir

    def test_job_status_rejects_paths_outside_project_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_data_dir = settings.DATA_DIR
            try:
                settings.DATA_DIR = Path(tmp)
                outside = Path(tmp) / "outside_report.json"
                outside.write_text("{}", encoding="utf-8")
                with self.assertRaises(ValueError):
                    get_compression_job_status(44, str(outside))
            finally:
                settings.DATA_DIR = previous_data_dir

    def test_job_status_includes_benchmark_payload_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_data_dir = settings.DATA_DIR
            try:
                settings.DATA_DIR = Path(tmp)
                report_path = (
                    Path(tmp) / "projects" / "55" / "compressed" / "benchmark_report.json"
                )
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(
                        {
                            "command": "python benchmark.py",
                            "returncode": 0,
                            "duration_seconds": 12.4,
                            "stdout": "done",
                            "stderr": "",
                            "benchmark_report_path": "/tmp/benchmark_results.json",
                            "benchmark": {
                                "runtime": {"engine": "transformers"},
                                "metrics": {"token_throughput_tps": 130.5},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                status = get_compression_job_status(55, str(report_path))
                self.assertEqual(status["status"], "completed")
                self.assertEqual(status["benchmark_report_path"], "/tmp/benchmark_results.json")
                self.assertEqual(status["benchmark"]["runtime"]["engine"], "transformers")
            finally:
                settings.DATA_DIR = previous_data_dir


if __name__ == "__main__":
    unittest.main()
