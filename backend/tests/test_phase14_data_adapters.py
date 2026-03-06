"""Phase 14 tests: data adapter SDK registry, auto-detect, and plugin loading."""

import os
import unittest

os.environ["DEBUG"] = "false"

from app.config import settings
from app.services.data_adapter_service import (
    clear_plugin_data_adapters,
    list_data_adapter_catalog,
    load_data_adapter_plugins_from_settings,
    preview_data_adapter,
)


class DataAdapterServiceTests(unittest.TestCase):
    def setUp(self):
        clear_plugin_data_adapters()

    def tearDown(self):
        clear_plugin_data_adapters()

    def test_catalog_includes_builtin_and_auto(self):
        catalog = list_data_adapter_catalog()
        self.assertEqual(catalog["default_adapter"], "default-canonical")
        self.assertIn("auto", catalog["adapters"])
        self.assertIn("default-canonical", catalog["adapters"])
        self.assertIn("qa-pair", catalog["adapters"])
        self.assertIn("seq2seq-pair", catalog["adapters"])
        self.assertIn("classification-label", catalog["adapters"])

    def test_preview_auto_detects_qa_pair(self):
        rows = [
            {"question": "What is tort law?", "answer": "Civil wrong law."},
            {"prompt": "What is negligence?", "response": "Failure to take proper care."},
        ]
        result = preview_data_adapter(rows, adapter_id="auto")
        self.assertEqual(result["resolved_adapter_id"], "qa-pair")
        self.assertEqual(result["mapped_records"], 2)
        self.assertEqual(result["dropped_records"], 0)

    def test_preview_auto_detects_classification(self):
        rows = [
            {"text": "The contract is enforceable.", "label": "legal"},
            {"content": "2 + 2 equals 4", "category": "math"},
        ]
        result = preview_data_adapter(rows, adapter_id="auto")
        self.assertEqual(result["resolved_adapter_id"], "classification-label")
        self.assertEqual(result["mapped_records"], 2)
        self.assertEqual(result["dropped_records"], 0)
        self.assertIn("label_distribution", result["validation_report"])

    def test_plugin_load_from_settings_registers_example_adapter(self):
        previous_modules = settings.DATA_ADAPTER_PLUGIN_MODULES
        try:
            settings.DATA_ADAPTER_PLUGIN_MODULES = ["app.plugins.data_adapters.example_adapters"]
            load_result = load_data_adapter_plugins_from_settings(force_reload=True)
            self.assertIn("app.plugins.data_adapters.example_adapters", load_result["loaded_modules"])
            catalog = list_data_adapter_catalog()
            self.assertIn("instruction-response", catalog["adapters"])
        finally:
            settings.DATA_ADAPTER_PLUGIN_MODULES = previous_modules


if __name__ == "__main__":
    unittest.main()

