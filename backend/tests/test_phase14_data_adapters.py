"""Phase 14 tests: data adapter SDK registry, auto-detect, and plugin loading."""

import os
import unittest

os.environ["DEBUG"] = "false"

from app.config import settings
from app.services.data_adapter_service import (
    clear_plugin_data_adapters,
    is_training_task_compatible,
    list_data_adapter_catalog,
    load_data_adapter_plugins_from_settings,
    normalize_task_profile,
    preview_data_adapter,
    resolve_data_adapter_contract,
    resolve_task_profile_for_adapter,
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
        self.assertIn("rag-grounded", catalog["adapters"])
        self.assertIn("seq2seq-pair", catalog["adapters"])
        self.assertIn("structured-extraction", catalog["adapters"])
        self.assertIn("classification-label", catalog["adapters"])
        self.assertIn("chat-messages", catalog["adapters"])
        self.assertIn("tool-call-json", catalog["adapters"])
        self.assertIn("preference-pair", catalog["adapters"])
        self.assertEqual(catalog.get("contract_version"), "slm.data_adapter/v3")
        contract = catalog["adapters"]["qa-pair"].get("contract") or {}
        self.assertEqual(contract.get("version"), "slm.data_adapter/v3")
        self.assertIn("qa", contract.get("task_profiles", []))

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

    def test_preview_with_task_profile_includes_contract_and_profile_resolution(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "What is negligence?"},
                    {"role": "assistant", "content": "Failure to exercise reasonable care."},
                ]
            }
        ]
        result = preview_data_adapter(rows, adapter_id="auto", task_profile="chat_sft")
        self.assertEqual(result["resolved_adapter_id"], "chat-messages")
        self.assertEqual(result["resolved_task_profile"], "chat_sft")
        self.assertTrue(bool(result.get("task_profile_compatible")))
        contract = result.get("adapter_contract") or {}
        self.assertIn("causal_lm", contract.get("preferred_training_tasks", []))

    def test_preview_returns_v3_conformance_and_inferred_profiles(self):
        rows = [
            {
                "context": "Negligence requires duty, breach, causation, and damages.",
                "question": "What are the elements of negligence?",
                "answer": "Duty, breach, causation, and damages.",
            }
        ]
        result = preview_data_adapter(rows, adapter_id="rag-grounded", task_profile="rag_qa")
        self.assertEqual(result["resolved_adapter_id"], "rag-grounded")
        conformance = result.get("conformance_report") or {}
        self.assertTrue(bool(conformance.get("contract_pass")))
        self.assertGreaterEqual(float(conformance.get("mapping_success_rate") or 0.0), 1.0)
        inferred = result.get("inferred_task_profiles") or []
        self.assertIn("rag_qa", inferred)

    def test_preview_surface_auto_fix_suggestions_on_low_mapping(self):
        rows = [
            {"prompt_text": "Define estoppel.", "gold_response": "A legal bar against asserting something."},
            {"prompt_text": "What is mens rea?", "gold_response": "A guilty mental state."},
        ]
        result = preview_data_adapter(rows, adapter_id="qa-pair", task_profile="qa")
        self.assertEqual(result["mapped_records"], 0)
        suggestions = result.get("auto_fix_suggestions") or []
        self.assertGreaterEqual(len(suggestions), 1)
        suggestion_kinds = {str(item.get("kind")) for item in suggestions if isinstance(item, dict)}
        self.assertIn("field_mapping", suggestion_kinds)
        field_mapping_suggestions = [
            item
            for item in suggestions
            if isinstance(item, dict) and item.get("kind") == "field_mapping"
        ]
        self.assertTrue(any("confidence" in item for item in field_mapping_suggestions))
        self.assertTrue(any("suggestion_id" in item for item in field_mapping_suggestions))
        auto_apply = result.get("auto_apply") or {}
        self.assertIn("confidence_threshold", auto_apply)
        self.assertIn("suggested_field_mapping", auto_apply)
        self.assertIsInstance(auto_apply.get("suggested_field_mapping"), dict)

    def test_contract_helpers_handle_profile_and_training_task_compatibility(self):
        self.assertEqual(normalize_task_profile("Causal_LM"), "instruction_sft")
        self.assertEqual(normalize_task_profile("rag"), "rag_qa")
        self.assertEqual(normalize_task_profile("function_calling"), "tool_calling")
        contract = resolve_data_adapter_contract("preference-pair")
        self.assertIn("preference", contract.get("task_profiles", []))
        self.assertEqual(
            resolve_task_profile_for_adapter("qa-pair", requested_task_profile="auto"),
            "qa",
        )
        self.assertTrue(is_training_task_compatible("classification", "classification"))
        self.assertFalse(is_training_task_compatible("classification", "causal_lm"))

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
