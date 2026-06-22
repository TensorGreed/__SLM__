"""Tests for natural language to pipeline recipe creation."""

import unittest
from unittest.mock import patch

from app.services.nl2pipeline_service import magic_create_pipeline_recipe


class TestNL2Pipeline(unittest.IsolatedAsyncioTestCase):
    @patch("app.services.nl2pipeline_service.call_teacher_model")
    async def test_magic_create_pipeline_recipe(self, mock_call_teacher):
        mock_call_teacher.return_value = {
            "content": '''```json
{
  "project_name": "Legal Bot",
  "project_description": "Extracts liabilities.",
  "domain_pack_id": "general-pack-v1",
  "adapter_id": "structured-extraction",
  "task_profile": "structured_extraction",
  "base_model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "pipeline_recipe_id": "recipe.pipeline.sft_default"
}
```'''
        }
        
        result = await magic_create_pipeline_recipe("I want to extract liabilities")
        
        self.assertEqual(result["project_name"], "Legal Bot")
        self.assertEqual(result["adapter_id"], "structured-extraction")
        self.assertEqual(result["task_profile"], "structured_extraction")

    @patch("app.services.nl2pipeline_service.call_teacher_model")
    async def test_magic_create_invalid_json(self, mock_call_teacher):
        mock_call_teacher.return_value = {
            "content": "Not JSON"
        }

        with self.assertRaises(ValueError):
            await magic_create_pipeline_recipe("Make it crash")

    def test_fallback_detects_classification(self):
        # Regression: a prompt that says "classification model" must NOT fall
        # through to instruction_sft on the heuristic fallback path.
        from app.services.nl2pipeline_service import _fallback_magic_recommendation
        rec = _fallback_magic_recommendation(
            "I have CSV data for customer transactions and I want a classification model to detect loyalty"
        )
        self.assertEqual(rec["task_profile"], "classification")
        self.assertEqual(rec["adapter_id"], "classification-label")
        self.assertEqual(rec["pipeline_recipe_id"], "recipe.pipeline.lora_fast")
        # Classification defaults to a small base.
        self.assertEqual(rec["base_model_name"], "Qwen/Qwen1.5-1.8B-Chat")

    def test_fallback_classification_keywords_and_vram(self):
        from app.services.nl2pipeline_service import _fallback_magic_recommendation
        for prompt in (
            "categorize incoming emails",
            "sentiment analysis of reviews",
            "intent detection for a chatbot",
        ):
            self.assertEqual(
                _fallback_magic_recommendation(prompt)["task_profile"],
                "classification", prompt,
            )
        # A hardware hint still drives base sizing even on the classification path.
        rec = _fallback_magic_recommendation("Classify tickets into categories for a 4090")
        self.assertEqual(rec["task_profile"], "classification")
        self.assertEqual(rec["base_model_name"], "meta-llama/Meta-Llama-3-8B-Instruct")
