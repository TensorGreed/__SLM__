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
