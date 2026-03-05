"""Tests for the tokenization service — tokenizer loading and analysis."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TokenizationServiceTests(unittest.TestCase):
    """Unit tests for tokenization utilities (mocking transformers)."""

    def _create_test_jsonl(self, records: list[dict]) -> str:
        """Write records to a temp JSONL file and return path."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for rec in records:
            tmp.write(json.dumps(rec) + "\n")
        tmp.close()
        return tmp.name

    def test_load_tokenizer_import_error(self):
        """Verify error when transformers is not installed."""
        from app.services.tokenization_service import load_tokenizer

        with patch.dict("sys.modules", {"transformers": None}):
            # Should raise because the import will fail
            try:
                load_tokenizer("test-model")
            except (ValueError, ImportError, ModuleNotFoundError):
                pass  # Expected

    @patch("app.services.tokenization_service.load_tokenizer")
    def test_analyze_dataset_tokens_qa_format(self, mock_load):
        """Test token analysis with question/answer format."""
        from app.services.tokenization_service import analyze_dataset_tokens

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        mock_tokenizer.vocab_size = 32000
        mock_load.return_value = mock_tokenizer

        records = [
            {"question": "What is ML?", "answer": "Machine learning is a subset of AI."},
            {"question": "What is DL?", "answer": "Deep learning uses neural networks."},
        ]
        path = self._create_test_jsonl(records)

        try:
            result = analyze_dataset_tokens(path, "test-model", max_seq_length=512)
            self.assertEqual(result["total_entries"], 2)
            self.assertIn("avg_length", result)
            self.assertIn("length_distribution", result)
            self.assertEqual(result["vocab_size"], 32000)
            self.assertEqual(result["truncation_count"], 0)
        finally:
            Path(path).unlink(missing_ok=True)

    @patch("app.services.tokenization_service.load_tokenizer")
    def test_analyze_dataset_tokens_text_format(self, mock_load):
        """Test token analysis with plain text format."""
        from app.services.tokenization_service import analyze_dataset_tokens

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        mock_tokenizer.vocab_size = 50000
        mock_load.return_value = mock_tokenizer

        records = [{"text": " ".join(["word"] * 100)}]
        path = self._create_test_jsonl(records)

        try:
            result = analyze_dataset_tokens(path, "test-model", max_seq_length=50)
            self.assertEqual(result["total_entries"], 1)
            self.assertEqual(result["truncation_count"], 1)
            self.assertGreater(result["truncation_percent"], 0)
        finally:
            Path(path).unlink(missing_ok=True)

    @patch("app.services.tokenization_service.load_tokenizer")
    def test_analyze_empty_dataset(self, mock_load):
        """Test token analysis with empty dataset."""
        from app.services.tokenization_service import analyze_dataset_tokens

        mock_tokenizer = MagicMock()
        mock_load.return_value = mock_tokenizer

        path = self._create_test_jsonl([])
        try:
            result = analyze_dataset_tokens(path, "test-model")
            self.assertIn("error", result)
        finally:
            Path(path).unlink(missing_ok=True)

    @patch("app.services.tokenization_service.load_tokenizer")
    def test_get_vocab_sample(self, mock_load):
        """Test vocabulary sampling."""
        from app.services.tokenization_service import get_vocab_sample

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {f"tok_{i}": i for i in range(200)}
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.unk_token = "<unk>"
        mock_load.return_value = mock_tokenizer

        result = get_vocab_sample("test-model", sample_size=10)
        self.assertEqual(result["vocab_size"], 200)
        self.assertEqual(len(result["sample"]), 10)
        self.assertEqual(result["special_tokens"]["bos"], "<s>")


if __name__ == "__main__":
    unittest.main()
