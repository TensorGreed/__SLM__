"""Compatibility tests for external training runtime adapters."""

import unittest

from scripts.train import (
    _coerce_trainer_kwargs,
    _coerce_training_arguments_kwargs,
)


class _DummyTrainingArgsV5:
    def __init__(self, output_dir, eval_strategy=None, save_strategy=None):
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy


class _DummyTrainerV5:
    def __init__(self, model, args, train_dataset=None, eval_dataset=None, processing_class=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class


class TrainingRuntimeCompatTests(unittest.TestCase):
    def test_training_arguments_alias_and_drop(self):
        raw = {
            "output_dir": "/tmp/out",
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "overwrite_output_dir": True,
        }
        filtered, dropped = _coerce_training_arguments_kwargs(raw, _DummyTrainingArgsV5)
        self.assertEqual(filtered["eval_strategy"], "steps")
        self.assertNotIn("evaluation_strategy", filtered)
        self.assertNotIn("overwrite_output_dir", filtered)
        self.assertIn("overwrite_output_dir", dropped)

    def test_trainer_alias_tokenizer_to_processing_class(self):
        raw = {
            "model": object(),
            "args": object(),
            "train_dataset": [],
            "eval_dataset": [],
            "tokenizer": object(),
        }
        filtered, dropped = _coerce_trainer_kwargs(raw, _DummyTrainerV5)
        self.assertFalse(dropped)
        self.assertIn("processing_class", filtered)
        self.assertNotIn("tokenizer", filtered)


if __name__ == "__main__":
    unittest.main()
