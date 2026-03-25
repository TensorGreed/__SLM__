"""Phase 40 tests: config bootstrap is resilient to malformed DEBUG env values."""

from __future__ import annotations

import os
import unittest
import warnings
from unittest.mock import patch

from pydantic import ValidationError

from app.config import Settings


class Phase40ConfigBootstrapBoolTests(unittest.TestCase):
    def test_malformed_debug_env_falls_back_without_crash(self):
        with patch.dict(os.environ, {"DEBUG": "release"}, clear=False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg = Settings(_env_file=None)

        self.assertFalse(bool(cfg.DEBUG))
        warning_messages = [str(item.message) for item in caught]
        self.assertTrue(
            any("Invalid DEBUG value" in message for message in warning_messages),
            warning_messages,
        )

    def test_valid_debug_env_values_remain_deterministic(self):
        with patch.dict(os.environ, {"DEBUG": "true"}, clear=False):
            cfg_true = Settings(_env_file=None)
        self.assertTrue(bool(cfg_true.DEBUG))

        with patch.dict(os.environ, {"DEBUG": "false"}, clear=False):
            cfg_false = Settings(_env_file=None)
        self.assertFalse(bool(cfg_false.DEBUG))

    def test_other_bool_fields_remain_strict(self):
        with patch.dict(
            os.environ,
            {
                "DEBUG": "false",
                "ALLOW_SIMULATED_TRAINING": "release",
            },
            clear=False,
        ):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)


if __name__ == "__main__":
    unittest.main()

