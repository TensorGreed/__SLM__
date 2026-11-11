"""train.py creates the CUDA context before the CPU model load (no-op without CUDA)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from scripts.train import _init_cuda_context_early


class InitCudaContextEarlyTests(unittest.TestCase):
    def test_no_cuda_is_a_no_op(self):
        torch_mod = SimpleNamespace(cuda=mock.Mock(), empty=mock.Mock())
        warnings: list[str] = []
        self.assertFalse(_init_cuda_context_early(torch_mod, False, warnings))
        torch_mod.cuda.init.assert_not_called()
        torch_mod.empty.assert_not_called()
        self.assertEqual(warnings, [])

    def test_cuda_creates_context_with_one_tiny_tensor(self):
        torch_mod = SimpleNamespace(cuda=mock.Mock(), empty=mock.Mock())
        warnings: list[str] = []
        self.assertTrue(_init_cuda_context_early(torch_mod, True, warnings))
        torch_mod.cuda.init.assert_called_once()
        torch_mod.empty.assert_called_once_with(1, device="cuda")

    def test_failure_is_a_warning_not_a_crash(self):
        torch_mod = SimpleNamespace(cuda=mock.Mock(), empty=mock.Mock(side_effect=RuntimeError("boom")))
        warnings: list[str] = []
        self.assertFalse(_init_cuda_context_early(torch_mod, True, warnings))
        self.assertTrue(any("early CUDA context init failed" in w for w in warnings))

    def test_called_before_the_model_load(self):
        import inspect

        import scripts.train as train_mod

        src = inspect.getsource(train_mod)
        call = src.index('runtime_environment["cuda_context_before_load"] = _init_cuda_context_early(')
        load = src.index("model, model_kwargs = _load_model_with_dtype_fallback(model_kwargs)")
        self.assertLess(call, load)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
