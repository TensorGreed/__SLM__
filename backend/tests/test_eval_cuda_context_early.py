"""Eval inference creates the CUDA context before the CPU model load (no-op without CUDA)."""

from __future__ import annotations

import inspect
import re
import unittest
from types import SimpleNamespace
from unittest import mock

from app.services import evaluation_service as svc


class EvalCudaContextEarlyTests(unittest.TestCase):
    def test_no_cuda_is_a_no_op(self):
        torch_mod = SimpleNamespace(cuda=mock.Mock(), empty=mock.Mock())
        self.assertFalse(svc._init_cuda_context_before_load(torch_mod, False))
        torch_mod.cuda.init.assert_not_called()
        torch_mod.empty.assert_not_called()

    def test_cuda_creates_context_with_one_tiny_tensor(self):
        torch_mod = SimpleNamespace(cuda=mock.Mock(), empty=mock.Mock())
        self.assertTrue(svc._init_cuda_context_before_load(torch_mod, True))
        torch_mod.cuda.init.assert_called_once()
        torch_mod.empty.assert_called_once_with(1, device="cuda")

    def test_failure_is_swallowed(self):
        torch_mod = SimpleNamespace(cuda=mock.Mock(), empty=mock.Mock(side_effect=RuntimeError("boom")))
        self.assertFalse(svc._init_cuda_context_before_load(torch_mod, True))

    def test_every_generation_loader_inits_before_its_from_pretrained(self):
        src = inspect.getsource(svc)
        calls = [m.start() for m in re.finditer(r"_init_cuda_context_before_load\(torch, use_cuda\)", src)]
        self.assertEqual(len(calls), 5)
        for pos in calls:
            nxt_load = src.index(".from_pretrained(", pos)
            prev_cuda = src.rindex("use_cuda = torch.cuda.is_available()", 0, pos)
            self.assertLess(prev_cuda, pos)
            self.assertLess(pos, nxt_load)
            # the next to("cuda") move comes after the load
            self.assertLess(nxt_load, src.index('"cuda"', pos + 60))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
