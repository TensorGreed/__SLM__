"""KD loss math + offline trainer compute_loss — Track 1, Epic A, slice 2.

Pure-tensor unit tests (CPU). No transformers / GPU: the trainer's compute_loss
is exercised through a stub Trainer base + a fake model, so the shift / gather /
scatter / loss pipeline is fully covered without the real HF Trainer.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from app.services.distillation.kd_loss import (
    kd_loss,
    scatter_topk_to_logits,
)
from app.services.distillation.kd_trainer import (
    compute_offline_kd_loss,
    make_offline_kd_trainer,
)


class KDLossTests(unittest.TestCase):
    def test_identical_teacher_gives_zero_soft(self):
        logits = torch.tensor([[2.0, 0.0, -1.0]])
        labels = torch.tensor([0])
        comp = kd_loss(logits, logits.clone(), labels, alpha=0.5, temperature=1.0)
        self.assertAlmostEqual(float(comp.soft), 0.0, places=5)
        # total == alpha * hard when soft is zero.
        self.assertAlmostEqual(
            float(comp.total), 0.5 * float(comp.hard), places=5
        )

    def test_hard_term_matches_cross_entropy(self):
        logits = torch.tensor([[2.0, 0.0]])
        labels = torch.tensor([0])
        comp = kd_loss(logits, logits.clone(), labels, alpha=1.0, temperature=1.0)
        # alpha=1 → total == hard == CE; CE = log(1 + e^-2).
        expected_ce = math.log(1.0 + math.exp(-2.0))
        self.assertAlmostEqual(float(comp.hard), expected_ce, places=5)
        self.assertAlmostEqual(float(comp.total), expected_ce, places=5)

    def test_alpha_zero_is_pure_soft(self):
        student = torch.tensor([[0.0, 0.0]])
        teacher = torch.tensor([[3.0, 0.0]])  # different → soft > 0
        labels = torch.tensor([0])
        comp = kd_loss(student, teacher, labels, alpha=0.0, temperature=1.0)
        self.assertGreater(float(comp.soft), 0.0)
        self.assertAlmostEqual(float(comp.total), float(comp.soft), places=5)

    def test_ignore_index_positions_dropped(self):
        # Two positions, second is ignored → loss equals loss of first only.
        student = torch.tensor([[2.0, 0.0], [5.0, 5.0]])
        teacher = student.clone()
        labels_full = torch.tensor([0, -100])
        labels_one = torch.tensor([0])
        full = kd_loss(student, teacher, labels_full, alpha=0.7, temperature=1.5)
        one = kd_loss(student[:1], teacher[:1], labels_one, alpha=0.7, temperature=1.5)
        self.assertAlmostEqual(float(full.total), float(one.total), places=5)

    def test_all_ignored_returns_zero(self):
        student = torch.tensor([[1.0, 2.0]])
        comp = kd_loss(student, student.clone(), torch.tensor([-100]), alpha=0.5)
        self.assertEqual(float(comp.total), 0.0)
        self.assertEqual(float(comp.hard), 0.0)
        self.assertEqual(float(comp.soft), 0.0)

    def test_temperature_squared_scaling(self):
        # soft term carries the T^2 factor; with a fixed logit gap, soft at T=2
        # should be larger than the unscaled KL would suggest. Just assert it's
        # positive, finite, and grows with the gap.
        student = torch.tensor([[0.0, 0.0, 0.0]])
        teacher_small = torch.tensor([[1.0, 0.0, 0.0]])
        teacher_big = torch.tensor([[5.0, 0.0, 0.0]])
        labels = torch.tensor([0])
        soft_small = float(kd_loss(student, teacher_small, labels, alpha=0.0, temperature=2.0).soft)
        soft_big = float(kd_loss(student, teacher_big, labels, alpha=0.0, temperature=2.0).soft)
        self.assertTrue(math.isfinite(soft_small) and soft_small > 0)
        self.assertGreater(soft_big, soft_small)

    def test_validation_errors(self):
        logits = torch.zeros((1, 2))
        labels = torch.tensor([0])
        with self.assertRaises(ValueError):
            kd_loss(logits, logits, labels, alpha=1.5)
        with self.assertRaises(ValueError):
            kd_loss(logits, logits, labels, temperature=0.0)
        with self.assertRaises(ValueError):
            kd_loss(logits, torch.zeros((1, 3)), labels)  # shape mismatch

    def test_total_stays_in_autograd_graph(self):
        student = torch.tensor([[2.0, 0.0]], requires_grad=True)
        teacher = torch.tensor([[1.0, 0.0]])
        comp = kd_loss(student, teacher, torch.tensor([0]), alpha=0.5)
        comp.total.backward()
        self.assertIsNotNone(student.grad)


class ScatterTopkTests(unittest.TestCase):
    def test_places_logprobs_and_fills_rest(self):
        ids = torch.tensor([[0, 2], [1, -1]])
        logprobs = torch.tensor([[-0.1, -0.5], [-0.2, 0.0]])
        out = scatter_topk_to_logits(ids, logprobs, vocab_size=3, fill_value=-30.0)
        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertAlmostEqual(float(out[0, 0]), -0.1, places=5)
        self.assertAlmostEqual(float(out[0, 2]), -0.5, places=5)
        self.assertAlmostEqual(float(out[0, 1]), -30.0, places=5)
        # Row 1: id 1 set; pad (-1) ignored; others fill.
        self.assertAlmostEqual(float(out[1, 1]), -0.2, places=5)
        self.assertAlmostEqual(float(out[1, 0]), -30.0, places=5)
        self.assertAlmostEqual(float(out[1, 2]), -30.0, places=5)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            scatter_topk_to_logits(
                torch.zeros((2, 2), dtype=torch.long),
                torch.zeros((2, 3)),
                vocab_size=4,
            )


class OfflineComputeTests(unittest.TestCase):
    def _batch(self):
        # B=1, S=3, V=4, k=2. Position 0 has no teacher (prompt); 1 and 2 do.
        logits = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]]]
        )
        labels = torch.tensor([[-100, 1, 2]])
        teacher_ids = torch.tensor([[[-1, -1], [1, 0], [2, 3]]])
        teacher_logprobs = torch.tensor([[[0.0, 0.0], [-0.1, -2.0], [-0.2, -1.5]]])
        return logits, labels, teacher_ids, teacher_logprobs

    def test_compute_offline_matches_manual_kd(self):
        logits, labels, t_ids, t_lp = self._batch()
        comp = compute_offline_kd_loss(
            logits, labels, t_ids, t_lp, alpha=0.5, temperature=2.0
        )
        # Manual: shift, gather valid positions (shifted labels [1, 2] both valid).
        student = logits[:, :-1, :].reshape(-1, 4)
        shift_labels = labels[:, 1:].reshape(-1)
        shift_ids = t_ids[:, 1:, :].reshape(-1, 2)
        shift_lp = t_lp[:, 1:, :].reshape(-1, 2)
        valid = shift_labels != -100
        teacher = scatter_topk_to_logits(shift_ids[valid], shift_lp[valid], 4)
        manual = kd_loss(
            student[valid], teacher, shift_labels[valid], alpha=0.5, temperature=2.0
        )
        self.assertAlmostEqual(float(comp.total), float(manual.total), places=5)
        self.assertGreater(float(comp.total), 0.0)


class OfflineTrainerComputeLossTests(unittest.TestCase):
    def _make_trainer(self):
        class _StubBase:
            def __init__(self, *args, **kwargs):
                self.state = SimpleNamespace(global_step=0)
                self.logged: list[dict] = []

            def log(self, metrics):
                self.logged.append(metrics)

        cls = make_offline_kd_trainer(_StubBase)
        return cls(kd_alpha=0.5, kd_temperature=2.0)

    def test_compute_loss_returns_loss_and_logs_components(self):
        trainer = self._make_trainer()

        logits = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]]]
        )

        class _FakeOut:
            def __init__(self, logits):
                self.logits = logits
                self.loss = None

        def fake_model(**inputs):
            return _FakeOut(logits)

        inputs = {
            "input_ids": torch.tensor([[5, 6, 7]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, 1, 2]]),
            "teacher_topk_ids": torch.tensor([[[-1, -1], [1, 0], [2, 3]]]),
            "teacher_topk_logprobs": torch.tensor(
                [[[0.0, 0.0], [-0.1, -2.0], [-0.2, -1.5]]]
            ),
        }
        loss = trainer.compute_loss(fake_model, inputs)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(float(loss), 0.0)
        # Teacher columns must be consumed before the (fake) forward.
        self.assertNotIn("teacher_topk_ids", inputs)
        # Step 0 logs the three components separately.
        self.assertEqual(len(trainer.logged), 1)
        logged = trainer.logged[0]
        for key in ("distill_total_loss", "distill_ce_loss", "distill_kd_loss"):
            self.assertIn(key, logged)
        self.assertEqual(logged["distill_mode"], "offline")

    def test_compute_loss_without_teacher_falls_back(self):
        trainer = self._make_trainer()

        class _FakeOut:
            def __init__(self):
                self.logits = torch.zeros((1, 3, 4))
                self.loss = torch.tensor(1.23)

        def fake_model(**inputs):
            return _FakeOut()

        inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, 3]])}
        loss = trainer.compute_loss(fake_model, inputs)
        self.assertAlmostEqual(float(loss), 1.23, places=4)


if __name__ == "__main__":
    unittest.main()
