"""Knowledge-distillation loss math (Track 1, Epic A, slice 2).

Pure, framework-level functions — no FastAPI / DB / file IO — so the loss is
trivially unit-testable on tiny CPU tensors. The trainer (``scripts/train.py``)
imports these for the *offline* KD path (teacher distribution comes from the
slice-1 capture artifact, not a live teacher model).

The objective is the textbook Hinton distillation loss:

    L = α · CE(student_logits, hard_labels)
        + (1 − α) · T² · KL( softmax(student_logits / T) ‖ softmax(teacher_logits / T) )

The T² factor restores the soft-target gradient magnitude that the 1/T²
softening would otherwise wash out, so α trades hard vs. soft on a comparable
scale. ``kd_loss`` returns the three components separately so callers can log
hard / soft / total independently (the slice-2 brief's "log … separately").

Note: the *online* DistillationTrainer in scripts/train.py predates this module
and inlines the same formula against a live teacher; it's intentionally left
as-is to avoid touching a GPU-only path that can't be tested here. New offline
work routes through this tested function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


@dataclass
class KDLossComponents:
    """Loss breakdown. Each field is a scalar ``torch.Tensor`` (so ``total``
    stays in the autograd graph); use ``.item()`` for logging."""

    total: "torch.Tensor"
    hard: "torch.Tensor"
    soft: "torch.Tensor"

    def as_floats(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "hard": float(self.hard.detach().cpu()),
            "soft": float(self.soft.detach().cpu()),
        }


def kd_loss(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    labels: "torch.Tensor",
    *,
    alpha: float = 0.5,
    temperature: float = 2.0,
    ignore_index: int = -100,
) -> KDLossComponents:
    """Compute the KD loss over a flat batch of scored positions.

    Shapes:
      - ``student_logits`` / ``teacher_logits``: ``[N, V]`` (N positions, V vocab).
      - ``labels``: ``[N]`` token ids; positions equal to ``ignore_index`` are
        dropped from *both* the hard and soft terms.

    Soft term uses ``KL(student ‖ teacher)`` via
    ``F.kl_div(student_log_softmax, teacher_softmax, reduction="batchmean")``,
    which matches the live-teacher trainer's convention.
    """
    import torch
    import torch.nn.functional as F

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    student = student_logits.float()
    teacher = teacher_logits.float()
    if student.shape != teacher.shape:
        raise ValueError(
            f"student/teacher logits shape mismatch: {tuple(student.shape)} vs "
            f"{tuple(teacher.shape)}"
        )

    valid = labels != ignore_index
    if bool(valid.any()):
        s = student[valid]
        t = teacher[valid]
        y = labels[valid]
        hard = F.cross_entropy(s, y)
        student_log = F.log_softmax(s / temperature, dim=-1)
        teacher_prob = F.softmax(t / temperature, dim=-1)
        soft = F.kl_div(student_log, teacher_prob, reduction="batchmean") * (
            temperature * temperature
        )
    else:
        hard = student.new_zeros(())
        soft = student.new_zeros(())

    total = alpha * hard + (1.0 - alpha) * soft
    return KDLossComponents(total=total, hard=hard, soft=soft)


def scatter_topk_to_logits(
    topk_ids: "torch.Tensor",
    topk_logprobs: "torch.Tensor",
    vocab_size: int,
    *,
    fill_value: float = -30.0,
    pad_id: int = -1,
) -> "torch.Tensor":
    """Densify a sparse top-k teacher distribution into a ``[N, V]`` logit matrix.

    The slice-1 capture stores, per position, the teacher's top-k
    ``(token, logprob)`` pairs. To compute KL against the student's full-vocab
    distribution we place those logprobs at their vocab ids and fill the rest
    with ``fill_value`` (a large negative so it's ~0 after softmax). Entries
    where ``topk_ids == pad_id`` (ragged padding) are ignored.

    Shapes: ``topk_ids`` / ``topk_logprobs`` are ``[N, k]``; returns ``[N, V]``.
    """
    import torch

    if topk_ids.shape != topk_logprobs.shape:
        raise ValueError(
            f"topk_ids/topk_logprobs shape mismatch: {tuple(topk_ids.shape)} vs "
            f"{tuple(topk_logprobs.shape)}"
        )
    if topk_ids.dim() != 2:
        raise ValueError(f"expected 2-D [N, k] tensors, got {topk_ids.dim()}-D")

    n_pos = topk_ids.shape[0]
    out = torch.full((n_pos, int(vocab_size)), float(fill_value), dtype=torch.float32)
    valid = topk_ids != pad_id
    if bool(valid.any()):
        rows = (
            torch.arange(n_pos, device=topk_ids.device)
            .unsqueeze(1)
            .expand_as(topk_ids)
        )
        safe_ids = topk_ids.clamp(min=0, max=int(vocab_size) - 1)
        out[rows[valid], safe_ids[valid]] = topk_logprobs[valid].float()
    return out
