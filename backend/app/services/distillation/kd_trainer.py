"""Offline-KD HF Trainer + collator (Track 1, Epic A, slice 2).

The trainer subclass is produced by a *factory* that takes the ``Trainer`` base
class as an argument, so this module imports cleanly without ``transformers``
installed (the real trainer passes it in; tests pass a tiny stub). All loss math
delegates to the unit-tested :mod:`app.services.distillation.kd_loss`, so the
only logic here is shift/gather/scatter plumbing — which is itself unit-tested
through the stub base.

Offline KD differs from the live-teacher ``DistillationTrainer`` in scripts/train.py
only in *where the teacher distribution comes from*: here it's carried on each
batch as ``teacher_topk_ids`` / ``teacher_topk_logprobs`` (built by
``kd_capture.build_offline_kd_records`` from the slice-1 capture artifact), so no
teacher model is loaded at train time.

train.py integration note: because ``teacher_topk_*`` aren't model.forward args,
the offline path must set ``TrainingArguments(remove_unused_columns=False)`` and
use ``OfflineKDCollator`` so HF Trainer doesn't strip those columns.
"""

from __future__ import annotations

from typing import Any, Callable

from app.services.distillation.kd_capture import TEACHER_PAD_ID
from app.services.distillation.kd_loss import KDLossComponents, kd_loss, scatter_topk_to_logits


def compute_offline_kd_loss(
    logits: "Any",
    labels: "Any",
    teacher_topk_ids: "Any",
    teacher_topk_logprobs: "Any",
    *,
    alpha: float,
    temperature: float,
    ignore_index: int = -100,
) -> KDLossComponents:
    """Shift (causal LM), gather scored positions, densify the teacher top-k,
    and return the KD loss components.

    ``logits``: ``[B, S, V]``; ``labels``: ``[B, S]``;
    ``teacher_topk_ids`` / ``teacher_topk_logprobs``: ``[B, S, k]`` aligned to
    ``labels`` (i.e. teacher distribution for the token *at* each position).
    """
    student = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_ids = teacher_topk_ids[:, 1:, :]
    shift_logprobs = teacher_topk_logprobs[:, 1:, :]

    vocab_size = int(student.shape[-1])
    top_k = int(shift_ids.shape[-1])
    student_flat = student.reshape(-1, vocab_size)
    labels_flat = shift_labels.reshape(-1)
    ids_flat = shift_ids.reshape(-1, top_k)
    logprobs_flat = shift_logprobs.reshape(-1, top_k)

    valid = labels_flat != ignore_index
    if not bool(valid.any()):
        zero = student_flat.new_zeros(())
        return KDLossComponents(total=zero, hard=zero, soft=zero)

    student_valid = student_flat[valid]
    teacher_valid = scatter_topk_to_logits(
        ids_flat[valid], logprobs_flat[valid], vocab_size, pad_id=TEACHER_PAD_ID
    ).to(student_valid.device)
    return kd_loss(
        student_valid,
        teacher_valid,
        labels_flat[valid],
        alpha=alpha,
        temperature=temperature,
        ignore_index=ignore_index,
    )


def make_offline_kd_trainer(trainer_base_cls: type) -> type:
    """Return an ``OfflineDistillationTrainer`` subclass of ``trainer_base_cls``.

    Factory form keeps this module import-safe without ``transformers``.
    """

    class OfflineDistillationTrainer(trainer_base_cls):  # type: ignore[valid-type, misc]
        def __init__(
            self,
            *args: Any,
            kd_alpha: float = 0.5,
            kd_temperature: float = 2.0,
            kd_log_every: int = 10,
            kd_ignore_index: int = -100,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.kd_alpha = max(0.0, min(float(kd_alpha), 1.0))
            self.kd_temperature = max(0.1, float(kd_temperature))
            self.kd_log_every = max(1, int(kd_log_every))
            self.kd_ignore_index = int(kd_ignore_index)
            self._last_kd_log_step = -1

        def compute_loss(  # noqa: D401
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ):
            # Teacher targets aren't model.forward args — pull them out before
            # the forward pass.
            teacher_ids = inputs.pop("teacher_topk_ids", None)
            teacher_logprobs = inputs.pop("teacher_topk_logprobs", None)
            labels = inputs.get("labels")

            outputs = model(**inputs)

            if teacher_ids is None or teacher_logprobs is None or labels is None:
                loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss")
                return (loss, outputs) if return_outputs else loss

            components = compute_offline_kd_loss(
                outputs.logits,
                labels,
                teacher_ids,
                teacher_logprobs,
                alpha=self.kd_alpha,
                temperature=self.kd_temperature,
                ignore_index=self.kd_ignore_index,
            )

            step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            if step != self._last_kd_log_step and step % self.kd_log_every == 0:
                floats = components.as_floats()
                self.log(
                    {
                        "distill_total_loss": floats["total"],
                        "distill_ce_loss": floats["hard"],
                        "distill_kd_loss": floats["soft"],
                        "distill_alpha": self.kd_alpha,
                        "distill_temperature": self.kd_temperature,
                        "distill_mode": "offline",
                    }
                )
                self._last_kd_log_step = step

            if return_outputs:
                if hasattr(outputs, "loss"):
                    outputs.loss = components.total
                return components.total, outputs
            return components.total

    return OfflineDistillationTrainer


class OfflineKDCollator:
    """Pad a batch of ``OfflineKDRecord``s into stacked tensors.

    Pads ``input_ids`` with ``pad_token_id``, ``attention_mask`` with 0,
    ``labels`` with ``label_pad`` (-100), and the per-position teacher top-k
    rows with the ragged sentinel. Requires torch at call time only.
    """

    def __init__(
        self,
        pad_token_id: int,
        *,
        top_k: int,
        label_pad: int = -100,
        teacher_pad_id: int = TEACHER_PAD_ID,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.top_k = int(top_k)
        self.label_pad = int(label_pad)
        self.teacher_pad_id = int(teacher_pad_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        pad_row = [self.teacher_pad_id] * self.top_k
        pad_lp_row = [0.0] * self.top_k

        input_ids, attention, labels, tk_ids, tk_lp = [], [], [], [], []
        for f in features:
            n = len(f["input_ids"])
            gap = max_len - n
            input_ids.append(list(f["input_ids"]) + [self.pad_token_id] * gap)
            attention.append(list(f["attention_mask"]) + [0] * gap)
            labels.append(list(f["labels"]) + [self.label_pad] * gap)
            tk_ids.append([list(r) for r in f["teacher_topk_ids"]] + [list(pad_row) for _ in range(gap)])
            tk_lp.append([list(r) for r in f["teacher_topk_logprobs"]] + [list(pad_lp_row) for _ in range(gap)])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "teacher_topk_ids": torch.tensor(tk_ids, dtype=torch.long),
            "teacher_topk_logprobs": torch.tensor(tk_lp, dtype=torch.float32),
        }
