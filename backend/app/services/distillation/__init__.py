"""Knowledge-distillation services (Track 1, Epic A).

Slice 1 (shipped): teacher logit capture — record a strong teacher's
top-k token log-probabilities for a source dataset.
Slice 2 (shipped): offline KD training — `kd_loss` (α·CE + (1−α)·T²·KL) +
the capture-reading / readiness-gate / teacher-target-alignment helpers the
trainer uses to learn from the captured soft targets without a live teacher.

Public surface re-exported for callers (API layer, trainer, tests).
"""

from app.services.distillation.kd_capture import (
    CaptureGate,
    build_teacher_target_topk,
    load_teacher_capture,
    verify_capture_artifact,
)
from app.services.distillation.kd_loss import (
    KDLossComponents,
    kd_loss,
    scatter_topk_to_logits,
)
from app.services.distillation.teacher_capture import (
    CaptureResult,
    DistillationCaptureTask,
    call_teacher_with_logprobs,
    capture_teacher_outputs,
    get_capture_task,
    get_capture_task_status,
    start_capture_task,
)

__all__ = [
    # slice 1 — capture
    "CaptureResult",
    "DistillationCaptureTask",
    "call_teacher_with_logprobs",
    "capture_teacher_outputs",
    "get_capture_task",
    "get_capture_task_status",
    "start_capture_task",
    # slice 2 — offline KD training
    "KDLossComponents",
    "kd_loss",
    "scatter_topk_to_logits",
    "CaptureGate",
    "load_teacher_capture",
    "verify_capture_artifact",
    "build_teacher_target_topk",
]
