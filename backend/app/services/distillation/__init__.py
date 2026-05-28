"""Knowledge-distillation services (Track 1, Epic A).

Slice 1 (shipped): teacher logit capture — record a strong teacher's
top-k token log-probabilities for a source dataset so a student model can
later train against soft targets.

Public surface re-exported for callers (API layer, tests):
- ``capture_teacher_outputs`` — the synchronous capture coroutine.
- ``call_teacher_with_logprobs`` — the mockable OpenAI-compatible call.
- ``start_capture_task`` / ``get_capture_task_status`` — background-task
  plumbing mirroring ``cleaning_service``.
"""

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
    "CaptureResult",
    "DistillationCaptureTask",
    "call_teacher_with_logprobs",
    "capture_teacher_outputs",
    "get_capture_task",
    "get_capture_task_status",
    "start_capture_task",
]
