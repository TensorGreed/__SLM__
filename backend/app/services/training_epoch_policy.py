"""Dataset-size-aware epoch policy (``auto_epochs``).

A fixed "3 epochs" is wrong at both ends: 50 rows × 3 epochs at an
effective batch of 16 is ~10 optimizer steps (the adapter barely moves),
while 50k rows × 3 epochs burns hours re-reading data the model already
fit after one pass. This policy targets a total optimizer-step budget.

Order of levers for a small dataset:
1. shrink the effective batch (gradient accumulation → 1) — more, smaller
   updates beat re-reading the same rows;
2. only then add epochs, up to a per-size ceiling.

The accumulation used to shrink only until an epoch had
``MIN_STEPS_PER_EPOCH`` steps, and the ceiling was 5 epochs: 108 rows got
9 steps × 5 epochs = 45 optimizer steps against a 200-step target, and the
run barely moved (train loss 3.9 → 3.6, held-out F1 unchanged vs base).
With accumulation 1 and the 8-epoch ceiling the same data gets 216 steps
(measured: eval loss 3.69 → 3.16, held-out F1 0.12 → 0.18).

Under 100 rows the ceiling stays below ``training_config_gap_service``'s
epochs-vs-small-data warn threshold — that few rows can't reach the step
target without memorising, and the scanner says so.

Pure + dependency-free: imported by ``scripts/train.py`` at run time and
by the config-gap scanner so both agree on the effective epoch count.
"""

from __future__ import annotations

import math
from typing import Any

TARGET_OPTIMIZER_STEPS = 200
# Below this many optimizer steps per epoch the effective batch is
# shrunk (via gradient accumulation) before adding epochs — more,
# smaller updates beat re-reading a tiny dataset many times.
MIN_STEPS_PER_EPOCH = 8


# The configured accumulation is kept only when the step target is
# reachable within this many epochs; otherwise it shrinks first.
COMFORTABLE_EPOCHS = 3


def max_epochs_for_rows(train_rows: int) -> int:
    if train_rows < 100:
        return 4
    if train_rows < 1000:
        return 8
    return 3


def _steps_per_epoch(rows: int, batch_size: int, accumulation: int) -> int:
    return max(1, math.ceil(rows / (batch_size * accumulation)))


def resolve_auto_epochs(
    *,
    train_rows: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    rows = max(1, int(train_rows))
    bs = max(1, int(batch_size))
    requested_ga = max(1, int(gradient_accumulation_steps))

    # Largest accumulation (≤ the configured one) that still gives enough
    # updates: a non-trivial epoch AND the step target within a few epochs.
    # Nothing qualifies on a small dataset → accumulation 1.
    ga = 1
    for candidate in range(requested_ga, 0, -1):
        per_epoch = _steps_per_epoch(rows, bs, candidate)
        if per_epoch >= MIN_STEPS_PER_EPOCH and per_epoch * COMFORTABLE_EPOCHS >= TARGET_OPTIMIZER_STEPS:
            ga = candidate
            break

    steps_per_epoch = max(1, math.ceil(rows / (bs * ga)))
    ceiling = max_epochs_for_rows(rows)
    epochs = max(1, min(ceiling, math.ceil(TARGET_OPTIMIZER_STEPS / steps_per_epoch)))
    total_steps = steps_per_epoch * epochs
    reason = (
        f"{rows} rows at effective batch {bs * ga} → {steps_per_epoch} steps/epoch; "
        f"{epochs} epoch(s) ≈ {total_steps} optimizer steps "
        f"(target {TARGET_OPTIMIZER_STEPS}, max {ceiling} epochs at this size)."
    )
    return {
        "num_epochs": epochs,
        "gradient_accumulation_steps": ga,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "reason": reason,
    }
