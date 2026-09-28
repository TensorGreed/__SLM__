"""Dataset-size-aware epoch policy (``auto_epochs``).

A fixed "3 epochs" is wrong at both ends: 50 rows × 3 epochs at an
effective batch of 16 is ~10 optimizer steps (the adapter barely moves),
while 50k rows × 3 epochs burns hours re-reading data the model already
fit after one pass. This policy targets a total optimizer-step budget,
clamped by a per-size epoch ceiling so small datasets aren't memorised
(the ceilings sit below ``training_config_gap_service``'s
epochs-vs-small-data warn threshold).

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


def max_epochs_for_rows(train_rows: int) -> int:
    if train_rows < 100:
        return 4
    if train_rows < 1000:
        return 5
    return 3


def resolve_auto_epochs(
    *,
    train_rows: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    rows = max(1, int(train_rows))
    bs = max(1, int(batch_size))
    ga = max(1, int(gradient_accumulation_steps))

    if math.ceil(rows / (bs * ga)) < MIN_STEPS_PER_EPOCH and ga > 1:
        ga = max(1, min(ga, rows // (bs * MIN_STEPS_PER_EPOCH)))

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
