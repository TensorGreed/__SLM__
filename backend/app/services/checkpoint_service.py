"""P16 — Checkpoint browser backend (priority.md, RM3).

Three operations on the checkpoints accumulated by a finished or in-flight
training run:

- ``list_checkpoints`` — read every checkpoint row written for a run, in
  step order, with the bookkeeping fields the UI needs (best flag,
  promotion timestamp, file path, metrics blob).
- ``promote_checkpoint`` — pick a specific step as the run's canonical
  "best". This is an operator override on top of whatever the runtime
  computed: it sets ``is_best=True`` on the chosen row, clears the flag
  on every other checkpoint of the same run (so promotion is exclusive
  per-run), and stamps ``promoted_at`` with the wall-clock. Re-promoting
  the same step is idempotent — the flag stays true and ``promoted_at``
  refreshes.
- ``resume_from_checkpoint`` — create a **new** experiment whose config
  is the parent run's manifest config + a ``_resume_from`` block pointing
  at the checkpoint to load. Like P15's clone-from-run, it does **not**
  start training — it returns a PENDING experiment so the caller decides
  when to fire ``/experiments/{new_id}/start``. That start handler will
  in turn write a fresh manifest for the resumed run, keeping the
  resume → manifest lineage walkable.

Stable error reason codes used by the API layer:
  - ``experiment_not_found`` — the run id doesn't exist in the project.
  - ``checkpoint_not_found`` — the run exists but no checkpoint matches
    the requested step.
  - ``manifest_not_captured`` — resume needs the parent's P14 manifest to
    rebuild the resumed config; raised when the parent ran before P14
    was hooked in (or capture failed silently).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Checkpoint, Experiment, TrainingMode
from app.models.training_manifest import TrainingManifest
from app.services.training_service import create_experiment


_RESUME_KEY = "_resume_from"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def serialize_checkpoint(ckpt: Checkpoint) -> dict[str, Any]:
    """Stable JSON shape for a single checkpoint row.

    The shape mirrors the spec's column list (``run_id, step, epoch,
    loss, metrics_blob, path, is_best, promoted_at``) and preserves both
    train and eval loss separately under ``loss`` for clients that want
    the structured form. ``loss`` is the canonical "the spec asked for a
    loss" field — train_loss when present, else eval_loss — so a CLI/UI
    can sort or display a single number per row.
    """
    train = float(ckpt.train_loss) if ckpt.train_loss is not None else None
    evl = float(ckpt.eval_loss) if ckpt.eval_loss is not None else None
    canonical_loss = train if train is not None else evl
    return {
        "id": int(ckpt.id),
        "run_id": int(ckpt.experiment_id),
        "experiment_id": int(ckpt.experiment_id),
        "step": int(ckpt.step),
        "epoch": int(ckpt.epoch),
        "loss": canonical_loss,
        "train_loss": train,
        "eval_loss": evl,
        "metrics_blob": dict(ckpt.metrics or {}),
        "path": str(ckpt.file_path or ""),
        "is_best": bool(ckpt.is_best),
        "promoted_at": ckpt.promoted_at.isoformat() if ckpt.promoted_at else None,
        "created_at": ckpt.created_at.isoformat() if ckpt.created_at else None,
    }


async def _load_experiment(
    db: AsyncSession, *, project_id: int, experiment_id: int
) -> Experiment:
    row = (
        await db.execute(
            select(Experiment).where(
                Experiment.id == experiment_id,
                Experiment.project_id == project_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise ValueError("experiment_not_found")
    return row


async def list_checkpoints(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
) -> dict[str, Any]:
    """Return every checkpoint for a run, ordered by step ascending."""
    exp = await _load_experiment(
        db, project_id=project_id, experiment_id=experiment_id
    )
    ckpt_result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.experiment_id == experiment_id)
        .order_by(Checkpoint.step.asc())
    )
    rows = list(ckpt_result.scalars().all())
    return {
        "project_id": int(project_id),
        "experiment_id": int(experiment_id),
        "run_status": exp.status.value if hasattr(exp.status, "value") else str(exp.status),
        "count": len(rows),
        "checkpoints": [serialize_checkpoint(r) for r in rows],
    }


async def promote_checkpoint(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    step: int,
) -> dict[str, Any]:
    """Mark a step as the run's canonical "best" and stamp ``promoted_at``.

    Promotion is **exclusive per-run** — every other checkpoint of the
    same run has its ``is_best`` flag cleared so the API contract holds
    that at most one checkpoint is ever flagged best. Re-promoting the
    same step is idempotent: the row stays best, ``promoted_at``
    refreshes (so the UI's "promoted X minutes ago" stays honest).
    """
    if step is None or int(step) < 0:
        raise ValueError("invalid_step")

    await _load_experiment(
        db, project_id=project_id, experiment_id=experiment_id
    )
    target = (
        await db.execute(
            select(Checkpoint).where(
                Checkpoint.experiment_id == experiment_id,
                Checkpoint.step == int(step),
            )
        )
    ).scalar_one_or_none()
    if target is None:
        raise ValueError("checkpoint_not_found")

    now = _utcnow()
    # Clear best on every other checkpoint of this run. We do this in a
    # single statement so we don't have to load every row, and it covers
    # the case where the runtime had already flagged a different step.
    await db.execute(
        update(Checkpoint)
        .where(
            Checkpoint.experiment_id == experiment_id,
            Checkpoint.id != target.id,
        )
        .values(is_best=False)
    )
    target.is_best = True
    target.promoted_at = now
    await db.commit()
    await db.refresh(target)
    return serialize_checkpoint(target)


def _normalize_training_mode(value: Any, fallback: TrainingMode) -> TrainingMode:
    if isinstance(value, TrainingMode):
        return value
    token = str(value or "").strip().lower()
    for mode in TrainingMode:
        if mode.value == token:
            return mode
    return fallback


async def resume_from_checkpoint(
    db: AsyncSession,
    *,
    project_id: int,
    parent_experiment_id: int,
    step: int,
    run_name: str | None = None,
    description: str | None = None,
) -> Experiment:
    """Create a new experiment that resumes from a captured checkpoint.

    Mirrors the structure of P15's ``clone_from_run``: pulls the parent
    run's P14 manifest, copies the resolved config, then layers a
    ``_resume_from`` parentage block carrying the parent run id, manifest
    id, step number, and the on-disk checkpoint path. The runtime will
    pick that up at start time to seed weights from the checkpoint.
    Like clone, this **does not** launch training — it returns a PENDING
    experiment row.
    """
    if int(step) < 0:
        raise ValueError("invalid_step")

    parent_exp = await _load_experiment(
        db, project_id=project_id, experiment_id=parent_experiment_id
    )

    ckpt = (
        await db.execute(
            select(Checkpoint).where(
                Checkpoint.experiment_id == parent_experiment_id,
                Checkpoint.step == int(step),
            )
        )
    ).scalar_one_or_none()
    if ckpt is None:
        raise ValueError("checkpoint_not_found")

    manifest = (
        await db.execute(
            select(TrainingManifest).where(
                TrainingManifest.experiment_id == parent_experiment_id,
                TrainingManifest.project_id == project_id,
            )
        )
    ).scalar_one_or_none()
    if manifest is None:
        raise ValueError("manifest_not_captured")

    resolved_config = copy.deepcopy(dict(manifest.resolved_config or {}))
    resolved_config[_RESUME_KEY] = {
        "parent_experiment_id": int(parent_experiment_id),
        "manifest_id": int(manifest.id),
        "checkpoint_step": int(ckpt.step),
        "checkpoint_epoch": int(ckpt.epoch),
        "checkpoint_path": str(ckpt.file_path or ""),
        "checkpoint_is_best": bool(ckpt.is_best),
        "reason": "resume-from-checkpoint",
    }

    base_model = manifest.base_model_source_ref or parent_exp.base_model
    mode = _normalize_training_mode(manifest.training_mode, parent_exp.training_mode)
    name = (run_name or f"{parent_exp.name} (resume@step{int(ckpt.step)})").strip()
    if not name:
        name = f"exp {parent_experiment_id} resume@step{int(ckpt.step)}"

    exp = await create_experiment(
        db,
        project_id=project_id,
        name=name,
        base_model=base_model,
        config=resolved_config,
        description=(description if description is not None else (parent_exp.description or "")),
        training_mode=mode,
    )
    await db.commit()
    await db.refresh(exp)
    return exp
