"""Experiment lifecycle recovery actions (Story 1.7).

Today an operator who wants to restart a FAILED training experiment
has to run hand-crafted SQL + ``mv`` commands to clear stale state
(see the experiment 9/10/11 incident series on 2026-05-15..17). This
module is the proper surface — three idempotent service functions
the API + CLI both call:

- :func:`reset_experiment` — flip FAILED → PENDING, archive the
  output dir to ``<dir>.bak.<ts>``, drop checkpoint rows. Safe to
  call repeatedly.
- :func:`delete_experiment` — hard delete: DB row + checkpoint rows
  + output dir gone for good.
- :func:`bulk_archive_failed` — sweep every FAILED experiment in a
  project through ``reset_experiment``. One-button cleanup after a
  chain of failures.

All three refuse to touch RUNNING experiments (the training
subprocess might still be writing to the output dir; archiving it
mid-run would corrupt the run). Cancel first, wait for status to
transition, then reset.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import (
    Checkpoint,
    Experiment,
    ExperimentStatus,
)


def _utc_stamp() -> str:
    """A second-precision UTC stamp suitable for backup-dir suffixes.
    Format: ``20260517T153012Z``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _archive_output_dir(output_dir: str | None) -> str | None:
    """Move the experiment's output dir aside so a fresh run starts
    clean. Returns the backup path (or None when there was nothing
    to archive)."""
    if not output_dir:
        return None
    src = Path(output_dir)
    if not src.exists():
        return None
    # Keep the suffix non-clashing in the unlikely case of two resets
    # within the same second.
    backup = src.with_name(f"{src.name}.bak.{_utc_stamp()}")
    counter = 0
    while backup.exists():
        counter += 1
        backup = src.with_name(
            f"{src.name}.bak.{_utc_stamp()}-{counter}"
        )
    shutil.move(str(src), str(backup))
    return str(backup)


def _purge_output_dir(output_dir: str | None) -> bool:
    """Permanently delete the experiment's output dir. Returns True
    when something was removed."""
    if not output_dir:
        return False
    src = Path(output_dir)
    if not src.exists():
        return False
    shutil.rmtree(src)
    return True


async def _get_experiment(
    db: AsyncSession, project_id: int, experiment_id: int
) -> Experiment | None:
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    return result.scalar_one_or_none()


async def _delete_checkpoint_rows(
    db: AsyncSession, experiment_id: int
) -> int:
    """Drop the experiment's checkpoint rows so the trainer's resume
    discovery starts from a clean slate. Returns the deleted count."""
    rows = await db.execute(
        select(Checkpoint).where(Checkpoint.experiment_id == experiment_id)
    )
    count = 0
    for ckpt in rows.scalars().all():
        await db.delete(ckpt)
        count += 1
    return count


async def reset_experiment(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    archive_output_dir: bool = True,
) -> dict[str, Any]:
    """Flip a FAILED experiment back to PENDING + clear stale state.

    Idempotent: calling twice on the same row is safe.

    Raises ``ValueError`` with stable codes the API layer can
    translate:
      - ``experiment_not_found``
      - ``experiment_running`` — caller must cancel + wait first
    """
    exp = await _get_experiment(db, project_id, experiment_id)
    if exp is None:
        raise ValueError("experiment_not_found")
    if exp.status == ExperimentStatus.RUNNING:
        raise ValueError("experiment_running")

    backup_path: str | None = None
    if archive_output_dir:
        backup_path = _archive_output_dir(exp.output_dir)

    checkpoints_deleted = await _delete_checkpoint_rows(db, experiment_id)

    previous_status = exp.status.value
    exp.status = ExperimentStatus.PENDING
    exp.final_train_loss = None
    exp.final_eval_loss = None
    exp.completed_at = None
    exp.started_at = None
    # Strip any runtime breadcrumbs from a prior attempt so the next
    # start_training sees a clean config.
    if isinstance(exp.config, dict):
        cfg = dict(exp.config)
        cfg.pop("_runtime", None)
        cfg.pop("resume_from_checkpoint", None)
        exp.config = cfg

    await db.flush()

    return {
        "experiment_id": exp.id,
        "previous_status": previous_status,
        "new_status": exp.status.value,
        "archived_output_dir": backup_path,
        "checkpoints_deleted": checkpoints_deleted,
    }


async def delete_experiment(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    purge_output_dir: bool = True,
) -> dict[str, Any]:
    """Hard delete an experiment + its on-disk artifacts.

    Refuses RUNNING experiments. Returns a small report so the API
    response shows what got cleaned up.
    """
    exp = await _get_experiment(db, project_id, experiment_id)
    if exp is None:
        raise ValueError("experiment_not_found")
    if exp.status == ExperimentStatus.RUNNING:
        raise ValueError("experiment_running")

    output_dir = exp.output_dir
    name = exp.name
    previous_status = exp.status.value
    checkpoints_deleted = await _delete_checkpoint_rows(db, experiment_id)

    dir_removed = False
    if purge_output_dir:
        dir_removed = _purge_output_dir(output_dir)

    await db.delete(exp)
    await db.flush()

    return {
        "experiment_id": experiment_id,
        "name": name,
        "previous_status": previous_status,
        "output_dir_removed": dir_removed,
        "output_dir_path": output_dir,
        "checkpoints_deleted": checkpoints_deleted,
    }


async def bulk_archive_failed(
    db: AsyncSession, *, project_id: int
) -> dict[str, Any]:
    """Reset every FAILED experiment in a project in one sweep. The
    "Archive all failed" header banner on the UI calls this."""
    result = await db.execute(
        select(Experiment).where(
            Experiment.project_id == project_id,
            Experiment.status == ExperimentStatus.FAILED,
        )
    )
    failed = list(result.scalars().all())

    reports: list[dict[str, Any]] = []
    for exp in failed:
        try:
            report = await reset_experiment(
                db,
                project_id=project_id,
                experiment_id=exp.id,
                archive_output_dir=True,
            )
            reports.append(report)
        except ValueError as exc:
            # An experiment flipped to RUNNING between our SELECT and
            # the reset call — skip and continue (rare race).
            reports.append(
                {
                    "experiment_id": exp.id,
                    "skipped": True,
                    "reason": str(exc),
                }
            )

    return {
        "project_id": project_id,
        "total_failed": len(failed),
        "reset_count": sum(1 for r in reports if not r.get("skipped")),
        "skipped_count": sum(1 for r in reports if r.get("skipped")),
        "reports": reports,
    }


__all__ = [
    "bulk_archive_failed",
    "delete_experiment",
    "reset_experiment",
]
