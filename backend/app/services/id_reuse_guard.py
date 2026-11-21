"""Id reuse guard: a new row never adopts files keyed by a deleted row's id (T1 / T3).

SQLite reuses the highest id after a delete (and a fresh DB over an old ``DATA_DIR`` restarts every id at
1), while many on-disk paths are keyed only by ids: ``projects/<pid>/``, ``exports/<pid>/``,
``projects/<pid>/experiments/<id>/``, ``projects/<pid>/compressed/exp-<id>/``,
``projects/<pid>/auto_rag/passage_reader/<id>.json``, ``exports/<pid>/<export_id>/``. A row created with a
reused id would otherwise read the old row's prepared split, weights, converted GGUF or reader record as its
own.

Nothing here deletes data: a stale path is renamed aside in its own parent to
``<name>.<tag>.<UTC stamp>[-n]`` (``tag`` = ``stale`` when a new row finds it, ``deleted`` when its row is
deleted). Those names never parse as ids (``exp-3.stale.…`` / ``7.deleted.…`` are not digits), so no
id-keyed reader picks them up again; project-wide scans skip them with ``is_archived_name`` /
``in_archived_folder``.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# Name markers of folders moved aside: ``.stale.`` / ``.deleted.`` (this module), ``.bak.`` (run reset).
ARCHIVED_NAME_MARKERS: tuple[str, ...] = (".stale.", ".deleted.", ".bak.")


def is_archived_name(name: str) -> bool:
    """Whether a file / folder name is one moved aside (``exp-3.stale.<stamp>``, ``7.deleted.<stamp>``,
    ``12.bak.<stamp>``): a reader scanning a whole tree must skip it, it belongs to no live row."""
    return any(marker in str(name) for marker in ARCHIVED_NAME_MARKERS)


def in_archived_folder(path: Path, root: Path) -> bool:
    """Whether ``path`` sits inside a moved-aside folder below ``root`` (its own name is not checked)."""
    try:
        parts = Path(path).relative_to(root).parts[:-1]
    except ValueError:
        return False
    return any(is_archived_name(part) for part in parts)


def _non_empty(path: Path) -> bool:
    if path.is_file() or path.is_symlink():
        return True
    if not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return False


def move_aside(path: Path, *, reason: str, tag: str = "stale") -> dict[str, Any] | None:
    """Rename a file or a non-empty dir to ``<name>.<tag>.<UTC stamp>[-n]`` next to it and return
    ``{"from", "to", "reason", "archived_at"}``; None when ``path`` is absent or an empty dir (an empty dir
    holds nothing to adopt and is left in place). Raises ``OSError`` when the rename fails."""
    path = Path(path)
    if not _non_empty(path):
        return None
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.{tag}.{stamp}")
    counter = 0
    while target.exists():
        counter += 1
        target = path.with_name(f"{path.name}.{tag}.{stamp}-{counter}")
    shutil.move(str(path), str(target))
    logger.warning("id_reuse_guard: moved %s aside to %s (%s)", path, target, reason)
    return {"from": str(path), "to": str(target), "reason": reason, "archived_at": now.isoformat()}


def _require_move_aside(path: Path, *, reason: str) -> dict[str, Any] | None:
    """For a NEW row: move a stale path aside or fail the create with the structured
    ``STALE_DIR_MOVE_FAILED`` error (same envelope as T1's ``OUTPUT_DIR_MOVE_FAILED``) -- adopting the
    stale files is worse than failing."""
    try:
        return move_aside(path, reason=reason, tag="stale")
    except OSError as exc:
        raise _move_failed(
            f"Could not start clean: {path} holds files from an earlier item with the same id and "
            f"could not be moved aside ({exc.__class__.__name__}: {exc}). Nothing was deleted. "
            "Fix the folder permissions or move it by hand, then try again.",
            {"from": str(path), "reason": reason},
        ) from exc


def _move_failed(message: str, metadata: dict[str, Any]):
    """The structured ``STALE_DIR_MOVE_FAILED`` error (an ``HTTPException`` whose ``detail`` carries
    ``error_code`` / ``message`` / ``metadata``; ``move_failed_message`` reads the message back)."""
    from fastapi import HTTPException

    return HTTPException(500, {"error_code": "STALE_DIR_MOVE_FAILED", "message": message, "metadata": metadata})


def move_failed_message(exc: BaseException) -> str | None:
    """The plain message of a structured error raised here (None for any other exception)."""
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict) and detail.get("error_code") == "STALE_DIR_MOVE_FAILED":
        return str(detail.get("message") or "")
    return None


def _safe_move_aside(path: Path, *, reason: str, tag: str) -> dict[str, Any] | None:
    try:
        return move_aside(path, reason=reason, tag=tag)
    except OSError as exc:
        logger.error("id_reuse_guard: could not move %s aside (%s): %s", path, reason, exc)
        return {"from": str(path), "to": None, "reason": reason, "error": str(exc)}


# ------------------------------------------------------------------ projects


def project_paths(project_id: int) -> list[Path]:
    """Every ``DATA_DIR`` tree keyed only by this project id."""
    pid = str(int(project_id))
    return [settings.DATA_DIR / "projects" / pid, settings.DATA_DIR / "exports" / pid]


def archive_project_paths(project_id: int, *, tag: str, reason: str) -> list[dict[str, Any]]:
    """Move every project-id-keyed tree aside (never raises: a failure is logged and reported)."""
    out = []
    for p in project_paths(project_id):
        rec = _safe_move_aside(p, reason=reason, tag=tag)
        if rec:
            out.append(rec)
    return out


def guard_new_project(project_id: int) -> list[dict[str, Any]]:
    """A NEW project row never adopts ``projects/<pid>/`` or ``exports/<pid>/`` from a deleted project
    (or an older DB): move them aside before the creator writes anything there."""
    out = []
    for p in project_paths(project_id):
        rec = _require_move_aside(p, reason="pre_existing_project_dir")
        if rec:
            out.append(rec)
    return out


# --------------------------------------------------------------- experiments


def experiment_paths(project_id: int, experiment_id: int) -> list[Path]:
    """The per-run paths OUTSIDE ``experiments/<id>/`` that are keyed only by the run id (the output dir
    itself is handled by ``training_service._archive_preexisting_experiment_dir`` / the delete purge)."""
    root = settings.DATA_DIR / "projects" / str(int(project_id))
    eid = int(experiment_id)
    return [
        root / "compressed" / f"exp-{eid}",
        root / "auto_rag" / "passage_reader" / f"{eid}.json",
    ]


def guard_new_experiment(project_id: int, experiment_id: int) -> list[dict[str, Any]]:
    """A NEW run never adopts a deleted run's converted GGUF / ONNX (``compressed/exp-<id>/``) or its
    passage-reader record."""
    out = []
    for p in experiment_paths(project_id, experiment_id):
        rec = _require_move_aside(p, reason="pre_existing_run_artifact")
        if rec:
            out.append(rec)
    return out


def archive_deleted_experiment(project_id: int, experiment_id: int) -> list[dict[str, Any]]:
    """On run delete: move its id-keyed side artifacts aside so the next run with the same id starts clean."""
    out = []
    for p in experiment_paths(project_id, experiment_id):
        rec = _safe_move_aside(p, reason="experiment_deleted", tag="deleted")
        if rec:
            out.append(rec)
    return out


def guard_new_export_dir(project_id: int, export_id: int) -> dict[str, Any] | None:
    """A NEW export row never adopts ``exports/<pid>/<export_id>/`` (its ``LATEST_RUN`` would point the
    download at another export's package)."""
    return _require_move_aside(
        settings.DATA_DIR / "exports" / str(int(project_id)) / str(int(export_id)),
        reason="pre_existing_export_dir",
    )


def archive_compression_files_predating_run(project_id: int, experiment_id: int, created_at: Any) -> dict[str, Any] | None:
    """Export-time provenance check for runs created before the create-time guard existed: a file in
    ``compressed/exp-<id>/`` last written BEFORE this run's row was created cannot be this run's output
    (a deleted run with the same id converted it). Move each such file (keeping its relative path) into
    ``compressed/exp-<id>.stale.<stamp>/`` and return ``{"from", "to", "files", "reason"}``; None when
    nothing predates the run or ``created_at`` is unknown. Raises the structured ``STALE_DIR_MOVE_FAILED``
    error when a move fails (shipping the stale file is worse than failing).

    The conversion lock file (``.convert.lock``) is never moved: every converter flocks that one inode, so
    moving it would let two conversions run at once. Callers hold ``run_conversion_lock`` (see
    ``export_service._archive_compression_files_predating_run``) so no conversion writes meanwhile.
    Known false positive: a file placed with its original mtime kept (``cp -p``, ``rsync -a``, a
    hand-dropped pre-converted GGUF) looks older than the run and is moved aside too -- never erased; the
    export then re-converts."""
    from app.services.compression_service import CONVERT_LOCK_NAME

    if not isinstance(created_at, datetime):
        return None
    if created_at.tzinfo is None:  # SQLite hands back naive UTC (the column default is datetime.now(utc))
        created_at = created_at.replace(tzinfo=timezone.utc)
    cutoff = created_at.timestamp()
    root = settings.DATA_DIR / "projects" / str(int(project_id)) / "compressed" / f"exp-{int(experiment_id)}"
    if not root.is_dir():
        return None
    stale = []
    for p in root.rglob("*"):
        try:
            if p.is_file() and p.name != CONVERT_LOCK_NAME and p.stat().st_mtime < cutoff:
                stale.append(p)
        except OSError:
            continue
    if not stale:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root.with_name(f"{root.name}.stale.{stamp}")
    counter = 0
    while target.exists():
        counter += 1
        target = root.with_name(f"{root.name}.stale.{stamp}-{counter}")
    moved = []
    for p in sorted(stale):
        rel = p.relative_to(root)
        dest = target / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
        except OSError as exc:
            raise _move_failed(
                f"{p} predates run #{experiment_id} (an earlier run with the same id wrote it) and could not "
                f"be moved aside ({exc.__class__.__name__}: {exc}). Nothing was deleted"
                + (f"; already moved to {target}: {', '.join(moved)}" if moved else "")
                + ". Fix the folder permissions or move it by hand, then export again.",
                {"from": str(p), "to": str(dest), "reason": "predates_run", "moved": list(moved)},
            ) from exc
        moved.append(str(rel))
    logger.warning("id_reuse_guard: moved %d file(s) predating run #%s from %s to %s", len(moved), experiment_id, root, target)
    return {"from": str(root), "to": str(target), "files": moved, "reason": "predates_run"}


# SQLAlchemy ``after_insert`` hooks (registered in the model modules): they run inside the flush that gives
# the row its id, so before any creator writes a file under that id. For a run, what was moved is left on the
# instance (not a mapped column) and ``create_experiment`` records it in the run config; for a project /
# export it is only logged.


def _project_after_insert(mapper, connection, target) -> None:  # noqa: ANN001 - SQLAlchemy event signature
    if getattr(target, "id", None) is None:
        return
    guard_new_project(int(target.id))  # what moved is logged (WARNING); nothing reads it back


def _experiment_after_insert(mapper, connection, target) -> None:  # noqa: ANN001
    if getattr(target, "id", None) is None or getattr(target, "project_id", None) is None:
        return
    archived = guard_new_experiment(int(target.project_id), int(target.id))
    if archived:
        target._stale_artifacts_archived = archived


def _export_after_insert(mapper, connection, target) -> None:  # noqa: ANN001
    if getattr(target, "id", None) is None or getattr(target, "project_id", None) is None:
        return
    guard_new_export_dir(int(target.project_id), int(target.id))  # logged (WARNING)


__all__ = [
    "archive_compression_files_predating_run",
    "archive_deleted_experiment",
    "archive_project_paths",
    "experiment_paths",
    "guard_new_experiment",
    "guard_new_export_dir",
    "guard_new_project",
    "in_archived_folder",
    "is_archived_name",
    "move_aside",
    "move_failed_message",
    "project_paths",
]
