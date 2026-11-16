"""SQL execution scoring settings API: opt-in execution match for projects whose answers are SQL queries
(the SQLite files to run them against, timeout, row cap, float tolerance, row-order rule, refusal
marker). Stored in ``Project.runtime_config["sql_execution"]``; defaults and the scorer live in
``services/sql_execution_scoring.py``. Database paths must resolve inside the project's data directory.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services import sql_execution_scoring as ses

router = APIRouter(prefix="/projects/{project_id}", tags=["SQL Execution Scoring"])

ERROR_CODE = "INVALID_SQL_EXECUTION_CONFIG"
_KEYS = ("enabled", "db_paths", "timeout_ms", "row_cap", "float_rel_tol", "order", "refusal_marker")


class SqlExecutionConfigBody(BaseModel):
    enabled: bool | None = None
    db_paths: list[str] | None = None
    timeout_ms: int | None = None
    row_cap: int | None = None
    float_rel_tol: float | None = None
    order: str | None = None
    refusal_marker: str | None = None


def _invalid(errors: list[str]) -> HTTPException:
    return HTTPException(422, {
        "error_code": ERROR_CODE,
        "message": errors[0] if errors else "Invalid SQL execution scoring config.",
        "metadata": {"errors": errors},
    })


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _databases(project_id: int, paths: list[str]) -> list[dict[str, Any]]:
    root = ses.project_data_dir(project_id)
    out: list[dict[str, Any]] = []
    for raw in paths:
        entry: dict[str, Any] = {"path": raw, "ok": False, "error": None, "size_bytes": None}
        try:
            real = ses.resolve_db_paths([raw], root)[0]
            entry.update({"ok": True, "size_bytes": os.path.getsize(real)})
        except (ses.SqlPathError, OSError) as exc:
            entry["error"] = str(exc)
        out.append(entry)
    return out


def _payload(project: Project) -> dict[str, Any]:
    rc = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    stored = ses.stored_config(rc)
    config = {key: stored.get(key) for key in _KEYS}
    config["updated_at"] = stored.get("updated_at")
    effective = ses.normalize_config(stored)
    # ``active`` = the scorer will run on the next eval (enabled with at least one database).
    effective["active"] = ses.effective_config(rc) is not None
    return {
        "config": config,
        "effective": effective,
        "databases": _databases(project.id, effective["db_paths"]),
        "data_dir": str(ses.project_data_dir(project.id)),
        "limits": {
            "max_db_paths": ses.MAX_DB_PATHS,
            "timeout_ms": list(ses.TIMEOUT_MS_RANGE),
            "row_cap": list(ses.ROW_CAP_RANGE),
            "float_rel_tol": list(ses.FLOAT_TOL_RANGE),
            "order": list(ses.ORDER_MODES),
        },
    }


@router.get("/sql-execution-config")
async def get_sql_execution_config(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return _payload(await _project(db, project_id))


@router.put("/sql-execution-config")
async def put_sql_execution_config(
    project_id: int, body: SqlExecutionConfigBody, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Partial merge (``null`` clears a key). 422 ``INVALID_SQL_EXECUTION_CONFIG`` with
    ``metadata.errors`` and nothing saved for a database path outside the project's data directory or
    missing, an out-of-range number, an unknown order rule or a bad refusal marker."""
    project = await _project(db, project_id)
    stored = dict(ses.stored_config(project.runtime_config))
    for key in body.model_fields_set:
        if key not in _KEYS:
            continue
        value = getattr(body, key)
        if value is None:
            stored.pop(key, None)
        else:
            stored[key] = value
    errors: list[str] = []
    if body.db_paths is not None:
        paths = [p.strip() for p in body.db_paths if p.strip()]
        if len(paths) > ses.MAX_DB_PATHS:
            errors.append(f"At most {ses.MAX_DB_PATHS} database files.")
        for entry in _databases(project_id, paths):
            if not entry["ok"]:
                errors.append(entry["error"])
        stored["db_paths"] = paths
    for key, (lo, hi) in (
        ("timeout_ms", ses.TIMEOUT_MS_RANGE), ("row_cap", ses.ROW_CAP_RANGE), ("float_rel_tol", ses.FLOAT_TOL_RANGE),
    ):
        value = getattr(body, key)
        if value is not None and not (lo <= value <= hi):
            errors.append(f"{key} must be between {lo} and {hi}.")
    if body.order is not None and body.order not in ses.ORDER_MODES:
        errors.append(f"order must be one of {list(ses.ORDER_MODES)}.")
    if body.refusal_marker is not None and (not body.refusal_marker.strip() or len(body.refusal_marker) > 64):
        errors.append("refusal_marker must be 1-64 characters.")
    if errors:
        raise _invalid(errors)

    cfg = dict(project.runtime_config or {})
    if stored:
        stored["updated_at"] = datetime.now(timezone.utc).isoformat()
        cfg[ses.CONFIG_KEY] = stored
    else:
        cfg.pop(ses.CONFIG_KEY, None)
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return _payload(project)
