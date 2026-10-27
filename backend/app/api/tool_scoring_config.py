"""Tool-scoring settings API: how predicted tool calls are scored against the project's tool catalog
(modes, where a plan keeps its steps, argument keys, the optional read-only dry run). Stored in
``Project.runtime_config["tool_scoring"]``; defaults and the scorer live in
``services/tool_call_scoring.py``. The settings only take effect when the project has a tool catalog.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services import tool_call_scoring as tcs
from app.services.json_pointer import split_json_pointer

router = APIRouter(prefix="/projects/{project_id}", tags=["Tool Scoring"])

ERROR_CODE = "INVALID_TOOL_SCORING_CONFIG"
_KEYS = (
    "enabled", "modes", "steps_pointer", "tool_field", "args_field", "id_field", "depends_field",
    "arg_keys", "dry_run",
)
_FIELD_KEYS = ("tool_field", "args_field", "id_field", "depends_field")


class ToolScoringConfigBody(BaseModel):
    enabled: bool | None = None
    modes: list[str] | None = None
    steps_pointer: str | None = None
    tool_field: str | None = None
    args_field: str | None = None
    id_field: str | None = None
    depends_field: str | None = None
    arg_keys: list[str] | None = None
    dry_run: bool | None = None


def _invalid(errors: list[str]) -> HTTPException:
    return HTTPException(422, {
        "error_code": ERROR_CODE,
        "message": errors[0] if errors else "Invalid tool-scoring config.",
        "metadata": {"errors": errors},
    })


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _payload(project: Project) -> dict[str, Any]:
    rc = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    stored = tcs.stored_config(rc)
    config = {key: stored.get(key) for key in _KEYS}
    config["updated_at"] = stored.get("updated_at")
    cat = rc.get("tool_catalog") if isinstance(rc.get("tool_catalog"), dict) else None
    catalog = None
    if cat and cat.get("tools"):
        source = cat.get("source")
        catalog = {
            "version": cat.get("version"),
            "tool_count": len(cat["tools"]),
            "source_kind": source.get("kind") if isinstance(source, dict) else None,
        }
    effective = tcs.normalize_config(stored)
    # ``active`` = the scorer will actually run (a catalog exists and the settings are enabled).
    effective["active"] = tcs.effective_tool_scoring_config(rc) is not None
    effective["dry_run_available"] = tcs.dry_run_env_enabled()
    return {"config": config, "effective": effective, "catalog": catalog}


@router.get("/tool-scoring-config")
async def get_tool_scoring_config(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return _payload(await _project(db, project_id))


@router.put("/tool-scoring-config")
async def put_tool_scoring_config(
    project_id: int, body: ToolScoringConfigBody, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Partial merge (``null`` clears a key). 422 ``INVALID_TOOL_SCORING_CONFIG`` with
    ``metadata.errors`` and nothing saved for a bad mode, pointer or field name, or for ``dry_run: true``
    while the admin switch ``MCP_DRY_RUN_ENABLED`` is off."""
    project = await _project(db, project_id)
    stored = dict(tcs.stored_config(project.runtime_config))
    errors: list[str] = []
    for key in body.model_fields_set:
        if key not in _KEYS:
            continue
        value = getattr(body, key)
        if value is None:
            stored.pop(key, None)
        else:
            stored[key] = value
    if body.modes is not None:
        bad = [m for m in body.modes if m not in tcs.ALL_MODES]
        if bad:
            errors.append(f"Unknown modes {bad}; use any of {list(tcs.ALL_MODES)}.")
    if body.steps_pointer is not None and split_json_pointer(body.steps_pointer) is None:
        errors.append(f"Invalid JSON pointer: {body.steps_pointer!r}.")
    for key in _FIELD_KEYS:
        value = getattr(body, key)
        if value is not None and (not value.strip() or len(value) > 64):
            errors.append(f"{key} must be 1-64 characters.")
    if body.arg_keys is not None:
        if len(body.arg_keys) > tcs.MAX_ARG_KEYS:
            errors.append(f"At most {tcs.MAX_ARG_KEYS} arg_keys.")
        if any(not k.strip() for k in body.arg_keys):
            errors.append("arg_keys must not contain empty names.")
    if body.dry_run and not tcs.dry_run_env_enabled():
        errors.append(
            "dry_run is disabled on this server: an administrator must set MCP_DRY_RUN_ENABLED=true first."
        )
    if errors:
        raise _invalid(errors)

    cfg = dict(project.runtime_config or {})
    if stored:
        stored["updated_at"] = datetime.now(timezone.utc).isoformat()
        cfg[tcs.CONFIG_KEY] = stored
    else:
        cfg.pop(tcs.CONFIG_KEY, None)
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return _payload(project)
