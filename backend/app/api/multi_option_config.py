"""``multi_option`` scoring settings API (opt-in): formats, option range, strategy taxonomy, the option
verifier the project picks (by id, among the operator-registered ones -- never a command line),
``require_agreement``, ``row_pass``. Stored in ``Project.runtime_config["multi_option"]``; see
``services/multi_option_scoring.py`` and ``services/external_checks.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services import external_checks
from app.services import multi_option_scoring as mos

router = APIRouter(prefix="/projects/{project_id}", tags=["Multi-option scoring"])

ERROR_CODE = "INVALID_MULTI_OPTION_CONFIG"
_KEYS = ("enabled", "formats", "min_options", "max_options", "strategy_taxonomy", "verifier", "require_agreement",
         "row_pass", "instruction")


class MultiOptionConfigBody(BaseModel):
    enabled: bool | None = None
    formats: list[str] | None = None
    min_options: int | None = None
    max_options: int | None = None
    strategy_taxonomy: list[str] | None = None
    verifier: dict[str, Any] | None = None
    require_agreement: bool | None = None
    row_pass: str | None = None
    instruction: str | None = None


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _payload(project: Project) -> dict[str, Any]:
    from app.services.eval_task_handler_service import read_prepared_manifest

    rc = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    manifest_schema = read_prepared_manifest(project.id).get("output_schema")
    effective = mos.effective_config(rc, manifest_schema if isinstance(manifest_schema, dict) else None)
    diagnostics = external_checks.ensure_loaded()
    return {
        "config": mos.stored_config(rc),
        "enabled": mos.project_enabled(rc),
        "effective": effective,
        # Read-only: what the operator registered (the project picks one by id).
        "verifiers": external_checks.list_option_verifiers(),
        "external_checks": {"configured": diagnostics["configured"], "errors": len(diagnostics["errors"])},
        "fingerprint": mos.multi_option_fingerprint(effective),
    }


@router.get("/multi-option-config")
async def get_multi_option_config(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return _payload(await _project(db, project_id))


@router.put("/multi-option-config")
async def put_multi_option_config(
    project_id: int, body: MultiOptionConfigBody, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Partial merge (``null`` clears a key). 422 ``INVALID_MULTI_OPTION_CONFIG`` with ``metadata.errors``
    and nothing saved when a value is invalid, the verifier id is not a registered option verifier, or
    the config is enabled with no format (here or in the prepared manifest)."""
    project = await _project(db, project_id)
    stored = mos.stored_config(project.runtime_config)
    for key in body.model_fields_set:
        if key not in _KEYS:
            continue
        value = getattr(body, key)
        if value is None:
            stored.pop(key, None)
        else:
            stored[key] = value
    if isinstance(stored.get("verifier"), dict):
        # Only the id is ever stored: a project never supplies a command line.
        stored["verifier"] = {"check_id": stored["verifier"].get("check_id")}
    # ``enabled`` is checked against the EFFECTIVE formats below (the manifest may supply them).
    _cfg, errors = mos.normalize_config({k: v for k, v in stored.items() if k != "enabled"})
    if "enabled" in stored and not isinstance(stored["enabled"], bool):
        errors.append("enabled must be true or false")
    if stored.get("enabled") is True and not errors:
        from app.services.eval_task_handler_service import read_prepared_manifest

        manifest_schema = read_prepared_manifest(project.id).get("output_schema")
        effective = mos.effective_config(
            {mos.CONFIG_KEY: stored}, manifest_schema if isinstance(manifest_schema, dict) else None
        )
        if not effective["formats"]:
            # An option with no format to check would pass vacuously.
            errors.append("formats: an enabled multi_option config needs at least one format")
    verifier_id = (stored.get("verifier") or {}).get("check_id") if isinstance(stored.get("verifier"), dict) else None
    if verifier_id:
        registered = {v["id"] for v in external_checks.list_option_verifiers()}
        if verifier_id not in registered:
            errors.append(f"'{verifier_id}' is not an option verifier registered by the operator")
    if errors:
        raise HTTPException(422, {
            "error_code": ERROR_CODE,
            "message": errors[0],
            "metadata": {"errors": errors},
        })
    cfg = dict(project.runtime_config or {})
    if stored:
        stored["updated_at"] = datetime.now(timezone.utc).isoformat()
        cfg[mos.CONFIG_KEY] = stored
    else:
        cfg.pop(mos.CONFIG_KEY, None)
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return _payload(project)
