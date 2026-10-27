"""Structured-plan settings API: the JSON Schema bundle plans must validate against, the optional
rubric, the JSON pointers that are field-matched, the judge toggle and passage retrieval. Stored in
``Project.runtime_config["structured_plan"]``; see ``services/structured_plan_prompt.py`` (defaults)
and ``services/structured_plan_scoring.py``.

A valid schema also writes ``runtime_config["guided_json"]`` (the dereferenced schema, source
``structured_plan``) so a serving layer can constrain generation; clearing the schema drops it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services import json_schema_bundle as jsb
from app.services import rubric_service as rs
from app.services import structured_plan_prompt as spp
from app.services.json_pointer import split_json_pointer

router = APIRouter(prefix="/projects/{project_id}", tags=["Structured Plan"])

ERROR_CODE = "INVALID_STRUCTURED_PLAN_CONFIG"
_KEYS = ("schema_files", "root_schema", "rubric_yaml", "pointers", "judge_enabled", "retrieve_passages", "top_k")


class StructuredPlanConfigBody(BaseModel):
    schema_files: dict[str, Any] | None = None
    root_schema: str | None = None
    rubric_yaml: str | None = None
    pointers: list[str] | None = None
    judge_enabled: bool | None = None
    retrieve_passages: bool | None = None
    top_k: int | None = None


class ValidateBody(BaseModel):
    value: Any = None


def _invalid(errors: list[str]) -> HTTPException:
    return HTTPException(422, {
        "error_code": ERROR_CODE,
        "message": errors[0] if errors else "Invalid structured-plan config.",
        "metadata": {"errors": errors},
    })


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _schema_block(stored: dict[str, Any]) -> dict[str, Any] | None:
    files, root = stored.get("schema_files"), stored.get("root_schema")
    if not isinstance(files, dict) or not files:
        return None
    names = sorted(files)
    block: dict[str, Any] = {"root": root, "files": names, "sha256": None, "valid": False, "errors": []}
    if not isinstance(root, str) or not root:
        block["errors"] = ["root_schema is not set."]
        return block
    try:
        bundle = jsb.validate_bundle(files, root)
    except jsb.SchemaBundleError as exc:
        block["errors"] = list(exc.errors)
        return block
    block.update({"sha256": bundle.sha256, "valid": True})
    return block


def _rubric_block(stored: dict[str, Any]) -> dict[str, Any] | None:
    text = stored.get("rubric_yaml")
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        rubric = rs.parse_rubric(text)
    except rs.RubricError:
        return None
    return {
        "id": rubric.id,
        "title": rubric.title,
        "sha256": rubric.sha256,
        "criteria": [
            {"id": c.id, "description": c.description, "weight": c.weight, "critical": c.critical}
            for c in rubric.criteria
        ],
    }


def _payload(project: Project) -> dict[str, Any]:
    rc = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    stored = spp.stored_config(rc)
    config = {
        "schema_files": stored.get("schema_files") if isinstance(stored.get("schema_files"), dict) else None,
        "root_schema": stored.get("root_schema") if isinstance(stored.get("root_schema"), str) else None,
        "rubric_yaml": stored.get("rubric_yaml") if isinstance(stored.get("rubric_yaml"), str) else None,
        "pointers": stored.get("pointers") if isinstance(stored.get("pointers"), list) else None,
        "judge_enabled": stored.get("judge_enabled") if isinstance(stored.get("judge_enabled"), bool) else None,
        "retrieve_passages": stored.get("retrieve_passages") if isinstance(stored.get("retrieve_passages"), bool) else None,
        "top_k": stored.get("top_k") if isinstance(stored.get("top_k"), int) else None,
        "updated_at": stored.get("updated_at"),
    }
    guided = rc.get("guided_json")
    return {
        "config": config,
        "effective": spp.effective_structured_plan_config(rc, project.id),
        "schema": _schema_block(stored),
        "rubric": _rubric_block(stored),
        "guided_json_present": isinstance(guided, dict) and bool(guided),
    }


@router.get("/structured-plan-config")
async def get_structured_plan_config(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return _payload(await _project(db, project_id))


@router.put("/structured-plan-config")
async def put_structured_plan_config(
    project_id: int, body: StructuredPlanConfigBody, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Partial merge (``null`` clears a key). 422 ``INVALID_STRUCTURED_PLAN_CONFIG`` with
    ``metadata.errors`` and nothing saved when the schema bundle, the rubric or a pointer is invalid;
    409 ``NOT_STRUCTURED_PLAN`` for a project on another recipe."""
    project = await _project(db, project_id)
    if not spp.project_is_structured_plan(project):
        raise HTTPException(409, {"error_code": "NOT_STRUCTURED_PLAN", "message": "Pick the structured-plan task type first."})
    stored = dict(spp.stored_config(project.runtime_config))
    errors: list[str] = []
    for key in body.model_fields_set:
        if key not in _KEYS:
            continue
        value = getattr(body, key)
        if value is None:
            stored.pop(key, None)
        else:
            stored[key] = value
    if "schema_files" in body.model_fields_set and body.schema_files is None and "root_schema" not in body.model_fields_set:
        stored.pop("root_schema", None)

    bundle = None
    touched_schema = bool({"schema_files", "root_schema"} & body.model_fields_set)
    if touched_schema and stored.get("schema_files"):
        try:
            if not isinstance(stored.get("root_schema"), str) or not stored.get("root_schema"):
                errors.append("root_schema is required with schema_files.")
            else:
                bundle = await asyncio.to_thread(jsb.validate_bundle, stored["schema_files"], stored["root_schema"])
        except jsb.SchemaBundleError as exc:
            errors.extend(exc.errors)
    elif touched_schema and stored.get("root_schema"):
        errors.append("root_schema needs schema_files.")
    if "rubric_yaml" in body.model_fields_set and stored.get("rubric_yaml"):
        try:
            rs.parse_rubric(stored["rubric_yaml"])
        except rs.RubricError as exc:
            errors.extend(exc.errors)
    if "pointers" in body.model_fields_set and body.pointers is not None:
        if len(body.pointers) > spp.MAX_POINTERS:
            errors.append(f"At most {spp.MAX_POINTERS} pointers.")
        for ptr in body.pointers:
            if split_json_pointer(ptr) is None:
                errors.append(f"Invalid JSON pointer: {ptr!r}.")
    if "top_k" in body.model_fields_set and body.top_k is not None and not 1 <= body.top_k <= spp.MAX_TOP_K:
        errors.append(f"top_k must be between 1 and {spp.MAX_TOP_K}.")
    if errors:
        raise _invalid(errors)

    now = datetime.now(timezone.utc).isoformat()
    cfg = dict(project.runtime_config or {})
    if stored:
        stored["updated_at"] = now
        cfg[spp.CONFIG_KEY] = stored
    else:
        cfg.pop(spp.CONFIG_KEY, None)
    if touched_schema:
        hint = cfg.get("guided_json")
        ours = isinstance(hint, dict) and hint.get("source") == spp.GUIDED_JSON_SOURCE
        if bundle is not None:
            # Never clobber another recipe's hint (a code-findings hint has a different source).
            if hint is None or ours:
                try:
                    flat = bundle.dereferenced()
                except jsb.SchemaBundleError as exc:
                    raise _invalid(exc.errors) from None
                cfg["guided_json"] = {
                    "schema": flat,
                    "source": spp.GUIDED_JSON_SOURCE,
                    "schema_sha256": bundle.sha256,
                    "updated_at": now,
                }
        elif ours:
            cfg.pop("guided_json", None)
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return _payload(project)


@router.post("/structured-plan/validate")
async def validate_structured_plan_value(
    project_id: int, body: ValidateBody, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Validate ``body.value`` against the saved schema bundle (first 20 errors)."""
    project = await _project(db, project_id)
    cfg = spp.effective_structured_plan_config(project.runtime_config, project_id)

    def _check() -> tuple[Any, list[dict[str, Any]]]:
        found = spp.bundle_for_config(cfg)
        return found, (found.validate(body.value) if found is not None else [])

    # Off the event loop: schema validation (regex patterns, deep documents) is CPU-bound.
    bundle, errors = await asyncio.to_thread(_check)
    if bundle is None:
        raise HTTPException(409, {"error_code": "NO_STRUCTURED_PLAN_SCHEMA", "message": "Save a valid schema first."})
    return {"valid": not errors, "errors": errors[:20]}
