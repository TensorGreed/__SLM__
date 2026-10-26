"""Label taxonomy API: the project's controlled vocabulary of labels, with
crosswalks to reference ids in other namespaces. See
``services/label_taxonomy_service.py``. Stored in ``Project.runtime_config``.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services import label_taxonomy_service as lts
from app.services.code_findings_prompt import refresh_guided_json

router = APIRouter(prefix="/projects/{project_id}/label-taxonomy", tags=["Label Taxonomy"])
guided_json_router = APIRouter(prefix="/projects/{project_id}/guided-json", tags=["Label Taxonomy"])


class TaxonomyBody(BaseModel):
    content: str | None = Field(default=None, description="Taxonomy text (YAML or JSON).")
    entries: list[dict[str, Any]] | None = Field(default=None, description="Entries as JSON; send either this or content.")
    format: Literal["auto", "yaml", "json"] = "auto"
    crosswalk_from: list[str] = Field(default_factory=list, description="Entry fields holding 'NAMESPACE:ref' strings to turn into crosswalk entries.")
    name: str | None = None


class RestoreBody(BaseModel):
    version: str = Field(..., min_length=1)


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


def _invalid(errors: list[dict[str, Any]]) -> HTTPException:
    first = errors[0] if errors else {"message": "Invalid taxonomy."}
    suffix = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    return HTTPException(422, {
        "error_code": "INVALID_TAXONOMY",
        "message": f"{first['message']}{suffix}",
        "metadata": {"errors": errors},
    })


def _parse(body: TaxonomyBody) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if (body.content is None) == (body.entries is None):
        return None, [{"code": "bad_request", "id": None, "message": "Send exactly one of 'content' or 'entries'."}]
    source: Any = body.content if body.content is not None else {"entries": body.entries}
    return lts.parse_taxonomy_lenient(source, fmt=body.format, crosswalk_from=body.crosswalk_from, name=body.name)


@router.get("")
async def get_taxonomy(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return lts.taxonomy_status(await _project(db, project_id))


@router.put("")
async def put_taxonomy(project_id: int, body: TaxonomyBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Parse, validate and store the taxonomy. 422 ``INVALID_TAXONOMY`` with
    ``metadata.errors`` when anything fails — nothing is saved."""
    project = await _project(db, project_id)
    tax, errors = _parse(body)
    if tax is not None:
        errors = errors + lts.validate_taxonomy(tax)
    if errors or tax is None:
        raise _invalid(errors)
    lts.set_project_taxonomy(project, tax)
    refresh_guided_json(project)
    await db.commit()
    return lts.taxonomy_status(project)


@router.post("/validate")
async def validate_taxonomy(project_id: int, body: TaxonomyBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    await _project(db, project_id)
    tax, errors = _parse(body)
    warnings: list[dict[str, Any]] = []
    if tax is not None:
        errors = errors + lts.validate_taxonomy(tax)
        warnings = lts.taxonomy_warnings(tax)
    return {
        "valid": not errors and tax is not None,
        "errors": errors,
        "warnings": warnings,
        "entry_count": len(tax["entries"]) if tax else 0,
        "version": lts.taxonomy_version(tax) if tax and not errors else None,
    }


@router.post("/restore")
async def restore_taxonomy(project_id: int, body: RestoreBody, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _project(db, project_id)
    outcome = lts.restore_version(project, body.version)
    if outcome == "unknown":
        raise HTTPException(404, f"Taxonomy version '{body.version}' not found")
    if outcome == "stub":
        raise HTTPException(409, f"Taxonomy version '{body.version}' can no longer be restored: its snapshot is no longer kept.")
    refresh_guided_json(project)
    await db.commit()
    return lts.taxonomy_status(project)


@router.delete("")
async def delete_taxonomy(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Remove the taxonomy (its versions stay in the history for restore)."""
    project = await _project(db, project_id)
    lts.set_project_taxonomy(project, None)
    refresh_guided_json(project)
    await db.commit()
    return lts.taxonomy_status(project)


@router.get("/resolve")
async def resolve(project_id: int, label: str = Query(..., min_length=0), db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    project = await _project(db, project_id)
    tax = lts.project_taxonomy(project)
    index = lts.TaxonomyIndex(tax) if tax else None
    label_id, matched_by = index.resolve(label) if index else (None, None)
    return {
        "raw": label, "id": label_id, "matched_by": matched_by,
        "title": index.title(label_id) if index and label_id else None,
        "candidates": index.candidates(label) if index and matched_by == "ambiguous" else [],
    }


@guided_json_router.get("")
async def get_guided_json(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """The JSON Schema a serving layer can constrain generation with (code-findings
    projects), or ``null``. Refreshed whenever the taxonomy changes."""
    project = await _project(db, project_id)
    cfg = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    hint = cfg.get("guided_json")
    return {"guided_json": hint if isinstance(hint, dict) else None}
