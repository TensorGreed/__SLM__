"""Project template catalog + instantiation API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.project import ProjectResponse
from app.services.project_template_service import (
    get_project_template,
    instantiate_project_template,
    list_project_templates,
)


router = APIRouter(prefix="/project-templates", tags=["Project Templates"])


class InstantiateTemplateRequest(BaseModel):
    project_name: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "Name for the new project. Defaults to the template's "
            "`name` (e.g. 'Ticket Router SLM') if omitted. Name "
            "collisions get a ' (2)', ' (3)' suffix automatically — "
            "templates are designed to spawn multiple projects."
        ),
    )


@router.get("")
async def list_templates_endpoint():
    """List every available project template."""
    templates = list_project_templates()
    return {
        "templates": templates,
        "count": len(templates),
    }


@router.get("/{slug}")
async def get_template_endpoint(slug: str):
    """Single-template detail. Returns 404 when the slug is unknown."""
    template = get_project_template(slug)
    if template is None:
        raise HTTPException(404, f"template_slug_unknown:{slug}")
    return template


@router.post("/{slug}/instantiate", status_code=201, response_model=ProjectResponse)
async def instantiate_template_endpoint(
    slug: str,
    data: InstantiateTemplateRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Create a new project from a template. The same template can
    back any number of projects — name collisions get a numeric
    suffix automatically."""
    requested_name = (data.project_name if data else None) or None
    try:
        project, _summary = await instantiate_project_template(
            db,
            slug,
            project_name=requested_name,
        )
    except ValueError as e:
        detail = str(e)
        if detail.startswith("template_slug_unknown:"):
            raise HTTPException(404, detail)
        if detail.startswith("template_manifest_invalid:"):
            raise HTTPException(400, detail)
        raise HTTPException(400, detail)
    return ProjectResponse.model_validate(project)
