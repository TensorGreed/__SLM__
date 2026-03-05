"""Project secret management API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.secret_service import (
    delete_project_secret,
    list_project_secrets,
    serialize_secret,
    upsert_project_secret,
)

router = APIRouter(prefix="/projects/{project_id}/secrets", tags=["Secrets"])


class SecretUpsertRequest(BaseModel):
    provider: str = Field(..., min_length=1, max_length=64)
    key_name: str = Field(..., min_length=1, max_length=64)
    value: str = Field(..., min_length=1, max_length=4096)


@router.get("")
async def list_secrets(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List configured project secrets with masked hints."""
    items = await list_project_secrets(db, project_id)
    return {"secrets": [serialize_secret(item) for item in items]}


@router.put("", status_code=201)
async def upsert_secret(
    project_id: int,
    req: SecretUpsertRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create/update a project secret value."""
    try:
        secret_obj = await upsert_project_secret(
            db=db,
            project_id=project_id,
            provider=req.provider,
            key_name=req.key_name,
            value=req.value,
        )
        return serialize_secret(secret_obj)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/{provider}/{key_name}")
async def delete_secret(
    project_id: int,
    provider: str,
    key_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a project secret."""
    deleted = await delete_project_secret(db, project_id, provider, key_name)
    if not deleted:
        raise HTTPException(404, "Secret not found")
    return {"status": "deleted", "provider": provider, "key_name": key_name}
