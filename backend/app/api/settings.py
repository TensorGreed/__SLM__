"""System settings API routes."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.models.auth import GlobalRole
from app.security import get_request_principal
from app.services.runtime_settings_service import (
    list_runtime_settings,
    update_runtime_settings,
)

router = APIRouter(prefix="/settings", tags=["Settings"])


class RuntimeSettingsUpdateRequest(BaseModel):
    updates: dict[str, object] = Field(default_factory=dict)


def _enforce_manage_settings_permission(request: Request) -> None:
    if not settings.AUTH_ENABLED:
        return
    principal = get_request_principal(request)
    if not principal:
        raise HTTPException(401, "Authentication required")
    if principal.role not in {GlobalRole.ADMIN, GlobalRole.ENGINEER}:
        raise HTTPException(403, "Only admin/engineer can manage system settings")


@router.get("/runtime")
async def get_runtime_settings(request: Request):
    """List runtime-manageable settings and their effective values."""
    _enforce_manage_settings_permission(request)
    return list_runtime_settings()


@router.put("/runtime")
async def put_runtime_settings(
    req: RuntimeSettingsUpdateRequest,
    request: Request,
):
    """Update runtime-manageable settings and persist overrides."""
    _enforce_manage_settings_permission(request)
    try:
        return update_runtime_settings(dict(req.updates or {}))
    except ValueError as e:
        raise HTTPException(400, str(e))

