"""Domain profile API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole
from app.schemas.domain_profile import (
    DomainProfileContract,
    DomainProfileResponse,
    DomainProfileSummaryResponse,
)
from app.security import get_request_principal
from app.services.domain_profile_service import (
    create_domain_profile,
    get_domain_profile,
    list_domain_profiles,
    update_domain_profile,
)

router = APIRouter(prefix="/domain-profiles", tags=["Domain Profiles"])


def _require_profile_write_access(request: Request) -> None:
    principal = get_request_principal(request)
    if not settings.AUTH_ENABLED:
        return
    if principal is None:
        raise HTTPException(401, "Authentication required")
    if principal.role not in {GlobalRole.ADMIN, GlobalRole.ENGINEER}:
        raise HTTPException(403, "Only admin or engineer can modify domain profiles")


@router.get("")
async def list_profiles(
    db: AsyncSession = Depends(get_db),
):
    """List available domain profiles."""
    profiles = await list_domain_profiles(db)
    return {
        "profiles": [DomainProfileSummaryResponse.model_validate(item) for item in profiles],
        "count": len(profiles),
    }


@router.get("/{profile_id}", response_model=DomainProfileResponse)
async def get_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a full domain profile contract by profile_id."""
    profile = await get_domain_profile(db, profile_id)
    if not profile:
        raise HTTPException(404, f"Domain profile '{profile_id}' not found")
    return DomainProfileResponse.model_validate(profile)


@router.post("", response_model=DomainProfileResponse, status_code=201)
async def create_profile(
    contract: DomainProfileContract,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a new domain profile contract."""
    _require_profile_write_access(request)
    try:
        profile = await create_domain_profile(db, contract)
        return DomainProfileResponse.model_validate(profile)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.put("/{profile_id}", response_model=DomainProfileResponse)
async def update_profile(
    profile_id: str,
    contract: DomainProfileContract,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Update an existing domain profile contract."""
    _require_profile_write_access(request)
    profile = await get_domain_profile(db, profile_id)
    if not profile:
        raise HTTPException(404, f"Domain profile '{profile_id}' not found")
    try:
        updated = await update_domain_profile(db, profile, contract)
        return DomainProfileResponse.model_validate(updated)
    except ValueError as e:
        raise HTTPException(400, str(e))
