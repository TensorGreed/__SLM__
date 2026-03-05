"""Domain pack API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole
from app.schemas.domain_pack import (
    DomainPackContract,
    DomainPackDuplicateRequest,
    DomainPackResponse,
    DomainPackSummaryResponse,
)
from app.security import get_request_principal
from app.services.domain_pack_service import (
    create_domain_pack,
    duplicate_domain_pack,
    get_domain_pack,
    list_domain_packs,
    update_domain_pack,
)

router = APIRouter(prefix="/domain-packs", tags=["Domain Packs"])


def _require_pack_write_access(request: Request) -> None:
    principal = get_request_principal(request)
    if not settings.AUTH_ENABLED:
        return
    if principal is None:
        raise HTTPException(401, "Authentication required")
    if principal.role not in {GlobalRole.ADMIN, GlobalRole.ENGINEER}:
        raise HTTPException(403, "Only admin or engineer can modify domain packs")


@router.get("")
async def list_packs(
    db: AsyncSession = Depends(get_db),
):
    """List available domain packs."""
    packs = await list_domain_packs(db)
    return {
        "packs": [DomainPackSummaryResponse.model_validate(item) for item in packs],
        "count": len(packs),
    }


@router.get("/{pack_id}", response_model=DomainPackResponse)
async def get_pack(
    pack_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a full domain pack contract by pack_id."""
    pack = await get_domain_pack(db, pack_id)
    if not pack:
        raise HTTPException(404, f"Domain pack '{pack_id}' not found")
    return DomainPackResponse.model_validate(pack)


@router.post("", response_model=DomainPackResponse, status_code=201)
async def create_pack(
    contract: DomainPackContract,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a new domain pack contract."""
    _require_pack_write_access(request)
    try:
        pack = await create_domain_pack(db, contract)
        return DomainPackResponse.model_validate(pack)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.put("/{pack_id}", response_model=DomainPackResponse)
async def update_pack(
    pack_id: str,
    contract: DomainPackContract,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Update an existing domain pack contract."""
    _require_pack_write_access(request)
    pack = await get_domain_pack(db, pack_id)
    if not pack:
        raise HTTPException(404, f"Domain pack '{pack_id}' not found")
    try:
        updated = await update_domain_pack(db, pack, contract)
        return DomainPackResponse.model_validate(updated)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{pack_id}/duplicate", response_model=DomainPackResponse, status_code=201)
async def duplicate_pack(
    pack_id: str,
    request_data: DomainPackDuplicateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Duplicate a domain pack with auto-generated pack_id/version unless overridden."""
    _require_pack_write_access(request)
    pack = await get_domain_pack(db, pack_id)
    if not pack:
        raise HTTPException(404, f"Domain pack '{pack_id}' not found")

    try:
        duplicated = await duplicate_domain_pack(
            db,
            pack,
            new_pack_id=request_data.new_pack_id,
            new_version=request_data.new_version,
            status_override=request_data.status.value,
        )
        return DomainPackResponse.model_validate(duplicated)
    except ValueError as e:
        raise HTTPException(400, str(e))
