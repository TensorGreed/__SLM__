"""Authentication, user management, and project membership routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole, ProjectMembership, ProjectRole, User
from app.security import (
    create_user_with_key,
    get_request_principal,
    upsert_project_membership,
)

router = APIRouter(prefix="/auth", tags=["Auth"])


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    role: GlobalRole = GlobalRole.ENGINEER
    key_name: str = Field(default="default", min_length=1, max_length=128)
    api_key: str | None = None


class SetMembershipRequest(BaseModel):
    user_id: int
    role: ProjectRole = ProjectRole.VIEWER


@router.get("/me")
async def me(request: Request, db: AsyncSession = Depends(get_db)):
    """Return current authenticated principal and project memberships."""
    principal = get_request_principal(request)
    if settings.AUTH_ENABLED and not principal:
        raise HTTPException(401, "Authentication required")

    if not settings.AUTH_ENABLED:
        return {"auth_enabled": False, "principal": None, "memberships": []}

    memberships_result = await db.execute(
        select(ProjectMembership).where(ProjectMembership.user_id == principal.user_id)
    )
    memberships = memberships_result.scalars().all()

    return {
        "auth_enabled": True,
        "principal": {
            "user_id": principal.user_id,
            "username": principal.username,
            "role": principal.role.value,
            "api_key_prefix": principal.api_key_prefix,
        },
        "memberships": [
            {
                "project_id": m.project_id,
                "role": m.role.value,
                "created_at": m.created_at.isoformat(),
            }
            for m in memberships
        ],
    }


@router.post("/users", status_code=201)
async def create_user(
    data: CreateUserRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a user and API key (admin only when auth is enabled)."""
    principal = get_request_principal(request)
    if settings.AUTH_ENABLED:
        if not principal:
            raise HTTPException(401, "Authentication required")
        if principal.role != GlobalRole.ADMIN:
            raise HTTPException(403, "Only admin can create users")

    try:
        created = await create_user_with_key(
            db=db,
            username=data.username,
            role=data.role,
            key_name=data.key_name,
            api_key=data.api_key,
        )
        return created
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/projects/{project_id}/members")
async def list_members(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """List project memberships."""
    principal = get_request_principal(request)
    if settings.AUTH_ENABLED:
        if not principal:
            raise HTTPException(401, "Authentication required")
        if principal.role != GlobalRole.ADMIN:
            access_result = await db.execute(
                select(ProjectMembership).where(
                    ProjectMembership.project_id == project_id,
                    ProjectMembership.user_id == principal.user_id,
                )
            )
            if not access_result.scalar_one_or_none():
                raise HTTPException(403, "No access to this project")

    result = await db.execute(
        select(ProjectMembership, User)
        .join(User, User.id == ProjectMembership.user_id)
        .where(ProjectMembership.project_id == project_id)
        .order_by(User.username.asc())
    )
    rows = result.all()
    return {
        "project_id": project_id,
        "members": [
            {
                "user_id": membership.user_id,
                "username": user.username,
                "global_role": user.role.value,
                "project_role": membership.role.value,
                "created_at": membership.created_at.isoformat(),
            }
            for membership, user in rows
        ],
    }


@router.post("/projects/{project_id}/members", status_code=201)
async def set_member(
    project_id: int,
    data: SetMembershipRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Upsert project membership (admin or project owner)."""
    principal = get_request_principal(request)
    if settings.AUTH_ENABLED:
        if not principal:
            raise HTTPException(401, "Authentication required")
        if principal.role != GlobalRole.ADMIN:
            membership_result = await db.execute(
                select(ProjectMembership).where(
                    ProjectMembership.project_id == project_id,
                    ProjectMembership.user_id == principal.user_id,
                    ProjectMembership.role == ProjectRole.OWNER,
                )
            )
            if not membership_result.scalar_one_or_none():
                raise HTTPException(403, "Only project owner or admin can update memberships")

    user_result = await db.execute(select(User).where(User.id == data.user_id))
    if not user_result.scalar_one_or_none():
        raise HTTPException(404, f"User {data.user_id} not found")

    membership = await upsert_project_membership(db, project_id, data.user_id, data.role)
    return {
        "project_id": membership.project_id,
        "user_id": membership.user_id,
        "role": membership.role.value,
    }
