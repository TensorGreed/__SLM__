"""Authentication, user management, and project membership routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from authlib.integrations.starlette_client import OAuth
import jwt
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole, ProjectMembership, ProjectRole, User
from app.security import (
    create_user_with_key,
    get_request_principal,
    upsert_project_membership,
)

router = APIRouter(prefix="/auth", tags=["Auth"])

oauth = OAuth()
if settings.OIDC_CLIENT_ID and settings.OIDC_CLIENT_SECRET and settings.OIDC_DISCOVERY_URL:
    oauth.register(
        name="sso",
        client_id=settings.OIDC_CLIENT_ID,
        client_secret=settings.OIDC_CLIENT_SECRET,
        server_metadata_url=settings.OIDC_DISCOVERY_URL,
        client_kwargs={"scope": "openid email profile"},
    )


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    role: GlobalRole = GlobalRole.ENGINEER
    key_name: str = Field(default="default", min_length=1, max_length=128)
    api_key: str | None = None


class SetMembershipRequest(BaseModel):
    user_id: int
    role: ProjectRole = ProjectRole.VIEWER


class LocalLoginRequest(BaseModel):
    username: str
    password: str


@router.get("/config")
async def get_config():
    """Return public authentication configuration."""
    return {
        "auth_enabled": settings.AUTH_ENABLED,
        "sso_enabled": bool(settings.OIDC_CLIENT_ID and settings.OIDC_CLIENT_SECRET and settings.OIDC_DISCOVERY_URL),
    }


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
        "sso_enabled": bool(settings.OIDC_CLIENT_ID and settings.OIDC_CLIENT_SECRET and settings.OIDC_DISCOVERY_URL),
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

@router.get("/sso/login")
async def sso_login(request: Request):
    """Initiates the SSO OAuth2 flow."""
    if not oauth.sso:
        raise HTTPException(status_code=400, detail="SSO not configured")
    
    # Needs to match the route we register below
    redirect_uri = request.url_for("sso_callback")
    return await oauth.sso.authorize_redirect(request, redirect_uri)

@router.get("/sso/callback", name="sso_callback")
async def sso_callback(request: Request, db: AsyncSession = Depends(get_db)):
    """Handles the SSO callback and issues a local JWT token."""
    if not oauth.sso:
        raise HTTPException(status_code=400, detail="SSO not configured")
        
    try:
        # Fetch the token and userinfo from the IdP
        token = await oauth.sso.authorize_access_token(request)
        userinfo = token.get("userinfo")
        if not userinfo:
            userinfo = await oauth.sso.userinfo(token=token)
            
        email = userinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="No email provided by SSO provider")
            
        # Here we upsert the user based on email (treat email as username for SSO)
        username = email
        user_result = await db.execute(select(User).where(User.username == username))
        user = user_result.scalar_one_or_none()
        
        if not user:
            # First time logging in with SSO -> auto provision as engineer
            user_info = await create_user_with_key(
                db=db,
                username=username,
                role=GlobalRole.ENGINEER,
                key_name="sso_default_key"
            )
            user_id = str(user_info["user_id"])
            user_name = user_info["username"]
            user_role = user_info["role"]
        else:
            user_id = str(user.id)
            user_name = user.username
            user_role = user.role.value
            
        # Issue our internal JWT for the frontend session
        payload = {
            "sub": user_id,
            "username": user_name,
            "role": user_role,
            "exp": datetime.now(timezone.utc) + timedelta(days=7)
        }
        internal_token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
        
        # In a real app we might redirect to frontend with ?token=... or set a cookie
        # For this implementation, redirect to frontend root with the token in URL
        # Assuming frontend runs on localhost:5173 for local dev 
        # (This should be configured via env var in production)
        frontend_url = "http://localhost:5173"
        return RedirectResponse(f"{frontend_url}/?token={internal_token}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SSO authentication failed: {str(e)}")


@router.post("/local/login")
async def local_login(data: LocalLoginRequest, db: AsyncSession = Depends(get_db)):
    """Local login fallback when SSO is not configured."""
    if not settings.AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Authentication is disabled")

    # Check password against global API_KEY for local dev mode
    if data.password != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user_result = await db.execute(select(User).where(User.username == data.username))
    user = user_result.scalar_one_or_none()
        
    if not user:
        user_info = await create_user_with_key(
            db=db,
            username=data.username,
            role=GlobalRole.ENGINEER,
            key_name="local_default_key"
        )
        user_id = str(user_info["user_id"])
        user_name = user_info["username"]
        user_role = user_info["role"]
    else:
        user_id = str(user.id)
        user_name = user.username
        user_role = user.role.value
        
    payload = {
        "sub": user_id,
        "username": user_name,
        "role": user_role,
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    internal_token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
    
    return {"token": internal_token}


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
