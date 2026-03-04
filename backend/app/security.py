"""Authentication, authorization, and audit helpers."""

import hashlib
import re
import secrets
import jwt
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory, get_db
from app.models.auth import ApiKey, GlobalRole, ProjectMembership, ProjectRole, User

PROJECT_PATH_RE = re.compile(r"^/api/projects/(?P<project_id>\d+)(?:/|$)")

PROJECT_ROLE_RANK: dict[ProjectRole, int] = {
    ProjectRole.VIEWER: 1,
    ProjectRole.EDITOR: 2,
    ProjectRole.OWNER: 3,
}


@dataclass
class Principal:
    user_id: int
    username: str
    role: GlobalRole
    api_key_id: int
    api_key_prefix: str


def hash_api_key(raw_key: str) -> str:
    """Hash API key for DB lookup."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _parse_api_key_from_headers(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key.strip()

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    return None


def extract_project_id_from_path(path: str) -> int | None:
    match = PROJECT_PATH_RE.match(path)
    if not match:
        return None
    return int(match.group("project_id"))


def get_request_principal(request: Request) -> Principal | None:
    principal = getattr(request.state, "principal", None)
    if isinstance(principal, Principal):
        return principal
    return None


async def ensure_bootstrap_auth() -> None:
    """
    Ensure one bootstrap admin account and API key exist when auth is enabled.

    This allows secure API usage in dev/prod with explicit key-based access.
    """
    if not settings.AUTH_ENABLED or not settings.AUTH_BOOTSTRAP_API_KEY:
        return

    try:
        bootstrap_role = GlobalRole(settings.AUTH_BOOTSTRAP_ROLE.lower())
    except ValueError:
        bootstrap_role = GlobalRole.ADMIN

    username = settings.AUTH_BOOTSTRAP_USERNAME.strip() or "admin"
    key = settings.AUTH_BOOTSTRAP_API_KEY.strip()
    key_hash = hash_api_key(key)
    key_prefix = key[:8]

    async with async_session_factory() as db:
        user_res = await db.execute(select(User).where(User.username == username))
        user = user_res.scalar_one_or_none()
        if not user:
            user = User(username=username, role=bootstrap_role, is_active=True)
            db.add(user)
            await db.flush()
            await db.refresh(user)

        key_res = await db.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
        key_obj = key_res.scalar_one_or_none()
        if not key_obj:
            key_obj = ApiKey(
                user_id=user.id,
                name="bootstrap",
                key_hash=key_hash,
                key_prefix=key_prefix,
                is_active=True,
            )
            db.add(key_obj)

        await db.commit()


async def create_user_with_key(
    db: AsyncSession,
    username: str,
    role: GlobalRole,
    key_name: str = "default",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Create a user and API key; returns generated key in plaintext once."""
    existing_user = await db.execute(select(User).where(User.username == username))
    if existing_user.scalar_one_or_none():
        raise ValueError(f"User '{username}' already exists")

    user = User(username=username, role=role, is_active=True)
    db.add(user)
    await db.flush()
    await db.refresh(user)

    plain_key = api_key or secrets.token_urlsafe(32)
    api_key_obj = ApiKey(
        user_id=user.id,
        name=key_name,
        key_hash=hash_api_key(plain_key),
        key_prefix=plain_key[:8],
        is_active=True,
    )
    db.add(api_key_obj)
    await db.flush()

    return {
        "user_id": user.id,
        "username": user.username,
        "role": user.role.value,
        "api_key": plain_key,
        "api_key_prefix": api_key_obj.key_prefix,
    }


async def upsert_project_membership(
    db: AsyncSession,
    project_id: int,
    user_id: int,
    role: ProjectRole,
) -> ProjectMembership:
    """Create or update project membership."""
    res = await db.execute(
        select(ProjectMembership).where(
            ProjectMembership.project_id == project_id,
            ProjectMembership.user_id == user_id,
        )
    )
    membership = res.scalar_one_or_none()
    if not membership:
        membership = ProjectMembership(project_id=project_id, user_id=user_id, role=role)
        db.add(membership)
    else:
        membership.role = role
    await db.flush()
    await db.refresh(membership)
    return membership


async def get_current_principal(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Principal | None:
    """Resolve current principal from API key or JWT headers."""
    if not settings.AUTH_ENABLED:
        request.state.principal = None
        return None

    token = _parse_api_key_from_headers(request)
    if not token:
        return None

    # Try JWT first (stateless local auth session)
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        principal = Principal(
            user_id=int(payload["sub"]),
            username=payload["username"],
            role=GlobalRole(payload["role"]),
            api_key_id=0,
            api_key_prefix="jwt",
        )
        request.state.principal = principal
        return principal
    except (jwt.PyJWTError, KeyError, ValueError):
        # Not a valid JWT or missing claims, fall back to API key lookup
        pass

    key_hash = hash_api_key(token)
    result = await db.execute(
        select(ApiKey, User)
        .join(User, User.id == ApiKey.user_id)
        .where(
            ApiKey.key_hash == key_hash,
            ApiKey.is_active.is_(True),
            User.is_active.is_(True),
        )
    )
    row = result.first()
    if not row:
        return None

    api_key_obj, user = row
    api_key_obj.last_used_at = datetime.now(timezone.utc)
    await db.flush()

    principal = Principal(
        user_id=user.id,
        username=user.username,
        role=user.role,
        api_key_id=api_key_obj.id,
        api_key_prefix=api_key_obj.key_prefix,
    )
    request.state.principal = principal
    return principal


def _require_global_role(principal: Principal, allowed: set[GlobalRole]) -> None:
    if principal.role not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")


async def _require_project_role(
    db: AsyncSession,
    principal: Principal,
    project_id: int,
    min_role: ProjectRole,
) -> None:
    if principal.role == GlobalRole.ADMIN:
        return

    membership_res = await db.execute(
        select(ProjectMembership).where(
            ProjectMembership.project_id == project_id,
            ProjectMembership.user_id == principal.user_id,
        )
    )
    membership = membership_res.scalar_one_or_none()
    if not membership:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No access to this project")

    if PROJECT_ROLE_RANK[membership.role] < PROJECT_ROLE_RANK[min_role]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient project permissions")


def _required_project_role(method: str, path: str) -> ProjectRole:
    if method in {"GET", "HEAD", "OPTIONS"}:
        return ProjectRole.VIEWER
    # Deleting a project itself requires owner permissions.
    if method == "DELETE" and re.fullmatch(r"^/api/projects/\d+$", path):
        return ProjectRole.OWNER
    return ProjectRole.EDITOR


async def authorize_request(
    request: Request,
    db: AsyncSession = Depends(get_db),
    principal: Principal | None = Depends(get_current_principal),
) -> Principal | None:
    """
    Global API authz dependency.

    Applied router-wide so every API request passes through role and project checks.
    """
    path = request.url.path
    method = request.method.upper()
    project_id = extract_project_id_from_path(path)
    request.state.project_id = project_id

    if not settings.AUTH_ENABLED:
        return principal

    # Pre-check public routes
    if path in {
        "/api/health",
        "/api/auth/local/login",
        "/api/auth/config",
        "/api/auth/sso/login",
        "/api/auth/sso/callback",
        "/api/auth/callback",
    }:
        return principal

    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    if path == "/api/projects" and method == "POST":
        _require_global_role(principal, {GlobalRole.ADMIN, GlobalRole.ENGINEER})
        return principal

    if path == "/api/projects" and method == "GET":
        _require_global_role(principal, {GlobalRole.ADMIN, GlobalRole.ENGINEER, GlobalRole.VIEWER})
        return principal

    if project_id is not None:
        required = _required_project_role(method, path)
        await _require_project_role(db, principal, project_id, required)

    return principal
