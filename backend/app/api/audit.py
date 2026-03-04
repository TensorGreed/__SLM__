"""Audit log query routes."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.auth import AuditLog

router = APIRouter(prefix="/projects/{project_id}/audit", tags=["Audit"])


@router.get("/logs")
async def list_audit_logs(
    project_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    method: str | None = None,
    status_code: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List recent audit log entries for a project."""
    query = (
        select(AuditLog)
        .where(AuditLog.project_id == project_id)
        .order_by(AuditLog.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if method:
        query = query.where(AuditLog.method == method.upper())
    if status_code is not None:
        query = query.where(AuditLog.status_code == status_code)

    result = await db.execute(query)
    logs = result.scalars().all()
    return {
        "project_id": project_id,
        "count": len(logs),
        "logs": [
            {
                "id": l.id,
                "request_id": l.request_id,
                "method": l.method,
                "path": l.path,
                "status_code": l.status_code,
                "user_id": l.user_id,
                "action": l.action,
                "ip_address": l.ip_address,
                "created_at": l.created_at.isoformat(),
                "metadata": l.metadata_ or {},
            }
            for l in logs
        ],
    }
