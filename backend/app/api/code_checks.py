"""Code-check registry API: list checks, run them over a snippet."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.models.auth import GlobalRole
from app.security import get_request_principal
from app.services import code_check_service, sandbox_exec

router = APIRouter(prefix="/code-checks", tags=["code-checks"])

MAX_CODE_CHARS = 200_000


class RunCodeChecksRequest(BaseModel):
    language: str = Field(min_length=1, max_length=40)
    code: str = Field(max_length=MAX_CODE_CHARS)
    check_ids: list[str] | None = Field(default=None, max_length=50)
    context: dict[str, Any] | None = None


def _require_run_access(request: Request) -> None:
    if not settings.AUTH_ENABLED:
        return
    principal = get_request_principal(request)
    if principal is None:
        raise HTTPException(401, "Authentication required")
    if principal.role not in {GlobalRole.ADMIN, GlobalRole.ENGINEER}:
        raise HTTPException(403, "Only admin or engineer can run code checks")


@router.get("")
async def list_checks() -> dict[str, Any]:
    return {
        "checks": code_check_service.list_code_checks(),
        "network_isolation": {
            "mode": sandbox_exec.network_isolation_mode(),
            "available": sandbox_exec.network_isolation_available(),
        },
    }


@router.post("/run")
async def run_checks(body: RunCodeChecksRequest, request: Request) -> dict[str, Any]:
    _require_run_access(request)
    ctx = {k: v for k, v in (body.context or {}).items() if k == "json_schema"}  # module_dir etc. are server-side only
    return await asyncio.to_thread(
        code_check_service.run_code_checks,
        body.code, language=body.language, check_ids=body.check_ids, context=ctx,
    )
