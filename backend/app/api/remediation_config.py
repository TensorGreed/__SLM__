"""Code-remediation settings API: the detector behind ``issue_resolved``, the code checks that
decide "the patched code still parses", the judge, passage retrieval and whether held-out
``code_example`` rows are scored (``score_code_examples``, default off) and whether their predicted code is
run through the language's canonical formatter before the checks (``format_before_check``, default off). Stored in
``Project.runtime_config["remediation"]``; see ``services/code_remediation_prompt.py``
(defaults) and ``services/remediation_detector.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole, ProjectMembership
from app.models.project import Project
from app.security import get_request_principal
from app.services import code_check_service
from app.services import remediation_detector as rd
from app.services.code_remediation_prompt import (
    CONFIG_KEY,
    MAX_TOP_K,
    effective_remediation_config,
    project_is_code_remediation,
)

router = APIRouter(prefix="/projects/{project_id}/remediation-config", tags=["Code Remediation"])


class DetectorBody(BaseModel):
    kind: str = Field(..., description="'project' (a code-findings project's run) or 'command' (an allow-listed detector check).")
    project_id: int | None = None
    experiment_id: int | None = None
    check_id: str | None = None


class RemediationConfigBody(BaseModel):
    detector: DetectorBody | None = None
    checks: list[str] | None = Field(default=None, description="Code-check ids; null = every runnable check for the language.")
    judge: bool | None = None
    retrieve_passages: bool | None = None
    top_k: int | None = None
    score_code_examples: bool | None = Field(
        default=None, description="Score held-out code_example (task -> code) rows instead of skipping them. Default off."
    )
    format_before_check: bool | None = Field(
        default=None,
        description="Run the language's canonical formatter (gofmt, terraform fmt, rustfmt) on predicted code before "
        "the code checks, so a format-only difference is not a failure. Default off.",
    )


def _invalid(code: str, message: str) -> HTTPException:
    return HTTPException(422, {"error_code": code, "message": message})


async def _project(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


async def _accessible_ids(db: AsyncSession, request: Request) -> set[int] | None:
    """Project ids the caller may see; None = all (auth off, or an admin)."""
    principal = get_request_principal(request)
    if not settings.AUTH_ENABLED or principal is None or principal.role == GlobalRole.ADMIN:
        return None
    rows = await db.execute(select(ProjectMembership.project_id).where(ProjectMembership.user_id == principal.user_id))
    return {int(r) for r in rows.scalars().all()}


async def _payload(db: AsyncSession, request: Request, project: Project) -> dict[str, Any]:
    rc = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    stored = rc.get(CONFIG_KEY) if isinstance(rc.get(CONFIG_KEY), dict) else {}
    effective = effective_remediation_config(rc, project.id)
    config = {
        "detector": stored.get("detector") if isinstance(stored.get("detector"), dict) else None,
        "checks": stored.get("checks") if isinstance(stored.get("checks"), list) else None,
        "judge": stored.get("judge") if isinstance(stored.get("judge"), bool) else None,
        "retrieve_passages": stored.get("retrieve_passages") if isinstance(stored.get("retrieve_passages"), bool) else None,
        "top_k": stored.get("top_k") if isinstance(stored.get("top_k"), int) else None,
        "score_code_examples": stored.get("score_code_examples") if isinstance(stored.get("score_code_examples"), bool) else None,
        "format_before_check": stored.get("format_before_check") if isinstance(stored.get("format_before_check"), bool) else None,
    }
    return {
        "config": config,
        "effective": effective,
        "detector_label": await rd.describe_detector(db, effective["detector"]),
        "available_detectors": await rd.list_available_detectors(db, await _accessible_ids(db, request)),
        "is_code_remediation": project_is_code_remediation(project),
    }


@router.get("")
async def get_remediation_config(project_id: int, request: Request, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return await _payload(db, request, await _project(db, project_id))


@router.put("")
async def put_remediation_config(
    project_id: int, body: RemediationConfigBody, request: Request, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Validate and store the settings (replacing the previous ones). 422 with ``error_code``
    ``INVALID_REMEDIATION_CONFIG`` and nothing saved when the detector project is not a
    code-findings project with a completed weighted run (or not accessible to the caller), a
    check id is unknown, or ``top_k`` is outside 1..10."""
    project = await _project(db, project_id)
    if not project_is_code_remediation(project):
        raise HTTPException(409, {"error_code": "NOT_CODE_REMEDIATION", "message": "Pick the code-remediation task type first."})
    stored: dict[str, Any] = {}

    if body.top_k is not None:
        if not 1 <= body.top_k <= MAX_TOP_K:
            raise _invalid("INVALID_REMEDIATION_CONFIG", f"top_k must be between 1 and {MAX_TOP_K}.")
        stored["top_k"] = body.top_k
    if body.checks is not None:
        unknown = [c for c in body.checks if code_check_service.get_code_check(c) is None]
        if unknown:
            raise _invalid("INVALID_REMEDIATION_CONFIG", f"Unknown code check: {unknown[0]}.")
        detectors = [c for c in body.checks if getattr(code_check_service.get_code_check(c), "kind", None) == "detector"]
        if detectors:
            raise _invalid("INVALID_REMEDIATION_CONFIG", f"'{detectors[0]}' is a detector, not a parse check; set it as the detector.")
        stored["checks"] = list(dict.fromkeys(body.checks))
    if body.judge is not None:
        stored["judge"] = body.judge
    if body.retrieve_passages is not None:
        stored["retrieve_passages"] = body.retrieve_passages
    if body.score_code_examples is not None:
        stored["score_code_examples"] = body.score_code_examples
    if body.format_before_check is not None:
        stored["format_before_check"] = body.format_before_check

    if body.detector is not None:
        det = body.detector
        if det.kind == "project":
            if det.project_id is None:
                raise _invalid("INVALID_REMEDIATION_CONFIG", "A project detector needs project_id.")
            target = await db.get(Project, det.project_id)
            accessible = await _accessible_ids(db, request)
            if target is None or (accessible is not None and target.id not in accessible):
                raise _invalid("INVALID_REMEDIATION_CONFIG", f"Detector project {det.project_id} not found.")
            if not rd._is_code_findings_project(target):
                raise _invalid("INVALID_REMEDIATION_CONFIG", f"Detector project '{target.name}' is not a code-findings project.")
            exp = await rd.latest_weighted_run(db, target.id, det.experiment_id)
            if exp is None:
                raise _invalid(
                    "INVALID_REMEDIATION_CONFIG",
                    f"Detector project '{target.name}' has no completed run with weights"
                    + (f" (run {det.experiment_id})." if det.experiment_id else "."),
                )
            stored["detector"] = {"kind": "project", "project_id": target.id, "experiment_id": det.experiment_id}
        elif det.kind == "command":
            check = code_check_service.get_code_check(det.check_id or "")
            if check is None or getattr(check, "kind", None) != "detector":
                raise _invalid("INVALID_REMEDIATION_CONFIG", f"'{det.check_id}' is not a registered detector check.")
            stored["detector"] = {"kind": "command", "check_id": check.id}
        else:
            raise _invalid("INVALID_REMEDIATION_CONFIG", "detector.kind must be 'project' or 'command'.")

    cfg = dict(project.runtime_config or {})
    if stored:
        cfg[CONFIG_KEY] = stored
    else:
        cfg.pop(CONFIG_KEY, None)
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return await _payload(db, request, project)
