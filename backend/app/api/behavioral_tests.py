"""Quality-Lift phase 7 slice 2 — Behavioral tests CRUD scoped to the
project's scaffolded eval pack.

Phase 5 slice 1 put behavioral_tests on the pack contract
(``task_specs[].behavioral_tests``). The pack-scaffold save endpoint
(``POST /api/projects/{id}/evaluation/pack-scaffold``) already
round-trips them via the existing draft_pack payload. This module adds
the FOCUSED endpoint the BehavioralTestsSection editor calls:

  ``GET /api/projects/{id}/behavioral-tests`` — returns the current
    classification task_spec's behavioral_tests list.
  ``PUT /api/projects/{id}/behavioral-tests`` — replaces the list,
    validates via ``validate_behavioral_tests``, and persists back
    into the scaffolded pack JSON on ``project.runtime_config``.

Why a focused endpoint rather than the full pack-scaffold POST: the
editor's UX is tightly scoped to the behavioral_tests section. Sending
the whole draft on every save (gates, metric_schema, etc.) would
require the editor to also track every other field — which it
doesn't (the gates editor owns its own draft state). The focused
endpoint loads the current pack, swaps just the behavioral_tests
array, and re-runs the full validator.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services.behavioral_test_schema import (
    BehavioralTestValidationError,
    validate_behavioral_tests,
)
from app.services.eval_pack_scaffold_service import (
    RUNTIME_CONFIG_KEY,
    SCAFFOLDED_PACK_ID,
    save_scaffolded_pack,
)
from app.services.evaluation_pack_service import (
    resolve_project_evaluation_pack,
)


router = APIRouter(
    prefix="/projects/{project_id}/behavioral-tests",
    tags=["BehavioralTests"],
)


class BehavioralTestsPayload(BaseModel):
    """Just the behavioral_tests array. The closed-grammar validator
    enforces test_id grammar, kind enum, perturbation/expectation
    shape, etc. — Pydantic just shapes the outer envelope."""

    behavioral_tests: list[dict[str, Any]] = Field(default_factory=list)


class BehavioralTestsResponse(BaseModel):
    project_id: int
    task_profile: str
    behavioral_tests: list[dict[str, Any]]


def _find_classification_task_spec(
    pack: dict[str, Any],
) -> tuple[int, dict[str, Any]] | None:
    """Locate the classification task_spec (if any) in the pack.

    STRICT classification match — slice 1's behavioral runner is
    classification-only (phase 5 design). The PUT handler materialises
    a fresh classification spec when the recipe-resolved pack has none,
    rather than falling back to e.g. a qa spec (which would
    persist behavioral tests on a task the runner won't execute).
    """
    task_specs = pack.get("task_specs")
    if not isinstance(task_specs, list):
        return None
    for idx, spec in enumerate(task_specs):
        if not isinstance(spec, dict):
            continue
        profile = str(spec.get("task_profile") or "").strip().lower()
        if profile == "classification":
            return idx, spec
    return None


async def _load_active_pack(
    db: AsyncSession, project_id: int,
) -> tuple[dict[str, Any], bool]:
    """Return ``(pack, came_from_scaffold)``. When no scaffolded pack
    exists yet (user hasn't clicked "Use scaffold"), we fall back to
    the recipe-resolved pack so the editor can still show the test
    list — saving will materialize the scaffold."""
    project_row = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = project_row.scalar_one_or_none()
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")

    rc = project.runtime_config or {}
    scaffolded = rc.get(RUNTIME_CONFIG_KEY) if isinstance(rc, dict) else None
    if isinstance(scaffolded, dict) and scaffolded.get("task_specs"):
        return dict(scaffolded), True

    # Fallback — resolve the recipe-derived pack so the editor still
    # has a list to show. Saving will then build the scaffold for
    # real via ``save_scaffolded_pack``.
    resolved = await resolve_project_evaluation_pack(db, project_id)
    pack = resolved.get("pack") if isinstance(resolved, dict) else None
    if not isinstance(pack, dict):
        # Project has no pack reachable at all — return an empty
        # placeholder so the editor renders without errors.
        return {
            "pack_id": SCAFFOLDED_PACK_ID,
            "task_specs": [{
                "task_profile": "classification",
                "behavioral_tests": [],
                "gates": [],
            }],
        }, False
    return dict(pack), False


@router.get("", response_model=BehavioralTestsResponse)
async def get_project_behavioral_tests(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Read the behavioral_tests array scoped to the project's
    classification task_spec. Returns an empty list when no scaffolded
    pack exists yet (editor renders the empty state)."""
    pack, _from_scaffold = await _load_active_pack(db, project_id)
    found = _find_classification_task_spec(pack)
    if found is None:
        return {
            "project_id": project_id,
            "task_profile": "classification",
            "behavioral_tests": [],
        }
    _idx, task_spec = found
    raw_tests = task_spec.get("behavioral_tests")
    return {
        "project_id": project_id,
        "task_profile": str(task_spec.get("task_profile") or "classification"),
        "behavioral_tests": (
            list(raw_tests) if isinstance(raw_tests, list) else []
        ),
    }


@router.put("", response_model=BehavioralTestsResponse)
async def set_project_behavioral_tests(
    project_id: int,
    payload: BehavioralTestsPayload,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Replace the behavioral_tests array on the project's
    classification task_spec. Validates via
    ``validate_behavioral_tests`` then persists through
    ``save_scaffolded_pack`` so the gate validator + pack JSON write
    both fire."""

    # Validate the payload BEFORE loading the pack so a malformed
    # payload doesn't risk a partial write.
    try:
        cleaned = validate_behavioral_tests(payload.behavioral_tests)
    except BehavioralTestValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    pack, _from_scaffold = await _load_active_pack(db, project_id)
    found = _find_classification_task_spec(pack)
    if found is None:
        # Materialise a fresh classification task_spec for the
        # behavioral_tests. We APPEND rather than replace any
        # existing non-classification specs (e.g. qa) — the user's
        # primary task still works; behavioral_tests live on a
        # parallel classification spec that the runner can execute.
        existing_specs = list(pack.get("task_specs") or [])
        existing_specs.append({
            "task_profile": "classification",
            "gates": [],
            "behavioral_tests": cleaned,
        })
        new_pack = dict(pack)
        new_pack["task_specs"] = existing_specs
    else:
        spec_idx, task_spec = found
        new_specs = list(pack.get("task_specs") or [])
        updated_spec = dict(task_spec)
        updated_spec["behavioral_tests"] = cleaned
        new_specs[spec_idx] = updated_spec
        new_pack = dict(pack)
        new_pack["task_specs"] = new_specs

    try:
        result = await save_scaffolded_pack(
            db, project_id=project_id, draft_pack=new_pack,
        )
    except ValueError as exc:
        # save_scaffolded_pack raises on missing task_specs /
        # validate_draft_pack_gates failure. Surface verbatim.
        raise HTTPException(400, str(exc)) from exc
    await db.commit()

    saved_pack = result.get("scaffolded_pack") or new_pack
    found_after = _find_classification_task_spec(saved_pack)
    saved_tests = (
        list(found_after[1].get("behavioral_tests") or [])
        if found_after is not None
        else cleaned
    )
    saved_profile = (
        str(found_after[1].get("task_profile") or "classification")
        if found_after is not None
        else "classification"
    )
    return {
        "project_id": project_id,
        "task_profile": saved_profile,
        "behavioral_tests": saved_tests,
    }
