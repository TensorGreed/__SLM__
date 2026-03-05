"""Runtime resolution for effective domain defaults (core + profile + pack overlay)."""

from __future__ import annotations

import copy
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain_pack import DomainPack
from app.models.domain_profile import DomainProfile
from app.models.project import Project
from app.services.domain_pack_service import (
    DEFAULT_DOMAIN_PACK_CONTRACT,
    ensure_default_domain_pack,
    get_domain_pack,
)
from app.services.domain_profile_service import (
    DEFAULT_DOMAIN_PROFILE_CONTRACT,
    ensure_default_domain_profile,
    get_domain_profile,
)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _extract_pack_overlay(contract: dict | None) -> dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    overlay = contract.get("overlay")
    if isinstance(overlay, dict):
        cleaned = copy.deepcopy(overlay)
        for reserved in {
            "$schema",
            "schema_ref",
            "profile_id",
            "version",
            "display_name",
            "description",
            "owner",
            "status",
        }:
            cleaned.pop(reserved, None)
        return cleaned
    return {}


async def _get_project(project_id: int, db: AsyncSession) -> Project | None:
    result = await db.execute(select(Project).where(Project.id == project_id))
    return result.scalar_one_or_none()


async def _get_profile_by_id(profile_id: int, db: AsyncSession) -> DomainProfile | None:
    result = await db.execute(select(DomainProfile).where(DomainProfile.id == profile_id))
    return result.scalar_one_or_none()


async def _get_pack_by_id(pack_id: int, db: AsyncSession) -> DomainPack | None:
    result = await db.execute(select(DomainPack).where(DomainPack.id == pack_id))
    return result.scalar_one_or_none()


async def resolve_project_domain_runtime(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Resolve effective domain contract with deterministic fallback chain.

    Merge precedence:
    1) Core platform profile defaults
    2) Resolved domain profile contract (project -> pack default -> platform default)
    3) Resolved domain pack overlay (project -> platform default)
    """

    project = await _get_project(project_id, db)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    pack: DomainPack | None = None
    pack_source = "none"
    if project.domain_pack_id is not None:
        pack = await _get_pack_by_id(project.domain_pack_id, db)
        if pack:
            pack_source = "project"

    if pack is None:
        pack = await get_domain_pack(db, str(DEFAULT_DOMAIN_PACK_CONTRACT["pack_id"]))
        if pack:
            pack_source = "platform_default"

    if pack is None:
        pack = await ensure_default_domain_pack(db)
        pack_source = "platform_default"

    default_profile_id = None
    if pack.default_profile_id and pack.default_profile_id.strip():
        default_profile_id = pack.default_profile_id.strip().lower()
    else:
        pack_contract = pack.contract if isinstance(pack.contract, dict) else {}
        contract_default = pack_contract.get("default_profile_id")
        if isinstance(contract_default, str) and contract_default.strip():
            default_profile_id = contract_default.strip().lower()

    profile: DomainProfile | None = None
    profile_source = "none"
    if project.domain_profile_id is not None:
        profile = await _get_profile_by_id(project.domain_profile_id, db)
        if profile:
            profile_source = "project"

    if profile is None and default_profile_id:
        profile = await get_domain_profile(db, default_profile_id)
        if profile:
            profile_source = "pack_default"

    if profile is None:
        profile = await get_domain_profile(db, str(DEFAULT_DOMAIN_PROFILE_CONTRACT["profile_id"]))
        if profile:
            profile_source = "platform_default"

    if profile is None:
        profile = await ensure_default_domain_profile(db)
        profile_source = "platform_default"

    core_contract = copy.deepcopy(DEFAULT_DOMAIN_PROFILE_CONTRACT)
    profile_contract = profile.contract if isinstance(profile.contract, dict) else {}
    pack_contract = pack.contract if isinstance(pack.contract, dict) else {}
    pack_overlay = _extract_pack_overlay(pack_contract)

    effective_contract = _deep_merge(core_contract, profile_contract)
    effective_contract = _deep_merge(effective_contract, pack_overlay)

    return {
        "project_id": project_id,
        "domain_pack_applied": pack.pack_id if pack else None,
        "domain_pack_source": pack_source,
        "domain_profile_applied": profile.profile_id if profile else None,
        "domain_profile_source": profile_source,
        "pack_default_profile_id": default_profile_id,
        "pack_overlay": pack_overlay or {},
        "effective_contract": effective_contract,
    }


async def get_project_effective_domain_contract(db: AsyncSession, project_id: int) -> dict | None:
    runtime = await resolve_project_domain_runtime(db, project_id)
    contract = runtime.get("effective_contract")
    return contract if isinstance(contract, dict) else None
