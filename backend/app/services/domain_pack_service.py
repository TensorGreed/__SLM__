"""Domain pack service for pack persistence and project assignment."""

from __future__ import annotations

import copy
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain_pack import DomainPack
from app.models.domain_profile import DomainProfile
from app.models.project import Project
from app.schemas.domain_pack import DomainPackContract

DEFAULT_DOMAIN_PACK_CONTRACT = {
    "$schema": "slm.domain-pack/v1",
    "pack_id": "general-pack-v1",
    "version": "1.0.0",
    "display_name": "General Domain Pack",
    "description": "Default fallback pack for any domain. Applies safe baseline overlays.",
    "owner": "platform",
    "status": "active",
    "default_profile_id": "generic-domain-v1",
    "tags": ["general", "fallback"],
    "overlay": {
        "dataset_split": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
            "seed": 42,
        },
        "training_defaults": {
            "training_mode": "sft",
            "chat_template": "llama3",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.0002,
            "use_lora": True,
        },
        "registry_gates": {
            "to_staging": {"min_metrics": {"f1": 0.65, "llm_judge_pass_rate": 0.75}},
            "to_production": {
                "min_metrics": {
                    "f1": 0.7,
                    "llm_judge_pass_rate": 0.8,
                    "safety_pass_rate": 0.92,
                },
                "max_regression_vs_prod": {"f1": 0.03, "exact_match": 0.03},
            },
        },
    },
}


def _hydrate_pack_from_contract(
    pack: DomainPack,
    contract: DomainPackContract,
    *,
    is_system: bool | None = None,
) -> DomainPack:
    payload = contract.model_dump(by_alias=True, exclude_none=True)
    pack.pack_id = contract.pack_id
    pack.version = contract.version
    pack.display_name = contract.display_name
    pack.description = contract.description
    pack.owner = contract.owner
    pack.status = contract.status
    pack.schema_ref = contract.schema_ref
    pack.default_profile_id = contract.default_profile_id
    pack.contract = payload
    if is_system is not None:
        pack.is_system = bool(is_system)
    return pack


async def list_domain_packs(db: AsyncSession) -> list[DomainPack]:
    result = await db.execute(select(DomainPack).order_by(DomainPack.updated_at.desc(), DomainPack.id.desc()))
    return list(result.scalars().all())


def _bump_patch_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) != 3:
        return version
    try:
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2])
    except ValueError:
        return version
    return f"{major}.{minor}.{patch + 1}"


async def _derive_next_pack_id(db: AsyncSession, source_pack_id: str) -> str:
    rows = await db.execute(select(DomainPack.pack_id))
    existing = {str(item).strip().lower() for item in rows.scalars().all()}

    match = re.match(r"^(.*)-v(\d+)$", source_pack_id)
    stem = source_pack_id
    candidate_num = 2
    if match:
        stem = match.group(1)
        candidate_num = int(match.group(2)) + 1

    candidate = f"{stem}-v{candidate_num}"
    while candidate in existing:
        candidate_num += 1
        candidate = f"{stem}-v{candidate_num}"
    return candidate


async def get_domain_pack(db: AsyncSession, pack_id: str) -> DomainPack | None:
    result = await db.execute(select(DomainPack).where(DomainPack.pack_id == pack_id.strip().lower()))
    return result.scalar_one_or_none()


async def get_project_domain_pack(db: AsyncSession, project_id: int) -> DomainPack | None:
    result = await db.execute(select(Project.domain_pack_id).where(Project.id == project_id))
    domain_pack_id = result.scalar_one_or_none()
    if domain_pack_id is None:
        return None
    pack_result = await db.execute(select(DomainPack).where(DomainPack.id == domain_pack_id))
    return pack_result.scalar_one_or_none()


async def create_domain_pack(
    db: AsyncSession,
    contract: DomainPackContract,
    *,
    is_system: bool = False,
) -> DomainPack:
    existing = await get_domain_pack(db, contract.pack_id)
    if existing:
        raise ValueError(f"Domain pack '{contract.pack_id}' already exists")

    pack = DomainPack()
    _hydrate_pack_from_contract(pack, contract, is_system=is_system)
    db.add(pack)
    await db.flush()
    await db.refresh(pack)
    return pack


async def duplicate_domain_pack(
    db: AsyncSession,
    source: DomainPack,
    *,
    new_pack_id: str | None = None,
    new_version: str | None = None,
    status_override: str | None = None,
) -> DomainPack:
    source_contract = source.contract if isinstance(source.contract, dict) else None
    if not source_contract:
        raise ValueError("Source domain pack has no valid contract")

    payload = copy.deepcopy(source_contract)
    payload["pack_id"] = new_pack_id or await _derive_next_pack_id(db, source.pack_id)
    payload["version"] = new_version or _bump_patch_version(str(payload.get("version", source.version or "1.0.0")))
    if status_override:
        payload["status"] = status_override

    contract = DomainPackContract.model_validate(payload)
    return await create_domain_pack(db, contract)


async def update_domain_pack(
    db: AsyncSession,
    pack: DomainPack,
    contract: DomainPackContract,
) -> DomainPack:
    if pack.pack_id != contract.pack_id:
        raise ValueError("pack_id in payload must match path pack_id")

    _hydrate_pack_from_contract(pack, contract)
    await db.flush()
    await db.refresh(pack)
    return pack


async def assign_project_domain_pack(
    db: AsyncSession,
    project_id: int,
    pack_id: str,
    *,
    adopt_pack_default_profile: bool = True,
) -> Project:
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    pack = await get_domain_pack(db, pack_id)
    if not pack:
        raise ValueError(f"Domain pack '{pack_id}' not found")

    project.domain_pack_id = pack.id

    if adopt_pack_default_profile and pack.default_profile_id:
        profile_result = await db.execute(
            select(DomainProfile).where(DomainProfile.profile_id == pack.default_profile_id)
        )
        profile = profile_result.scalar_one_or_none()
        if profile:
            project.domain_profile_id = profile.id

    await db.flush()
    await db.refresh(project)
    return project


async def ensure_default_domain_pack(db: AsyncSession) -> DomainPack:
    contract = DomainPackContract.model_validate(DEFAULT_DOMAIN_PACK_CONTRACT)
    existing = await get_domain_pack(db, contract.pack_id)
    if existing:
        return existing
    return await create_domain_pack(db, contract, is_system=True)
