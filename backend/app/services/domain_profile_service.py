"""Domain profile service for contract persistence and project assignment."""

from __future__ import annotations

import copy
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain_profile import DomainProfile
from app.models.project import Project
from app.schemas.domain_profile import DomainProfileContract

DEFAULT_DOMAIN_PROFILE_CONTRACT = {
    "$schema": "slm.domain-profile/v1",
    "profile_id": "generic-domain-v1",
    "version": "1.0.0",
    "display_name": "Generic Domain",
    "description": "Domain-agnostic profile for QA, classification, extraction, and generation workloads.",
    "owner": "platform",
    "status": "active",
    "tasks": [
        {
            "task_id": "qa",
            "output_mode": "text",
            "required_fields": ["question", "answer"],
            "optional_fields": ["context", "source", "difficulty", "labels"],
        },
        {
            "task_id": "classification",
            "output_mode": "label",
            "required_fields": ["input", "label"],
            "optional_fields": ["label_set", "rationale"],
        },
    ],
    "canonical_schema": {
        "required": ["input_text", "target_text"],
        "aliases": {
            "input_text": ["question", "prompt", "instruction", "input", "text"],
            "target_text": ["answer", "output", "completion", "label", "response"],
            "context": ["context", "passage", "document"],
            "metadata": ["meta", "attributes", "tags"],
        },
    },
    "normalization": {
        "trim_whitespace": True,
        "drop_empty_records": True,
        "dedupe": {"enabled": True, "method": "hash(input_text,target_text)"},
        "pii_redaction": {"enabled": False, "policy": "default"},
    },
    "data_quality": {
        "min_records": 500,
        "max_null_ratio": 0.1,
        "max_duplicate_ratio": 0.2,
        "required_coverage": {
            "input_text": 0.99,
            "target_text": 0.99,
        },
    },
    "dataset_split": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
        "stratify_by": [],
        "seed": 42,
        "leakage_checks": ["exact_text_overlap"],
    },
    "training_defaults": {
        "training_mode": "sft",
        "chat_template": "llama3",
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 0.0002,
        "use_lora": True,
    },
    "evaluation": {
        "metrics": [
            {"metric_id": "exact_match", "weight": 0.35, "threshold": 0.55},
            {"metric_id": "f1", "weight": 0.35, "threshold": 0.65},
            {"metric_id": "llm_judge_pass_rate", "weight": 0.2, "threshold": 0.75},
            {"metric_id": "safety_pass_rate", "weight": 0.1, "threshold": 0.9},
        ],
        "required_metrics_for_promotion": ["f1", "llm_judge_pass_rate", "safety_pass_rate"],
    },
    "tools": {
        "retrieval": {"enabled": False, "adapter": None},
        "function_calling": {"enabled": False, "adapter": None},
        "required_secrets": [],
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
    "audit": {
        "require_human_approval_for_production": True,
        "notes_required_on_force_promotion": True,
    },
}


def _hydrate_profile_from_contract(
    profile: DomainProfile,
    contract: DomainProfileContract,
    *,
    is_system: bool | None = None,
) -> DomainProfile:
    payload = contract.model_dump(by_alias=True, exclude_none=True)
    profile.profile_id = contract.profile_id
    profile.version = contract.version
    profile.display_name = contract.display_name
    profile.description = contract.description
    profile.owner = contract.owner
    profile.status = contract.status
    profile.schema_ref = contract.schema_ref
    profile.contract = payload
    if is_system is not None:
        profile.is_system = bool(is_system)
    return profile


async def list_domain_profiles(db: AsyncSession) -> list[DomainProfile]:
    result = await db.execute(
        select(DomainProfile).order_by(DomainProfile.updated_at.desc(), DomainProfile.id.desc())
    )
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


async def _derive_next_profile_id(db: AsyncSession, source_profile_id: str) -> str:
    rows = await db.execute(select(DomainProfile.profile_id))
    existing = {str(item).strip().lower() for item in rows.scalars().all()}

    match = re.match(r"^(.*)-v(\d+)$", source_profile_id)
    stem = source_profile_id
    candidate_num = 2
    if match:
        stem = match.group(1)
        candidate_num = int(match.group(2)) + 1

    candidate = f"{stem}-v{candidate_num}"
    while candidate in existing:
        candidate_num += 1
        candidate = f"{stem}-v{candidate_num}"
    return candidate


async def get_domain_profile(db: AsyncSession, profile_id: str) -> DomainProfile | None:
    result = await db.execute(
        select(DomainProfile).where(DomainProfile.profile_id == profile_id.strip().lower())
    )
    return result.scalar_one_or_none()


async def get_project_domain_profile(db: AsyncSession, project_id: int) -> DomainProfile | None:
    result = await db.execute(select(Project.domain_profile_id).where(Project.id == project_id))
    domain_profile_id = result.scalar_one_or_none()
    if domain_profile_id is None:
        return None
    profile_result = await db.execute(select(DomainProfile).where(DomainProfile.id == domain_profile_id))
    return profile_result.scalar_one_or_none()


async def get_project_domain_profile_contract(db: AsyncSession, project_id: int) -> dict | None:
    profile = await get_project_domain_profile(db, project_id)
    if not profile:
        return None
    contract = profile.contract if isinstance(profile.contract, dict) else None
    if not contract:
        return None
    return contract


def get_dataset_split_defaults(contract: dict | None) -> dict[str, float | int | str]:
    if not isinstance(contract, dict):
        return {}

    result: dict[str, float | int | str] = {}
    split_cfg = contract.get("dataset_split")
    if isinstance(split_cfg, dict):
        train = split_cfg.get("train")
        val = split_cfg.get("val")
        test = split_cfg.get("test")
        seed = split_cfg.get("seed")
        if isinstance(train, (int, float)):
            result["train_ratio"] = float(train)
        if isinstance(val, (int, float)):
            result["val_ratio"] = float(val)
        if isinstance(test, (int, float)):
            result["test_ratio"] = float(test)
        if isinstance(seed, int):
            result["seed"] = seed

    training_defaults = contract.get("training_defaults")
    if isinstance(training_defaults, dict):
        chat_template = training_defaults.get("chat_template")
        if isinstance(chat_template, str) and chat_template.strip():
            result["chat_template"] = chat_template.strip()

    return result


def get_training_defaults(contract: dict | None) -> dict:
    if not isinstance(contract, dict):
        return {}
    defaults = contract.get("training_defaults")
    if not isinstance(defaults, dict):
        return {}
    return dict(defaults)


def get_registry_gate_defaults(
    contract: dict | None,
    target_stage: str,
) -> dict[str, float]:
    if not isinstance(contract, dict):
        return {}

    registry_gates = contract.get("registry_gates")
    if not isinstance(registry_gates, dict):
        return {}

    stage_key = "to_production" if target_stage == "production" else "to_staging"
    stage_cfg = registry_gates.get(stage_key)
    if not isinstance(stage_cfg, dict):
        return {}

    defaults: dict[str, float] = {}
    min_metrics = stage_cfg.get("min_metrics")
    if isinstance(min_metrics, dict):
        metric_map = {
            "exact_match": "min_exact_match",
            "f1": "min_f1",
            "llm_judge_pass_rate": "min_llm_judge_pass_rate",
            "safety_pass_rate": "min_safety_pass_rate",
        }
        for metric_name, gate_name in metric_map.items():
            value = min_metrics.get(metric_name)
            if isinstance(value, (int, float)):
                defaults[gate_name] = float(value)

    regressions = stage_cfg.get("max_regression_vs_prod")
    if isinstance(regressions, dict):
        exact_reg = regressions.get("exact_match")
        f1_reg = regressions.get("f1")
        if isinstance(exact_reg, (int, float)):
            defaults["max_exact_match_regression"] = float(exact_reg)
        if isinstance(f1_reg, (int, float)):
            defaults["max_f1_regression"] = float(f1_reg)

    return defaults


async def create_domain_profile(
    db: AsyncSession,
    contract: DomainProfileContract,
    *,
    is_system: bool = False,
) -> DomainProfile:
    existing = await get_domain_profile(db, contract.profile_id)
    if existing:
        raise ValueError(f"Domain profile '{contract.profile_id}' already exists")

    profile = DomainProfile()
    _hydrate_profile_from_contract(profile, contract, is_system=is_system)
    db.add(profile)
    await db.flush()
    await db.refresh(profile)
    return profile


async def duplicate_domain_profile(
    db: AsyncSession,
    source: DomainProfile,
    *,
    new_profile_id: str | None = None,
    new_version: str | None = None,
    status_override: str | None = None,
) -> DomainProfile:
    source_contract = source.contract if isinstance(source.contract, dict) else None
    if not source_contract:
        raise ValueError("Source domain profile has no valid contract")

    payload = copy.deepcopy(source_contract)
    payload["profile_id"] = new_profile_id or await _derive_next_profile_id(db, source.profile_id)
    payload["version"] = new_version or _bump_patch_version(str(payload.get("version", source.version or "1.0.0")))
    if status_override:
        payload["status"] = status_override

    contract = DomainProfileContract.model_validate(payload)
    return await create_domain_profile(db, contract)


async def update_domain_profile(
    db: AsyncSession,
    profile: DomainProfile,
    contract: DomainProfileContract,
) -> DomainProfile:
    if profile.profile_id != contract.profile_id:
        raise ValueError("profile_id in payload must match path profile_id")

    _hydrate_profile_from_contract(profile, contract)
    await db.flush()
    await db.refresh(profile)
    return profile


async def assign_project_domain_profile(
    db: AsyncSession,
    project_id: int,
    profile_id: str,
) -> Project:
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalar_one_or_none()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    profile = await get_domain_profile(db, profile_id)
    if not profile:
        raise ValueError(f"Domain profile '{profile_id}' not found")

    project.domain_profile_id = profile.id
    await db.flush()
    await db.refresh(project)
    return project


async def ensure_default_domain_profile(db: AsyncSession) -> DomainProfile:
    contract = DomainProfileContract.model_validate(DEFAULT_DOMAIN_PROFILE_CONTRACT)
    existing = await get_domain_profile(db, contract.profile_id)
    if existing:
        return existing
    return await create_domain_profile(db, contract, is_system=True)
