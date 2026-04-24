"""P14 — Capture + read immutable training-run manifests.

Called from the training launch path (``training_service.start_training``)
after the runtime successfully dispatches, so every successful run has a
manifest row. Capture is best-effort: subprocess-level lookups (git,
pip) record ``capture_warnings`` instead of failing the launch.

Public entry points:
- ``capture_training_manifest(db, *, experiment_id, project_id)`` — writes
  a new row (idempotent on ``experiment_id`` unique constraint; rewrites
  the row if already present).
- ``get_training_manifest(db, *, project_id, experiment_id)`` — reads
  the row and returns a serialized dict, or raises
  ``ValueError("manifest_not_captured")`` for a missing manifest.
- ``serialize_training_manifest(row)`` — shared serializer used by both
  the service and the API response path.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base_model_registry import BaseModelRegistryEntry
from app.models.dataset import Dataset, DatasetVersion
from app.models.dataset_adapter_definition import DatasetAdapterDefinition
from app.models.domain_blueprint import DomainBlueprintRevision
from app.models.experiment import Experiment
from app.models.training_manifest import TrainingManifest


_MANIFEST_SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/ root; git repo root is the parent


def _find_git_repo_root() -> Path | None:
    """Walk up from the backend dir to find the enclosing .git directory."""
    for candidate in (_REPO_ROOT, _REPO_ROOT.parent):
        if (candidate / ".git").exists():
            return candidate
    # Fall back to searching: start from this file and walk up.
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _collect_git_sha(warnings: list[str]) -> str | None:
    root = _find_git_repo_root()
    if root is None:
        warnings.append("git_repo_root_not_found")
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        warnings.append(f"git_unavailable:{type(exc).__name__}")
        return None
    if completed.returncode != 0:
        warnings.append("git_rev_parse_failed")
        return None
    sha = completed.stdout.strip()
    return sha or None


def _collect_pip_freeze(warnings: list[str]) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        warnings.append(f"pip_freeze_unavailable:{type(exc).__name__}")
        return None, None
    if completed.returncode != 0:
        warnings.append("pip_freeze_failed")
        return None, None
    blob = completed.stdout or ""
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest() if blob else None
    return blob or None, digest


def _env_digest(git_sha: str | None, pip_hash: str | None, runtime_id: str | None) -> str:
    material = f"{git_sha or ''}|{pip_hash or ''}|{runtime_id or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _extract_tokenizer_hint(config: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (tokenizer_name, tokenizer_config_hash).

    Tokenizer name lives under a handful of keys depending on which code
    path populated ``exp.config``. The hash is a quick fingerprint of the
    chat template + tokenizer name so rerun can compare without touching
    the filesystem.
    """
    name = None
    for key in ("tokenizer", "tokenizer_name", "tokenizer_id", "chat_template_tokenizer"):
        value = str(config.get(key) or "").strip()
        if value:
            name = value
            break

    hash_material = {
        "tokenizer": name or "",
        "chat_template": str(config.get("chat_template") or ""),
        "max_seq_length": config.get("max_seq_length"),
        "pad_token": str(config.get("pad_token") or ""),
    }
    digest = hashlib.sha256(
        "|".join(f"{k}={hash_material[k]}" for k in sorted(hash_material)).encode("utf-8")
    ).hexdigest()
    return name, digest


async def _resolve_base_model_reference(
    db: AsyncSession, base_model: str | None
) -> tuple[int | None, str | None, str | None]:
    """Look up the latest base-model registry entry matching the run's base_model.

    Returns ``(registry_id, cache_fingerprint, source_ref)``.
    """
    ref = str(base_model or "").strip()
    if not ref:
        return None, None, None
    stmt = (
        select(BaseModelRegistryEntry)
        .where(BaseModelRegistryEntry.source_ref == ref)
        .order_by(BaseModelRegistryEntry.updated_at.desc(), BaseModelRegistryEntry.id.desc())
        .limit(1)
    )
    row = (await db.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None, None, ref
    return int(row.id), row.cache_fingerprint or None, row.source_ref or ref


async def _resolve_dataset_adapter_reference(
    db: AsyncSession, config: dict[str, Any], project_id: int
) -> tuple[int | None, int | None]:
    """Resolve (adapter_id, version) from the resolved config or project state."""
    adapter_id_candidate = config.get("dataset_adapter_id") or config.get("adapter_id")
    if adapter_id_candidate is not None:
        try:
            adapter_id = int(adapter_id_candidate)
        except (TypeError, ValueError):
            adapter_id = None
        if adapter_id is not None:
            row = (
                await db.execute(
                    select(DatasetAdapterDefinition).where(
                        DatasetAdapterDefinition.id == adapter_id
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                return int(row.id), int(row.version or 0) or None

    # Fall back to the latest active adapter for the project.
    row = (
        await db.execute(
            select(DatasetAdapterDefinition)
            .where(DatasetAdapterDefinition.project_id == project_id)
            .order_by(
                DatasetAdapterDefinition.updated_at.desc(),
                DatasetAdapterDefinition.id.desc(),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None:
        return None, None
    return int(row.id), int(row.version or 0) or None


async def _resolve_blueprint_reference(
    db: AsyncSession, project_id: int
) -> tuple[int | None, int | None]:
    row = (
        await db.execute(
            select(DomainBlueprintRevision)
            .where(DomainBlueprintRevision.project_id == project_id)
            .order_by(
                DomainBlueprintRevision.version.desc(),
                DomainBlueprintRevision.id.desc(),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None:
        return None, None
    return int(row.id), int(row.version or 0) or None


async def _resolve_dataset_snapshot_ids(
    db: AsyncSession, project_id: int
) -> list[dict[str, Any]]:
    """Return the latest ``DatasetVersion`` id per eligible dataset.

    We pick train / validation / test / gold_dev / gold_test snapshots where
    they exist — anything more exotic is still captured as ``role=other``.
    """
    datasets = (
        await db.execute(
            select(Dataset).where(Dataset.project_id == project_id)
        )
    ).scalars().all()

    snapshots: list[dict[str, Any]] = []
    for ds in datasets:
        latest_version = (
            await db.execute(
                select(DatasetVersion)
                .where(DatasetVersion.dataset_id == ds.id)
                .order_by(DatasetVersion.version.desc(), DatasetVersion.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        snapshots.append(
            {
                "dataset_id": int(ds.id),
                "dataset_type": ds.dataset_type.value if ds.dataset_type else None,
                "dataset_name": ds.name,
                "dataset_version_id": int(latest_version.id) if latest_version else None,
                "dataset_version": int(latest_version.version) if latest_version else None,
                "record_count": int(getattr(latest_version, "record_count", 0) or 0),
            }
        )
    return snapshots


def serialize_training_manifest(row: TrainingManifest) -> dict[str, Any]:
    return {
        "id": int(row.id),
        "experiment_id": int(row.experiment_id),
        "project_id": int(row.project_id),
        "schema_version": int(row.schema_version),
        "captured_at": row.captured_at.isoformat() if row.captured_at else None,
        "base_model": {
            "registry_id": row.base_model_registry_id,
            "cache_fingerprint": row.base_model_cache_fingerprint,
            "source_ref": row.base_model_source_ref,
        },
        "dataset_adapter": {
            "id": row.dataset_adapter_id,
            "version": row.dataset_adapter_version,
        },
        "blueprint": {
            "revision_id": row.blueprint_revision_id,
            "version": row.blueprint_version,
        },
        "datasets": list(row.dataset_snapshot_ids or []),
        "recipe_id": row.recipe_id,
        "runtime_id": row.runtime_id,
        "training_mode": row.training_mode,
        "tokenizer": {
            "name": row.tokenizer_name,
            "config_hash": row.tokenizer_config_hash,
        },
        "seed": row.seed,
        "resolved_config": dict(row.resolved_config or {}),
        "env": {
            "git_sha": row.git_sha,
            "pip_freeze_hash": row.pip_freeze_hash,
            "pip_freeze_blob_length": len(row.pip_freeze_blob or ""),
            "env_digest": row.env_digest,
        },
        "artifact_ids": dict(row.artifact_ids or {}),
        "capture_warnings": list(row.capture_warnings or []),
    }


async def capture_training_manifest(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
    resolved_config: dict[str, Any] | None = None,
    artifact_ids: dict[str, Any] | None = None,
    collect_env: bool = True,
) -> TrainingManifest:
    """Capture a manifest row for ``experiment_id``.

    If the experiment already has a manifest (unique constraint), the
    existing row is **rewritten** — this keeps the capture idempotent so
    a re-launch after a failed dispatch leaves a clean record.
    """
    exp_row = (
        await db.execute(
            select(Experiment).where(
                Experiment.id == experiment_id,
                Experiment.project_id == project_id,
            )
        )
    ).scalar_one_or_none()
    if exp_row is None:
        raise ValueError("experiment_not_found")

    config = dict(resolved_config or exp_row.config or {})
    # Strip the transient ``_runtime`` sub-block from the snapshot so reruns
    # replay the authored config without task ids / worker PIDs.
    sanitized_config = {k: v for k, v in config.items() if k != "_runtime"}

    warnings: list[str] = []
    registry_id, cache_fingerprint, source_ref = await _resolve_base_model_reference(
        db, exp_row.base_model
    )
    adapter_id, adapter_version = await _resolve_dataset_adapter_reference(
        db, sanitized_config, project_id
    )
    blueprint_id, blueprint_version = await _resolve_blueprint_reference(db, project_id)
    dataset_snapshots = await _resolve_dataset_snapshot_ids(db, project_id)

    tokenizer_name, tokenizer_hash = _extract_tokenizer_hint(sanitized_config)

    git_sha = _collect_git_sha(warnings) if collect_env else None
    pip_blob, pip_hash = _collect_pip_freeze(warnings) if collect_env else (None, None)
    runtime_id = str(sanitized_config.get("training_runtime_id") or "").strip() or None
    env_digest = _env_digest(git_sha, pip_hash, runtime_id)

    training_mode_raw = sanitized_config.get("training_mode")
    training_mode = None
    if training_mode_raw:
        training_mode = str(training_mode_raw)
    elif exp_row.training_mode is not None:
        training_mode = (
            exp_row.training_mode.value
            if hasattr(exp_row.training_mode, "value")
            else str(exp_row.training_mode)
        )

    seed_raw = sanitized_config.get("seed")
    try:
        seed_value = int(seed_raw) if seed_raw is not None else None
    except (TypeError, ValueError):
        seed_value = None

    recipe_id = str(sanitized_config.get("recipe") or sanitized_config.get("recipe_id") or "").strip() or None

    # Idempotency: if a manifest already exists for this experiment_id,
    # update in place (unique constraint protects us from two concurrent
    # captures; the second one just overwrites).
    existing = (
        await db.execute(
            select(TrainingManifest).where(TrainingManifest.experiment_id == experiment_id)
        )
    ).scalar_one_or_none()

    if existing is None:
        row = TrainingManifest(
            experiment_id=int(experiment_id),
            project_id=int(project_id),
            base_model_registry_id=registry_id,
            base_model_cache_fingerprint=cache_fingerprint,
            base_model_source_ref=source_ref,
            dataset_adapter_id=adapter_id,
            dataset_adapter_version=adapter_version,
            blueprint_revision_id=blueprint_id,
            blueprint_version=blueprint_version,
            dataset_snapshot_ids=dataset_snapshots,
            recipe_id=recipe_id,
            runtime_id=runtime_id,
            training_mode=training_mode,
            tokenizer_name=tokenizer_name,
            tokenizer_config_hash=tokenizer_hash,
            seed=seed_value,
            resolved_config=sanitized_config,
            git_sha=git_sha,
            pip_freeze_blob=pip_blob,
            pip_freeze_hash=pip_hash,
            env_digest=env_digest,
            artifact_ids=dict(artifact_ids or {}),
            capture_warnings=warnings,
            schema_version=_MANIFEST_SCHEMA_VERSION,
        )
        db.add(row)
    else:
        existing.base_model_registry_id = registry_id
        existing.base_model_cache_fingerprint = cache_fingerprint
        existing.base_model_source_ref = source_ref
        existing.dataset_adapter_id = adapter_id
        existing.dataset_adapter_version = adapter_version
        existing.blueprint_revision_id = blueprint_id
        existing.blueprint_version = blueprint_version
        existing.dataset_snapshot_ids = dataset_snapshots
        existing.recipe_id = recipe_id
        existing.runtime_id = runtime_id
        existing.training_mode = training_mode
        existing.tokenizer_name = tokenizer_name
        existing.tokenizer_config_hash = tokenizer_hash
        existing.seed = seed_value
        existing.resolved_config = sanitized_config
        existing.git_sha = git_sha
        existing.pip_freeze_blob = pip_blob
        existing.pip_freeze_hash = pip_hash
        existing.env_digest = env_digest
        existing.artifact_ids = dict(artifact_ids or {})
        existing.capture_warnings = warnings
        existing.schema_version = _MANIFEST_SCHEMA_VERSION
        row = existing

    await db.flush()
    await db.commit()
    await db.refresh(row)
    return row


async def get_training_manifest(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int,
) -> dict[str, Any]:
    exp_row = (
        await db.execute(
            select(Experiment).where(
                Experiment.id == experiment_id,
                Experiment.project_id == project_id,
            )
        )
    ).scalar_one_or_none()
    if exp_row is None:
        raise ValueError("experiment_not_found")

    row = (
        await db.execute(
            select(TrainingManifest).where(
                TrainingManifest.experiment_id == experiment_id,
                TrainingManifest.project_id == project_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise ValueError("manifest_not_captured")
    return serialize_training_manifest(row)
