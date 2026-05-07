"""Support-bundle service with redaction (priority.md P34, Wave G).

Packages a project's recent operational state into a single zip:

- ``manifest.json`` — top-level summary (sections, counts, redactions).
- ``project.json`` — :class:`Project` row (redacted).
- ``experiments.json`` — :class:`Experiment` rows + configs (redacted).
- ``training_manifests.json`` — P14 :class:`TrainingManifest` rows.
- ``autopilot_decisions.json`` — P1 :class:`AutopilotDecision` rows.
- ``run_events.json`` — P31 :class:`RunEvent` rows for the project.
- ``deployment_versions.json`` — P25 :class:`DeploymentVersion` +
  audit log + telemetry sample counts.
- ``failure_clusters.json`` — P33 :class:`FailureCluster` rows.
- ``model_registry.json`` — :class:`ModelRegistryEntry` rows.
- ``env.txt`` — Python / OS / runtime metadata. **No env vars.**

Two-layer redaction is applied to every JSON section:

1. **Key blocklist** — keys whose lowercase form contains any of
   ``token``, ``secret``, ``password``, ``credential``, ``api_key``,
   ``private_key``, ``access_key``, ``auth`` are scrubbed regardless
   of value.
2. **Value patterns** — string values matching known secret shapes
   (HF token ``hf_...``, OpenAI ``sk-...``, AWS access key
   ``AKIA...``, bearer prefix, JWT-ish) are scrubbed.

Scrubbed values become ``"***REDACTED:<reason>***"`` and the per-
section count lands in the bundle's ``redactions_applied`` block so
the operator can verify the bundle was scrubbed before forwarding it
to support.

Stable reason codes (``ValueError``) → API:
- ``project_not_found`` (404)
- ``support_bundle_not_found`` (404)
- ``support_bundle_expired`` (410)
- ``support_bundle_invalid_token`` (403)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import secrets
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.autopilot_decision import AutopilotDecision
from app.models.deployment_version import (
    DeploymentRollback,
    DeploymentVersion,
)
from app.models.experiment import Experiment
from app.models.failure_cluster import FailureCluster
from app.models.project import Project
from app.models.registry import ModelRegistryEntry
from app.models.run_event import RunEvent
from app.models.support_bundle import SupportBundle
from app.models.training_manifest import TrainingManifest


_DEFAULT_TTL_SECONDS = 24 * 3600  # bundles expire after 1 day
_RUN_EVENTS_LIMIT = 5000  # cap rows per section
_DEPLOYMENTS_LIMIT = 200
_DECISIONS_LIMIT = 5000
_FAILURE_CLUSTERS_LIMIT = 500


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


_KEY_BLOCKLIST_TOKENS: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "credential",
    "api_key",
    "apikey",
    "private_key",
    "access_key",
    "auth",
    "session",
    "bearer",
)

# Compiled patterns for value-based scrubbing. Each entry is
# (label, compiled_pattern). The label appears in the redacted value.
_VALUE_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
    ),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
        ),
    ),
    (
        "url_with_credentials",
        re.compile(r"https?://[^/\s:]+:[^/@\s]+@[^\s]+"),
    ),
    (
        "ssh_private_key",
        re.compile(
            r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
        ),
    ),
)


def _key_is_sensitive(key: str | None) -> bool:
    if not key:
        return False
    lowered = str(key).lower()
    return any(token in lowered for token in _KEY_BLOCKLIST_TOKENS)


def _scrub_string(value: str) -> tuple[str, str | None]:
    """Return ``(maybe_scrubbed_value, redaction_label or None)``."""
    if not value:
        return value, None
    for label, pattern in _VALUE_PATTERNS:
        if pattern.search(value):
            return f"***REDACTED:{label}***", label
    return value, None


class _RedactionCounter:
    """Mutable counter passed through the recursive walker."""

    __slots__ = ("by_reason",)

    def __init__(self) -> None:
        self.by_reason: dict[str, int] = {}

    def bump(self, reason: str) -> None:
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.by_reason.values())

    def asdict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_reason": dict(self.by_reason),
        }


def _redact(
    value: Any, *, counter: _RedactionCounter, key: str | None = None
) -> Any:
    """Recursively redact a JSON-serialisable value."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if _key_is_sensitive(k):
                # Drop the value entirely; preserve the key + a marker
                # so the downstream consumer knows something was here.
                if v is None or v == "":
                    out[k] = v
                else:
                    out[k] = "***REDACTED:sensitive_key***"
                    counter.bump("sensitive_key")
                continue
            out[k] = _redact(v, counter=counter, key=k)
        return out
    if isinstance(value, list):
        return [_redact(item, counter=counter, key=key) for item in value]
    if isinstance(value, str):
        scrubbed, label = _scrub_string(value)
        if label is not None:
            counter.bump(label)
        return scrubbed
    return value


def redact_payload(value: Any) -> tuple[Any, dict[str, Any]]:
    """Public entry point — redacts and returns (clean_value, stats)."""
    counter = _RedactionCounter()
    cleaned = _redact(value, counter=counter, key=None)
    return cleaned, counter.asdict()


# ---------------------------------------------------------------------------
# Section collectors
# ---------------------------------------------------------------------------


def _serialise_dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _serialise_project(row: Project) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "beginner_mode": getattr(row, "beginner_mode", None),
        "created_at": _serialise_dt(getattr(row, "created_at", None)),
        "updated_at": _serialise_dt(getattr(row, "updated_at", None)),
    }


def _serialise_experiment(row: Experiment) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "name": row.name,
        "description": row.description,
        "status": row.status.value if row.status else None,
        "base_model": row.base_model,
        "training_mode": getattr(row, "training_mode", None),
        "config": row.config or {},
        "metrics": row.metrics or {},
        "final_train_loss": row.final_train_loss,
        "final_eval_loss": row.final_eval_loss,
        "started_at": _serialise_dt(row.started_at),
        "completed_at": _serialise_dt(row.completed_at),
        "created_at": _serialise_dt(row.created_at),
    }


def _serialise_training_manifest(row: TrainingManifest) -> dict[str, Any]:
    return {
        "id": row.id,
        "experiment_id": row.experiment_id,
        "project_id": row.project_id,
        "base_model_registry_id": row.base_model_registry_id,
        "base_model_cache_fingerprint": row.base_model_cache_fingerprint,
        "base_model_source_ref": row.base_model_source_ref,
        "dataset_adapter_id": row.dataset_adapter_id,
        "dataset_adapter_version": row.dataset_adapter_version,
        "blueprint_revision_id": row.blueprint_revision_id,
        "blueprint_version": row.blueprint_version,
        "dataset_snapshot_ids": row.dataset_snapshot_ids or [],
        "recipe_id": row.recipe_id,
        "runtime_id": row.runtime_id,
        "training_mode": row.training_mode,
        "tokenizer_name": row.tokenizer_name,
        "tokenizer_config_hash": row.tokenizer_config_hash,
        "seed": row.seed,
        "resolved_config": row.resolved_config or {},
        "git_sha": row.git_sha,
        # pip_freeze_blob can be large; keep its hash but drop the body
        # to keep bundles small. The hash is what reproducibility needs.
        "pip_freeze_hash": row.pip_freeze_hash,
        "env_digest": row.env_digest,
        "artifact_ids": row.artifact_ids or {},
        "capture_warnings": row.capture_warnings or [],
        "captured_at": _serialise_dt(row.captured_at),
    }


def _serialise_autopilot_decision(row: AutopilotDecision) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "project_id": row.project_id,
        "sequence": row.sequence,
        "stage": row.stage,
        "status": row.status,
        "action": row.action,
        "reason_code": row.reason_code,
        "rationale": row.rationale,
        "summary": row.summary,
        "actor": row.actor,
        "changed": row.changed,
        "safe": row.safe,
        "blocker": row.blocker,
        "dry_run": row.dry_run,
        "intent": row.intent,
        "payload": row.payload or {},
        "created_at": _serialise_dt(row.created_at),
    }


def _serialise_run_event(row: RunEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "run_id": row.run_id,
        "parent_run_id": row.parent_run_id,
        "stage": row.stage,
        "severity": row.severity,
        "reason_code": row.reason_code,
        "actor": row.actor,
        "summary": row.summary,
        "payload": row.payload or {},
        "ts": _serialise_dt(row.ts),
        "created_at": _serialise_dt(row.created_at),
    }


def _serialise_deployment_version(row: DeploymentVersion) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "export_id": row.export_id,
        "registry_entry_id": row.registry_entry_id,
        "version": row.version,
        "target_id": row.target_id,
        "target_kind": row.target_kind,
        "endpoint_name": row.endpoint_name,
        "endpoint_handle": row.endpoint_handle,
        "region": row.region,
        "instance_type": row.instance_type,
        "status": row.status.value if row.status else None,
        "plan_payload": row.plan_payload or {},
        "promoted_reason": row.promoted_reason,
        "rejected_reason": row.rejected_reason,
        "rolled_back_reason": row.rolled_back_reason,
        "rolled_back_to_id": row.rolled_back_to_id,
        "actor": row.actor,
        "created_at": _serialise_dt(row.created_at),
        "promoted_at": _serialise_dt(row.promoted_at),
        "rejected_at": _serialise_dt(row.rejected_at),
        "rolled_back_at": _serialise_dt(row.rolled_back_at),
        "superseded_at": _serialise_dt(row.superseded_at),
    }


def _serialise_audit(row: DeploymentRollback) -> dict[str, Any]:
    return {
        "id": row.id,
        "deployment_version_id": row.deployment_version_id,
        "project_id": row.project_id,
        "sequence": row.sequence,
        "action": row.action.value if row.action else None,
        "reason": row.reason,
        "actor": row.actor,
        "status_after": row.status_after,
        "rolled_back_to_id": row.rolled_back_to_id,
        "payload": row.payload or {},
        "created_at": _serialise_dt(row.created_at),
    }


def _serialise_failure_cluster(row: FailureCluster) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "stage": row.stage,
        "reason_code": row.reason_code,
        "signature": row.signature,
        "failure_count": row.failure_count,
        "first_seen_at": _serialise_dt(row.first_seen_at),
        "last_seen_at": _serialise_dt(row.last_seen_at),
        "exemplar_event_ids": list(row.exemplar_event_ids or []),
        "exemplar_summaries": list(row.exemplar_summaries or []),
        "last_computed_at": _serialise_dt(row.last_computed_at),
    }


def _serialise_registry_entry(row: ModelRegistryEntry) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "experiment_id": row.experiment_id,
        "export_id": row.export_id,
        "name": row.name,
        "version": row.version,
        "stage": row.stage.value if row.stage else None,
        "deployment_status": (
            row.deployment_status.value if row.deployment_status else None
        ),
        "artifact_path": row.artifact_path,
        "readiness": row.readiness or {},
        "deployment": row.deployment or {},
        "created_at": _serialise_dt(row.created_at),
        "promoted_at": _serialise_dt(row.promoted_at),
        "deployed_at": _serialise_dt(row.deployed_at),
    }


def _build_env_text() -> str:
    lines = [
        "# Support bundle environment summary",
        "# (intentionally excludes os.environ — env vars often carry secrets)",
        f"python_version: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor()}",
        f"hostname: {platform.node()}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


async def _collect_sections(
    db: AsyncSession, *, project_id: int
) -> dict[str, list[dict[str, Any]]]:
    project = (
        await db.execute(select(Project).where(Project.id == project_id))
    ).scalar_one_or_none()
    if project is None:
        raise ValueError("project_not_found")

    experiments = list(
        (
            await db.execute(
                select(Experiment).where(Experiment.project_id == project_id)
                .order_by(Experiment.id.desc())
            )
        )
        .scalars()
        .all()
    )

    training_manifests = list(
        (
            await db.execute(
                select(TrainingManifest)
                .where(TrainingManifest.project_id == project_id)
                .order_by(TrainingManifest.captured_at.desc())
            )
        )
        .scalars()
        .all()
    )

    decisions = list(
        (
            await db.execute(
                select(AutopilotDecision)
                .where(AutopilotDecision.project_id == project_id)
                .order_by(AutopilotDecision.created_at.desc())
                .limit(_DECISIONS_LIMIT)
            )
        )
        .scalars()
        .all()
    )

    run_events = list(
        (
            await db.execute(
                select(RunEvent)
                .where(RunEvent.project_id == project_id)
                .order_by(RunEvent.ts.desc(), RunEvent.id.desc())
                .limit(_RUN_EVENTS_LIMIT)
            )
        )
        .scalars()
        .all()
    )

    deployment_versions = list(
        (
            await db.execute(
                select(DeploymentVersion)
                .where(DeploymentVersion.project_id == project_id)
                .order_by(DeploymentVersion.created_at.desc())
                .limit(_DEPLOYMENTS_LIMIT)
            )
        )
        .scalars()
        .all()
    )

    deployment_audit = list(
        (
            await db.execute(
                select(DeploymentRollback)
                .where(DeploymentRollback.project_id == project_id)
                .order_by(DeploymentRollback.created_at.desc())
                .limit(_DEPLOYMENTS_LIMIT * 4)
            )
        )
        .scalars()
        .all()
    )

    failure_clusters = list(
        (
            await db.execute(
                select(FailureCluster)
                .where(FailureCluster.project_id == project_id)
                .order_by(FailureCluster.failure_count.desc())
                .limit(_FAILURE_CLUSTERS_LIMIT)
            )
        )
        .scalars()
        .all()
    )

    registry_entries = list(
        (
            await db.execute(
                select(ModelRegistryEntry).where(
                    ModelRegistryEntry.project_id == project_id
                )
            )
        )
        .scalars()
        .all()
    )

    return {
        "project": [_serialise_project(project)],
        "experiments": [_serialise_experiment(e) for e in experiments],
        "training_manifests": [
            _serialise_training_manifest(m) for m in training_manifests
        ],
        "autopilot_decisions": [
            _serialise_autopilot_decision(d) for d in decisions
        ],
        "run_events": [_serialise_run_event(r) for r in run_events],
        "deployment_versions": [
            _serialise_deployment_version(d) for d in deployment_versions
        ],
        "deployment_audit": [
            _serialise_audit(a) for a in deployment_audit
        ],
        "failure_clusters": [
            _serialise_failure_cluster(c) for c in failure_clusters
        ],
        "model_registry": [
            _serialise_registry_entry(r) for r in registry_entries
        ],
    }


# ---------------------------------------------------------------------------
# Bundle creation
# ---------------------------------------------------------------------------


def _bundle_dir() -> Path:
    base = Path(settings.DATA_DIR) / "support_bundles"
    base.mkdir(parents=True, exist_ok=True)
    return base


async def create_support_bundle(
    db: AsyncSession,
    *,
    project_id: int,
    actor: str | None = None,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    """Build a redacted support bundle and persist its metadata row."""
    sections = await _collect_sections(db, project_id=project_id)

    redactions_by_section: dict[str, dict[str, Any]] = {}
    redacted_sections: dict[str, list[dict[str, Any]]] = {}
    section_counts: dict[str, int] = {}
    for name, rows in sections.items():
        cleaned, stats = redact_payload(rows)
        redacted_sections[name] = cleaned
        redactions_by_section[name] = stats
        section_counts[name] = len(rows)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=int(ttl_seconds))
    bundle_uid = secrets.token_hex(16)
    download_token = secrets.token_hex(24)
    actor_str = (actor or "system").strip()[:128] or "system"

    bundle_dir = _bundle_dir() / str(project_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    file_path = bundle_dir / f"{bundle_uid}.zip"

    manifest = {
        "bundle_uid": bundle_uid,
        "project_id": project_id,
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "actor": actor_str,
        "section_counts": section_counts,
        "redactions_applied": redactions_by_section,
        "schema_version": 1,
        "format": "brewslm-support-bundle/v1",
    }

    # Materialise the zip on disk. Each section is its own JSON file
    # under ``sections/<name>.json`` so support engineers can read them
    # independently without parsing a megablob.
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("env.txt", _build_env_text())
        for name, rows in redacted_sections.items():
            zf.writestr(
                f"sections/{name}.json",
                json.dumps(rows, indent=2, default=str),
            )

    body = buffer.getvalue()
    file_path.write_bytes(body)
    sha256 = hashlib.sha256(body).hexdigest()

    row = SupportBundle(
        project_id=project_id,
        bundle_uid=bundle_uid,
        download_token=download_token,
        file_path=str(file_path),
        size_bytes=len(body),
        sha256=sha256,
        actor=actor_str,
        redactions_applied=redactions_by_section,
        section_counts=section_counts,
        expires_at=expires_at,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)

    return {
        "bundle_uid": bundle_uid,
        "project_id": project_id,
        "size_bytes": len(body),
        "sha256": sha256,
        "section_counts": section_counts,
        "redactions_applied": redactions_by_section,
        "expires_at": expires_at.isoformat(),
        "created_at": now.isoformat(),
        "download_url": (
            f"/api/support-bundles/{bundle_uid}/download"
            f"?token={download_token}"
        ),
        "download_token": download_token,
        "actor": actor_str,
    }


# ---------------------------------------------------------------------------
# Download (verification + path resolution)
# ---------------------------------------------------------------------------


async def resolve_bundle_for_download(
    db: AsyncSession, *, bundle_uid: str, token: str
) -> tuple[SupportBundle, Path]:
    result = await db.execute(
        select(SupportBundle).where(SupportBundle.bundle_uid == bundle_uid)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("support_bundle_not_found")

    # Constant-time comparison to avoid timing leakage on the token.
    if not secrets.compare_digest(token or "", row.download_token):
        raise ValueError("support_bundle_invalid_token")

    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < datetime.now(timezone.utc):
        raise ValueError("support_bundle_expired")

    path = Path(row.file_path)
    if not path.exists() or not path.is_file():
        raise ValueError("support_bundle_not_found")

    return row, path


async def list_support_bundles(
    db: AsyncSession, *, project_id: int, limit: int = 50
) -> dict[str, Any]:
    project = (
        await db.execute(select(Project).where(Project.id == project_id))
    ).scalar_one_or_none()
    if project is None:
        raise ValueError("project_not_found")

    bounded = max(1, min(int(limit), 200))
    result = await db.execute(
        select(SupportBundle)
        .where(SupportBundle.project_id == project_id)
        .order_by(SupportBundle.created_at.desc())
        .limit(bounded)
    )
    rows = list(result.scalars().all())
    return {
        "project_id": project_id,
        "limit": bounded,
        "bundles": [
            {
                "bundle_uid": r.bundle_uid,
                "size_bytes": r.size_bytes,
                "sha256": r.sha256,
                "section_counts": dict(r.section_counts or {}),
                "redactions_applied": dict(r.redactions_applied or {}),
                "actor": r.actor,
                "created_at": _serialise_dt(r.created_at),
                "expires_at": _serialise_dt(r.expires_at),
            }
            for r in rows
        ],
    }
