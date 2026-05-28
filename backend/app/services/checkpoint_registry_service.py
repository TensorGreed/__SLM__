"""Local pre-fine-tuned warm-start checkpoint registry (Track 1, Epic B).

A curated registry of task-pretuned starting checkpoints. Each entry is a
directory under ``backend/data/pretrained_checkpoints/<name>/`` holding a
``manifest.json`` descriptor. Recipes recommend a checkpoint by ``name`` via
``recommended_starting_checkpoint``; training resolves it to the local artifact
path when the weights are present and compatible, and falls back to the
configured ``base_model`` otherwise.

The heavyweight checkpoint *artifacts* (the ~200 MB warm-start weights produced
by the deferred ~32 GPU-hour job) are not committed — only the small manifests
are. A manifest with ``status="planned"`` (or absent weights) resolves to a
clean base-model fallback, so the wiring is exercised before any weights exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# backend/app/services/checkpoint_registry_service.py -> backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_ROOT = _BACKEND_ROOT / "data" / "pretrained_checkpoints"

MANIFEST_FILENAME = "manifest.json"

STATUS_AVAILABLE = "available"
STATUS_PLANNED = "planned"


def _registry_root(root: Path | str | None) -> Path:
    return DEFAULT_REGISTRY_ROOT if root is None else Path(root)


def _normalize_name(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_model_id(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_manifest(
    raw: dict[str, Any],
    *,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    name = _normalize_name(raw.get("name")) or _normalize_name(checkpoint_dir.name)
    status = str(raw.get("status") or "").strip().lower() or STATUS_AVAILABLE

    artifact_rel = str(raw.get("artifact_path") or "").strip()
    resolved_artifact_path: str | None = None
    if artifact_rel:
        candidate = Path(artifact_rel)
        resolved = candidate if candidate.is_absolute() else (checkpoint_dir / candidate)
        resolved_artifact_path = str(resolved)
    artifact_exists = bool(resolved_artifact_path) and Path(resolved_artifact_path).exists()
    available = status != STATUS_PLANNED and artifact_exists

    return {
        "name": name,
        "display_name": str(raw.get("display_name") or name),
        "description": str(raw.get("description") or ""),
        "base_model": str(raw.get("base_model") or "").strip(),
        "task_shape": str(raw.get("task_shape") or "").strip(),
        "status": status,
        "artifact_path": artifact_rel,
        "resolved_artifact_path": resolved_artifact_path,
        "artifact_exists": artifact_exists,
        "available": available,
        "hf_repo_id": str(raw.get("hf_repo_id") or "").strip(),
        "checkpoint_dir": str(checkpoint_dir),
        "metadata": dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
    }


def _load_manifest_file(manifest_path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return _normalize_manifest(raw, checkpoint_dir=manifest_path.parent)


def list_checkpoints(*, root: Path | str | None = None) -> list[dict[str, Any]]:
    """List normalized manifests for every registered checkpoint directory."""
    registry_root = _registry_root(root)
    if not registry_root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(registry_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        manifest = _load_manifest_file(manifest_path)
        if manifest is not None:
            out.append(manifest)
    return out


def load_checkpoint(name: str, *, root: Path | str | None = None) -> dict[str, Any] | None:
    """Look up a single checkpoint manifest by its registry ``name``."""
    token = _normalize_name(name)
    if not token:
        return None
    registry_root = _registry_root(root)
    direct = registry_root / token / MANIFEST_FILENAME
    if direct.is_file():
        manifest = _load_manifest_file(direct)
        if manifest is not None and manifest["name"] == token:
            return manifest
    for manifest in list_checkpoints(root=registry_root):
        if manifest["name"] == token:
            return manifest
    return None


def resolve_starting_checkpoint(
    *,
    base_model: str,
    recommended_checkpoint: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve the effective starting weights for a training run.

    Defaults to the recommended warm-start checkpoint when it is registered,
    architecture-compatible with ``base_model``, and its weights exist locally.
    Falls back to ``base_model`` (with a ``reason``) in every other case.
    """
    base = str(base_model or "").strip()
    result: dict[str, Any] = {
        "effective_base_model": base,
        "source": "base_model",
        "checkpoint_name": None,
        "manifest": None,
        "reason": "",
    }

    name = _normalize_name(recommended_checkpoint)
    if not name:
        result["reason"] = "no_checkpoint_recommended"
        return result

    manifest = load_checkpoint(name, root=root)
    if manifest is None:
        result["checkpoint_name"] = name
        result["reason"] = f"checkpoint_not_registered:{name}"
        return result

    result["checkpoint_name"] = manifest["name"]
    result["manifest"] = manifest

    manifest_base = _normalize_model_id(manifest.get("base_model"))
    if manifest_base and base and manifest_base != _normalize_model_id(base):
        result["reason"] = f"checkpoint_base_model_mismatch:{manifest['name']}"
        return result

    if manifest["status"] == STATUS_PLANNED:
        result["reason"] = f"checkpoint_planned:{manifest['name']}"
        return result

    if not manifest["artifact_exists"]:
        result["reason"] = f"checkpoint_artifact_missing:{manifest['name']}"
        return result

    result["effective_base_model"] = manifest["resolved_artifact_path"]
    result["source"] = "checkpoint"
    result["reason"] = f"warm_start:{manifest['name']}"
    return result
