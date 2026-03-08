"""Runtime-manageable system settings service.

This service provides a controlled subset of app settings that can be edited
from the UI and persisted to a JSON override file under DATA_DIR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class RuntimeSettingSpec:
    key: str
    label: str
    description: str
    category: str
    value_type: str  # string | bool | int
    requires_restart: bool
    options: tuple[str, ...] = ()
    multiline: bool = False


_RUNTIME_SETTING_SPECS: tuple[RuntimeSettingSpec, ...] = (
    RuntimeSettingSpec(
        key="TEACHER_MODEL_API_URL",
        label="Teacher Model API URL",
        description="Synthetic generation model endpoint.",
        category="integrations",
        value_type="string",
        requires_restart=False,
    ),
    RuntimeSettingSpec(
        key="JUDGE_MODEL_API_URL",
        label="Judge Model API URL",
        description="Evaluation judge model endpoint.",
        category="integrations",
        value_type="string",
        requires_restart=False,
    ),
    RuntimeSettingSpec(
        key="ALLOW_SYNTHETIC_DEMO_FALLBACK",
        label="Allow Synthetic Demo Fallback",
        description="Allow synthetic generation fallback when provider calls fail.",
        category="integrations",
        value_type="bool",
        requires_restart=False,
    ),
    RuntimeSettingSpec(
        key="ALLOW_SIMULATED_INGESTION_FALLBACK",
        label="Allow Simulated Ingestion Fallback",
        description="Allow ingestion fallback behavior for unsupported/failed remote import flows.",
        category="integrations",
        value_type="bool",
        requires_restart=False,
    ),
    RuntimeSettingSpec(
        key="TRAINING_BACKEND",
        label="Training Backend",
        description="Default training backend mode.",
        category="training",
        value_type="string",
        requires_restart=True,
        options=("simulate", "external"),
    ),
    RuntimeSettingSpec(
        key="ALLOW_SIMULATED_TRAINING",
        label="Allow Simulated Training",
        description="Permit fallback to simulated training runtime in unsupported environments.",
        category="training",
        value_type="bool",
        requires_restart=False,
    ),
    RuntimeSettingSpec(
        key="TRAINING_EXTERNAL_CMD",
        label="Training External Command",
        description="Command template used by external training runtime.",
        category="training",
        value_type="string",
        requires_restart=True,
        multiline=True,
    ),
    RuntimeSettingSpec(
        key="COMPRESSION_BACKEND",
        label="Compression Backend",
        description="Default compression backend mode.",
        category="compression",
        value_type="string",
        requires_restart=True,
        options=("external", "stub"),
    ),
    RuntimeSettingSpec(
        key="ALLOW_STUB_COMPRESSION",
        label="Allow Stub Compression",
        description="Allow stub compression outputs when full runtime/tooling is unavailable.",
        category="compression",
        value_type="bool",
        requires_restart=False,
    ),
    RuntimeSettingSpec(
        key="QUANTIZE_EXTERNAL_CMD",
        label="Quantize External Command",
        description="Command template used for quantization jobs.",
        category="compression",
        value_type="string",
        requires_restart=True,
        multiline=True,
    ),
    RuntimeSettingSpec(
        key="MERGE_LORA_EXTERNAL_CMD",
        label="Merge LoRA External Command",
        description="Command template used for LoRA merge jobs.",
        category="compression",
        value_type="string",
        requires_restart=True,
        multiline=True,
    ),
    RuntimeSettingSpec(
        key="BENCHMARK_EXTERNAL_CMD",
        label="Benchmark External Command",
        description="Command template used for benchmark jobs.",
        category="compression",
        value_type="string",
        requires_restart=True,
        multiline=True,
    ),
    RuntimeSettingSpec(
        key="EXTERNAL_COMMAND_TIMEOUT_SECONDS",
        label="External Command Timeout (seconds)",
        description="Timeout applied to external command execution.",
        category="worker",
        value_type="int",
        requires_restart=False,
    ),
    RuntimeSettingSpec(
        key="REDIS_URL",
        label="Redis URL",
        description="Redis endpoint for pubsub/log streams.",
        category="worker",
        value_type="string",
        requires_restart=True,
    ),
    RuntimeSettingSpec(
        key="CELERY_BROKER_URL",
        label="Celery Broker URL",
        description="Broker connection used by Celery workers.",
        category="worker",
        value_type="string",
        requires_restart=True,
    ),
    RuntimeSettingSpec(
        key="CELERY_RESULT_BACKEND",
        label="Celery Result Backend",
        description="Result backend used by Celery workers.",
        category="worker",
        value_type="string",
        requires_restart=True,
    ),
)

_SPEC_BY_KEY = {item.key: item for item in _RUNTIME_SETTING_SPECS}


def _settings_dir() -> Path:
    path = settings.DATA_DIR / "system"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _settings_path() -> Path:
    return _settings_dir() / "runtime_overrides.json"


def _load_persisted_overrides() -> dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items() if str(k) in _SPEC_BY_KEY}


def _save_persisted_overrides(payload: dict[str, Any]) -> str:
    path = _settings_path()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


def _normalize_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    token = str(value).strip()
    return int(token)


def _coerce_value(spec: RuntimeSettingSpec, value: Any) -> Any:
    if spec.value_type == "bool":
        return _normalize_bool(value)
    if spec.value_type == "int":
        normalized = _normalize_int(value)
        if spec.key == "EXTERNAL_COMMAND_TIMEOUT_SECONDS":
            return max(1, normalized)
        return normalized
    normalized = str(value).strip()
    if spec.options and normalized not in set(spec.options):
        raise ValueError(f"{spec.key} must be one of: {', '.join(spec.options)}")
    return normalized


def _apply_overrides(overrides: dict[str, Any]) -> tuple[list[str], list[str]]:
    applied: list[str] = []
    errors: list[str] = []
    for key, raw in overrides.items():
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            continue
        try:
            value = _coerce_value(spec, raw)
            setattr(settings, key, value)
            applied.append(key)
        except Exception as e:
            errors.append(f"{key}: {e}")
    return applied, errors


def apply_persisted_runtime_overrides() -> dict[str, Any]:
    """Load persisted overrides and apply them to in-memory settings."""
    overrides = _load_persisted_overrides()
    applied, errors = _apply_overrides(overrides)
    return {
        "path": str(_settings_path()),
        "count": len(overrides),
        "applied_keys": applied,
        "errors": errors,
    }


def list_runtime_settings() -> dict[str, Any]:
    overrides = _load_persisted_overrides()
    fields: list[dict[str, Any]] = []
    for spec in _RUNTIME_SETTING_SPECS:
        fields.append(
            {
                "key": spec.key,
                "label": spec.label,
                "description": spec.description,
                "category": spec.category,
                "type": spec.value_type,
                "requires_restart": spec.requires_restart,
                "options": list(spec.options),
                "multiline": spec.multiline,
                "source": "override" if spec.key in overrides else "env",
                "value": getattr(settings, spec.key),
            }
        )
    return {
        "path": str(_settings_path()),
        "count": len(fields),
        "fields": fields,
    }


def update_runtime_settings(updates: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(updates, dict):
        raise ValueError("updates payload must be an object")

    unknown = [str(k) for k in updates.keys() if str(k) not in _SPEC_BY_KEY]
    if unknown:
        raise ValueError(f"Unsupported setting keys: {', '.join(sorted(unknown))}")

    normalized_updates: dict[str, Any] = {}
    for key, raw in updates.items():
        spec = _SPEC_BY_KEY[str(key)]
        normalized_updates[key] = _coerce_value(spec, raw)

    overrides = _load_persisted_overrides()
    overrides.update(normalized_updates)
    path = _save_persisted_overrides(overrides)
    applied, errors = _apply_overrides(normalized_updates)

    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(f"Failed to apply one or more settings: {preview}")

    requires_restart = [
        key
        for key in normalized_updates
        if _SPEC_BY_KEY[key].requires_restart
    ]

    return {
        "updated_keys": sorted(applied),
        "requires_restart_keys": sorted(requires_restart),
        "path": path,
        "settings": list_runtime_settings(),
    }

