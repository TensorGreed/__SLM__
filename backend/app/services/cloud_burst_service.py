"""Cloud GPU burst planning: provider catalog, quote estimation, and launch plans."""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import StrictExecutionError
from app.models.experiment import Experiment
from app.services.secret_service import get_project_secret_value


_PROVIDER_CATALOG: dict[str, dict[str, Any]] = {
    "runpod": {
        "provider_id": "runpod",
        "display_name": "RunPod",
        "description": "On-demand and spot GPU pods with API-driven provisioning.",
        "credential_provider": "runpod",
        "credential_keys": ["api_key"],
        "regions": ["US", "EU"],
        "supports_spot": True,
        "supports_live_execution": True,
        "supports_managed_cancel": True,
        "supports_live_logs": True,
        "api_docs": "https://docs.runpod.io/",
    },
    "lambda_labs": {
        "provider_id": "lambda_labs",
        "display_name": "Lambda Labs",
        "description": "Cloud GPU instances for ML training and inference.",
        "credential_provider": "lambda_labs",
        "credential_keys": ["api_key"],
        "regions": ["US-WEST", "US-EAST"],
        "supports_spot": False,
        "supports_live_execution": False,
        "supports_managed_cancel": False,
        "supports_live_logs": False,
        "api_docs": "https://docs.lambda.ai/public-cloud/",
    },
    "aws_sagemaker": {
        "provider_id": "aws_sagemaker",
        "display_name": "AWS SageMaker",
        "description": "Managed training jobs and distributed training orchestration.",
        "credential_provider": "aws",
        "credential_keys": ["access_key_id", "secret_access_key", "region"],
        "regions": ["us-east-1", "us-west-2", "eu-west-1"],
        "supports_spot": True,
        "supports_live_execution": False,
        "supports_managed_cancel": False,
        "supports_live_logs": False,
        "api_docs": "https://docs.aws.amazon.com/sagemaker/",
    },
}


_GPU_SKUS: dict[str, dict[str, Any]] = {
    "a10g.24gb": {
        "display_name": "A10G 24GB",
        "vram_gb": 24,
        "hourly_usd": {
            "runpod": 0.59,
            "lambda_labs": 0.75,
            "aws_sagemaker": 1.20,
        },
    },
    "l40s.48gb": {
        "display_name": "L40S 48GB",
        "vram_gb": 48,
        "hourly_usd": {
            "runpod": 1.49,
            "lambda_labs": 1.65,
            "aws_sagemaker": 2.25,
        },
    },
    "h100.80gb": {
        "display_name": "H100 80GB",
        "vram_gb": 80,
        "hourly_usd": {
            "runpod": 2.49,
            "lambda_labs": 2.99,
            "aws_sagemaker": 4.10,
        },
    },
}

_TERMINAL_CLOUD_BURST_JOB_STATES: set[str] = {
    "completed",
    "failed",
    "cancelled",
}

_CLOUD_BURST_ACTIVE_STATES: set[str] = {
    "submitted",
    "provisioning",
    "syncing",
    "running",
    "cancel_requested",
}

_DEFAULT_SYNC_INCLUDE_GLOBS_SMART: tuple[str, ...] = (
    "training_config.json",
    "training_report.json",
    "metrics.json",
    "events.jsonl",
    "model/**",
    "checkpoints/**",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.ckpt",
    "*.gguf",
    "*.json",
)
_DEFAULT_SYNC_EXCLUDE_GLOBS: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/*.tmp",
    "**/*.temp",
    "**/*.lock",
)
_MAX_SYNC_HISTORY_ENTRIES = 20
_DEFAULT_API_TIMEOUT_SECONDS = 12.0
_DEFAULT_POLL_INTERVAL_SECONDS = 4.0
_MAX_PROVIDER_RETRIES = 3
_RETRY_BACKOFF_SECONDS = (0.35, 0.8, 1.5)
_RUNPOD_GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql"
_RUNPOD_LOG_FETCH_LIMIT = 64
_METRIC_LINE_PREFIX = "SLM_METRIC "
_MAX_METRIC_HISTORY = 1000
_SYNC_MANIFEST_FILE = ".sync_manifest.v2.json"


@dataclass
class _ManagedCloudBurstTask:
    task: asyncio.Task[None]
    cancel_event: asyncio.Event
    provider_id: str | None = None
    provider_job_id: str | None = None
    provider_api_key: str | None = None


_MANAGED_CLOUD_BURST_TASKS: dict[str, _ManagedCloudBurstTask] = {}
_MANAGED_CLOUD_BURST_TASK_LOCK = asyncio.Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_cloud_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "cloud_burst"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_cloud_runs_dir(project_id: int) -> Path:
    path = _project_cloud_dir(project_id) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_cloud_audit_path(project_id: int) -> Path:
    return _project_cloud_dir(project_id) / "audit.jsonl"


def _append_audit_event(
    project_id: int,
    event: dict[str, Any],
) -> None:
    payload = dict(event)
    payload.setdefault("at", _utcnow_iso())
    path = _project_cloud_audit_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _cloud_run_record_path(project_id: int, run_id: str) -> Path:
    return _project_cloud_runs_dir(project_id) / f"{run_id}.json"


def _cloud_run_logs_path(project_id: int, run_id: str) -> Path:
    return _project_cloud_runs_dir(project_id) / f"{run_id}.log"


def _cloud_run_artifacts_dir(project_id: int, run_id: str) -> Path:
    path = _project_cloud_runs_dir(project_id) / run_id / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_glob_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    out: list[str] = []
    for item in list(values or []):
        token = str(item or "").strip()
        if token and token not in out:
            out.append(token)
    return out


def _normalize_sync_policy(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"smart", "all"}:
        return token
    return "smart"


def _run_status_timeline_append(
    run: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> None:
    items = list(run.get("status_timeline") or [])
    items.append(
        {
            "status": status,
            "at": _utcnow_iso(),
            "reason": str(reason or "").strip(),
        }
    )
    run["status_timeline"] = items[-200:]


def _persist_run_record(
    project_id: int,
    run_id: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(run)
    payload["updated_at"] = _utcnow_iso()
    _write_json_payload(_cloud_run_record_path(project_id, run_id), payload)
    return payload


def _load_run_record(project_id: int, run_id: str) -> dict[str, Any]:
    token = str(run_id or "").strip()
    if not token:
        raise ValueError("run_id is required")
    path = _cloud_run_record_path(project_id, token)
    payload = _read_json_payload(path)
    if not payload:
        raise ValueError(f"Cloud burst run {token} not found in project {project_id}")
    payload.setdefault("run_id", token)
    payload.setdefault("project_id", int(project_id))
    payload.setdefault("record_path", str(path))
    payload.setdefault("logs_path", str(_cloud_run_logs_path(project_id, token)))
    return payload


def _append_run_log(
    project_id: int,
    run_id: str,
    line: str,
) -> None:
    message = str(line or "").strip()
    if not message:
        return
    path = _cloud_run_logs_path(project_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def _read_run_logs_tail(project_id: int, run_id: str, *, tail: int = 200) -> list[str]:
    path = _cloud_run_logs_path(project_id, run_id)
    if not path.exists():
        return []
    safe_tail = max(0, min(int(tail or 200), 5000))
    if safe_tail <= 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [str(item).rstrip("\n") for item in handle.readlines()]
    except Exception:
        return []
    return lines[-safe_tail:]


def _coerce_finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_metric_line(line: str) -> dict[str, Any] | None:
    text = str(line or "").strip()
    if not text:
        return None

    payload_text = ""
    marker_index = text.find(_METRIC_LINE_PREFIX)
    if marker_index >= 0:
        payload_text = text[marker_index + len(_METRIC_LINE_PREFIX):].strip()
    elif text.startswith("{") and text.endswith("}"):
        payload_text = text

    if not payload_text:
        return None

    parsed: dict[str, Any] | None = None
    try:
        candidate = json.loads(payload_text)
        if isinstance(candidate, dict):
            parsed = candidate
    except Exception:
        parsed = None

    if parsed is None and payload_text.startswith("{") and payload_text.endswith("}"):
        # Legacy trainer logs can be pseudo-JSON with single quotes.
        normalized = payload_text.replace("'", "\"")
        try:
            candidate = json.loads(normalized)
            if isinstance(candidate, dict):
                parsed = candidate
        except Exception:
            parsed = None

    if parsed is None:
        return None

    metric: dict[str, Any] = {}
    for key in ("step", "epoch", "train_loss", "eval_loss", "learning_rate", "throughput_tps"):
        if key not in parsed:
            continue
        numeric = _coerce_finite_number(parsed.get(key))
        if numeric is not None:
            if key in {"step"}:
                metric[key] = int(max(0, round(numeric)))
            else:
                metric[key] = numeric
    if not metric:
        return None
    return metric


def _ingest_metric_history(
    run: dict[str, Any],
    *,
    metrics: list[dict[str, Any]],
) -> None:
    if not metrics:
        return
    existing = list(run.get("metrics_tail") or [])
    existing_hashes = set(str(item) for item in list(run.get("metrics_hashes") or []))
    merged = list(existing)
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        metric_payload = dict(metric)
        metric_payload.setdefault("at", _utcnow_iso())
        fingerprint = hashlib.sha1(
            json.dumps(metric_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if fingerprint in existing_hashes:
            continue
        existing_hashes.add(fingerprint)
        merged.append(metric_payload)
    run["metrics_tail"] = merged[-_MAX_METRIC_HISTORY:]
    run["metrics_hashes"] = list(existing_hashes)[-_MAX_METRIC_HISTORY:]


def _ingest_provider_log_lines(
    project_id: int,
    run_id: str,
    run: dict[str, Any],
    *,
    lines: list[str],
    source: str,
) -> None:
    if not lines:
        return
    log_bridge = dict(run.get("log_bridge") or {})
    seen_hashes = set(str(item) for item in list(log_bridge.get("seen_hashes") or []))

    accepted_lines: list[str] = []
    parsed_metrics: list[dict[str, Any]] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        fingerprint = hashlib.sha1(line.encode("utf-8")).hexdigest()
        if fingerprint in seen_hashes:
            continue
        seen_hashes.add(fingerprint)
        decorated = f"{source}: {line}" if source else line
        accepted_lines.append(decorated)
        metric = _parse_metric_line(line)
        if metric is not None:
            parsed_metrics.append(metric)

    for line in accepted_lines:
        _append_run_log(project_id, run_id, line)
    _ingest_metric_history(run, metrics=parsed_metrics)
    log_bridge["seen_hashes"] = list(seen_hashes)[-1000:]
    if accepted_lines:
        log_bridge["last_ingested_at"] = _utcnow_iso()
        log_bridge["last_ingested_count"] = len(accepted_lines)
    run["log_bridge"] = log_bridge


def _serialize_run(
    run: dict[str, Any],
    *,
    logs_tail: int = 200,
) -> dict[str, Any]:
    project_id = int(run.get("project_id") or 0)
    run_id = str(run.get("run_id") or "").strip()
    status = str(run.get("status") or "unknown").strip().lower() or "unknown"
    payload = dict(run)
    payload["status"] = status
    payload["can_cancel"] = status in _CLOUD_BURST_ACTIVE_STATES
    payload["logs_tail"] = _read_run_logs_tail(project_id, run_id, tail=logs_tail)
    payload["logs_tail_count"] = len(list(payload.get("logs_tail") or []))
    payload["metrics_tail"] = list(run.get("metrics_tail") or [])[-200:]
    payload["metrics_tail_count"] = len(list(payload.get("metrics_tail") or []))
    payload.pop("metrics_hashes", None)
    log_bridge = payload.get("log_bridge")
    if isinstance(log_bridge, dict):
        seen_hashes = list(log_bridge.get("seen_hashes") or [])
        payload["log_bridge"] = {
            **{k: v for k, v in log_bridge.items() if k != "seen_hashes"},
            "seen_hash_count": len(seen_hashes),
        }
    return payload


def _mark_run_status(
    project_id: int,
    run_id: str,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    run = _load_run_record(project_id, run_id)
    prev_status = str(run.get("status") or "unknown").strip().lower() or "unknown"
    next_status = str(status or "").strip().lower() or "unknown"
    run["status"] = next_status
    run["status_reason"] = str(reason or "").strip()
    if next_status in {"provisioning", "syncing", "running"} and not run.get("started_at"):
        run["started_at"] = _utcnow_iso()
    if next_status in _TERMINAL_CLOUD_BURST_JOB_STATES:
        run["finished_at"] = run.get("finished_at") or _utcnow_iso()
    _run_status_timeline_append(run, status=next_status, reason=reason)
    persisted = _persist_run_record(project_id, run_id, run)
    _append_run_log(project_id, run_id, f"status={next_status} ({reason})")
    _append_audit_event(
        project_id,
        {
            "event": "status_transition",
            "run_id": run_id,
            "from_status": prev_status,
            "to_status": next_status,
            "reason": str(reason or "").strip(),
        },
    )
    return persisted


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lower()


def _require_provider(provider_id: str) -> dict[str, Any]:
    token = _normalize_token(provider_id)
    item = _PROVIDER_CATALOG.get(token)
    if item is None:
        allowed = ", ".join(sorted(_PROVIDER_CATALOG.keys()))
        raise ValueError(f"Unsupported provider '{provider_id}'. Available: {allowed}")
    return item


def _require_gpu_sku(gpu_sku: str) -> dict[str, Any]:
    token = _normalize_token(gpu_sku)
    item = _GPU_SKUS.get(token)
    if item is None:
        allowed = ", ".join(sorted(_GPU_SKUS.keys()))
        raise ValueError(f"Unsupported gpu_sku '{gpu_sku}'. Available: {allowed}")
    return item


def list_cloud_burst_catalog() -> dict[str, Any]:
    providers = [
        {
            **item,
        }
        for item in _PROVIDER_CATALOG.values()
    ]
    gpu_skus = [
        {
            "gpu_sku": gpu_sku,
            **item,
        }
        for gpu_sku, item in _GPU_SKUS.items()
    ]
    return {
        "generated_at": _utcnow_iso(),
        "provider_count": len(providers),
        "gpu_sku_count": len(gpu_skus),
        "providers": providers,
        "gpu_skus": gpu_skus,
    }


def estimate_cloud_burst_quote(
    *,
    provider_id: str,
    gpu_sku: str,
    duration_hours: float,
    storage_gb: int = 50,
    egress_gb: float = 0.0,
    spot: bool = True,
) -> dict[str, Any]:
    provider = _require_provider(provider_id)
    sku = _require_gpu_sku(gpu_sku)
    safe_hours = max(0.25, min(float(duration_hours), 72.0))
    safe_storage = max(10, min(int(storage_gb), 2000))
    safe_egress = max(0.0, float(egress_gb))

    hourly_map = dict(sku.get("hourly_usd") or {})
    base_hourly = float(hourly_map.get(provider["provider_id"], 0.0))
    if base_hourly <= 0:
        raise ValueError(
            f"No pricing configured for provider={provider['provider_id']} gpu_sku={gpu_sku}."
        )

    spot_requested = bool(spot)
    spot_supported = bool(provider.get("supports_spot"))
    spot_effective = spot_requested and spot_supported
    effective_hourly = base_hourly * (0.7 if spot_effective else 1.0)

    compute_cost = effective_hourly * safe_hours
    storage_cost = (safe_storage * 0.00015) * safe_hours
    egress_cost = safe_egress * 0.09
    total = compute_cost + storage_cost + egress_cost

    warnings: list[str] = []
    if spot_requested and not spot_supported:
        warnings.append(f"{provider['display_name']} does not support spot in this planner.")
    if safe_hours > 24:
        warnings.append("Long-running lease requested (>24h). Consider checkpoints and preemption strategy.")

    return {
        "generated_at": _utcnow_iso(),
        "provider_id": provider["provider_id"],
        "provider_name": provider["display_name"],
        "gpu_sku": _normalize_token(gpu_sku),
        "gpu_display_name": sku.get("display_name"),
        "duration_hours": round(safe_hours, 2),
        "spot_requested": spot_requested,
        "spot_effective": spot_effective,
        "effective_hourly_usd": round(effective_hourly, 4),
        "cost_breakdown_usd": {
            "compute": round(compute_cost, 4),
            "storage": round(storage_cost, 4),
            "egress": round(egress_cost, 4),
            "total": round(total, 4),
        },
        "warnings": warnings,
    }


async def _credential_status(
    db: AsyncSession,
    project_id: int,
    *,
    provider_id: str,
) -> dict[str, Any]:
    provider = _require_provider(provider_id)
    credential_provider = str(provider.get("credential_provider") or provider["provider_id"])
    required_keys = [str(item) for item in list(provider.get("credential_keys") or []) if str(item).strip()]

    present: list[str] = []
    missing: list[str] = []
    masked: dict[str, str] = {}
    for key in required_keys:
        value = await get_project_secret_value(
            db,
            project_id,
            credential_provider,
            key,
            touch=False,
        )
        if value:
            present.append(key)
            masked[key] = f"{value[:2]}***{value[-2:]}" if len(value) >= 4 else "***"
        else:
            missing.append(key)
    return {
        "credential_provider": credential_provider,
        "required_keys": required_keys,
        "present_keys": present,
        "missing_keys": missing,
        "ready": len(missing) == 0,
        "masked_hints": masked,
    }


async def _credential_values(
    db: AsyncSession,
    project_id: int,
    *,
    provider_id: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    provider = _require_provider(provider_id)
    credential_provider = str(provider.get("credential_provider") or provider["provider_id"])
    required_keys = [
        str(item).strip()
        for item in list(provider.get("credential_keys") or [])
        if str(item).strip()
    ]
    values: dict[str, str] = {}
    status = await _credential_status(db, project_id, provider_id=provider_id)
    for key in required_keys:
        value = await get_project_secret_value(
            db,
            project_id,
            credential_provider,
            key,
            touch=True,
        )
        if isinstance(value, str) and value.strip():
            values[key] = value.strip()
    return values, status


def _normalize_execution_mode(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"auto", "live", "simulate"}:
        return token
    return "auto"


def _normalize_idempotency_key(value: str | None) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    if len(token) <= 160:
        return token
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{token[:80]}:{digest[:16]}"


def _provider_supports_live(provider_id: str) -> bool:
    provider = _require_provider(provider_id)
    return bool(provider.get("supports_live_execution", False))


def _provider_supports_cancel(provider_id: str) -> bool:
    provider = _require_provider(provider_id)
    return bool(provider.get("supports_managed_cancel", False))


def _provider_supports_logs(provider_id: str) -> bool:
    provider = _require_provider(provider_id)
    return bool(provider.get("supports_live_logs", False))


def _find_run_by_idempotency_key(
    *,
    project_id: int,
    idempotency_key: str,
) -> dict[str, Any] | None:
    token = str(idempotency_key or "").strip()
    if not token:
        return None
    runs_dir = _project_cloud_runs_dir(project_id)
    matches: list[dict[str, Any]] = []
    for path in runs_dir.glob("*.json"):
        payload = _read_json_payload(path)
        if not payload:
            continue
        if str(payload.get("idempotency_key") or "").strip() != token:
            continue
        payload.setdefault("run_id", path.stem)
        matches.append(payload)
    if not matches:
        return None
    matches.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return matches[0]


def _provider_request_template(
    *,
    provider_id: str,
    gpu_sku: str,
    region: str | None,
    image: str,
    startup_script: str,
    duration_hours: float,
    spot: bool,
) -> dict[str, Any]:
    token = _normalize_token(provider_id)
    safe_region = str(region or "").strip()
    safe_image = str(image or "").strip() or "ghcr.io/slm/platform-trainer:latest"
    safe_script = str(startup_script or "").strip() or "bash /workspace/entrypoint.sh"

    if token == "runpod":
        return {
            "endpoint": "https://api.runpod.io/graphql",
            "method": "POST",
            "body": {
                "query": (
                    "mutation { podFindAndDeployOnDemand("
                    f"gpuTypeId:\"{gpu_sku}\",name:\"slm-train\",containerDiskInGb:50,"
                    f"dockerArgs:\"{safe_script}\",imageName:\"{safe_image}\",supportPublicIp:true) {{ id }}"
                    " }"
                ),
            },
            "headers": {"Authorization": "Bearer <RUNPOD_API_KEY>"},
        }

    if token == "lambda_labs":
        return {
            "endpoint": "https://cloud.lambda.ai/api/v1/instances",
            "method": "POST",
            "body": {
                "instance_type_name": gpu_sku,
                "region_name": safe_region or "US-WEST",
                "ssh_key_names": ["<your-ssh-key>"],
                "name": "slm-train",
                "file_system_names": ["<optional-volume>"],
            },
            "headers": {"Authorization": "Bearer <LAMBDA_API_KEY>"},
        }

    return {
        "endpoint": "aws sagemaker create-training-job (CLI/SDK)",
        "method": "AWS_SIGV4",
        "body": {
            "TrainingJobName": "slm-train",
            "RoleArn": "<sagemaker-execution-role-arn>",
            "AlgorithmSpecification": {"TrainingImage": safe_image, "TrainingInputMode": "File"},
            "ResourceConfig": {
                "InstanceType": gpu_sku,
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": int(duration_hours * 3600)},
            "EnableManagedSpotTraining": bool(spot),
            "Region": safe_region or "us-east-1",
        },
        "headers": {"X-Aws-Auth": "AWS4-HMAC-SHA256 <derived-signature>"},
    }


def _backoff_delay(attempt: int) -> float:
    idx = max(0, min(int(attempt), len(_RETRY_BACKOFF_SECONDS) - 1))
    return float(_RETRY_BACKOFF_SECONDS[idx])


async def _runpod_graphql_request(
    *,
    api_key: str,
    query: str,
    variables: dict[str, Any] | None = None,
    timeout_seconds: float = _DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {str(api_key or '').strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": str(query or "").strip(),
        "variables": dict(variables or {}),
    }
    if not payload["query"]:
        raise ValueError("RunPod GraphQL query is required")

    try:
        async with httpx.AsyncClient(timeout=max(1.0, float(timeout_seconds))) as client:
            response = await client.post(_RUNPOD_GRAPHQL_ENDPOINT, json=payload, headers=headers)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"RunPod request failed: {exc}") from exc

    if int(response.status_code) >= 400:
        preview = response.text[:400]
        raise ValueError(f"RunPod API returned HTTP {response.status_code}: {preview}")

    try:
        body = response.json()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"RunPod API returned invalid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError("RunPod API returned non-object JSON payload")

    errors = list(body.get("errors") or [])
    if errors:
        head = errors[0] if isinstance(errors[0], dict) else {"message": str(errors[0])}
        message = str(head.get("message") or errors[0]).strip()
        raise ValueError(f"RunPod GraphQL error: {message}")

    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("RunPod GraphQL response is missing data object")
    return data


async def _retryable_provider_call(
    *,
    op_name: str,
    call: Callable[[], Awaitable[dict[str, Any]]],
    retries: int = _MAX_PROVIDER_RETRIES,
) -> dict[str, Any]:
    last_error: Exception | None = None
    safe_retries = max(1, int(retries))
    for attempt in range(1, safe_retries + 1):
        try:
            payload = await call()
            if isinstance(payload, dict):
                return payload
            raise ValueError(f"{op_name} returned invalid payload shape")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= safe_retries:
                break
            await asyncio.sleep(_backoff_delay(attempt - 1))
    if last_error is None:
        raise ValueError(f"{op_name} failed")
    raise ValueError(f"{op_name} failed after {safe_retries} attempts: {last_error}") from last_error


async def _runpod_submit_job(
    *,
    api_key: str,
    gpu_sku: str,
    image: str,
    startup_script: str,
    run_name: str,
) -> dict[str, Any]:
    mutation = """
mutation DeployPod($gpuTypeId: String!, $name: String!, $imageName: String!, $dockerArgs: String!, $containerDiskInGb: Int!, $supportPublicIp: Boolean!) {
  podFindAndDeployOnDemand(
    gpuTypeId: $gpuTypeId,
    name: $name,
    imageName: $imageName,
    dockerArgs: $dockerArgs,
    containerDiskInGb: $containerDiskInGb,
    supportPublicIp: $supportPublicIp
  ) {
    id
    desiredStatus
  }
}
"""

    variables = {
        "gpuTypeId": str(gpu_sku or "").strip(),
        "name": str(run_name or "").strip() or "slm-cloud-burst",
        "imageName": str(image or "").strip() or "ghcr.io/slm/platform-trainer:latest",
        "dockerArgs": str(startup_script or "").strip() or "bash /workspace/entrypoint.sh",
        "containerDiskInGb": 50,
        "supportPublicIp": True,
    }
    data = await _runpod_graphql_request(
        api_key=api_key,
        query=mutation,
        variables=variables,
    )
    pod_payload = data.get("podFindAndDeployOnDemand")
    if not isinstance(pod_payload, dict):
        raise ValueError("RunPod submit response is missing pod deployment payload")
    pod_id = str(pod_payload.get("id") or "").strip()
    if not pod_id:
        raise ValueError("RunPod submit response missing pod id")
    return {
        "provider_job_id": pod_id,
        "provider_status_raw": str(pod_payload.get("desiredStatus") or "").strip(),
        "provider_payload": pod_payload,
    }


async def _runpod_get_job_status(
    *,
    api_key: str,
    provider_job_id: str,
) -> dict[str, Any]:
    query = """
query PodStatus($podId: String!) {
  pod(input: { podId: $podId }) {
    id
    desiredStatus
    lastStatusChange
    runtime {
      uptimeInSeconds
    }
  }
}
"""
    data = await _runpod_graphql_request(
        api_key=api_key,
        query=query,
        variables={"podId": str(provider_job_id or "").strip()},
    )
    pod_payload = data.get("pod")
    if not isinstance(pod_payload, dict):
        raise ValueError("RunPod status response missing pod payload")
    status_raw = str(
        pod_payload.get("desiredStatus")
        or pod_payload.get("status")
        or ""
    ).strip()
    runtime_payload = pod_payload.get("runtime")
    uptime_seconds: int | None = None
    if isinstance(runtime_payload, dict):
        try:
            uptime_value = runtime_payload.get("uptimeInSeconds")
            if uptime_value is not None:
                uptime_seconds = max(0, int(uptime_value))
        except Exception:
            uptime_seconds = None
    return {
        "provider_job_id": str(pod_payload.get("id") or provider_job_id).strip(),
        "provider_status_raw": status_raw,
        "provider_uptime_seconds": uptime_seconds,
        "provider_payload": pod_payload,
    }


async def _runpod_get_job_logs(
    *,
    api_key: str,
    provider_job_id: str,
    limit: int = _RUNPOD_LOG_FETCH_LIMIT,
) -> dict[str, Any]:
    safe_limit = max(10, min(int(limit or _RUNPOD_LOG_FETCH_LIMIT), 500))
    query_candidates = [
        """
query PodRuntimeLogs($podId: String!, $limit: Int!) {
  pod(input: { podId: $podId }) {
    id
    runtime {
      logs(limit: $limit)
    }
  }
}
""",
        """
query PodRuntimeLogStream($podId: String!, $limit: Int!) {
  pod(input: { podId: $podId }) {
    id
    runtime {
      logStream(limit: $limit)
    }
  }
}
""",
    ]
    last_error: Exception | None = None
    for query in query_candidates:
        try:
            data = await _runpod_graphql_request(
                api_key=api_key,
                query=query,
                variables={
                    "podId": str(provider_job_id or "").strip(),
                    "limit": safe_limit,
                },
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        pod_payload = data.get("pod")
        if not isinstance(pod_payload, dict):
            continue
        runtime_payload = pod_payload.get("runtime")
        if not isinstance(runtime_payload, dict):
            continue

        raw_stream = runtime_payload.get("logs")
        if raw_stream is None:
            raw_stream = runtime_payload.get("logStream")

        lines: list[str] = []
        if isinstance(raw_stream, str):
            lines = [item.strip() for item in raw_stream.splitlines() if str(item or "").strip()]
        elif isinstance(raw_stream, list):
            for item in raw_stream:
                if isinstance(item, dict):
                    line = str(
                        item.get("line")
                        or item.get("message")
                        or item.get("text")
                        or ""
                    ).strip()
                else:
                    line = str(item or "").strip()
                if line:
                    lines.append(line)
        return {
            "provider_job_id": str(pod_payload.get("id") or provider_job_id).strip(),
            "logs": lines[-safe_limit:],
            "provider_payload": pod_payload,
        }

    if last_error is not None:
        raise ValueError(f"RunPod logs query failed: {last_error}") from last_error
    return {
        "provider_job_id": str(provider_job_id or "").strip(),
        "logs": [],
        "provider_payload": {},
    }


async def _runpod_cancel_job(
    *,
    api_key: str,
    provider_job_id: str,
) -> dict[str, Any]:
    mutation_candidates = [
        """
mutation StopPod($podId: String!) {
  podStop(input: { podId: $podId }) {
    id
    desiredStatus
  }
}
""",
        """
mutation TerminatePod($podId: String!) {
  podTerminate(input: { podId: $podId }) {
    id
    desiredStatus
  }
}
""",
    ]
    last_error: Exception | None = None
    for mutation in mutation_candidates:
        try:
            data = await _runpod_graphql_request(
                api_key=api_key,
                query=mutation,
                variables={"podId": str(provider_job_id or "").strip()},
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        payload = (
            data.get("podStop")
            if isinstance(data.get("podStop"), dict)
            else data.get("podTerminate")
        )
        if not isinstance(payload, dict):
            continue
        return {
            "provider_job_id": str(payload.get("id") or provider_job_id).strip(),
            "provider_status_raw": str(payload.get("desiredStatus") or "").strip(),
            "provider_payload": payload,
        }
    if last_error is not None:
        raise ValueError(f"RunPod cancel failed: {last_error}") from last_error
    raise ValueError("RunPod cancel failed: no compatible mutation response")


async def _submit_provider_job(
    *,
    provider_id: str,
    api_key: str,
    gpu_sku: str,
    image: str,
    startup_script: str,
    run_name: str,
) -> dict[str, Any]:
    provider = _normalize_token(provider_id)
    if provider == "runpod":
        return await _retryable_provider_call(
            op_name="runpod.submit",
            call=lambda: _runpod_submit_job(
                api_key=api_key,
                gpu_sku=gpu_sku,
                image=image,
                startup_script=startup_script,
                run_name=run_name,
            ),
        )
    raise ValueError(f"Provider {provider_id} does not support live submit yet")


async def _fetch_provider_job_status(
    *,
    provider_id: str,
    api_key: str,
    provider_job_id: str,
) -> dict[str, Any]:
    provider = _normalize_token(provider_id)
    if provider == "runpod":
        return await _retryable_provider_call(
            op_name="runpod.status",
            call=lambda: _runpod_get_job_status(
                api_key=api_key,
                provider_job_id=provider_job_id,
            ),
        )
    raise ValueError(f"Provider {provider_id} does not support live status yet")


async def _cancel_provider_job(
    *,
    provider_id: str,
    api_key: str,
    provider_job_id: str,
) -> dict[str, Any]:
    provider = _normalize_token(provider_id)
    if provider == "runpod":
        return await _retryable_provider_call(
            op_name="runpod.cancel",
            call=lambda: _runpod_cancel_job(
                api_key=api_key,
                provider_job_id=provider_job_id,
            ),
        )
    raise ValueError(f"Provider {provider_id} does not support managed cancel yet")


async def _fetch_provider_job_logs(
    *,
    provider_id: str,
    api_key: str,
    provider_job_id: str,
    limit: int = _RUNPOD_LOG_FETCH_LIMIT,
) -> dict[str, Any]:
    provider = _normalize_token(provider_id)
    if provider == "runpod":
        try:
            return await _runpod_get_job_logs(
                api_key=api_key,
                provider_job_id=provider_job_id,
                limit=limit,
            )
        except Exception:
            return {
                "provider_job_id": provider_job_id,
                "logs": [],
                "provider_payload": {},
            }
    return {
        "provider_job_id": provider_job_id,
        "logs": [],
        "provider_payload": {},
    }


def _map_provider_status_to_run_status(provider_status_raw: str) -> str:
    token = str(provider_status_raw or "").strip().lower()
    if token in {"pending", "created", "starting", "initializing", "queued"}:
        return "provisioning"
    if token in {"running", "ready"}:
        return "running"
    if token in {"stopping", "terminating", "cancel_requested"}:
        return "cancel_requested"
    if token in {"stopped", "terminated", "cancelled"}:
        return "cancelled"
    if token in {"failed", "error", "unhealthy"}:
        return "failed"
    if token in {"completed", "exited", "finished"}:
        return "completed"
    return "running"


async def build_cloud_burst_launch_plan(
    db: AsyncSession,
    *,
    project_id: int,
    provider_id: str,
    gpu_sku: str,
    duration_hours: float = 2.0,
    experiment_id: int | None = None,
    region: str | None = None,
    image: str = "",
    startup_script: str = "",
    spot: bool = True,
) -> dict[str, Any]:
    provider = _require_provider(provider_id)
    quote = estimate_cloud_burst_quote(
        provider_id=provider["provider_id"],
        gpu_sku=gpu_sku,
        duration_hours=duration_hours,
        spot=spot,
    )
    credentials = await _credential_status(
        db,
        project_id,
        provider_id=provider["provider_id"],
    )
    request_template = _provider_request_template(
        provider_id=provider["provider_id"],
        gpu_sku=gpu_sku,
        region=region,
        image=image,
        startup_script=startup_script,
        duration_hours=quote["duration_hours"],
        spot=bool(quote.get("spot_effective")),
    )

    prep_dir = _project_cloud_dir(project_id)
    launch_id = f"{provider['provider_id']}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    record = {
        "launch_id": launch_id,
        "project_id": project_id,
        "experiment_id": experiment_id,
        "provider_id": provider["provider_id"],
        "gpu_sku": _normalize_token(gpu_sku),
        "created_at": _utcnow_iso(),
        "quote": quote,
        "credentials": credentials,
        "request_template": request_template,
        "bootstrap_steps": [
            "Provision the GPU lease with provider API request template.",
            "Attach project artifact storage and sync training datasets/config.",
            "Install runtime deps (torch/transformers/datasets/accelerate/peft/trl as needed).",
            "Run training command and stream logs/metrics back to SLM platform.",
        ],
        "suggested_training_command_template": settings.TRAINING_EXTERNAL_CMD,
    }
    record_path = prep_dir / f"{launch_id}.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    record["record_path"] = str(record_path)
    return record


async def _resolve_experiment_output_dir(
    db: AsyncSession,
    *,
    project_id: int,
    experiment_id: int | None,
) -> str | None:
    if not experiment_id:
        return None
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == int(experiment_id),
            Experiment.project_id == int(project_id),
        )
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        return None
    output_dir_token = str(exp.output_dir or "").strip()
    if not output_dir_token:
        return None
    try:
        output_dir = Path(output_dir_token).expanduser().resolve()
    except Exception:
        return None
    return str(output_dir)


def _matches_any_pattern(path_token: str, name_token: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    for pattern in patterns:
        candidate = str(pattern or "").strip()
        if not candidate:
            continue
        if fnmatch.fnmatch(path_token, candidate) or fnmatch.fnmatch(name_token, candidate):
            return True
    return False


def _effective_sync_include_globs(policy: str, override: list[str] | None) -> list[str]:
    normalized_override = _normalize_glob_list(override)
    if normalized_override:
        return normalized_override
    if policy == "all":
        return ["**"]
    return list(_DEFAULT_SYNC_INCLUDE_GLOBS_SMART)


def _effective_sync_exclude_globs(override: list[str] | None) -> list[str]:
    normalized_override = _normalize_glob_list(override)
    if normalized_override:
        return normalized_override
    return list(_DEFAULT_SYNC_EXCLUDE_GLOBS)


def _sync_manifest_path(target_dir: Path) -> Path:
    return target_dir / _SYNC_MANIFEST_FILE


def _load_sync_manifest(target_dir: Path) -> dict[str, Any]:
    payload = _read_json_payload(_sync_manifest_path(target_dir))
    files = payload.get("files")
    if not isinstance(files, dict):
        payload["files"] = {}
    return payload


def _write_sync_manifest(target_dir: Path, payload: dict[str, Any]) -> None:
    _write_json_payload(_sync_manifest_path(target_dir), payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sync_cloud_burst_job_artifacts(
    *,
    project_id: int,
    run_id: str,
    policy: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    dry_run: bool = False,
    max_files: int = 2000,
    cursor: str | None = None,
    fail_if_missing_source: bool = True,
    reason: str = "manual",
) -> dict[str, Any]:
    run = _load_run_record(project_id, run_id)
    artifacts = dict(run.get("artifacts") or {})

    source_dir_token = str(artifacts.get("source_dir") or "").strip()
    source_dir = Path(source_dir_token).expanduser() if source_dir_token else None
    if source_dir is None or not source_dir.exists():
        message = (
            "No source artifact directory is available for this run. "
            "Provide experiment_id during submit once training artifacts exist."
        )
        if fail_if_missing_source:
            raise ValueError(message)
        summary = {
            "status": "skipped",
            "reason": message,
            "synced_at": _utcnow_iso(),
            "policy": _normalize_sync_policy(policy or artifacts.get("policy")),
            "file_count": 0,
            "copied_count": 0,
            "unchanged_count": 0,
            "would_copy_count": 0,
            "total_bytes": 0,
            "cursor": str(cursor or "").strip() or None,
            "next_cursor": None,
            "sampled_files": [],
        }
        artifacts["last_sync_at"] = summary["synced_at"]
        artifacts["last_sync_summary"] = summary
        history = list(artifacts.get("sync_history") or [])
        history.append(summary)
        artifacts["sync_history"] = history[-_MAX_SYNC_HISTORY_ENTRIES:]
        run["artifacts"] = artifacts
        _persist_run_record(project_id, run_id, run)
        _append_run_log(project_id, run_id, f"artifact_sync skipped ({reason}): {message}")
        _append_audit_event(
            project_id,
            {
                "event": "artifact_sync",
                "run_id": run_id,
                "status": "skipped",
                "reason": str(reason or "").strip() or "manual",
                "message": message,
            },
        )
        return {
            "run_id": run_id,
            "project_id": project_id,
            "status": run.get("status"),
            "artifacts": artifacts,
            "sync": summary,
        }

    sync_policy = _normalize_sync_policy(policy or artifacts.get("policy"))
    include_patterns = _effective_sync_include_globs(sync_policy, include_globs or artifacts.get("include_globs"))
    exclude_patterns = _effective_sync_exclude_globs(exclude_globs or artifacts.get("exclude_globs"))
    safe_max_files = max(1, min(int(max_files or 2000), 10000))
    cursor_token = str(cursor or "").strip()

    source_root = source_dir.resolve()
    all_rel_paths: list[str] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        rel_posix = path.relative_to(source_root).as_posix()
        file_name = path.name
        include_ok = (
            include_patterns == ["**"]
            or _matches_any_pattern(rel_posix, file_name, include_patterns)
        )
        if not include_ok:
            continue
        if _matches_any_pattern(rel_posix, file_name, exclude_patterns):
            continue
        all_rel_paths.append(rel_posix)

    all_rel_paths.sort()
    if cursor_token:
        candidate_rel_paths = [item for item in all_rel_paths if item > cursor_token]
    else:
        candidate_rel_paths = list(all_rel_paths)
    limited = len(candidate_rel_paths) > safe_max_files
    selected_rel_paths = candidate_rel_paths[:safe_max_files]
    next_cursor = selected_rel_paths[-1] if limited and selected_rel_paths else None

    target_dir = _cloud_run_artifacts_dir(project_id, run_id)
    manifest_path = _sync_manifest_path(target_dir)
    manifest_payload = _load_sync_manifest(target_dir)
    manifest_files = dict(manifest_payload.get("files") or {})

    copied_files: list[str] = []
    unchanged_files: list[str] = []
    copied_count = 0
    unchanged_count = 0
    would_copy_count = 0
    deleted_count = 0
    total_bytes = 0
    errors: list[str] = []
    updated_manifest_files = dict(manifest_files)
    synced_at = _utcnow_iso()
    for rel_posix in selected_rel_paths:
        path = source_root / rel_posix
        rel_path = Path(rel_posix)
        try:
            stat_info = path.stat()
        except Exception:
            errors.append(f"{rel_posix}: source stat unavailable")
            pass
            continue

        size = int(stat_info.st_size)
        mtime_ns = int(getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000)))
        total_bytes += size
        quick_sig = f"{size}:{mtime_ns}"

        destination = target_dir / rel_path
        previous = dict(manifest_files.get(rel_posix) or {})
        previous_quick_sig = str(previous.get("quick_sig") or "")
        previous_sha = str(previous.get("sha256") or "")
        destination_exists = destination.exists()

        needs_copy = True
        sha256 = previous_sha if previous_quick_sig == quick_sig else ""
        if destination_exists and previous_quick_sig == quick_sig and previous_sha:
            try:
                if int(destination.stat().st_size) == size:
                    needs_copy = False
            except Exception:
                needs_copy = True

        if needs_copy:
            try:
                if not sha256:
                    sha256 = _file_sha256(path)
            except Exception as exc:
                errors.append(f"{rel_posix}: checksum failed ({exc})")
                continue
            if destination_exists and previous_sha and previous_sha == sha256:
                needs_copy = False

        if needs_copy:
            if dry_run:
                would_copy_count += 1
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(path, destination)
                    copied_count += 1
                    if len(copied_files) < 50:
                        copied_files.append(rel_posix)
                except Exception as exc:
                    errors.append(f"{rel_posix}: {exc}")
                    continue
        else:
            unchanged_count += 1
            if len(unchanged_files) < 50:
                unchanged_files.append(rel_posix)

        updated_manifest_files[rel_posix] = {
            "path": rel_posix,
            "size": size,
            "mtime_ns": mtime_ns,
            "quick_sig": quick_sig,
            "sha256": sha256,
            "synced_at": synced_at,
        }

    status = "dry_run" if dry_run else "copied"
    if errors:
        status = "partial_error"
    elif not dry_run and copied_count == 0:
        status = "up_to_date"

    full_scan_completed = (not cursor_token) and (not limited)
    if not dry_run and full_scan_completed:
        source_paths = set(all_rel_paths)
        for key in list(updated_manifest_files.keys()):
            if key not in source_paths:
                deleted_count += 1
                updated_manifest_files.pop(key, None)
                destination = target_dir / Path(key)
                if destination.exists():
                    try:
                        destination.unlink()
                    except Exception:
                        pass

    manifest_updated = False
    if not dry_run:
        manifest_payload["schema"] = "slm.cloud_burst.sync_manifest/v2"
        manifest_payload["project_id"] = int(project_id)
        manifest_payload["run_id"] = str(run_id)
        manifest_payload["updated_at"] = synced_at
        manifest_payload["file_count"] = len(updated_manifest_files)
        manifest_payload["files"] = updated_manifest_files
        _write_sync_manifest(target_dir, manifest_payload)
        manifest_updated = True

    summary = {
        "status": status,
        "reason": str(reason or "").strip() or "manual",
        "synced_at": synced_at,
        "policy": sync_policy,
        "source_dir": str(source_root),
        "target_dir": str(target_dir),
        "manifest_path": str(manifest_path),
        "manifest_updated": manifest_updated,
        "include_globs": include_patterns,
        "exclude_globs": exclude_patterns,
        "file_count": len(selected_rel_paths),
        "candidate_count": len(candidate_rel_paths),
        "copied_count": copied_count,
        "unchanged_count": unchanged_count,
        "would_copy_count": would_copy_count,
        "deleted_count": deleted_count,
        "total_bytes": int(total_bytes),
        "sampled_files": copied_files,
        "sampled_unchanged_files": unchanged_files,
        "errors": errors[:20],
        "limited": limited,
        "cursor": cursor_token or None,
        "next_cursor": next_cursor,
        "remaining_count": max(0, len(candidate_rel_paths) - len(selected_rel_paths)),
        "dry_run": bool(dry_run),
    }

    artifacts["policy"] = sync_policy
    artifacts["include_globs"] = include_patterns
    artifacts["exclude_globs"] = exclude_patterns
    artifacts["last_sync_at"] = summary["synced_at"]
    artifacts["last_sync_summary"] = summary
    history = list(artifacts.get("sync_history") or [])
    history.append(summary)
    artifacts["sync_history"] = history[-_MAX_SYNC_HISTORY_ENTRIES:]
    run["artifacts"] = artifacts
    _persist_run_record(project_id, run_id, run)
    _append_run_log(
        project_id,
        run_id,
        (
            "artifact_sync "
            f"{status} ({reason}) copied={copied_count} unchanged={unchanged_count} "
            f"would_copy={would_copy_count} deleted={deleted_count} limited={limited}"
        ),
    )
    _append_audit_event(
        project_id,
        {
            "event": "artifact_sync",
            "run_id": run_id,
            "status": status,
            "reason": str(reason or "").strip() or "manual",
            "copied_count": copied_count,
            "unchanged_count": unchanged_count,
            "would_copy_count": would_copy_count,
            "deleted_count": deleted_count,
            "limited": limited,
            "cursor": cursor_token or None,
            "next_cursor": next_cursor,
            "dry_run": bool(dry_run),
        },
    )

    return {
        "run_id": run_id,
        "project_id": project_id,
        "status": run.get("status"),
        "artifacts": artifacts,
        "sync": summary,
    }


def list_cloud_burst_jobs(
    *,
    project_id: int,
    limit: int = 20,
) -> dict[str, Any]:
    runs_dir = _project_cloud_runs_dir(project_id)
    items: list[dict[str, Any]] = []
    for path in runs_dir.glob("*.json"):
        payload = _read_json_payload(path)
        if not payload:
            continue
        payload.setdefault("run_id", path.stem)
        payload.setdefault("project_id", int(project_id))
        payload.setdefault("record_path", str(path))
        payload.setdefault("logs_path", str(_cloud_run_logs_path(project_id, path.stem)))
        status = str(payload.get("status") or "unknown").strip().lower()
        payload["status"] = status or "unknown"
        payload["can_cancel"] = payload["status"] in _CLOUD_BURST_ACTIVE_STATES
        payload.pop("metrics_hashes", None)
        bridge_payload = payload.get("log_bridge")
        if isinstance(bridge_payload, dict):
            payload["log_bridge"] = {
                **{k: v for k, v in bridge_payload.items() if k != "seen_hashes"},
                "seen_hash_count": len(list(bridge_payload.get("seen_hashes") or [])),
            }
        items.append(payload)
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)

    safe_limit = max(1, min(int(limit or 20), 200))
    listed = items[:safe_limit]
    return {
        "project_id": project_id,
        "count": len(items),
        "limit": safe_limit,
        "runs": listed,
    }


def get_cloud_burst_job_status(
    *,
    project_id: int,
    run_id: str,
    logs_tail: int = 200,
) -> dict[str, Any]:
    run = _load_run_record(project_id, run_id)
    return _serialize_run(run, logs_tail=logs_tail)


def get_cloud_burst_job_logs(
    *,
    project_id: int,
    run_id: str,
    tail: int = 200,
) -> dict[str, Any]:
    run = _load_run_record(project_id, run_id)
    safe_tail = max(1, min(int(tail or 200), 5000))
    return {
        "project_id": project_id,
        "run_id": str(run.get("run_id") or run_id),
        "status": str(run.get("status") or "unknown"),
        "provider_status_raw": str(run.get("provider_status_raw") or ""),
        "tail": safe_tail,
        "logs": _read_run_logs_tail(project_id, run_id, tail=safe_tail),
        "metrics_tail": list(run.get("metrics_tail") or [])[-200:],
        "metrics_tail_count": len(list(run.get("metrics_tail") or [])),
    }


async def _set_active_task(
    run_id: str,
    task_state: _ManagedCloudBurstTask,
) -> None:
    async with _MANAGED_CLOUD_BURST_TASK_LOCK:
        _MANAGED_CLOUD_BURST_TASKS[str(run_id)] = task_state


async def _get_active_task(run_id: str) -> _ManagedCloudBurstTask | None:
    async with _MANAGED_CLOUD_BURST_TASK_LOCK:
        return _MANAGED_CLOUD_BURST_TASKS.get(str(run_id))


async def _clear_active_task(run_id: str) -> None:
    async with _MANAGED_CLOUD_BURST_TASK_LOCK:
        _MANAGED_CLOUD_BURST_TASKS.pop(str(run_id), None)


async def _sleep_with_cancel(cancel_event: asyncio.Event, seconds: float) -> bool:
    remaining = max(0.0, float(seconds))
    while remaining > 0.0:
        if cancel_event.is_set():
            return False
        step = min(0.1, remaining)
        await asyncio.sleep(step)
        remaining -= step
    return not cancel_event.is_set()


def _mark_cancel_requested(project_id: int, run_id: str) -> dict[str, Any]:
    run = _load_run_record(project_id, run_id)
    run["cancel_requested"] = True
    status = str(run.get("status") or "").strip().lower()
    if status not in _TERMINAL_CLOUD_BURST_JOB_STATES and status != "cancel_requested":
        run["status"] = "cancel_requested"
        run["status_reason"] = "Cancellation requested by user."
        _run_status_timeline_append(
            run,
            status="cancel_requested",
            reason="Cancellation requested by user.",
        )
    persisted = _persist_run_record(project_id, run_id, run)
    _append_run_log(project_id, run_id, "cancel_requested=true")
    _append_audit_event(
        project_id,
        {
            "event": "cancel_requested",
            "run_id": run_id,
            "status": str(persisted.get("status") or "").strip().lower() or "unknown",
        },
    )
    return persisted


def _sync_artifacts_on_completion(
    *,
    project_id: int,
    run_id: str,
    auto_artifact_sync: bool,
    reason: str,
) -> None:
    if not auto_artifact_sync:
        return
    sync_result = sync_cloud_burst_job_artifacts(
        project_id=project_id,
        run_id=run_id,
        fail_if_missing_source=False,
        reason=reason,
    )
    sync_state = str(dict(sync_result.get("sync") or {}).get("status") or "unknown")
    _append_run_log(project_id, run_id, f"auto artifact sync status={sync_state}")


async def _run_managed_cloud_burst_job_simulated(
    *,
    project_id: int,
    run_id: str,
    cancel_event: asyncio.Event,
    auto_artifact_sync: bool,
) -> None:
    _mark_run_status(
        project_id,
        run_id,
        status="provisioning",
        reason="Provisioning remote cloud burst lease (simulated).",
    )
    if not await _sleep_with_cancel(cancel_event, 0.45):
        _mark_run_status(project_id, run_id, status="cancelled", reason="Cancelled during provisioning.")
        return

    _mark_run_status(
        project_id,
        run_id,
        status="syncing",
        reason="Syncing configs/datasets to remote workspace (simulated).",
    )
    if not await _sleep_with_cancel(cancel_event, 0.45):
        _mark_run_status(project_id, run_id, status="cancelled", reason="Cancelled during sync stage.")
        return

    _mark_run_status(
        project_id,
        run_id,
        status="running",
        reason="Managed cloud burst job is running (simulated).",
    )
    for step in range(1, 7):
        if not await _sleep_with_cancel(cancel_event, 0.35):
            _mark_run_status(project_id, run_id, status="cancelled", reason="Cancelled while running.")
            return
        metric_line = (
            f"{_METRIC_LINE_PREFIX}"
            + json.dumps(
                {
                    "step": step,
                    "epoch": round(step / 2.0, 2),
                    "train_loss": round(max(0.05, 1.0 - (step * 0.12)), 4),
                    "eval_loss": round(max(0.05, 1.12 - (step * 0.11)), 4),
                },
                ensure_ascii=False,
            )
        )
        run = _load_run_record(project_id, run_id)
        _ingest_provider_log_lines(
            project_id,
            run_id,
            run,
            lines=[f"heartbeat step={step}/6", metric_line],
            source="simulated",
        )
        _persist_run_record(project_id, run_id, run)

    _sync_artifacts_on_completion(
        project_id=project_id,
        run_id=run_id,
        auto_artifact_sync=auto_artifact_sync,
        reason="auto_on_completion",
    )
    _mark_run_status(
        project_id,
        run_id,
        status="completed",
        reason="Managed cloud burst lifecycle completed (simulated).",
    )


async def _run_managed_cloud_burst_job_live(
    *,
    project_id: int,
    run_id: str,
    cancel_event: asyncio.Event,
    provider_id: str,
    provider_job_id: str,
    provider_api_key: str,
    auto_artifact_sync: bool,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> None:
    _mark_run_status(
        project_id,
        run_id,
        status="provisioning",
        reason=f"Provisioning provider job on {provider_id}.",
    )
    started = datetime.now(timezone.utc)
    duration_hours = float(
        dict(_load_run_record(project_id, run_id).get("runtime") or {}).get("duration_hours") or 2.0
    )
    timeout_seconds = max(600.0, (duration_hours * 3600.0) + 900.0)

    while True:
        if cancel_event.is_set():
            try:
                await _cancel_provider_job(
                    provider_id=provider_id,
                    api_key=provider_api_key,
                    provider_job_id=provider_job_id,
                )
            except Exception as exc:  # noqa: BLE001
                _append_run_log(project_id, run_id, f"provider_cancel_warning: {exc}")
            _mark_run_status(project_id, run_id, status="cancelled", reason="Cancelled by user request.")
            return

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        if elapsed > timeout_seconds:
            _mark_run_status(
                project_id,
                run_id,
                status="failed",
                reason=f"Provider poll timeout exceeded ({int(timeout_seconds)}s).",
            )
            return

        status_payload = await _fetch_provider_job_status(
            provider_id=provider_id,
            api_key=provider_api_key,
            provider_job_id=provider_job_id,
        )
        provider_status_raw = str(status_payload.get("provider_status_raw") or "").strip()
        mapped_status = _map_provider_status_to_run_status(provider_status_raw)

        run = _load_run_record(project_id, run_id)
        run["provider_id"] = _normalize_token(provider_id)
        run["provider_job_id"] = str(status_payload.get("provider_job_id") or provider_job_id).strip()
        run["provider_status_raw"] = provider_status_raw
        run["provider_uptime_seconds"] = status_payload.get("provider_uptime_seconds")
        run["provider_last_status_at"] = _utcnow_iso()
        run["provider_poll_count"] = int(run.get("provider_poll_count") or 0) + 1

        if _provider_supports_logs(provider_id):
            logs_payload = await _fetch_provider_job_logs(
                provider_id=provider_id,
                api_key=provider_api_key,
                provider_job_id=run["provider_job_id"],
                limit=_RUNPOD_LOG_FETCH_LIMIT,
            )
            log_lines = [
                str(item).strip()
                for item in list(logs_payload.get("logs") or [])
                if str(item or "").strip()
            ]
            _ingest_provider_log_lines(
                project_id,
                run_id,
                run,
                lines=log_lines,
                source=f"{provider_id}:{run['provider_job_id']}",
            )

        _persist_run_record(project_id, run_id, run)
        current_status = str(run.get("status") or "").strip().lower()
        if mapped_status != current_status:
            reason = f"Provider status={provider_status_raw or 'unknown'}."
            _mark_run_status(project_id, run_id, status=mapped_status, reason=reason)

        if mapped_status in _TERMINAL_CLOUD_BURST_JOB_STATES:
            if mapped_status == "completed":
                _sync_artifacts_on_completion(
                    project_id=project_id,
                    run_id=run_id,
                    auto_artifact_sync=auto_artifact_sync,
                    reason="auto_on_completion_live",
                )
            return

        if not await _sleep_with_cancel(cancel_event, max(0.5, float(poll_interval_seconds))):
            continue


async def _run_managed_cloud_burst_job(
    *,
    project_id: int,
    run_id: str,
    cancel_event: asyncio.Event,
    auto_artifact_sync: bool,
    provider_id: str | None = None,
    provider_job_id: str | None = None,
    provider_api_key: str | None = None,
) -> None:
    run = _load_run_record(project_id, run_id)
    mode = _normalize_execution_mode(str(run.get("execution_mode_effective") or "simulate"))
    try:
        if mode == "live":
            if not provider_id or not provider_job_id or not provider_api_key:
                raise ValueError("Live execution requires provider_id, provider_job_id, and provider_api_key.")
            await _run_managed_cloud_burst_job_live(
                project_id=project_id,
                run_id=run_id,
                cancel_event=cancel_event,
                provider_id=provider_id,
                provider_job_id=provider_job_id,
                provider_api_key=provider_api_key,
                auto_artifact_sync=auto_artifact_sync,
            )
        else:
            await _run_managed_cloud_burst_job_simulated(
                project_id=project_id,
                run_id=run_id,
                cancel_event=cancel_event,
                auto_artifact_sync=auto_artifact_sync,
            )
    except Exception as exc:  # noqa: BLE001
        run = _load_run_record(project_id, run_id)
        run["status"] = "failed"
        run["status_reason"] = f"Managed cloud burst worker failed: {exc}"
        run["error"] = str(exc)
        run["finished_at"] = _utcnow_iso()
        _run_status_timeline_append(
            run,
            status="failed",
            reason=f"Managed cloud burst worker failed: {exc}",
        )
        _persist_run_record(project_id, run_id, run)
        _append_run_log(project_id, run_id, f"worker_error: {exc}")
    finally:
        await _clear_active_task(run_id)


async def submit_cloud_burst_job(
    db: AsyncSession,
    *,
    project_id: int,
    provider_id: str,
    gpu_sku: str,
    duration_hours: float = 2.0,
    experiment_id: int | None = None,
    region: str | None = None,
    image: str = "",
    startup_script: str = "",
    spot: bool = True,
    auto_artifact_sync: bool = True,
    artifact_sync_policy: str = "smart",
    artifact_include_globs: list[str] | None = None,
    artifact_exclude_globs: list[str] | None = None,
    execution_mode: str = "auto",
    allow_fallback_to_simulation: bool = True,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    normalized_idempotency_key = _normalize_idempotency_key(idempotency_key)
    if normalized_idempotency_key:
        existing = _find_run_by_idempotency_key(
            project_id=project_id,
            idempotency_key=normalized_idempotency_key,
        )
        if existing is not None:
            _append_audit_event(
                project_id,
                {
                    "event": "submit_idempotent_replay",
                    "run_id": str(existing.get("run_id") or ""),
                    "idempotency_key": normalized_idempotency_key,
                },
            )
            payload = _serialize_run(existing, logs_tail=120)
            payload["idempotent_replay"] = True
            return payload

    plan = await build_cloud_burst_launch_plan(
        db,
        project_id=project_id,
        provider_id=provider_id,
        gpu_sku=gpu_sku,
        duration_hours=duration_hours,
        experiment_id=experiment_id,
        region=region,
        image=image,
        startup_script=startup_script,
        spot=spot,
    )

    resolved_experiment_id = experiment_id
    if resolved_experiment_id is None:
        candidate = plan.get("experiment_id")
        try:
            resolved_experiment_id = int(candidate) if candidate is not None else None
        except (TypeError, ValueError):
            resolved_experiment_id = None

    source_dir = await _resolve_experiment_output_dir(
        db,
        project_id=project_id,
        experiment_id=resolved_experiment_id,
    )
    sync_policy = _normalize_sync_policy(artifact_sync_policy)
    include_patterns = _effective_sync_include_globs(sync_policy, artifact_include_globs)
    exclude_patterns = _effective_sync_exclude_globs(artifact_exclude_globs)

    provider_token = str(plan.get("provider_id") or provider_id).strip().lower() or "provider"
    requested_execution_mode = _normalize_execution_mode(execution_mode)
    fallback_allowed = bool(allow_fallback_to_simulation) and not settings.STRICT_EXECUTION_MODE
    credential_snapshot = dict(plan.get("credentials") or {})
    credentials_ready = bool(credential_snapshot.get("ready"))
    provider_live_supported = _provider_supports_live(provider_token)

    provider_api_key: str | None = None
    provider_job_id: str | None = None
    provider_status_raw: str | None = None
    provider_submit_payload: dict[str, Any] | None = None
    provider_submit_error: str | None = None
    execution_mode_effective = "simulate"
    fallback_reason = ""

    if requested_execution_mode == "simulate":
        if settings.STRICT_EXECUTION_MODE:
            raise StrictExecutionError(
                "training",
                "Cloud burst simulate mode is blocked because STRICT_EXECUTION_MODE is enabled. "
                "Use execution_mode=live with provider credentials configured.",
            )
        execution_mode_effective = "simulate"
        fallback_reason = "Explicit simulate mode requested."
    elif requested_execution_mode == "live":
        if not provider_live_supported:
            if not fallback_allowed:
                if settings.STRICT_EXECUTION_MODE:
                    raise StrictExecutionError(
                        "training",
                        f"Cloud burst live execution is required in strict mode, but provider '{provider_token}' "
                        "does not support live execution.",
                    )
                raise ValueError(f"Provider '{provider_token}' does not support live execution yet.")
            execution_mode_effective = "simulate"
            fallback_reason = "Provider does not support live execution; fallback to simulation."
        elif not credentials_ready:
            if not fallback_allowed:
                if settings.STRICT_EXECUTION_MODE:
                    raise StrictExecutionError(
                        "training",
                        "Cloud burst live execution is required in strict mode, but provider credentials are missing.",
                    )
                missing = ", ".join(list(credential_snapshot.get("missing_keys") or []))
                raise ValueError(
                    f"Live mode requires provider credentials. Missing keys: {missing or 'unknown'}."
                )
            execution_mode_effective = "simulate"
            fallback_reason = "Credentials not ready; fallback to simulation."
        else:
            execution_mode_effective = "live"
    else:
        if provider_live_supported and credentials_ready:
            execution_mode_effective = "live"
        else:
            if not fallback_allowed:
                if settings.STRICT_EXECUTION_MODE:
                    if not provider_live_supported:
                        raise StrictExecutionError(
                            "training",
                            f"Cloud burst strict mode requires live execution, but provider '{provider_token}' "
                            "does not support live execution.",
                        )
                    raise StrictExecutionError(
                        "training",
                        "Cloud burst strict mode requires provider credentials for live execution.",
                    )
                if not provider_live_supported:
                    raise ValueError(f"Provider '{provider_token}' does not support live execution yet.")
                missing = ", ".join(list(credential_snapshot.get("missing_keys") or []))
                raise ValueError(
                    f"Live mode requires provider credentials. Missing keys: {missing or 'unknown'}."
                )
            execution_mode_effective = "simulate"
            if not provider_live_supported:
                fallback_reason = "Provider live execution is not available; auto-fallback to simulation."
            elif not credentials_ready:
                fallback_reason = "Credentials not ready; auto-fallback to simulation."

    if execution_mode_effective == "live":
        credential_values, credential_status = await _credential_values(
            db,
            project_id,
            provider_id=provider_token,
        )
        credential_snapshot = credential_status
        provider_api_key = str(credential_values.get("api_key") or "").strip() or None
        if not provider_api_key:
            if not fallback_allowed:
                if settings.STRICT_EXECUTION_MODE:
                    raise StrictExecutionError(
                        "training",
                        "Cloud burst strict mode requires provider api_key credential for live execution.",
                    )
                raise ValueError("Live mode requires provider api_key credential.")
            execution_mode_effective = "simulate"
            fallback_reason = "Provider api_key missing at submit time; fallback to simulation."
        else:
            try:
                provider_submit_payload = await _submit_provider_job(
                    provider_id=provider_token,
                    api_key=provider_api_key,
                    gpu_sku=str(plan.get("gpu_sku") or gpu_sku).strip().lower(),
                    image=str(image or "").strip() or "ghcr.io/slm/platform-trainer:latest",
                    startup_script=str(startup_script or "").strip() or "bash /workspace/entrypoint.sh",
                    run_name=f"slm-{project_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                )
                provider_job_id = str(provider_submit_payload.get("provider_job_id") or "").strip() or None
                provider_status_raw = str(provider_submit_payload.get("provider_status_raw") or "").strip() or None
                if not provider_job_id:
                    raise ValueError("Provider submit did not return provider job id.")
            except Exception as exc:  # noqa: BLE001
                provider_submit_error = str(exc)
                if not fallback_allowed:
                    if settings.STRICT_EXECUTION_MODE:
                        raise StrictExecutionError(
                            "training",
                            "Cloud burst provider submit failed in strict mode; simulation fallback is blocked.",
                        ) from exc
                    raise ValueError(
                        f"Live provider submit failed and fallback disabled: {provider_submit_error}"
                    ) from exc
                execution_mode_effective = "simulate"
                fallback_reason = f"Provider submit failed; fallback to simulation ({provider_submit_error})."
                provider_job_id = None
                provider_status_raw = None

    run_id = (
        f"cbr-{provider_token}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}"
    )
    target_artifact_dir = _cloud_run_artifacts_dir(project_id, run_id)
    record = {
        "schema": "slm.cloud_burst.managed_run/v2",
        "run_id": run_id,
        "project_id": int(project_id),
        "launch_id": str(plan.get("launch_id") or ""),
        "idempotency_key": normalized_idempotency_key,
        "idempotent_replay": False,
        "provider_id": provider_token,
        "provider_job_id": provider_job_id,
        "provider_status_raw": provider_status_raw,
        "provider_submit_error": provider_submit_error,
        "provider_submit_payload": provider_submit_payload or {},
        "gpu_sku": str(plan.get("gpu_sku") or gpu_sku).strip().lower(),
        "experiment_id": resolved_experiment_id,
        "status": "submitted",
        "status_reason": "Cloud burst job submitted to managed lifecycle orchestrator.",
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
        "started_at": None,
        "finished_at": None,
        "cancel_requested": False,
        "error": None,
        "quote": dict(plan.get("quote") or {}),
        "credentials": credential_snapshot,
        "request_template": dict(plan.get("request_template") or {}),
        "execution_mode_requested": requested_execution_mode,
        "execution_mode_effective": execution_mode_effective,
        "execution_mode_fallback_reason": fallback_reason or None,
        "runtime": {
            "duration_hours": float(duration_hours),
            "region": str(region or "").strip() or None,
            "image": str(image or "").strip() or "ghcr.io/slm/platform-trainer:latest",
            "startup_script": str(startup_script or "").strip() or "bash /workspace/entrypoint.sh",
            "spot": bool(spot),
        },
        "artifacts": {
            "sync_enabled": bool(auto_artifact_sync),
            "source_dir": source_dir,
            "target_dir": str(target_artifact_dir),
            "policy": sync_policy,
            "include_globs": include_patterns,
            "exclude_globs": exclude_patterns,
            "last_sync_at": None,
            "last_sync_summary": None,
            "sync_history": [],
        },
        "status_timeline": [
            {
                "status": "submitted",
                "at": _utcnow_iso(),
                "reason": "Cloud burst job submitted to managed lifecycle orchestrator.",
            }
        ],
        "metrics_tail": [],
        "metrics_hashes": [],
        "log_bridge": {"seen_hashes": []},
        "record_path": str(_cloud_run_record_path(project_id, run_id)),
        "logs_path": str(_cloud_run_logs_path(project_id, run_id)),
    }
    _persist_run_record(project_id, run_id, record)
    _append_run_log(
        project_id,
        run_id,
        (
            "managed job submitted "
            f"(provider={provider_token}, gpu={gpu_sku}, mode={execution_mode_effective})"
        ),
    )
    _append_audit_event(
        project_id,
        {
            "event": "submit",
            "run_id": run_id,
            "provider_id": provider_token,
            "gpu_sku": str(plan.get("gpu_sku") or gpu_sku).strip().lower(),
            "execution_mode_requested": requested_execution_mode,
            "execution_mode_effective": execution_mode_effective,
            "execution_mode_fallback_reason": fallback_reason or None,
            "idempotency_key": normalized_idempotency_key,
            "provider_job_id": provider_job_id,
            "provider_submit_error": provider_submit_error,
        },
    )
    if execution_mode_effective == "simulate" and fallback_reason:
        _append_run_log(project_id, run_id, fallback_reason)
    if not bool(dict(credential_snapshot or {}).get("ready", False)):
        _append_run_log(
            project_id,
            run_id,
            "credential readiness is false; running simulated managed lifecycle (no provider API call).",
        )

    cancel_event = asyncio.Event()
    task = asyncio.create_task(
        _run_managed_cloud_burst_job(
            project_id=project_id,
            run_id=run_id,
            cancel_event=cancel_event,
            auto_artifact_sync=bool(auto_artifact_sync),
            provider_id=provider_token if execution_mode_effective == "live" else None,
            provider_job_id=provider_job_id if execution_mode_effective == "live" else None,
            provider_api_key=provider_api_key if execution_mode_effective == "live" else None,
        )
    )
    await _set_active_task(
        run_id,
        _ManagedCloudBurstTask(
            task=task,
            cancel_event=cancel_event,
            provider_id=provider_token if execution_mode_effective == "live" else None,
            provider_job_id=provider_job_id if execution_mode_effective == "live" else None,
            provider_api_key=provider_api_key if execution_mode_effective == "live" else None,
        ),
    )

    return get_cloud_burst_job_status(
        project_id=project_id,
        run_id=run_id,
        logs_tail=120,
    )


async def cancel_cloud_burst_job(
    *,
    project_id: int,
    run_id: str,
) -> dict[str, Any]:
    run = _load_run_record(project_id, run_id)
    status = str(run.get("status") or "unknown").strip().lower()
    if status in _TERMINAL_CLOUD_BURST_JOB_STATES:
        return _serialize_run(run, logs_tail=120)

    _mark_cancel_requested(project_id, run_id)
    active = await _get_active_task(run_id)
    provider_id = str(
        (active.provider_id if active is not None else run.get("provider_id")) or ""
    ).strip().lower()
    provider_job_id = str(
        (active.provider_job_id if active is not None else run.get("provider_job_id")) or ""
    ).strip()
    provider_api_key = str(
        (active.provider_api_key if active is not None else "") or ""
    ).strip()
    if (
        provider_id
        and provider_job_id
        and provider_api_key
        and _provider_supports_cancel(provider_id)
    ):
        try:
            provider_cancel = await _cancel_provider_job(
                provider_id=provider_id,
                api_key=provider_api_key,
                provider_job_id=provider_job_id,
            )
            run = _load_run_record(project_id, run_id)
            run["provider_status_raw"] = str(provider_cancel.get("provider_status_raw") or "").strip()
            _persist_run_record(project_id, run_id, run)
            _append_run_log(
                project_id,
                run_id,
                f"provider cancel sent (provider={provider_id}, provider_job_id={provider_job_id})",
            )
        except Exception as exc:  # noqa: BLE001
            _append_run_log(project_id, run_id, f"provider_cancel_warning: {exc}")
            _append_audit_event(
                project_id,
                {
                    "event": "cancel_provider_error",
                    "run_id": run_id,
                    "provider_id": provider_id,
                    "provider_job_id": provider_job_id,
                    "error": str(exc),
                },
            )
    if active is not None:
        active.cancel_event.set()
    else:
        current = _load_run_record(project_id, run_id)
        current_status = str(current.get("status") or "").strip().lower()
        if current_status not in _TERMINAL_CLOUD_BURST_JOB_STATES:
            _mark_run_status(
                project_id,
                run_id,
                status="cancelled",
                reason="Cancelled without active worker task.",
            )
    _append_audit_event(
        project_id,
        {
            "event": "cancel",
            "run_id": run_id,
            "provider_id": provider_id or None,
            "provider_job_id": provider_job_id or None,
            "has_active_worker": active is not None,
        },
    )
    return get_cloud_burst_job_status(project_id=project_id, run_id=run_id, logs_tail=120)
