"""Cloud GPU burst planning: provider catalog, quote estimation, and launch plans."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_cloud_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "cloud_burst"
    path.mkdir(parents=True, exist_ok=True)
    return path


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

