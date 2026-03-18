#!/usr/bin/env python3
"""BrewSLM CLI for ingestion, preflight, training, and export workflows."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - runtime guard for bare Python envs
    httpx = None  # type: ignore[assignment]


DEFAULT_API_BASE = os.environ.get("BREWSLM_API_BASE", "http://127.0.0.1:8000/api")
DEFAULT_TOKEN = os.environ.get("BREWSLM_TOKEN", "").strip()
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("BREWSLM_TIMEOUT_SECONDS", "60"))
DEFAULT_INTENT = "Fine-tune a practical assistant on my imported dataset."
TERMINAL_IMPORT_STATES = {"completed", "failed", "error", "cancelled"}

_SCRIPT_PATH = Path(__file__).resolve()
_SCRIPT_ROOT = _SCRIPT_PATH.parent
_BACKEND_ROOT = _SCRIPT_ROOT.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_QUICKSTART_BASE_ENV = "BREWSLM_QUICKSTART_DIR"

_TEMPLATE_ALIASES = {
    "general": "general",
    "legal": "legal",
    "support": "support",
}

_SAMPLE_DATASET_ALIASES = {
    "general-chat-v1": "general-sample.csv",
    "general-sample-v1": "general-sample.csv",
    "general": "general-sample.csv",
    "support-chat-v1": "support-sample.csv",
    "support-sample-v1": "support-sample.csv",
    "support": "support-sample.csv",
    "legal-contract-v1": "legal-sample.csv",
    "legal-sample-v1": "legal-sample.csv",
    "legal": "legal-sample.csv",
}

SOURCE_ALIASES = {
    "hf": "huggingface",
    "huggingface": "huggingface",
    "kaggle": "kaggle",
    "url": "url",
}

TARGET_ALIASES = {
    "vllm": "runner.vllm",
    "tgi": "runner.tgi",
    "ollama": "runner.ollama",
    "managed-vllm": "deployment.vllm_managed",
    "vllm-managed": "deployment.vllm_managed",
    "hf-endpoint": "deployment.hf_inference_endpoint",
    "huggingface-endpoint": "deployment.hf_inference_endpoint",
    "sagemaker": "deployment.aws_sagemaker",
    "aws-sagemaker": "deployment.aws_sagemaker",
}


class ApiClient:
    """Small synchronous HTTP client for BrewSLM API."""

    def __init__(self, *, api_base: str, token: str, timeout_seconds: float):
        if httpx is None:
            raise ValueError(
                "Missing dependency 'httpx'. Install backend requirements or run with backend/.venv/bin/python."
            )
        api_base_normalized = str(api_base or "").strip().rstrip("/")
        if not api_base_normalized:
            raise ValueError("api_base cannot be empty")
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-API-Key"] = token
        self._client = httpx.Client(
            base_url=api_base_normalized,
            headers=headers,
            timeout=max(timeout_seconds, 1.0),
        )

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._client.request(method.upper(), path, json=json_body, params=params)
        except httpx.RequestError as e:
            raise RuntimeError(f"Request failed: {e}") from e
        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise RuntimeError(f"{method.upper()} {path} failed ({response.status_code}): {detail}")
        if not response.content:
            return {}
        try:
            return response.json()
        except Exception:
            return {"raw": response.text}

    def upload_file(
        self,
        path: str,
        *,
        file_path: Path,
        form_fields: dict[str, Any] | None = None,
    ) -> Any:
        payload = {
            str(key): str(value)
            for key, value in dict(form_fields or {}).items()
            if value is not None
        }
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with open(file_path, "rb") as handle:
            files = {"file": (file_path.name, handle, mime_type)}
            try:
                response = self._client.request(
                    "POST",
                    path,
                    data=payload,
                    files=files,
                )
            except httpx.RequestError as e:
                raise RuntimeError(f"Request failed: {e}") from e
        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise RuntimeError(f"POST {path} failed ({response.status_code}): {detail}")
        if not response.content:
            return {}
        try:
            return response.json()
        except Exception:
            return {"raw": response.text}


def _extract_error_detail(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        text = str(response.text or "").strip()
        return text or "unknown error"
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        return json.dumps(payload, ensure_ascii=True)
    return str(payload)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))


def _parse_json_object(text: str, *, label: str) -> dict[str, Any]:
    token = str(text or "").strip()
    if not token:
        return {}
    try:
        payload = json.loads(token)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} must be valid JSON object: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_json_object_file(path_value: str, *, label: str) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.exists():
        raise ValueError(f"{label} file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} file must contain valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a JSON object")
    return payload


def _quickstart_base_candidates() -> list[Path]:
    env_dir = str(os.environ.get(_QUICKSTART_BASE_ENV, "")).strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(
        [
            _REPO_ROOT / "data" / "quickstart",
            _BACKEND_ROOT / "data" / "quickstart",
            Path.cwd() / "data" / "quickstart",
        ]
    )
    # Preserve order and remove duplicates
    resolved: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        token = str(item.resolve() if item.exists() else item.expanduser())
        if token in seen:
            continue
        seen.add(token)
        resolved.append(item)
    return resolved


def _resolve_quickstart_templates_dir() -> Path:
    for base in _quickstart_base_candidates():
        candidate = base / "templates"
        if candidate.exists():
            return candidate
    expected = _quickstart_base_candidates()[0] / "templates"
    raise ValueError(
        "Quickstart templates directory not found. "
        f"Set {_QUICKSTART_BASE_ENV} or ensure {expected} exists."
    )


def _resolve_quickstart_datasets_dir() -> Path:
    for base in _quickstart_base_candidates():
        candidate = base / "datasets"
        if candidate.exists():
            return candidate
    expected = _quickstart_base_candidates()[0] / "datasets"
    raise ValueError(
        "Quickstart datasets directory not found. "
        f"Set {_QUICKSTART_BASE_ENV} or ensure {expected} exists."
    )


def _load_quickstart_template(template: str) -> tuple[str, dict[str, Any], Path]:
    token = str(template or "").strip()
    if not token:
        raise ValueError("template cannot be empty")
    normalized = token.lower().replace("_", "-").replace(" ", "-")
    alias = _TEMPLATE_ALIASES.get(normalized)

    if alias:
        template_path = _resolve_quickstart_templates_dir() / f"{alias}.json"
        if not template_path.exists():
            raise ValueError(f"Quickstart template file not found: {template_path}")
        payload = _load_json_object_file(str(template_path), label=f"template:{alias}")
        return alias, payload, template_path

    custom = Path(token).expanduser()
    if not custom.is_absolute():
        cwd_candidate = (Path.cwd() / custom).resolve()
        repo_candidate = (_REPO_ROOT / custom).resolve()
        custom = cwd_candidate if cwd_candidate.exists() else repo_candidate
    if not custom.exists():
        raise ValueError(
            f"Unknown template '{template}'. Use one of {sorted(_TEMPLATE_ALIASES)} or provide a valid JSON path."
        )
    payload = _load_json_object_file(str(custom), label=f"template:{template}")
    return custom.stem, payload, custom


def _resolve_sample_dataset(sample: str) -> tuple[str, Path]:
    token = str(sample or "").strip()
    if not token:
        raise ValueError("sample cannot be empty")
    normalized = token.lower().replace("_", "-").replace(" ", "-")
    file_name = _SAMPLE_DATASET_ALIASES.get(normalized)
    if file_name is None:
        supported = ", ".join(sorted(_SAMPLE_DATASET_ALIASES.keys()))
        raise ValueError(f"Unknown sample '{sample}'. Supported samples: {supported}")
    dataset_path = _resolve_quickstart_datasets_dir() / file_name
    if not dataset_path.exists():
        raise ValueError(f"Quickstart sample file not found: {dataset_path}")
    return normalized, dataset_path


def _resolve_local_input_file(path_value: str, *, label: str) -> Path:
    token = str(path_value or "").strip()
    if not token:
        raise ValueError(f"{label} cannot be empty")
    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        fallback = (_REPO_ROOT / token).resolve()
        if fallback.exists():
            candidate = fallback
    if not candidate.exists():
        raise ValueError(f"{label} file not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"{label} must be a file: {candidate}")
    return candidate


def _normalize_source(source: str) -> str:
    token = str(source or "").strip().lower().replace("_", "").replace("-", "")
    if token == "huggingface":
        return "huggingface"
    if token in {"hf", "kaggle", "url"}:
        return SOURCE_ALIASES[token]
    if token in SOURCE_ALIASES:
        return SOURCE_ALIASES[token]
    raise ValueError("source must be one of: hf, huggingface, kaggle, url")


def _normalize_target(target: str) -> str:
    token = str(target or "").strip()
    if not token:
        raise ValueError("target cannot be empty")
    if "." in token:
        return token
    compact = token.lower().replace("_", "-").replace(" ", "-")
    return TARGET_ALIASES.get(compact, token)


def _pick_experiment_id(experiments: list[dict[str, Any]]) -> int:
    if not experiments:
        raise ValueError("No experiments found. Run training first or pass --experiment.")
    preferred_order = ["completed", "running", "pending", "failed", "cancelled"]
    for status in preferred_order:
        for row in experiments:
            if str(row.get("status", "")).lower() == status:
                return int(row["id"])
    return int(experiments[0]["id"])


def run_ingest(args: argparse.Namespace, client: ApiClient) -> int:
    source_type = _normalize_source(args.source)
    field_mapping = _parse_json_object(args.field_map, label="field-map")
    adapter_config = _parse_json_object(args.adapter_config, label="adapter-config")
    payload: dict[str, Any] = {
        "source_type": source_type,
        "identifier": args.identifier,
        "split": args.split,
        "max_samples": args.max_samples,
        "config_name": args.config_name or None,
        "field_mapping": field_mapping,
        "adapter_id": args.adapter_id,
        "task_profile": args.task_profile or None,
        "adapter_config": adapter_config,
        "normalize_for_training": not args.no_normalize,
        "hf_token": args.hf_token or None,
        "kaggle_username": args.kaggle_username or None,
        "kaggle_key": args.kaggle_key or None,
        "use_saved_secrets": not args.no_saved_secrets,
    }

    if args.sync:
        result = client.request(
            "POST",
            f"/projects/{args.project_id}/ingestion/import-remote",
            json_body=payload,
        )
        _print_json(result)
        status = str(result.get("status", "")).lower()
        if status in {"failed", "error", "cancelled"}:
            return 1
        return 0

    queued = client.request(
        "POST",
        f"/projects/{args.project_id}/ingestion/import-remote/queue",
        json_body=payload,
    )
    _print_json(queued)

    if not args.wait:
        return 0

    report_path = str(queued.get("report_path") or "").strip()
    if not report_path:
        raise RuntimeError("Queued response missing report_path; cannot poll status")

    deadline = None
    if args.max_wait_seconds > 0:
        deadline = time.monotonic() + float(args.max_wait_seconds)

    while True:
        status_payload = client.request(
            "GET",
            f"/projects/{args.project_id}/ingestion/imports/status",
            params={"report_path": report_path},
        )
        state = str(status_payload.get("status", "")).strip().lower()
        if state in TERMINAL_IMPORT_STATES:
            _print_json(status_payload)
            return 0 if state == "completed" else 1

        if deadline is not None and time.monotonic() >= deadline:
            _print_json(
                {
                    "status": "timeout",
                    "message": "Timed out while waiting for remote import job completion.",
                    "last_status": status_payload,
                }
            )
            return 1

        time.sleep(max(float(args.poll_interval), 0.5))


def run_preflight(args: argparse.Namespace, client: ApiClient) -> int:
    config = _load_json_object_file(args.config_file, label="config")
    if args.config_json:
        config.update(_parse_json_object(args.config_json, label="config-json"))
    if args.base_model:
        config["base_model"] = args.base_model
    if args.task_type:
        config["task_type"] = args.task_type
    if args.training_runtime_id:
        config["training_runtime_id"] = args.training_runtime_id

    endpoint = "/experiments/preflight/plan" if args.plan else "/experiments/preflight"
    result = client.request(
        "POST",
        f"/projects/{args.project_id}/training{endpoint}",
        json_body={"config": config},
    )
    _print_json(result)
    if args.plan:
        return 0
    preflight = dict(result.get("preflight") or {})
    return 0 if bool(preflight.get("ok", False)) else 1


def run_train(args: argparse.Namespace, client: ApiClient) -> int:
    if not args.autopilot and not args.one_click:
        print(
            "Note: defaulting to autopilot one-click mode. "
            "Pass --autopilot --one-click explicitly for clarity.",
            file=sys.stderr,
        )

    payload: dict[str, Any] = {
        "intent": args.intent or DEFAULT_INTENT,
        "target_device": args.target_device,
        "primary_language": args.primary_language,
        "available_vram_gb": args.available_vram_gb,
        "base_model": args.base_model or None,
        "run_name": args.run_name or None,
        "description": args.description or None,
        "auto_apply_rewrite": not args.no_auto_rewrite,
        "intent_rewrite": args.intent_rewrite or None,
    }

    result = client.request(
        "POST",
        f"/projects/{args.project_id}/training/autopilot/one-click-run",
        json_body=payload,
    )
    _print_json(result)
    started = bool(result.get("started", False))
    return 0 if started else 1


def run_export(args: argparse.Namespace, client: ApiClient) -> int:
    experiment_id = args.experiment_id
    if experiment_id is None:
        experiments_payload = client.request(
            "GET",
            f"/projects/{args.project_id}/training/experiments",
        )
        if not isinstance(experiments_payload, list):
            raise RuntimeError("Unexpected experiment list response")
        experiment_id = _pick_experiment_id([dict(row) for row in experiments_payload])

    create_payload = {
        "experiment_id": experiment_id,
        "export_format": args.export_format,
        "quantization": args.quantization or None,
    }
    created = client.request(
        "POST",
        f"/projects/{args.project_id}/export/create",
        json_body=create_payload,
    )
    _print_json(created)

    if args.no_run:
        return 0

    export_id = int(created.get("id"))
    deployment_targets = [_normalize_target(target) for target in list(args.targets or [])]
    run_payload: dict[str, Any] = {
        "deployment_targets": deployment_targets or None,
        "run_smoke_tests": not args.no_smoke_tests,
    }
    if args.eval_report:
        run_payload["eval_report"] = _load_json_object_file(args.eval_report, label="eval-report")
    if args.safety_scorecard:
        run_payload["safety_scorecard"] = _load_json_object_file(
            args.safety_scorecard,
            label="safety-scorecard",
        )

    run_result = client.request(
        "POST",
        f"/projects/{args.project_id}/export/{export_id}/run",
        json_body=run_payload,
    )
    _print_json(run_result)
    status = str(run_result.get("status", "")).lower()
    return 0 if status in {"completed", "success"} else 1


def run_doctor(args: argparse.Namespace, client: ApiClient) -> int:
    result = client.request(
        "GET",
        f"/projects/{args.project_id}/runtime/readiness",
    )
    
    status = result.get("status", "unknown")
    strict = result.get("strict_mode", False)
    
    print(f"BrewSLM Doctor - Project {args.project_id}")
    print(f"Overall Status: {status.upper()}")
    print(f"Strict Mode: {'ENABLED' if strict else 'DISABLED'}")
    print("-" * 40)
    
    checks = result.get("checks", [])
    for check in checks:
        c_status = check.get("status", "unknown").upper()
        c_name = check.get("name", "Unknown Check")
        c_msg = check.get("message", "")
        c_fix = check.get("fix", "")
        
        icon = "✅" if c_status == "PASS" else "⚠️" if c_status == "WARN" else "❌"
        print(f"{icon} {c_name}: {c_status}")
        print(f"   {c_msg}")
        if c_fix:
            print(f"   FIX: {c_fix}")
        print()

    return 0 if status == "pass" else 1


def run_optimize(args: argparse.Namespace, client: ApiClient) -> int:
    payload = {
        "target_id": _normalize_target(args.target),
    }
    result = client.request(
        "POST",
        f"/projects/{args.project_id}/export/optimize",
        json_body=payload,
    )

    candidates = result.get("candidates", [])
    if not candidates:
        print("No optimization candidates found.")
        return 1

    print(f"Inference Optimization Results for Target: {result.get('target_id')}")
    print(f"{'ID':<20} {'Name':<25} {'Latency (ms)':<15} {'Memory (GB)':<15} {'Quality':<10} {'Recommended'}")
    print("-" * 100)
    for c in candidates:
        m = c.get("metrics", {})
        rec = "★ YES" if c.get("is_recommended") else ""
        print(
            f"{c.get('id'):<20} {c.get('name'):<25} {m.get('latency_ms'):<15} {m.get('memory_gb'):<15} {m.get('quality_score'):<10} {rec}"
        )

    return 0


def run_project(args: argparse.Namespace, client: ApiClient) -> int:
    if args.subcommand == "create":
        template_payload: dict[str, Any] = {}
        template_id = ""
        template_path: Path | None = None
        if not args.no_template:
            template_id, template_payload, template_path = _load_quickstart_template(args.template)

        gate_policy = {}
        if isinstance(template_payload.get("gate_policy"), dict):
            gate_policy.update(dict(template_payload.get("gate_policy") or {}))
        if args.gate_policy:
            gate_policy.update(_parse_json_object(args.gate_policy, label="gate-policy"))

        budget_settings = {}
        budget_candidate = template_payload.get("budget_settings")
        if not isinstance(budget_candidate, dict):
            budget_candidate = template_payload.get("budget")
        if isinstance(budget_candidate, dict):
            budget_settings.update(dict(budget_candidate))
        if args.budget_settings:
            budget_settings.update(_parse_json_object(args.budget_settings, label="budget-settings"))

        payload: dict[str, Any] = {
            "name": args.name,
            "description": args.description or str(template_payload.get("description") or ""),
            "base_model_name": args.base_model_name
            or str(
                template_payload.get("base_model_name")
                or template_payload.get("base_model")
                or ""
            ),
        }
        if args.domain_pack_id is not None:
            payload["domain_pack_id"] = int(args.domain_pack_id)
        if args.domain_profile_id is not None:
            payload["domain_profile_id"] = int(args.domain_profile_id)
        if args.target_profile_id:
            payload["target_profile_id"] = args.target_profile_id
        if gate_policy:
            payload["gate_policy"] = gate_policy
        if budget_settings:
            payload["budget_settings"] = budget_settings

        created = client.request("POST", "/projects", json_body=payload)
        output = {
            "project": created,
            "quickstart": {
                "template_id": template_id or None,
                "template_path": str(template_path) if template_path else None,
            },
        }
        _print_json(output)
        return 0

    pid = getattr(args, "project_id", None)
    if pid is None:
        print("Project ID required.")
        return 1

    if args.subcommand == "budget":
        project = client.request("GET", f"/projects/{pid}")
        budget = project.get("budget_settings") or {}
        print(f"Project: {project.get('name')} (ID: {project.get('id')})")
        print(f"Monthly Cap: ${budget.get('monthly_cap', 0.0):.2f}")
        print(f"Current Spend: ${budget.get('current_spend', 0.0):.2f}")
        print(f"Alert Threshold: {int(budget.get('alert_threshold', 0.8) * 100)}%")
        print(f"Auto-Cancel: {budget.get('auto_cancel', True)}")
    elif args.subcommand == "budget-set":
        project = client.request("GET", f"/projects/{pid}")
        budget = project.get("budget_settings") or {}
        if args.cap is not None:
            budget["monthly_cap"] = args.cap
        if args.threshold is not None:
            budget["alert_threshold"] = args.threshold
        if args.auto_cancel is not None:
            budget["auto_cancel"] = args.auto_cancel.lower() == "true"
        client.request("PUT", f"/projects/{pid}", json_body={"budget_settings": budget})
        print(f"Updated budget settings for project {pid}")
    return 0


def run_dataset(args: argparse.Namespace, client: ApiClient) -> int:
    if args.subcommand != "import":
        raise ValueError(f"Unsupported dataset subcommand '{args.subcommand}'.")

    sample_token = str(args.sample or "").strip()
    file_token = str(args.file or "").strip()
    if bool(sample_token) == bool(file_token):
        raise ValueError("Provide exactly one of --sample or --file.")

    sample_id = ""
    source_label = str(args.source or "").strip()
    input_path: Path
    if sample_token:
        sample_id, input_path = _resolve_sample_dataset(sample_token)
        if not source_label:
            source_label = f"quickstart:{sample_id}"
    else:
        input_path = _resolve_local_input_file(file_token, label="dataset")
        if not source_label:
            source_label = "upload"

    uploaded = client.upload_file(
        f"/projects/{args.project_id}/ingestion/upload",
        file_path=input_path,
        form_fields={
            "source": source_label,
            "sensitivity": args.sensitivity,
            "license_info": args.license_info,
        },
    )

    output: dict[str, Any] = {
        "upload": uploaded,
        "input_file": str(input_path),
        "sample": sample_id or None,
    }

    if args.process:
        document_id = int(uploaded.get("id") or 0)
        if document_id <= 0:
            raise RuntimeError("Upload response missing document id; cannot process document.")
        processed = client.request(
            "POST",
            f"/projects/{args.project_id}/ingestion/documents/{document_id}/process",
        )
        output["process"] = processed
        status = str(processed.get("status") or "").strip().lower()
        _print_json(output)
        return 0 if status in {"accepted", "completed", "done"} else 1

    _print_json(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brewslm",
        description="BrewSLM command line client",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL (default: %(default)s)")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="API token (or set BREWSLM_TOKEN)")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP request timeout in seconds (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Project commands
    project_parser = subparsers.add_parser("project", help="Manage project settings (budget, etc)")
    project_sub = project_parser.add_subparsers(dest="subcommand", required=True)

    p_budget = project_sub.add_parser("budget", help="Show project budget and spend")
    p_budget.add_argument("--id", "--project-id", dest="project_id", type=int, required=True)

    p_create = project_sub.add_parser("create", help="Create a project (quickstart template supported)")
    p_create.add_argument("--name", required=True, help="Project name")
    p_create.add_argument(
        "--template",
        default="general",
        help="Template alias (general, support, legal) or path to template JSON",
    )
    p_create.add_argument(
        "--no-template",
        action="store_true",
        help="Skip template loading and use only explicit CLI fields.",
    )
    p_create.add_argument("--description", default="", help="Project description override")
    p_create.add_argument("--base-model", dest="base_model_name", default="", help="Base model name override")
    p_create.add_argument("--domain-pack-id", type=int, default=None)
    p_create.add_argument("--domain-profile-id", type=int, default=None)
    p_create.add_argument("--target-profile-id", default="", help="Target profile id (default from API)")
    p_create.add_argument("--gate-policy", default="", help="Inline JSON object overriding gate policy")
    p_create.add_argument(
        "--budget-settings",
        default="",
        help="Inline JSON object overriding budget settings",
    )

    p_budget_set = project_sub.add_parser("budget-set", help="Update project budget policy")
    p_budget_set.add_argument("--id", "--project-id", dest="project_id", type=int, required=True)
    p_budget_set.add_argument("--cap", type=float, help="Monthly cap (USD)")
    p_budget_set.add_argument("--threshold", type=float, help="Alert threshold (0.0-1.0)")
    p_budget_set.add_argument("--auto-cancel", choices=["true", "false"], help="Auto-cancel on cap")
    project_parser.set_defaults(func=run_project)

    dataset_parser = subparsers.add_parser("dataset", help="Dataset upload/import helpers")
    dataset_sub = dataset_parser.add_subparsers(dest="subcommand", required=True)
    dataset_import = dataset_sub.add_parser("import", help="Upload a dataset file or quickstart sample")
    dataset_import.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    dataset_import.add_argument(
        "--sample",
        default="",
        help="Quickstart sample alias (general-chat-v1, support-chat-v1, legal-contract-v1)",
    )
    dataset_import.add_argument("--file", default="", help="Path to local dataset file")
    dataset_import.add_argument("--source", default="", help="Source label for ingestion metadata")
    dataset_import.add_argument("--sensitivity", default="internal")
    dataset_import.add_argument("--license-info", default="")
    dataset_import.add_argument(
        "--no-process",
        dest="process",
        action="store_false",
        help="Skip immediate document processing after upload",
    )
    dataset_import.set_defaults(process=True)
    dataset_parser.set_defaults(func=run_dataset)

    ingest_parser = subparsers.add_parser("ingest", help="Import remote dataset (HF/Kaggle/URL)")
    ingest_parser.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    ingest_parser.add_argument("--source", required=True, help="Data source (hf, huggingface, kaggle, url)")
    ingest_parser.add_argument("--id", "--identifier", dest="identifier", required=True, help="Dataset ID or URL")
    ingest_parser.add_argument("--split", default="train")
    ingest_parser.add_argument("--max-samples", type=int, default=None)
    ingest_parser.add_argument("--config-name", default="")
    ingest_parser.add_argument("--field-map", default="", help="JSON object for source field mapping")
    ingest_parser.add_argument("--adapter-id", default="default-canonical")
    ingest_parser.add_argument("--task-profile", default="")
    ingest_parser.add_argument("--adapter-config", default="", help="JSON object for adapter config")
    ingest_parser.add_argument("--hf-token", default="")
    ingest_parser.add_argument("--kaggle-username", default="")
    ingest_parser.add_argument("--kaggle-key", default="")
    ingest_parser.add_argument("--no-saved-secrets", action="store_true")
    ingest_parser.add_argument("--no-normalize", action="store_true")
    ingest_parser.add_argument("--sync", action="store_true", help="Run import synchronously via API")
    ingest_parser.add_argument("--wait", action="store_true", help="Poll queued import status until terminal")
    ingest_parser.add_argument("--poll-interval", type=float, default=3.0)
    ingest_parser.add_argument("--max-wait-seconds", type=float, default=1800.0)
    ingest_parser.set_defaults(func=run_ingest)

    preflight_parser = subparsers.add_parser("preflight", help="Run training preflight for project config")
    preflight_parser.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    preflight_parser.add_argument("--config-file", default="", help="Path to JSON object config")
    preflight_parser.add_argument("--config-json", default="", help="Inline JSON object config")
    preflight_parser.add_argument("--base-model", default="")
    preflight_parser.add_argument("--task", dest="task_type", default="")
    preflight_parser.add_argument("--runtime-id", dest="training_runtime_id", default="")
    preflight_parser.add_argument("--plan", action="store_true", help="Run preflight-plan suggestions endpoint")
    preflight_parser.set_defaults(func=run_preflight)

    train_parser = subparsers.add_parser("train", help="Launch training (autopilot one-click)")
    train_parser.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    train_parser.add_argument("--autopilot", action="store_true")
    train_parser.add_argument("--one-click", action="store_true")
    train_parser.add_argument("--intent", default=DEFAULT_INTENT)
    train_parser.add_argument(
        "--target-device",
        default="laptop",
        choices=["mobile", "laptop", "server"],
    )
    train_parser.add_argument("--primary-language", default="english")
    train_parser.add_argument("--available-vram-gb", type=float, default=None)
    train_parser.add_argument("--base-model", default="", help="Optional explicit base model override")
    train_parser.add_argument("--run-name", default="")
    train_parser.add_argument("--description", default="")
    train_parser.add_argument("--intent-rewrite", default="")
    train_parser.add_argument("--no-auto-rewrite", action="store_true")
    train_parser.set_defaults(func=run_train)

    export_parser = subparsers.add_parser("export", help="Create and run an export job")
    export_parser.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    export_parser.add_argument("--experiment", "--experiment-id", dest="experiment_id", type=int, default=None)
    export_parser.add_argument(
        "--format",
        dest="export_format",
        default="huggingface",
        choices=["huggingface", "gguf", "onnx", "tensorrt", "docker"],
    )
    export_parser.add_argument("--quantization", default="")
    export_parser.add_argument(
        "--target",
        dest="targets",
        action="append",
        default=[],
        help="Deployment/runner target id or alias (repeat for multiple)",
    )
    export_parser.add_argument("--eval-report", default="", help="Path to eval report JSON object")
    export_parser.add_argument("--safety-scorecard", default="", help="Path to safety scorecard JSON object")
    export_parser.add_argument("--no-run", action="store_true", help="Only create export; skip run step")
    export_parser.add_argument("--no-smoke-tests", action="store_true")
    export_parser.set_defaults(func=run_export)

    doctor_parser = subparsers.add_parser("doctor", help="Check project readiness (GPU/deps/secrets)")
    doctor_parser.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    doctor_parser.set_defaults(func=run_doctor)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Search for optimal quantization + runtime combinations"
    )
    optimize_parser.add_argument(
        "--project", "--project-id", dest="project_id", type=int, required=True
    )
    optimize_parser.add_argument(
        "--target",
        required=True,
        help="Target deployment profile (e.g., mobile_cpu, edge_gpu)",
    )
    optimize_parser.set_defaults(func=run_optimize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        client = ApiClient(
            api_base=args.api_base,
            token=str(args.token or "").strip(),
            timeout_seconds=float(args.timeout_seconds),
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        return int(args.func(args, client))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
