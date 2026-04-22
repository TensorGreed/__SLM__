#!/usr/bin/env python3
"""BrewSLM CLI for ingestion, preflight, training, and export workflows."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
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


def _load_text_file(path_value: str, *, label: str) -> str:
    token = str(path_value or "").strip()
    if not token:
        return ""
    path = Path(token).expanduser()
    if not path.exists():
        raise ValueError(f"{label} file not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path.read_text(encoding="utf-8").strip()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in values:
        token = _normalize_text(raw)
        if not token:
            continue
        marker = token.lower()
        if marker in seen:
            continue
        seen.add(marker)
        cleaned.append(token)
    return cleaned


def _parse_line_list(value: str) -> list[str]:
    token = str(value or "").strip()
    if not token:
        return []
    return [item.strip() for item in token.splitlines() if item.strip()]


def _derive_project_name_from_brief(brief_text: str) -> str:
    words = [item for item in re.split(r"[^a-zA-Z0-9]+", str(brief_text or "").strip()) if item]
    if not words:
        return "Beginner Mode Project"
    return " ".join(words[:6])[:80].strip() or "Beginner Mode Project"


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

    if bool(args.autopilot_v2) or bool(args.dry_run):
        payload.update(
            {
                "dry_run": bool(args.dry_run),
                "allow_target_fallback": not bool(args.no_target_fallback),
                "allow_profile_autotune": not bool(args.no_profile_autotune),
                "plan_profile": args.plan_profile,
            }
        )
        result = client.request(
            "POST",
            f"/projects/{args.project_id}/training/autopilot/v2/orchestrate",
            json_body=payload,
        )
        _print_autopilot_v2_decision_log(result)
        _print_json(result)
        guardrails = dict(result.get("guardrails") or {})
        if bool(args.dry_run):
            return 0 if bool(guardrails.get("can_run", False)) else 1
        started = bool(result.get("started", False))
        return 0 if started else 1

    result = client.request(
        "POST",
        f"/projects/{args.project_id}/training/autopilot/one-click-run",
        json_body=payload,
    )
    _print_json(result)
    started = bool(result.get("started", False))
    return 0 if started else 1


def _print_autopilot_v2_decision_log(payload: dict[str, Any]) -> None:
    rows = [row for row in list(payload.get("decision_log") or []) if isinstance(row, dict)]
    if not rows:
        return
    print("Autopilot v2 Decision Log:")
    for idx, row in enumerate(rows, start=1):
        step = str(row.get("step") or f"step_{idx}").strip() or f"step_{idx}"
        status = str(row.get("status") or "info").strip().upper() or "INFO"
        summary = str(row.get("summary") or "").strip()
        changed = bool(row.get("changed"))
        suffix = " (changed)" if changed else ""
        if summary:
            print(f"{idx:02d}. [{status}] {step}{suffix}: {summary}")
        else:
            print(f"{idx:02d}. [{status}] {step}{suffix}")


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


def _format_float(value: Any, *, precision: int = 3) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "n/a"


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

    run_meta = dict(result.get("optimization_run") or {})
    measured_count = int(run_meta.get("measured_candidate_count") or 0) if run_meta else 0
    estimated_count = int(run_meta.get("estimated_candidate_count") or 0) if run_meta else 0

    print(f"Inference Optimization Results for Target: {result.get('target_id')}")
    if run_meta:
        print(
            "Evidence Run: "
            f"{run_meta.get('run_id')} "
            f"(measured={measured_count}, estimated={estimated_count})"
        )
    print(
        "Metric Source: MEASURED=live probe, MIXED=partial probe + estimates, "
        "ESTIMATED=artifact/profile estimate."
    )
    if measured_count == 0:
        print(
            "Tip: all candidates are estimated. Install benchmark runtime dependencies "
            "or run 'brewslm doctor --project <id>' to identify blockers."
        )
    print(
        f"{'ID':<18} {'Name':<24} {'Source':<10} {'Latency (ms)':<13} "
        f"{'Memory (GB)':<12} {'Quality':<10} {'Recommended'}"
    )
    print("-" * 120)
    for c in candidates:
        m = c.get("metrics", {})
        source = str(c.get("metric_source") or "estimated").strip().lower() or "estimated"
        source_label = source.upper()
        rec = "★ YES" if c.get("is_recommended") else ""
        candidate_id = str(c.get("id") or "")[:18]
        candidate_name = str(c.get("name") or "")[:24]
        print(
            f"{candidate_id:<18} {candidate_name:<24} {source_label:<10} "
            f"{_format_float(m.get('latency_ms'), precision=3):<13} "
            f"{_format_float(m.get('memory_gb'), precision=4):<12} "
            f"{_format_float(m.get('quality_score'), precision=4):<10} {rec}"
        )
        measurement = c.get("measurement") or {}
        fallback_reason = str(measurement.get("fallback_reason") or "").strip()
        if source != "measured" and fallback_reason:
            print(f"  note: {fallback_reason}")

    return 0


def run_project(args: argparse.Namespace, client: ApiClient) -> int:
    if args.subcommand == "bootstrap":
        brief_text = _normalize_text(args.brief)
        if args.brief_file:
            file_brief = _load_text_file(args.brief_file, label="brief")
            brief_text = f"{brief_text}\n{file_brief}".strip() if brief_text else file_brief
        if not brief_text:
            raise ValueError("Provide --brief or --brief-file for project bootstrap.")

        sample_inputs = _dedupe_preserve_order(
            list(args.sample_input or []) + _parse_line_list(_load_text_file(args.sample_inputs_file, label="sample-inputs"))
        )
        sample_outputs = _dedupe_preserve_order(
            list(args.sample_output or []) + _parse_line_list(_load_text_file(args.sample_outputs_file, label="sample-outputs"))
        )
        risk_constraints = _dedupe_preserve_order(list(args.risk_note or []))

        analysis_payload: dict[str, Any] = {
            "brief_text": brief_text,
            "sample_inputs": sample_inputs,
            "sample_outputs": sample_outputs,
            "risk_constraints": risk_constraints,
            "deployment_target": args.target or None,
            "llm_enrich": not bool(args.no_llm_enrich),
        }
        analyzed = client.request("POST", "/domain-blueprints/analyze", json_body=analysis_payload)

        output: dict[str, Any] = {
            "analysis": analyzed,
        }

        if args.create_project:
            project_name = _normalize_text(args.name) or _derive_project_name_from_brief(brief_text)
            create_payload: dict[str, Any] = {
                "name": project_name,
                "description": _normalize_text(args.description),
                "base_model_name": _normalize_text(args.base_model),
                "beginner_mode": True,
                "brief_text": brief_text,
                "sample_inputs": sample_inputs,
                "sample_outputs": sample_outputs,
                "domain_blueprint": dict(analyzed.get("blueprint") or {}),
            }
            target_profile_id = _normalize_text(args.target_profile_id)
            if target_profile_id:
                create_payload["target_profile_id"] = target_profile_id
            created = client.request("POST", "/projects", json_body=create_payload)
            output["project"] = created
            output["mode"] = "create_project"
        else:
            if args.project_id is None:
                raise ValueError("Provide --project when not using --create-project.")
            save_payload = {
                "blueprint": dict(analyzed.get("blueprint") or {}),
                "source": "cli.bootstrap",
                "brief_text": brief_text,
                "analysis_metadata": {
                    "guidance": analyzed.get("guidance") or {},
                    "validation": analyzed.get("validation") or {},
                    "llm_enrichment": analyzed.get("llm_enrichment") or {},
                },
                "apply_immediately": bool(args.apply),
            }
            saved = client.request(
                "POST",
                f"/projects/{args.project_id}/domain-blueprints",
                json_body=save_payload,
            )
            output["project_id"] = int(args.project_id)
            output["saved_revision"] = saved
            output["mode"] = "save_revision"

        _print_json(output)
        return 0

    if args.subcommand == "blueprint":
        pid = getattr(args, "project_id", None)
        if pid is None:
            raise ValueError("Project ID is required for blueprint commands.")
        if args.blueprint_subcommand == "show":
            if bool(args.latest):
                payload = client.request("GET", f"/projects/{pid}/domain-blueprints/latest")
            elif args.version is not None:
                payload = client.request("GET", f"/projects/{pid}/domain-blueprints/{int(args.version)}")
            else:
                payload = client.request("GET", f"/projects/{pid}/domain-blueprints")
            _print_json(payload)
            return 0
        if args.blueprint_subcommand == "diff":
            payload = client.request(
                "GET",
                f"/projects/{pid}/domain-blueprints/diff",
                params={
                    "from_version": int(args.from_version),
                    "to_version": int(args.to_version),
                },
            )
            _print_json(payload)
            return 0
        raise ValueError(f"Unsupported blueprint subcommand '{args.blueprint_subcommand}'.")

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
    else:
        raise ValueError(f"Unsupported project subcommand '{args.subcommand}'.")
    return 0


def run_dataset(args: argparse.Namespace, client: ApiClient) -> int:
    subcommand = str(getattr(args, "subcommand", "") or "").strip().lower()
    if subcommand == "import":
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

    if subcommand == "profile":
        project_id = int(getattr(args, "project_id", 0) or 0)
        if project_id <= 0:
            raise ValueError("--project is required for dataset profile.")
        source: dict[str, Any] = {
            "source_type": _normalize_text(getattr(args, "source_type", "")) or "project_dataset",
        }
        if _normalize_text(getattr(args, "source_ref", "")):
            source["source_ref"] = _normalize_text(getattr(args, "source_ref", ""))
        if _normalize_text(getattr(args, "dataset_type", "")):
            source["dataset_type"] = _normalize_text(getattr(args, "dataset_type", ""))
        if _normalize_text(getattr(args, "split", "")):
            source["split"] = _normalize_text(getattr(args, "split", ""))
        document_id = getattr(args, "document_id", None)
        if document_id is not None:
            source["document_id"] = int(document_id)
        payload = {
            "source": source,
            "sample_size": int(getattr(args, "sample_size", 500) or 500),
        }
        resp = client.request(
            "POST",
            f"/projects/{project_id}/adapter-studio/profile",
            json_body=payload,
        )
        if bool(getattr(args, "json", False)):
            _print_json(resp)
            return 0
        schema = dict(resp.get("schema") or {})
        print(f"Sampled Rows: {resp.get('sampled_rows')}")
        print(f"Field Count: {schema.get('field_count')}")
        hints = list((resp.get("dataset_characteristics") or {}).get("task_shape_hints") or [])
        if hints:
            print(f"Task Shape Hints: {', '.join(hints)}")
        sensitive = list((resp.get("dataset_characteristics") or {}).get("potential_sensitive_columns") or [])
        if sensitive:
            print(f"Sensitive Columns: {', '.join(sensitive[:10])}")
        return 0

    raise ValueError(f"Unsupported dataset subcommand '{subcommand}'.")


def run_adapter(args: argparse.Namespace, client: ApiClient) -> int:
    subcommand = str(getattr(args, "adapter_subcommand", "") or "").strip().lower()
    as_json = bool(getattr(args, "json", False))
    project_id = int(getattr(args, "project_id", 0) or 0)
    if project_id <= 0:
        raise ValueError("--project is required.")

    source: dict[str, Any] = {
        "source_type": _normalize_text(getattr(args, "source_type", "")) or "project_dataset",
    }
    if _normalize_text(getattr(args, "source_ref", "")):
        source["source_ref"] = _normalize_text(getattr(args, "source_ref", ""))
    if _normalize_text(getattr(args, "dataset_type", "")):
        source["dataset_type"] = _normalize_text(getattr(args, "dataset_type", ""))
    if _normalize_text(getattr(args, "split", "")):
        source["split"] = _normalize_text(getattr(args, "split", ""))
    document_id = getattr(args, "document_id", None)
    if document_id is not None:
        source["document_id"] = int(document_id)

    field_mapping_payload = _parse_json_object(
        _normalize_text(getattr(args, "field_mapping", "")),
        label="field-mapping",
    )
    adapter_config_payload = _parse_json_object(
        _normalize_text(getattr(args, "adapter_config", "")),
        label="adapter-config",
    )

    if subcommand == "infer":
        payload = {
            "source": source,
            "sample_size": int(getattr(args, "sample_size", 400) or 400),
            "task_profile": _normalize_text(getattr(args, "task_profile", "")) or None,
        }
        resp = client.request("POST", f"/projects/{project_id}/adapter-studio/infer", json_body=payload)
        if as_json:
            _print_json(resp)
            return 0
        inference = dict(resp.get("inference") or {})
        print(f"Resolved Adapter: {inference.get('resolved_adapter_id')}")
        print(f"Task Profile: {inference.get('resolved_task_profile') or 'n/a'}")
        print(f"Confidence: {inference.get('confidence')}")
        drop = dict(inference.get("drop_analysis") or {})
        print(f"Drop Rate: {round(float(drop.get('drop_rate') or 0.0) * 100, 2)}%")
        return 0

    if subcommand == "preview":
        payload = {
            "source": source,
            "adapter_id": _normalize_text(getattr(args, "adapter_id", "")) or "auto",
            "field_mapping": field_mapping_payload,
            "adapter_config": adapter_config_payload,
            "task_profile": _normalize_text(getattr(args, "task_profile", "")) or None,
            "sample_size": int(getattr(args, "sample_size", 300) or 300),
            "preview_limit": int(getattr(args, "preview_limit", 25) or 25),
        }
        resp = client.request("POST", f"/projects/{project_id}/adapter-studio/preview", json_body=payload)
        if as_json:
            _print_json(resp)
            return 0
        preview = dict(resp.get("preview") or {})
        print(f"Resolved Adapter: {preview.get('resolved_adapter_id')}")
        print(f"Sampled: {preview.get('sampled_records')}  Mapped: {preview.get('mapped_records')}  Dropped: {preview.get('dropped_records')}")
        return 0

    if subcommand == "validate":
        payload = {
            "source": source,
            "adapter_id": _normalize_text(getattr(args, "adapter_id", "")) or "auto",
            "field_mapping": field_mapping_payload,
            "adapter_config": adapter_config_payload,
            "task_profile": _normalize_text(getattr(args, "task_profile", "")) or None,
            "sample_size": int(getattr(args, "sample_size", 300) or 300),
            "preview_limit": int(getattr(args, "preview_limit", 25) or 25),
        }
        resp = client.request("POST", f"/projects/{project_id}/adapter-studio/validate", json_body=payload)
        if as_json:
            _print_json(resp)
            return 0
        status = str(resp.get("status") or "").strip().upper()
        print(f"Validation Status: {status}")
        reason_codes = [str(item) for item in list(resp.get("reason_codes") or []) if str(item).strip()]
        if reason_codes:
            print(f"Reason Codes: {', '.join(reason_codes)}")
        actions = [str(item) for item in list(resp.get("recommended_next_actions") or []) if str(item).strip()]
        if actions:
            print("Recommended Actions:")
            for item in actions[:6]:
                print(f"- {item}")
        return 0 if str(resp.get("status") or "").strip().lower() == "pass" else 1

    if subcommand == "export":
        adapter_name = _normalize_text(getattr(args, "adapter_name", ""))
        version = int(getattr(args, "version", 0) or 0)
        if not adapter_name or version <= 0:
            raise ValueError("--adapter-name and --version are required for adapter export.")
        payload = {"export_dir": _normalize_text(getattr(args, "export_dir", "")) or None}
        resp = client.request(
            "POST",
            f"/projects/{project_id}/adapter-studio/adapters/{adapter_name}/versions/{version}/export",
            json_body=payload,
        )
        if as_json:
            _print_json(resp)
            return 0
        written = dict(resp.get("written_files") or {})
        print(f"Exported adapter scaffold: {adapter_name} v{version}")
        print(f"Template JSON: {written.get('template_json')}")
        print(f"Plugin Python: {written.get('plugin_python')}")
        return 0

    raise ValueError(f"Unsupported adapter subcommand '{subcommand}'.")


def run_models(args: argparse.Namespace, client: ApiClient) -> int:
    subcommand = str(getattr(args, "models_subcommand", "") or "").strip().lower()
    as_json = bool(getattr(args, "json", False))

    if subcommand == "import":
        source_tokens = [
            ("huggingface", _normalize_text(getattr(args, "hf_id", ""))),
            ("local_path", _normalize_text(getattr(args, "path", ""))),
            ("catalog", _normalize_text(getattr(args, "catalog_id", ""))),
        ]
        selected = [(kind, ref) for kind, ref in source_tokens if ref]
        if len(selected) != 1:
            raise ValueError("Provide exactly one source selector: --hf-id, --path, or --catalog-id.")
        source_type, source_ref = selected[0]
        payload = {
            "source_type": source_type,
            "source_ref": source_ref,
            "allow_network": bool(getattr(args, "allow_network", False)),
            "overwrite": not bool(getattr(args, "no_overwrite", False)),
        }
        resp = client.request("POST", "/models/import", json_body=payload)
        if as_json:
            _print_json(resp)
            return 0
        model = dict(resp.get("model") or {})
        print(f"Imported model: {model.get('display_name') or model.get('model_key')}")
        print(f"Model Key: {model.get('model_key')}")
        print(f"Architecture: {model.get('architecture') or 'unknown'}")
        print(f"Tasks: {', '.join(list(model.get('supported_task_families') or [])[:6]) or 'n/a'}")
        return 0

    if subcommand == "refresh":
        model_token = _normalize_text(getattr(args, "model", ""))
        if not model_token:
            raise ValueError("--model is required for models refresh.")
        payload: dict[str, Any] = {
            "allow_network": bool(getattr(args, "allow_network", False)),
        }
        if model_token.isdigit():
            payload["model_id"] = int(model_token)
        else:
            payload["model_key"] = model_token
        resp = client.request("POST", "/models/refresh", json_body=payload)
        if as_json:
            _print_json(resp)
            return 0
        model = dict(resp.get("model") or {})
        print(f"Refreshed model: {model.get('display_name') or model.get('model_key')}")
        print(f"Refresh Count: {model.get('refresh_count')}")
        return 0

    if subcommand == "list":
        params: dict[str, Any] = {}
        for key, arg_name in (
            ("family", "family"),
            ("license", "license"),
            ("hardware_fit", "hardware_fit"),
            ("min_context_length", "min_context_length"),
            ("max_params_b", "max_params_b"),
            ("training_mode", "training_mode"),
            ("search", "search"),
        ):
            value = getattr(args, arg_name, None)
            if value is None:
                continue
            token = str(value).strip() if isinstance(value, str) else value
            if token in {"", None}:
                continue
            params[key] = token
        resp = client.request("GET", "/models", params=params)
        if as_json:
            _print_json(resp)
            return 0
        rows = [item for item in list(resp.get("models") or []) if isinstance(item, dict)]
        print(f"Registered Models: {len(rows)}")
        if not rows:
            print("No model records found.")
            return 0
        print(f"{'ID':<6} {'Key':<46} {'Family':<14} {'Arch':<14} {'Ctx':<8} {'PEFT'}")
        print("-" * 98)
        for row in rows[:50]:
            rid = str(row.get("id") or "")
            key = str(row.get("model_key") or "")[:46]
            family = str(row.get("model_family") or "unknown")[:14]
            arch = str(row.get("architecture") or "unknown")[:14]
            ctx = str(row.get("context_length") or "n/a")[:8]
            peft = "yes" if bool(row.get("peft_support")) else "no"
            print(f"{rid:<6} {key:<46} {family:<14} {arch:<14} {ctx:<8} {peft}")
        return 0

    if subcommand == "recommend":
        project_id = int(getattr(args, "project_id", 0) or 0)
        if project_id <= 0:
            raise ValueError("--project is required for models recommend.")
        params = {
            "limit": int(getattr(args, "limit", 10) or 10),
            "include_incompatible": bool(getattr(args, "include_incompatible", False)),
            "allow_network": bool(getattr(args, "allow_network", False)),
        }
        for key, arg_name in (
            ("family", "family"),
            ("license", "license"),
            ("hardware_fit", "hardware_fit"),
            ("min_context_length", "min_context_length"),
            ("max_params_b", "max_params_b"),
            ("training_mode", "training_mode"),
            ("search", "search"),
            ("target_profile_id", "target"),
            ("runtime_id", "runtime_id"),
            ("dataset_adapter_id", "adapter_id"),
        ):
            value = getattr(args, arg_name, None)
            if value is None:
                continue
            token = str(value).strip() if isinstance(value, str) else value
            if token in {"", None}:
                continue
            params[key] = token
        resp = client.request("GET", f"/projects/{project_id}/models/compatible", params=params)
        if as_json:
            _print_json(resp)
            return 0
        rows = [item for item in list(resp.get("models") or []) if isinstance(item, dict)]
        print(f"Project {project_id} Compatible Models: {len(rows)}")
        if not rows:
            print("No compatible models found.")
            return 1
        for idx, item in enumerate(rows[:20], start=1):
            model = dict(item.get("model") or {})
            score = float(item.get("compatibility_score") or 0.0)
            print(f"{idx}. {model.get('display_name') or model.get('model_key')} (score={score:.3f})")
            rec_reasons = [str(r.get("message") or "") for r in list(item.get("why_recommended") or []) if isinstance(r, dict)]
            risk_reasons = [str(r.get("message") or "") for r in list(item.get("why_risky") or []) if isinstance(r, dict)]
            if rec_reasons:
                print(f"   Why recommended: {rec_reasons[0]}")
            if risk_reasons:
                print(f"   Why risky: {risk_reasons[0]}")
        return 0

    if subcommand == "validate":
        project_id = int(getattr(args, "project_id", 0) or 0)
        model_token = _normalize_text(getattr(args, "model", ""))
        if project_id <= 0:
            raise ValueError("--project is required for models validate.")
        if not model_token:
            raise ValueError("--model is required for models validate.")
        payload: dict[str, Any] = {
            "allow_network": bool(getattr(args, "allow_network", False)),
            "dataset_adapter_id": _normalize_text(getattr(args, "adapter_id", "")) or None,
            "runtime_id": _normalize_text(getattr(args, "runtime_id", "")) or None,
            "target_profile_id": _normalize_text(getattr(args, "target", "")) or None,
        }
        if model_token.isdigit():
            payload["model_id"] = int(model_token)
        else:
            payload["model_key"] = model_token
        resp = client.request("POST", f"/projects/{project_id}/models/validate", json_body=payload)
        if as_json:
            _print_json(resp)
            return 0
        compatible = bool(resp.get("compatible"))
        score = float(resp.get("compatibility_score") or 0.0)
        print(f"Model Compatibility: {'PASS' if compatible else 'BLOCKED'} (score={score:.3f})")
        risky = [
            item for item in list(resp.get("why_risky") or [])
            if isinstance(item, dict)
        ]
        if risky:
            print("Risks:")
            for item in risky[:5]:
                sev = str(item.get("severity") or "warning").upper()
                msg = str(item.get("message") or "")
                print(f"- [{sev}] {msg}")
        actions = [str(item) for item in list(resp.get("recommended_next_actions") or []) if str(item).strip()]
        if actions:
            print("Recommended Actions:")
            for action in actions[:5]:
                print(f"- {action}")
        return 0 if compatible else 1

    raise ValueError(f"Unsupported models subcommand '{subcommand}'.")


# ---------------------------------------------------------------------------
# Autopilot command group (priority.md P4 — plan / run / repair / rollback /
# decisions). Builds on the P1 decision-log API, P2 rollback, and P3
# repair-preview/apply endpoints.
# ---------------------------------------------------------------------------


_AUTOPILOT_PLAN_PROFILES = (
    "safe",
    "balanced",
    "max_quality",
    "fastest",
    "best_quality",
)


def _build_orchestrate_body(args: argparse.Namespace) -> dict[str, Any]:
    """Turn shared `plan` / `run` CLI flags into an orchestrate request body."""
    body: dict[str, Any] = {
        "intent": _normalize_text(getattr(args, "intent", "")) or DEFAULT_INTENT,
        "target_profile_id": _normalize_text(getattr(args, "target_profile_id", "")) or "vllm_server",
        "primary_language": _normalize_text(getattr(args, "primary_language", "")) or "english",
        "auto_prepare_data": not bool(getattr(args, "no_auto_prepare_data", False)),
        "auto_apply_rewrite": not bool(getattr(args, "no_auto_rewrite", False)),
        "allow_target_fallback": not bool(getattr(args, "no_target_fallback", False)),
        "allow_profile_autotune": not bool(getattr(args, "no_profile_autotune", False)),
        "plan_profile": _normalize_text(getattr(args, "plan_profile", "")) or "balanced",
    }
    target_device = _normalize_text(getattr(args, "target_device", ""))
    if target_device:
        body["target_device"] = target_device
    available_vram_gb = getattr(args, "available_vram_gb", None)
    if available_vram_gb is not None:
        body["available_vram_gb"] = float(available_vram_gb)
    base_model = _normalize_text(getattr(args, "base_model", ""))
    if base_model:
        body["base_model"] = base_model
    intent_rewrite = _normalize_text(getattr(args, "intent_rewrite", ""))
    if intent_rewrite:
        body["intent_rewrite"] = intent_rewrite
    run_name = _normalize_text(getattr(args, "run_name", ""))
    if run_name:
        body["run_name"] = run_name
    description = _normalize_text(getattr(args, "description", ""))
    if description:
        body["description"] = description
    return body


def _render_autopilot_plan_summary(payload: dict[str, Any]) -> None:
    preview = dict(payload.get("preview") or {})
    diff = dict(payload.get("config_diff") or {})
    guardrails = dict(diff.get("guardrails") or {})
    token = str(preview.get("plan_token") or "")
    expires_at = str(preview.get("expires_at") or "")
    state_hash = str(payload.get("state_hash") or preview.get("state_hash") or "")

    print(f"Plan token: {token}")
    print(f"State hash: {state_hash}")
    if expires_at:
        print(f"Expires at: {expires_at}")
    summary = str(diff.get("summary") or "").strip()
    if summary:
        print(f"Summary: {summary}")
    can_run = bool(diff.get("would_create_experiment"))
    print(f"Would create experiment: {'yes' if can_run else 'no'}")
    profile = str(diff.get("selected_profile") or "").strip()
    if profile:
        print(f"Selected profile: {profile}")
    target = str(diff.get("effective_target_profile_id") or "").strip()
    if target:
        print(f"Target profile: {target}")

    repairs = [row for row in list(diff.get("repairs_planned") or []) if isinstance(row, dict)]
    if repairs:
        print("Repairs planned:")
        for entry in repairs:
            kind = str(entry.get("kind") or "").strip()
            marker = "applied" if entry.get("applied") or entry.get("succeeded") else "skipped"
            extras = []
            if entry.get("from_profile") and entry.get("to_profile"):
                extras.append(f"profile {entry['from_profile']} -> {entry['to_profile']}")
            if entry.get("from_target_profile_id") and entry.get("to_target_profile_id"):
                extras.append(
                    f"target {entry['from_target_profile_id']} -> {entry['to_target_profile_id']}"
                )
            tail = f" ({', '.join(extras)})" if extras else ""
            print(f"  - {kind} [{marker}]{tail}")

    blockers = [str(x).strip() for x in list(guardrails.get("blockers") or []) if str(x).strip()]
    warnings = [str(x).strip() for x in list(guardrails.get("warnings") or []) if str(x).strip()]
    print(f"Guardrails: {len(blockers)} blocker(s), {len(warnings)} warning(s)")
    for blocker in blockers[:3]:
        print(f"  [BLOCKER] {blocker}")
    for warning in warnings[:3]:
        print(f"  [WARN] {warning}")


def _render_autopilot_run_summary(response: dict[str, Any]) -> None:
    run_id = str(response.get("run_id") or "")
    started = bool(response.get("started"))
    experiment = dict(response.get("experiment") or {})
    dry_run = bool(response.get("dry_run"))
    print(f"Run ID: {run_id}")
    print(f"Dry run: {'yes' if dry_run else 'no'}")
    print(f"Training started: {'yes' if started else 'no'}")
    if experiment:
        print(f"Experiment id: {experiment.get('id')}")
        print(f"Experiment name: {experiment.get('name')}")
    _print_autopilot_v2_decision_log(response)


def _autopilot_plan(args: argparse.Namespace, client: ApiClient, *, as_json: bool) -> int:
    project_id = int(getattr(args, "project_id", 0) or 0)
    if project_id <= 0:
        raise ValueError("--project is required for autopilot plan.")
    body = _build_orchestrate_body(args)
    body["project_id"] = project_id
    payload = client.request("POST", "/autopilot/repair-preview", json_body=body)
    if as_json:
        _print_json(payload)
    else:
        _render_autopilot_plan_summary(payload)
    diff = dict(dict(payload or {}).get("config_diff") or {})
    return 0 if bool(diff.get("would_create_experiment")) else 1


def _autopilot_run(args: argparse.Namespace, client: ApiClient, *, as_json: bool) -> int:
    project_id = int(getattr(args, "project_id", 0) or 0)
    if project_id <= 0:
        raise ValueError("--project is required for autopilot run.")
    body = _build_orchestrate_body(args)
    body["dry_run"] = False
    response = client.request(
        "POST",
        f"/projects/{project_id}/training/autopilot/v2/orchestrate/run",
        json_body=body,
    )
    if as_json:
        _print_json(response)
    else:
        _render_autopilot_run_summary(response)
    return 0 if bool(response.get("started")) else 1


def _autopilot_repair(args: argparse.Namespace, client: ApiClient, *, as_json: bool) -> int:
    token = _normalize_text(getattr(args, "plan_token", ""))
    if not token:
        raise ValueError("--plan-token is required for autopilot repair.")
    body: dict[str, Any] = {"plan_token": token, "force": bool(getattr(args, "force", False))}
    actor = _normalize_text(getattr(args, "actor", ""))
    if actor:
        body["actor"] = actor
    reason = _normalize_text(getattr(args, "reason", ""))
    if reason:
        body["reason"] = reason
    expected_hash = _normalize_text(getattr(args, "expected_state_hash", ""))
    if expected_hash:
        body["expected_state_hash"] = expected_hash
    payload = client.request("POST", "/autopilot/repair-apply", json_body=body)
    if as_json:
        _print_json(payload)
        return 0 if bool(payload.get("ok")) else 1
    response = dict(payload.get("response") or {})
    preview = dict(payload.get("preview") or {})
    print(f"Applied plan token: {preview.get('plan_token')}")
    if preview.get("applied_at"):
        print(f"Applied at: {preview['applied_at']}")
    _render_autopilot_run_summary(response)
    return 0 if bool(payload.get("ok")) and bool(response.get("started")) else 1


def _autopilot_rollback(args: argparse.Namespace, client: ApiClient, *, as_json: bool) -> int:
    decision_id = int(getattr(args, "decision_id", 0) or 0)
    if decision_id <= 0:
        raise ValueError("decision_id must be a positive integer.")

    if bool(getattr(args, "preview", False)):
        payload = client.request(
            "POST", f"/autopilot/rollback/{decision_id}/preview"
        )
        if as_json:
            _print_json(payload)
        else:
            reversible = bool(payload.get("reversible"))
            print(f"Reversible: {'yes' if reversible else 'no'}")
            if not reversible:
                print(f"Reason: {payload.get('reason')}")
                print(f"Detail: {payload.get('message') or ''}")
            steps = [row for row in list(payload.get("steps") or []) if isinstance(row, dict)]
            if steps:
                print(f"Steps ({len(steps)}):")
                for step in steps:
                    kind = str(step.get("kind") or "").strip()
                    extras = {k: v for k, v in step.items() if k != "kind"}
                    print(f"  - {kind}: {json.dumps(extras, ensure_ascii=True)}")
        return 0 if bool(payload.get("reversible")) else 1

    body: dict[str, Any] = {}
    actor = _normalize_text(getattr(args, "actor", ""))
    if actor:
        body["actor"] = actor
    reason = _normalize_text(getattr(args, "reason", ""))
    if reason:
        body["reason"] = reason
    payload = client.request(
        "POST",
        f"/autopilot/rollback/{decision_id}",
        json_body=body,
    )
    if as_json:
        _print_json(payload)
        return 0 if bool(payload.get("ok")) else 1
    ok = bool(payload.get("ok"))
    rollback_decision = dict(payload.get("rollback_decision") or {})
    print(f"Rollback succeeded: {'yes' if ok else 'no'}")
    if rollback_decision:
        print(f"Rollback decision id: {rollback_decision.get('id')}")
        print(f"Rollback stage: {rollback_decision.get('stage')} / {rollback_decision.get('action')}")
    outcomes = [row for row in list(payload.get("outcomes") or []) if isinstance(row, dict)]
    if outcomes:
        print(f"Outcomes ({len(outcomes)}):")
        for row in outcomes:
            kind = str(row.get("kind") or "").strip()
            status = str(row.get("status") or "").strip()
            message = str(row.get("message") or "").strip()
            tail = f": {message}" if message else ""
            print(f"  - {kind}: {status}{tail}")
    return 0 if ok else 1


def _autopilot_decisions(args: argparse.Namespace, client: ApiClient, *, as_json: bool) -> int:
    params: dict[str, Any] = {}
    if getattr(args, "project_id", None):
        params["project_id"] = int(args.project_id)
    for key, attr in (
        ("run_id", "run_id"),
        ("stage", "stage"),
        ("status", "status"),
        ("action", "action"),
        ("reason_code", "reason_code"),
        ("since", "since"),
        ("until", "until"),
    ):
        value = _normalize_text(getattr(args, attr, ""))
        if value:
            params[key] = value
    params["limit"] = int(getattr(args, "limit", 100) or 100)
    params["offset"] = int(getattr(args, "offset", 0) or 0)

    payload = client.request("GET", "/autopilot/decisions", params=params)
    if as_json:
        _print_json(payload)
        return 0
    items = [row for row in list(payload.get("items") or []) if isinstance(row, dict)]
    print(f"Autopilot decisions: {len(items)} returned")
    if not items:
        return 0
    print(f"{'ID':<6} {'Run':<10} {'Seq':<4} {'Stage':<26} {'Status':<12} {'Action':<10} {'Actor'}")
    print("-" * 90)
    for row in items:
        rid = str(row.get("id") or "")
        run_token = str(row.get("run_id") or "")[:10]
        seq = str(row.get("sequence") or 0)[:4]
        stage = str(row.get("stage") or "")[:26]
        status = str(row.get("status") or "")[:12]
        action = str(row.get("action") or "")[:10]
        actor = str(row.get("actor") or "")
        print(f"{rid:<6} {run_token:<10} {seq:<4} {stage:<26} {status:<12} {action:<10} {actor}")
    return 0


def run_autopilot(args: argparse.Namespace, client: ApiClient) -> int:
    subcommand = str(getattr(args, "autopilot_subcommand", "") or "").strip().lower()
    as_json = bool(getattr(args, "json", False))
    if subcommand == "plan":
        return _autopilot_plan(args, client, as_json=as_json)
    if subcommand == "run":
        return _autopilot_run(args, client, as_json=as_json)
    if subcommand == "repair":
        return _autopilot_repair(args, client, as_json=as_json)
    if subcommand == "rollback":
        return _autopilot_rollback(args, client, as_json=as_json)
    if subcommand == "decisions":
        return _autopilot_decisions(args, client, as_json=as_json)
    raise ValueError(f"Unsupported autopilot subcommand '{subcommand}'.")


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

    p_bootstrap = project_sub.add_parser(
        "bootstrap",
        help="Analyze a plain-English brief and bootstrap a Domain Blueprint workflow.",
    )
    p_bootstrap.add_argument("--name", default="", help="Project name (optional with --create-project)")
    p_bootstrap.add_argument("--description", default="", help="Project description override")
    p_bootstrap.add_argument("--base-model", default="", help="Optional base model override for project creation")
    p_bootstrap.add_argument("--project", "--project-id", dest="project_id", type=int, default=None)
    p_bootstrap.add_argument("--brief", default="", help="Plain-English brief text")
    p_bootstrap.add_argument("--brief-file", default="", help="Path to plain-text brief file")
    p_bootstrap.add_argument(
        "--sample-input",
        action="append",
        default=[],
        help="Representative sample input (repeatable)",
    )
    p_bootstrap.add_argument(
        "--sample-output",
        action="append",
        default=[],
        help="Representative sample output (repeatable)",
    )
    p_bootstrap.add_argument("--sample-inputs-file", default="", help="Path to newline-delimited sample inputs")
    p_bootstrap.add_argument("--sample-outputs-file", default="", help="Path to newline-delimited sample outputs")
    p_bootstrap.add_argument(
        "--risk-note",
        action="append",
        default=[],
        help="Safety/compliance note (repeatable)",
    )
    p_bootstrap.add_argument(
        "--target",
        default="",
        help="Deployment target hint for blueprint analysis (for example edge_gpu, mobile_cpu, vllm_server).",
    )
    p_bootstrap.add_argument(
        "--target-profile-id",
        default="",
        help="Project target profile id when using --create-project.",
    )
    p_bootstrap.add_argument("--create-project", action="store_true", help="Create a new project from analyzed blueprint")
    p_bootstrap.add_argument("--apply", action="store_true", help="Apply saved blueprint when targeting existing project")
    p_bootstrap.add_argument("--no-llm-enrich", action="store_true", help="Force deterministic-only analysis path")

    p_blueprint = project_sub.add_parser(
        "blueprint",
        help="Inspect project Domain Blueprint revisions.",
    )
    p_blueprint_sub = p_blueprint.add_subparsers(dest="blueprint_subcommand", required=True)

    p_blueprint_show = p_blueprint_sub.add_parser("show", help="Show latest/list/specific blueprint revision")
    p_blueprint_show.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    p_blueprint_show.add_argument("--version", type=int, default=None, help="Specific version to fetch")
    p_blueprint_show.add_argument("--latest", action="store_true", help="Fetch latest revision")

    p_blueprint_diff = p_blueprint_sub.add_parser("diff", help="Diff two blueprint revisions")
    p_blueprint_diff.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    p_blueprint_diff.add_argument("--from-version", type=int, required=True)
    p_blueprint_diff.add_argument("--to-version", type=int, required=True)

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

    dataset_profile = dataset_sub.add_parser("profile", help="Profile dataset structure and sensitive columns")
    dataset_profile.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    dataset_profile.add_argument(
        "--source-type",
        default="project_dataset",
        help=(
            "Source type: project_dataset, csv, tsv, json, jsonl, parquet, huggingface, "
            "sql_snapshot, document_corpus, chunk_corpus, chat_transcripts, pairwise_preference"
        ),
    )
    dataset_profile.add_argument("--source-ref", default="", help="Source path/ref (required for non-project_dataset)")
    dataset_profile.add_argument("--dataset-type", default="raw", help="Project dataset type when source-type=project_dataset")
    dataset_profile.add_argument("--document-id", type=int, default=None, help="Optional raw document id for project_dataset")
    dataset_profile.add_argument("--split", default="", help="Dataset split (Hugging Face only)")
    dataset_profile.add_argument("--sample-size", type=int, default=500)
    dataset_profile.add_argument("--json", action="store_true", help="Emit JSON output")
    dataset_parser.set_defaults(func=run_dataset)

    adapter_parser = subparsers.add_parser("adapter", help="Dataset Adapter Studio workflows")
    adapter_sub = adapter_parser.add_subparsers(dest="adapter_subcommand", required=True)

    adapter_infer = adapter_sub.add_parser("infer", help="Infer adapter definition from source structure")
    adapter_infer.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    adapter_infer.add_argument("--source-type", default="project_dataset")
    adapter_infer.add_argument("--source-ref", default="")
    adapter_infer.add_argument("--dataset-type", default="raw")
    adapter_infer.add_argument("--document-id", type=int, default=None)
    adapter_infer.add_argument("--split", default="")
    adapter_infer.add_argument("--task-profile", default="")
    adapter_infer.add_argument("--sample-size", type=int, default=400)
    adapter_infer.add_argument("--json", action="store_true", help="Emit JSON output")

    adapter_preview = adapter_sub.add_parser("preview", help="Preview transformed rows with adapter config")
    adapter_preview.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    adapter_preview.add_argument("--source-type", default="project_dataset")
    adapter_preview.add_argument("--source-ref", default="")
    adapter_preview.add_argument("--dataset-type", default="raw")
    adapter_preview.add_argument("--document-id", type=int, default=None)
    adapter_preview.add_argument("--split", default="")
    adapter_preview.add_argument("--adapter-id", default="auto")
    adapter_preview.add_argument("--field-mapping", default="", help="JSON object canonical->source mapping")
    adapter_preview.add_argument("--adapter-config", default="", help="JSON object adapter config")
    adapter_preview.add_argument("--task-profile", default="")
    adapter_preview.add_argument("--sample-size", type=int, default=300)
    adapter_preview.add_argument("--preview-limit", type=int, default=25)
    adapter_preview.add_argument("--json", action="store_true", help="Emit JSON output")

    adapter_validate = adapter_sub.add_parser("validate", help="Validate adapter coverage against sampled rows")
    adapter_validate.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    adapter_validate.add_argument("--source-type", default="project_dataset")
    adapter_validate.add_argument("--source-ref", default="")
    adapter_validate.add_argument("--dataset-type", default="raw")
    adapter_validate.add_argument("--document-id", type=int, default=None)
    adapter_validate.add_argument("--split", default="")
    adapter_validate.add_argument("--adapter-id", default="auto")
    adapter_validate.add_argument("--field-mapping", default="", help="JSON object canonical->source mapping")
    adapter_validate.add_argument("--adapter-config", default="", help="JSON object adapter config")
    adapter_validate.add_argument("--task-profile", default="")
    adapter_validate.add_argument("--sample-size", type=int, default=300)
    adapter_validate.add_argument("--preview-limit", type=int, default=25)
    adapter_validate.add_argument("--json", action="store_true", help="Emit JSON output")

    adapter_export = adapter_sub.add_parser("export", help="Export saved adapter definition as plugin/template scaffold")
    adapter_export.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    adapter_export.add_argument("--adapter-name", required=True)
    adapter_export.add_argument("--version", type=int, required=True)
    adapter_export.add_argument("--export-dir", default="", help="Optional output directory")
    adapter_export.add_argument("--json", action="store_true", help="Emit JSON output")
    adapter_parser.set_defaults(func=run_adapter)

    models_parser = subparsers.add_parser(
        "models",
        help="Universal base model registry and compatibility commands",
    )
    models_sub = models_parser.add_subparsers(dest="models_subcommand", required=True)

    models_import = models_sub.add_parser("import", help="Import model metadata into universal registry")
    models_import.add_argument("--hf-id", default="", help="Hugging Face model id")
    models_import.add_argument("--path", default="", help="Local model directory/config path")
    models_import.add_argument("--catalog-id", default="", help="Internal catalog model id")
    models_import.add_argument("--allow-network", action="store_true", help="Allow network metadata enrichment")
    models_import.add_argument("--no-overwrite", action="store_true", help="Do not overwrite an existing model_key")
    models_import.add_argument("--json", action="store_true", help="Emit JSON output")

    models_refresh = models_sub.add_parser("refresh", help="Refresh imported model metadata")
    models_refresh.add_argument("--model", required=True, help="Model id or model_key")
    models_refresh.add_argument("--allow-network", action="store_true", help="Allow network metadata refresh")
    models_refresh.add_argument("--json", action="store_true", help="Emit JSON output")

    models_list = models_sub.add_parser("list", help="List imported base models")
    models_list.add_argument("--family", default="", help="Family filter")
    models_list.add_argument("--license", default="", help="License token filter")
    models_list.add_argument(
        "--hardware-fit",
        default="",
        help="Hardware fit filter (mobile, laptop, server)",
    )
    models_list.add_argument("--min-context-length", type=int, default=None)
    models_list.add_argument("--max-params-b", type=float, default=None)
    models_list.add_argument("--training-mode", default="", help="Training mode filter")
    models_list.add_argument("--search", default="", help="Free-text search")
    models_list.add_argument("--json", action="store_true", help="Emit JSON output")

    models_recommend = models_sub.add_parser("recommend", help="List project-compatible base models")
    models_recommend.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    models_recommend.add_argument("--limit", type=int, default=10)
    models_recommend.add_argument("--include-incompatible", action="store_true")
    models_recommend.add_argument("--family", default="", help="Family filter")
    models_recommend.add_argument("--license", default="", help="License token filter")
    models_recommend.add_argument(
        "--hardware-fit",
        default="",
        help="Hardware fit filter (mobile, laptop, server)",
    )
    models_recommend.add_argument("--min-context-length", type=int, default=None)
    models_recommend.add_argument("--max-params-b", type=float, default=None)
    models_recommend.add_argument("--training-mode", default="", help="Training mode filter")
    models_recommend.add_argument("--search", default="", help="Free-text search")
    models_recommend.add_argument("--target", default="", help="Override project target profile id")
    models_recommend.add_argument("--runtime-id", default="", help="Override runtime id")
    models_recommend.add_argument("--adapter-id", default="", help="Override dataset adapter id")
    models_recommend.add_argument("--allow-network", action="store_true", help="Allow network checks")
    models_recommend.add_argument("--json", action="store_true", help="Emit JSON output")

    models_validate = models_sub.add_parser("validate", help="Validate one model against project compatibility")
    models_validate.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
    models_validate.add_argument("--model", required=True, help="Model id or model_key")
    models_validate.add_argument("--target", default="", help="Override target profile id")
    models_validate.add_argument("--runtime-id", default="", help="Override runtime id")
    models_validate.add_argument("--adapter-id", default="", help="Override dataset adapter id")
    models_validate.add_argument("--allow-network", action="store_true", help="Allow network checks")
    models_validate.add_argument("--json", action="store_true", help="Emit JSON output")
    models_parser.set_defaults(func=run_models)

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
    train_parser.add_argument(
        "--autopilot-v2",
        action="store_true",
        help="Use autopilot v2 orchestration (readiness + auto-repair + decision log).",
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use autopilot v2 dry-run mode (no experiment is started).",
    )
    train_parser.add_argument(
        "--plan-profile",
        default="balanced",
        choices=["safe", "balanced", "max_quality", "fastest", "best_quality"],
        help="Preferred training plan profile for autopilot selection.",
    )
    train_parser.add_argument(
        "--no-target-fallback",
        action="store_true",
        help="Disable automatic target-profile fallback during autopilot v2 orchestration.",
    )
    train_parser.add_argument(
        "--no-profile-autotune",
        action="store_true",
        help="Disable automatic profile tuning to the first runnable preflight plan.",
    )
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
        "optimize",
        help="Rank export candidates with measured vs estimated metrics",
        description=(
            "Evaluate project artifacts for a deployment target and print metric provenance "
            "(measured, mixed, or estimated) for each candidate."
        ),
    )
    optimize_parser.add_argument(
        "--project", "--project-id", dest="project_id", type=int, required=True
    )
    optimize_parser.add_argument(
        "--target",
        required=True,
        help="Target profile id or alias (for example: mobile_cpu, edge_gpu, vllm_server).",
    )
    optimize_parser.set_defaults(func=run_optimize)

    # ------------------------------------------------------------------
    # Autopilot command group (P4).
    # ------------------------------------------------------------------
    autopilot_parser = subparsers.add_parser(
        "autopilot",
        help="Autopilot plan/run/repair/rollback/decisions commands.",
    )
    autopilot_sub = autopilot_parser.add_subparsers(
        dest="autopilot_subcommand", required=True
    )

    def _add_orchestrate_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--project", "--project-id", dest="project_id", type=int, required=True)
        p.add_argument("--intent", default=DEFAULT_INTENT, help="Plain-language training intent.")
        p.add_argument(
            "--target-profile",
            dest="target_profile_id",
            default="vllm_server",
            help="Target profile id (e.g. mobile_cpu, edge_gpu, vllm_server).",
        )
        p.add_argument(
            "--target-device",
            dest="target_device",
            default="",
            choices=["", "mobile", "laptop", "server"],
            help="Target device hint for autopilot planning.",
        )
        p.add_argument("--primary-language", default="english")
        p.add_argument("--available-vram-gb", type=float, default=None)
        p.add_argument("--base-model", default="", help="Optional explicit base model override.")
        p.add_argument(
            "--plan-profile",
            default="balanced",
            choices=list(_AUTOPILOT_PLAN_PROFILES),
        )
        p.add_argument("--intent-rewrite", default="", help="Optional replacement intent text.")
        p.add_argument(
            "--no-auto-rewrite",
            action="store_true",
            help="Disable autopilot auto-applied intent rewrites.",
        )
        p.add_argument(
            "--no-target-fallback",
            action="store_true",
            help="Disable automatic target-profile fallback.",
        )
        p.add_argument(
            "--no-profile-autotune",
            action="store_true",
            help="Disable automatic profile tuning.",
        )
        p.add_argument(
            "--no-auto-prepare-data",
            action="store_true",
            help="Disable autopilot dataset auto-prepare step.",
        )
        p.add_argument("--run-name", default="", help="Optional experiment name override.")
        p.add_argument("--description", default="", help="Optional experiment description override.")
        p.add_argument("--json", action="store_true", help="Emit raw JSON instead of a human summary.")

    plan_parser = autopilot_sub.add_parser(
        "plan",
        help="Dry-run autopilot and return a plan_token (alias for /autopilot/repair-preview).",
    )
    _add_orchestrate_args(plan_parser)

    run_parser = autopilot_sub.add_parser(
        "run",
        help="Execute autopilot end-to-end in one shot (no persisted preview).",
    )
    _add_orchestrate_args(run_parser)

    repair_parser = autopilot_sub.add_parser(
        "repair",
        help="Apply a previously issued plan_token via /autopilot/repair-apply.",
    )
    repair_parser.add_argument("--plan-token", dest="plan_token", required=True)
    repair_parser.add_argument("--actor", default="", help="Actor recorded with the apply.")
    repair_parser.add_argument("--reason", default="", help="Reason recorded with the apply.")
    repair_parser.add_argument(
        "--expected-state-hash",
        dest="expected_state_hash",
        default="",
        help="Reject if current preview state_hash differs from this value.",
    )
    repair_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip state-drift detection (advanced).",
    )
    repair_parser.add_argument("--json", action="store_true", help="Emit raw JSON.")

    rollback_parser = autopilot_sub.add_parser(
        "rollback",
        help="Roll back an autopilot decision by id (or preview the rollback).",
    )
    rollback_parser.add_argument("decision_id", type=int, help="autopilot_decisions.id")
    rollback_parser.add_argument(
        "--preview",
        action="store_true",
        help="Dry-run: return what the rollback would do without mutating.",
    )
    rollback_parser.add_argument("--actor", default="", help="Actor recorded with the rollback.")
    rollback_parser.add_argument("--reason", default="", help="Reason recorded with the rollback.")
    rollback_parser.add_argument("--json", action="store_true", help="Emit raw JSON.")

    decisions_parser = autopilot_sub.add_parser(
        "decisions",
        help="List persisted autopilot decisions (P1 /autopilot/decisions).",
    )
    decisions_parser.add_argument(
        "--project", "--project-id", dest="project_id", type=int, default=None
    )
    decisions_parser.add_argument("--run-id", dest="run_id", default="", help="Filter by run id.")
    decisions_parser.add_argument("--stage", default="", help="Filter by stage.")
    decisions_parser.add_argument("--status", default="", help="Filter by status.")
    decisions_parser.add_argument("--action", default="", help="Filter by derived action.")
    decisions_parser.add_argument(
        "--reason-code",
        dest="reason_code",
        default="",
        help="Filter by reason code.",
    )
    decisions_parser.add_argument("--since", default="", help="created_at >= ISO-8601 timestamp.")
    decisions_parser.add_argument("--until", default="", help="created_at <= ISO-8601 timestamp.")
    decisions_parser.add_argument("--limit", type=int, default=100)
    decisions_parser.add_argument("--offset", type=int, default=0)
    decisions_parser.add_argument("--json", action="store_true", help="Emit raw JSON.")

    for sub in (plan_parser, run_parser, repair_parser, rollback_parser, decisions_parser):
        sub.set_defaults(func=run_autopilot)

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
