"""Deployment Target SDK: artifact/runnable target validation + smoke checks.

This module defines first-class exporter/runner targets and validates whether
an export bundle is truly deployable (artifact-accurate) for the requested
format/runtime.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.export import ExportFormat


DEPLOYMENT_TARGET_CONTRACT_VERSION = "slm.deployment-target/v1"


_TARGET_CATALOG: list[dict[str, Any]] = [
    {
        "target_id": "exporter.huggingface",
        "kind": "exporter",
        "display_name": "HuggingFace Exporter",
        "description": "Transformer-compatible model directory export.",
        "artifact_profiles": ["huggingface"],
        "supported_export_formats": ["huggingface", "docker"],
        "smoke_supported": False,
        "launch_example": "python serve.py --model ./model",
    },
    {
        "target_id": "exporter.gguf",
        "kind": "exporter",
        "display_name": "GGUF Exporter",
        "description": "llama.cpp-compatible GGUF artifact export.",
        "artifact_profiles": ["gguf"],
        "supported_export_formats": ["gguf"],
        "smoke_supported": False,
        "launch_example": "llama-cli -m ./model/model.gguf -p 'Hello'",
    },
    {
        "target_id": "exporter.onnx",
        "kind": "exporter",
        "display_name": "ONNX Exporter",
        "description": "ONNX runtime package export with model graph files.",
        "artifact_profiles": ["onnx"],
        "supported_export_formats": ["onnx"],
        "smoke_supported": False,
        "launch_example": "python serve.py --provider onnxruntime --model ./model",
    },
    {
        "target_id": "exporter.tensorrt",
        "kind": "exporter",
        "display_name": "TensorRT Exporter",
        "description": "TensorRT engine package export.",
        "artifact_profiles": ["tensorrt"],
        "supported_export_formats": ["tensorrt"],
        "smoke_supported": False,
        "launch_example": "trtexec --loadEngine=./model/model.engine --verbose",
    },
    {
        "target_id": "runner.vllm",
        "kind": "runner",
        "display_name": "vLLM Runner",
        "description": "High-throughput OpenAI-compatible serving for HF models.",
        "artifact_profiles": ["huggingface"],
        "supported_export_formats": ["huggingface", "docker"],
        "smoke_supported": True,
        "launch_example": "python -m vllm.entrypoints.openai.api_server --model ./model",
    },
    {
        "target_id": "runner.tgi",
        "kind": "runner",
        "display_name": "TGI Runner",
        "description": "Text Generation Inference runner for HF-compatible exports.",
        "artifact_profiles": ["huggingface"],
        "supported_export_formats": ["huggingface", "docker"],
        "smoke_supported": True,
        "launch_example": "text-generation-launcher --model-id ./model",
    },
    {
        "target_id": "runner.ollama",
        "kind": "runner",
        "display_name": "Ollama Runner",
        "description": "Local Ollama runtime for GGUF artifacts.",
        "artifact_profiles": ["gguf"],
        "supported_export_formats": ["gguf"],
        "smoke_supported": True,
        "launch_example": "ollama create my-model -f Modelfile && ollama run my-model",
    },
    {
        "target_id": "deployment.hf_inference_endpoint",
        "kind": "deployment",
        "display_name": "HuggingFace Inference Endpoint",
        "description": "Managed HTTPS endpoint deployment on Hugging Face.",
        "artifact_profiles": ["huggingface"],
        "supported_export_formats": ["huggingface", "docker"],
        "smoke_supported": False,
        "launch_example": "curl -X POST https://api.endpoints.huggingface.cloud/v2/endpoint/.../resume",
    },
    {
        "target_id": "deployment.aws_sagemaker",
        "kind": "deployment",
        "display_name": "AWS SageMaker Endpoint",
        "description": "Managed real-time inference endpoint on SageMaker.",
        "artifact_profiles": ["huggingface", "onnx", "tensorrt"],
        "supported_export_formats": ["huggingface", "onnx", "tensorrt", "docker"],
        "smoke_supported": False,
        "launch_example": "aws sagemaker create-endpoint --endpoint-name my-slm-endpoint ...",
    },
    {
        "target_id": "deployment.vllm_managed",
        "kind": "deployment",
        "display_name": "Managed vLLM API",
        "description": "Provision a managed vLLM instance and expose OpenAI-compatible API.",
        "artifact_profiles": ["huggingface", "gguf"],
        "supported_export_formats": ["huggingface", "gguf", "docker"],
        "smoke_supported": False,
        "launch_example": "curl -X POST https://managed-vllm.example.com/provision ...",
    },
    {
        "target_id": "sdk.apple_coreml_stub",
        "kind": "sdk",
        "display_name": "Apple CoreML Stub App",
        "description": "Generate iOS starter app scaffold to load exported model.",
        "artifact_profiles": ["huggingface", "onnx", "gguf"],
        "supported_export_formats": ["huggingface", "onnx", "gguf", "docker"],
        "smoke_supported": False,
        "launch_example": "Open iOS stub project in Xcode and run on device.",
    },
    {
        "target_id": "sdk.android_executorch_stub",
        "kind": "sdk",
        "display_name": "Android ExecuTorch Stub App",
        "description": "Generate Android starter app scaffold for edge runtime integration.",
        "artifact_profiles": ["huggingface", "onnx", "gguf"],
        "supported_export_formats": ["huggingface", "onnx", "gguf", "docker"],
        "smoke_supported": False,
        "launch_example": "Open Android stub project in Android Studio and run on device.",
    },
]


_ARTIFACT_PROFILE_BY_FORMAT: dict[str, str] = {
    "huggingface": "huggingface",
    "docker": "huggingface",
    "gguf": "gguf",
    "onnx": "onnx",
    "tensorrt": "tensorrt",
}


_DEFAULT_RUNNERS_BY_PROFILE: dict[str, list[str]] = {
    "huggingface": ["runner.vllm", "runner.tgi"],
    "gguf": ["runner.ollama"],
    "onnx": [],
    "tensorrt": [],
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lower()


def _target_by_id() -> dict[str, dict[str, Any]]:
    return {
        str(item.get("target_id") or "").strip().lower(): item
        for item in _TARGET_CATALOG
        if str(item.get("target_id") or "").strip()
    }


def resolve_artifact_profile(export_format: ExportFormat | str) -> str:
    token = _normalize_token(export_format.value if isinstance(export_format, ExportFormat) else str(export_format))
    return _ARTIFACT_PROFILE_BY_FORMAT.get(token, token or "unknown")


def default_deployment_targets_for_format(export_format: ExportFormat | str) -> list[str]:
    token = _normalize_token(export_format.value if isinstance(export_format, ExportFormat) else str(export_format))
    profile = resolve_artifact_profile(token)
    exporter = f"exporter.{profile}"
    targets = [exporter]
    targets.extend(_DEFAULT_RUNNERS_BY_PROFILE.get(profile, []))
    return [item for item in targets if item in _target_by_id()]


def list_deployment_targets(
    *,
    export_format: ExportFormat | str | None = None,
) -> dict[str, Any]:
    profile = resolve_artifact_profile(export_format) if export_format is not None else None
    format_token = _normalize_token(export_format.value if isinstance(export_format, ExportFormat) else str(export_format or ""))

    rows: list[dict[str, Any]] = []
    for target in _TARGET_CATALOG:
        target_id = str(target.get("target_id") or "")
        artifact_profiles = [str(item) for item in list(target.get("artifact_profiles") or []) if str(item).strip()]
        supported_formats = [str(item) for item in list(target.get("supported_export_formats") or []) if str(item).strip()]
        compatible = True
        if profile is not None:
            compatible = profile in artifact_profiles
        if compatible and format_token:
            compatible = format_token in supported_formats
        rows.append(
            {
                "target_id": target_id,
                "kind": str(target.get("kind") or ""),
                "display_name": str(target.get("display_name") or target_id),
                "description": str(target.get("description") or ""),
                "artifact_profiles": artifact_profiles,
                "supported_export_formats": supported_formats,
                "smoke_supported": bool(target.get("smoke_supported")),
                "launch_example": str(target.get("launch_example") or ""),
                "compatible": compatible,
            }
        )

    return {
        "contract_version": DEPLOYMENT_TARGET_CONTRACT_VERSION,
        "export_format": format_token or None,
        "artifact_profile": profile,
        "default_target_ids": default_deployment_targets_for_format(export_format) if export_format is not None else [],
        "targets": rows,
    }


def _check_result(
    *,
    check_id: str,
    passed: bool,
    severity: str = "error",
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = "pass" if passed else ("warn" if severity == "warning" else "fail")
    return {
        "check_id": check_id,
        "passed": passed,
        "status": status,
        "severity": severity,
        "message": message,
        "details": details or {},
    }


def _summarize_checks(checks: list[dict[str, Any]]) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for item in checks:
        if bool(item.get("passed")):
            continue
        text = str(item.get("message") or "").strip()
        if str(item.get("severity") or "error") == "warning":
            if text:
                warnings.append(text)
        else:
            if text:
                errors.append(text)
    return not errors, errors, warnings


def _model_dir(run_dir: Path) -> Path:
    return run_dir / "model"


def _all_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [item for item in root.rglob("*") if item.is_file()]


def _files_with_suffix(files: list[Path], suffixes: set[str]) -> list[Path]:
    allowed = {item.lower() for item in suffixes}
    return [item for item in files if item.suffix.lower() in allowed]


def validate_export_artifacts(
    *,
    run_dir: Path,
    export_format: ExportFormat | str,
) -> dict[str, Any]:
    """Validate that export artifacts match requested export format profile."""
    profile = resolve_artifact_profile(export_format)
    checks: list[dict[str, Any]] = []
    model_dir = _model_dir(run_dir)
    files = _all_files(model_dir)
    model_bytes = sum(item.stat().st_size for item in files) if files else 0

    checks.append(
        _check_result(
            check_id="model_dir_exists",
            passed=model_dir.exists() and model_dir.is_dir(),
            message=f"model directory must exist at '{model_dir}'.",
        )
    )
    checks.append(
        _check_result(
            check_id="model_files_present",
            passed=bool(files),
            message="model directory must contain at least one file.",
            details={"file_count": len(files)},
        )
    )
    checks.append(
        _check_result(
            check_id="model_bytes_positive",
            passed=model_bytes > 0,
            message="model artifact bytes must be > 0.",
            details={"model_bytes": model_bytes},
        )
    )

    if profile == "huggingface":
        config_file = model_dir / "config.json"
        weight_files = _files_with_suffix(files, {".safetensors", ".bin", ".pt"})
        tokenizer_files = _files_with_suffix(files, {".model", ".json", ".txt"})
        checks.append(
            _check_result(
                check_id="hf_config_present",
                passed=config_file.exists() and config_file.is_file(),
                message="HuggingFace export requires model/config.json.",
            )
        )
        checks.append(
            _check_result(
                check_id="hf_weights_present",
                passed=bool(weight_files),
                message="HuggingFace export requires at least one weights file (.safetensors/.bin/.pt).",
                details={"weights_count": len(weight_files)},
            )
        )
        checks.append(
            _check_result(
                check_id="hf_tokenizer_present",
                passed=bool(tokenizer_files),
                severity="warning",
                message="Tokenizer metadata is recommended for serving compatibility.",
                details={"tokenizer_like_files": len(tokenizer_files)},
            )
        )
    elif profile == "gguf":
        gguf_files = _files_with_suffix(files, {".gguf"})
        checks.append(
            _check_result(
                check_id="gguf_file_present",
                passed=bool(gguf_files),
                message="GGUF export requires at least one .gguf file under model/.",
                details={"gguf_count": len(gguf_files)},
            )
        )
        if gguf_files:
            checks.append(
                _check_result(
                    check_id="gguf_nontrivial_size",
                    passed=any(item.stat().st_size > 1024 for item in gguf_files),
                    message="GGUF file size appears too small to be a real model artifact.",
                    details={"largest_gguf_bytes": max(item.stat().st_size for item in gguf_files)},
                )
            )
    elif profile == "onnx":
        onnx_files = _files_with_suffix(files, {".onnx"})
        checks.append(
            _check_result(
                check_id="onnx_file_present",
                passed=bool(onnx_files),
                message="ONNX export requires at least one .onnx file under model/.",
                details={"onnx_count": len(onnx_files)},
            )
        )
        checks.append(
            _check_result(
                check_id="onnx_config_present",
                passed=(model_dir / "config.json").exists(),
                severity="warning",
                message="ONNX export is recommended to include config.json.",
            )
        )
    elif profile == "tensorrt":
        engine_files = _files_with_suffix(files, {".engine", ".plan"})
        checks.append(
            _check_result(
                check_id="tensorrt_engine_present",
                passed=bool(engine_files),
                message="TensorRT export requires at least one .engine/.plan file under model/.",
                details={"engine_count": len(engine_files)},
            )
        )
        if engine_files:
            checks.append(
                _check_result(
                    check_id="tensorrt_nontrivial_size",
                    passed=any(item.stat().st_size > 1024 for item in engine_files),
                    message="TensorRT engine size appears too small to be a real deployable artifact.",
                    details={"largest_engine_bytes": max(item.stat().st_size for item in engine_files)},
                )
            )
    else:
        checks.append(
            _check_result(
                check_id="profile_supported",
                passed=False,
                message=f"Artifact profile '{profile}' is not supported by Deployment Target SDK.",
            )
        )

    passed, errors, warnings = _summarize_checks(checks)
    return {
        "artifact_profile": profile,
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "file_count": len(files),
        "model_bytes": model_bytes,
        "captured_at": _utcnow_iso(),
    }


def _command_exists(command_name: str) -> bool:
    return shutil.which(command_name) is not None


def _python_module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _run_smoke_command(cmd: list[str], timeout_seconds: int = 8) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        stderr = str(proc.stderr or "").strip()
        stdout = str(proc.stdout or "").strip()
        return False, (stderr or stdout or f"command exited with code {proc.returncode}")[:1000]
    return True, str(proc.stdout or "").strip()[:1000]


def _validate_runner_target(
    *,
    target: dict[str, Any],
    artifact_profile: str,
    run_smoke_tests: bool,
) -> dict[str, Any]:
    target_id = str(target.get("target_id") or "")
    checks: list[dict[str, Any]] = []
    compatible = artifact_profile in [str(item) for item in list(target.get("artifact_profiles") or [])]

    checks.append(
        _check_result(
            check_id="artifact_profile_compatible",
            passed=compatible,
            message=(
                f"Target '{target_id}' is not compatible with artifact profile '{artifact_profile}'."
                if not compatible
                else f"Target '{target_id}' is compatible with artifact profile '{artifact_profile}'."
            ),
        )
    )

    smoke_supported = bool(target.get("smoke_supported"))
    smoke_executed = False
    smoke_passed: bool | None = None

    if target_id == "runner.vllm":
        module_ok = _python_module_exists("vllm")
        checks.append(
            _check_result(
                check_id="vllm_module_installed",
                passed=module_ok,
                message="Python module 'vllm' must be importable for local vLLM smoke tests.",
            )
        )
        if run_smoke_tests and smoke_supported and module_ok:
            smoke_executed = True
            smoke_passed, smoke_output = _run_smoke_command([sys.executable, "-c", "import vllm; print('ok')"])
            checks.append(
                _check_result(
                    check_id="vllm_smoke_import",
                    passed=bool(smoke_passed),
                    message="vLLM import smoke test failed." if not smoke_passed else "vLLM smoke import succeeded.",
                    details={"output": smoke_output},
                )
            )
    elif target_id == "runner.tgi":
        launcher_ok = _command_exists("text-generation-launcher")
        docker_ok = _command_exists("docker")
        checks.append(
            _check_result(
                check_id="tgi_runtime_present",
                passed=launcher_ok or docker_ok,
                message="TGI runtime requires `text-generation-launcher` or `docker` binary.",
                details={"launcher": launcher_ok, "docker": docker_ok},
            )
        )
        if run_smoke_tests and smoke_supported and (launcher_ok or docker_ok):
            smoke_executed = True
            if launcher_ok:
                smoke_passed, smoke_output = _run_smoke_command(["text-generation-launcher", "--help"])
            else:
                smoke_passed, smoke_output = _run_smoke_command(["docker", "--version"])
            checks.append(
                _check_result(
                    check_id="tgi_smoke_runtime",
                    passed=bool(smoke_passed),
                    message="TGI runtime smoke test failed." if not smoke_passed else "TGI runtime smoke test succeeded.",
                    details={"output": smoke_output},
                )
            )
    elif target_id == "runner.ollama":
        ollama_ok = _command_exists("ollama")
        checks.append(
            _check_result(
                check_id="ollama_binary_present",
                passed=ollama_ok,
                message="Ollama runner requires `ollama` binary in PATH.",
            )
        )
        if run_smoke_tests and smoke_supported and ollama_ok:
            smoke_executed = True
            smoke_passed, smoke_output = _run_smoke_command(["ollama", "--version"])
            checks.append(
                _check_result(
                    check_id="ollama_smoke_version",
                    passed=bool(smoke_passed),
                    message="Ollama smoke test failed." if not smoke_passed else "Ollama smoke test succeeded.",
                    details={"output": smoke_output},
                )
            )
    else:
        checks.append(
            _check_result(
                check_id="runner_supported",
                passed=False,
                message=f"Runner target '{target_id}' is not implemented.",
            )
        )

    passed, errors, warnings = _summarize_checks(checks)
    return {
        "target_id": target_id,
        "kind": "runner",
        "compatible": compatible,
        "passed": passed and compatible,
        "local_ready": passed and compatible,
        "smoke_supported": smoke_supported,
        "smoke_executed": smoke_executed,
        "smoke_passed": smoke_passed,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "launch_example": str(target.get("launch_example") or ""),
    }


def run_deployment_target_suite(
    *,
    run_dir: Path,
    export_format: ExportFormat | str,
    deployment_targets: list[str] | None = None,
    run_smoke_tests: bool = True,
) -> dict[str, Any]:
    """Validate export artifact deployability and runtime targets."""
    format_token = _normalize_token(export_format.value if isinstance(export_format, ExportFormat) else str(export_format))
    artifact_profile = resolve_artifact_profile(format_token)
    artifact_validation = validate_export_artifacts(
        run_dir=run_dir,
        export_format=format_token,
    )

    known_targets = _target_by_id()
    selected = [str(item).strip().lower() for item in list(deployment_targets or []) if str(item).strip()]
    if not selected:
        selected = default_deployment_targets_for_format(format_token)

    target_reports: list[dict[str, Any]] = []
    unknown_targets: list[str] = []

    for target_id in selected:
        target = known_targets.get(target_id)
        if target is None:
            unknown_targets.append(target_id)
            target_reports.append(
                {
                    "target_id": target_id,
                    "kind": "unknown",
                    "compatible": False,
                    "passed": False,
                    "local_ready": False,
                    "smoke_supported": False,
                    "smoke_executed": False,
                    "smoke_passed": None,
                    "checks": [
                        _check_result(
                            check_id="target_known",
                            passed=False,
                            message=f"Deployment target '{target_id}' is not registered.",
                        )
                    ],
                    "errors": [f"Deployment target '{target_id}' is not registered."],
                    "warnings": [],
                    "launch_example": "",
                }
            )
            continue

        kind = str(target.get("kind") or "")
        if kind == "exporter":
            compatible = artifact_profile in [str(item) for item in list(target.get("artifact_profiles") or [])]
            checks = [
                _check_result(
                    check_id="artifact_profile_compatible",
                    passed=compatible,
                    message=(
                        f"Target '{target_id}' is compatible with artifact profile '{artifact_profile}'."
                        if compatible
                        else f"Target '{target_id}' is not compatible with artifact profile '{artifact_profile}'."
                    ),
                ),
                _check_result(
                    check_id="artifact_validation_passed",
                    passed=bool(artifact_validation.get("passed")),
                    message=(
                        "Artifact validation passed for exporter target."
                        if bool(artifact_validation.get("passed"))
                        else "Artifact validation failed for exporter target."
                    ),
                ),
            ]
            passed, errors, warnings = _summarize_checks(checks)
            target_reports.append(
                {
                    "target_id": target_id,
                    "kind": "exporter",
                    "compatible": compatible,
                    "passed": bool(compatible and passed and artifact_validation.get("passed")),
                    "local_ready": bool(compatible and passed and artifact_validation.get("passed")),
                    "smoke_supported": False,
                    "smoke_executed": False,
                    "smoke_passed": None,
                    "checks": checks,
                    "errors": errors,
                    "warnings": warnings,
                    "launch_example": str(target.get("launch_example") or ""),
                }
            )
            continue

        target_reports.append(
            _validate_runner_target(
                target=target,
                artifact_profile=artifact_profile,
                run_smoke_tests=run_smoke_tests,
            )
        )

    exporter_reports = [item for item in target_reports if str(item.get("kind")) == "exporter"]
    runner_reports = [item for item in target_reports if str(item.get("kind")) == "runner"]
    compatible_runners = [item for item in runner_reports if bool(item.get("compatible"))]
    runner_local_ready = [item for item in compatible_runners if bool(item.get("local_ready"))]
    runner_smoke_success = [item for item in compatible_runners if item.get("smoke_passed") is True]

    deployable_artifact = bool(artifact_validation.get("passed"))
    if exporter_reports:
        deployable_artifact = deployable_artifact and all(bool(item.get("passed")) for item in exporter_reports)

    return {
        "contract_version": DEPLOYMENT_TARGET_CONTRACT_VERSION,
        "captured_at": _utcnow_iso(),
        "run_dir": str(run_dir),
        "export_format": format_token,
        "artifact_profile": artifact_profile,
        "selected_target_ids": selected,
        "unknown_target_ids": unknown_targets,
        "artifact_validation": artifact_validation,
        "target_reports": target_reports,
        "summary": {
            "deployable_artifact": deployable_artifact,
            "selected_target_count": len(selected),
            "compatible_runner_count": len(compatible_runners),
            "runner_local_ready_count": len(runner_local_ready),
            "runner_smoke_success_count": len(runner_smoke_success),
            "local_smoke_passed": (len(runner_smoke_success) > 0) if run_smoke_tests and compatible_runners else None,
        },
    }


def _safe_slug(value: str, *, fallback: str = "slm-endpoint") -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value or "").strip().lower())
    token = "-".join(part for part in token.split("-") if part)
    return token[:96] or fallback


def _write_mobile_stub_bundle(
    *,
    run_dir: Path,
    target_id: str,
    model_name: str,
) -> dict[str, Any]:
    base_dir = run_dir / "mobile_sdk" / _safe_slug(target_id, fallback="mobile-sdk")
    base_dir.mkdir(parents=True, exist_ok=True)

    if target_id == "sdk.apple_coreml_stub":
        readme = base_dir / "README.md"
        app_file = base_dir / "SLMStubApp.swift"
        readme.write_text(
            (
                "# iOS CoreML Stub App\n\n"
                f"Model: `{model_name}`\n\n"
                "1. Drag your exported model artifact into this Xcode project.\n"
                "2. Replace `ModelLoader.load()` with your CoreML model class.\n"
                "3. Run on iPhone/iPad and call `generate(prompt:)`.\n"
            ),
            encoding="utf-8",
        )
        app_file.write_text(
            (
                "import Foundation\n"
                "import CoreML\n\n"
                "final class ModelLoader {\n"
                "    func generate(prompt: String) -> String {\n"
                "        // TODO: wire CoreML model inference here.\n"
                "        return \"Stub response for: \\(prompt)\"\n"
                "    }\n"
                "}\n"
            ),
            encoding="utf-8",
        )
    else:
        readme = base_dir / "README.md"
        app_file = base_dir / "SLMStubApp.kt"
        readme.write_text(
            (
                "# Android ExecuTorch Stub App\n\n"
                f"Model: `{model_name}`\n\n"
                "1. Copy exported model artifact into `app/src/main/assets/`.\n"
                "2. Replace stub inference bridge with ExecuTorch runtime calls.\n"
                "3. Run on Android device and call `generate(prompt)`.\n"
            ),
            encoding="utf-8",
        )
        app_file.write_text(
            (
                "package com.example.slmstub\n\n"
                "class ModelLoader {\n"
                "    fun generate(prompt: String): String {\n"
                "        // TODO: wire ExecuTorch inference here.\n"
                "        return \"Stub response for: $prompt\"\n"
                "    }\n"
                "}\n"
            ),
            encoding="utf-8",
        )

    zip_path = base_dir / "stub_app.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in [readme, app_file]:
            archive.write(file_path, arcname=file_path.name)

    return {
        "bundle_dir": str(base_dir),
        "readme_path": str(readme),
        "entrypoint_path": str(app_file),
        "zip_path": str(zip_path),
    }


def build_deploy_target_plan(
    *,
    run_dir: Path,
    export_format: ExportFormat | str,
    target_id: str,
    model_name: str = "",
    endpoint_name: str | None = None,
    region: str | None = None,
    instance_type: str | None = None,
) -> dict[str, Any]:
    """Build actionable deploy plan for managed API targets and mobile SDK stubs."""
    known = _target_by_id()
    normalized_target_id = str(target_id or "").strip().lower()
    target = known.get(normalized_target_id)
    if target is None:
        raise ValueError(f"Deployment target '{target_id}' is not registered.")

    format_token = _normalize_token(export_format.value if isinstance(export_format, ExportFormat) else str(export_format))
    profile = resolve_artifact_profile(format_token)
    compatible = profile in [str(item) for item in list(target.get("artifact_profiles") or [])]
    if not compatible:
        raise ValueError(
            f"Target '{normalized_target_id}' is not compatible with export format '{format_token}' (profile={profile})."
        )

    model_token = model_name.strip() or "exported-model"
    endpoint_token = _safe_slug(endpoint_name or model_token, fallback="slm-endpoint")
    region_token = str(region or "us-east-1").strip() or "us-east-1"
    instance_token = str(instance_type or "ml.g5.xlarge").strip() or "ml.g5.xlarge"
    kind = str(target.get("kind") or "")

    if kind == "sdk":
        sdk_artifact = _write_mobile_stub_bundle(
            run_dir=run_dir,
            target_id=normalized_target_id,
            model_name=model_token,
        )
        return {
            "target_id": normalized_target_id,
            "target_kind": kind,
            "display_name": str(target.get("display_name") or normalized_target_id),
            "summary": "Mobile SDK stub generated.",
            "steps": [
                "Download the generated stub zip.",
                "Open it in Xcode/Android Studio.",
                "Replace the stub inference bridge with runtime-specific model execution.",
            ],
            "sdk_artifact": sdk_artifact,
        }

    if normalized_target_id == "deployment.hf_inference_endpoint":
        curl = (
            "curl -X POST \"https://api.endpoints.huggingface.cloud/v2/endpoint/{endpoint}\" "
            "-H \"Authorization: Bearer <HF_TOKEN>\" -H \"Content-Type: application/json\" "
            "--data '{{\"action\":\"resume\"}}'"
        ).format(endpoint=endpoint_token)
        steps = [
            f"Upload export bundle from `{run_dir}` to HuggingFace model repo.",
            "Create (or select) a HuggingFace Inference Endpoint linked to that repo.",
            "Use the generated curl command to resume/deploy endpoint.",
        ]
    elif normalized_target_id == "deployment.aws_sagemaker":
        curl = (
            "aws sagemaker create-endpoint "
            f"--endpoint-name {endpoint_token} "
            f"--region {region_token} "
            f"--endpoint-config-name {endpoint_token}-cfg"
        )
        steps = [
            f"Push container/model artifact from `{run_dir}` to ECR/S3.",
            f"Create SageMaker model and endpoint config using instance `{instance_token}`.",
            "Call `create-endpoint` (command below) and wait for InService status.",
        ]
    else:
        curl = (
            "curl -X POST \"https://managed-vllm.example.com/provision\" "
            "-H \"Authorization: Bearer <MANAGED_VLLM_TOKEN>\" "
            "-H \"Content-Type: application/json\" "
            f"--data '{{\"name\":\"{endpoint_token}\",\"model_path\":\"{run_dir}/model\"}}'"
        )
        steps = [
            f"Upload model artifact from `{run_dir}/model` to your managed vLLM provider.",
            "Provision API endpoint with selected model and autoscaling policy.",
            "Use returned OpenAI-compatible URL with generated curl template.",
        ]

    return {
        "target_id": normalized_target_id,
        "target_kind": kind or "deployment",
        "display_name": str(target.get("display_name") or normalized_target_id),
        "summary": "Managed deployment plan generated.",
        "endpoint_name": endpoint_token,
        "region": region_token,
        "instance_type": instance_token,
        "steps": steps,
        "curl_example": curl,
        "source_run_dir": str(run_dir),
    }
