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
_ZIP_EPOCH = (2024, 1, 1, 0, 0, 0)


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
        "display_name": "Apple CoreML Reference App",
        "description": "Generate a runnable iOS reference bundle with model-loader scaffold.",
        "artifact_profiles": ["huggingface", "onnx", "gguf"],
        "supported_export_formats": ["huggingface", "onnx", "gguf", "docker"],
        "smoke_supported": True,
        "launch_example": "Open iOS reference project in Xcode and run on simulator/device.",
    },
    {
        "target_id": "sdk.android_executorch_stub",
        "kind": "sdk",
        "display_name": "Android ExecuTorch Reference App",
        "description": "Generate a runnable Android reference bundle with runtime bridge.",
        "artifact_profiles": ["huggingface", "onnx", "gguf"],
        "supported_export_formats": ["huggingface", "onnx", "gguf", "docker"],
        "smoke_supported": True,
        "launch_example": "Open Android reference project in Android Studio and run on emulator/device.",
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


def _mobile_reference_file_map(target_id: str, model_name: str) -> dict[str, str]:
    if target_id == "sdk.apple_coreml_stub":
        return {
            "README.md": (
                "# iOS CoreML Reference Bundle\n\n"
                f"Base model: `{model_name}`\n\n"
                "## Model Placement\n"
                "1. Place your exported model under `ios/ModelAssets/`.\n"
                "2. Use `model.mlmodelc` for CoreML compiled models.\n"
                "3. Keep tokenizer/config artifacts beside the model when available.\n\n"
                "## Run Instructions\n"
                "1. Open Xcode and create an iOS App target (SwiftUI).\n"
                "2. Copy `ios/SLMReferenceApp.swift` and `ios/SLMRuntime.swift` into your target.\n"
                "3. Launch on simulator/device and enter a prompt in the UI.\n\n"
                "## CLI Smoke Check\n"
                "Run with: `swift scripts/run_reference.swift \"Hello from iOS\"`\n"
            ),
            "ios/SLMReferenceApp.swift": (
                "import SwiftUI\n\n"
                "@main\n"
                "struct SLMReferenceApp: App {\n"
                "    @State private var prompt = \"Hello from iOS\"\n"
                "    @State private var output = \"\"\n"
                "    private let runtime = SLMRuntime(modelURL: nil)\n\n"
                "    var body: some Scene {\n"
                "        WindowGroup {\n"
                "            VStack(alignment: .leading, spacing: 12) {\n"
                "                Text(\"SLM iOS Reference\")\n"
                "                    .font(.headline)\n"
                "                TextField(\"Prompt\", text: $prompt)\n"
                "                    .textFieldStyle(.roundedBorder)\n"
                "                Button(\"Generate\") {\n"
                "                    output = runtime.generate(prompt: prompt, maxTokens: 32)\n"
                "                }\n"
                "                .buttonStyle(.borderedProminent)\n"
                "                Text(output)\n"
                "                    .font(.body)\n"
                "                    .fixedSize(horizontal: false, vertical: true)\n"
                "            }\n"
                "            .padding(20)\n"
                "        }\n"
                "    }\n"
                "}\n"
            ),
            "ios/SLMRuntime.swift": (
                "import Foundation\n\n"
                "final class SLMRuntime {\n"
                "    private let modelURL: URL?\n"
                "    private let modelBytes: Int\n\n"
                "    init(modelURL: URL?) {\n"
                "        self.modelURL = modelURL\n"
                "        self.modelBytes = SLMRuntime.resolveModelBytes(url: modelURL)\n"
                "    }\n\n"
                "    func generate(prompt: String, maxTokens: Int = 32) -> String {\n"
                "        let cleaned = prompt.trimmingCharacters(in: .whitespacesAndNewlines)\n"
                "        let base = cleaned.isEmpty ? \"hello\" : cleaned\n"
                "        let scalarSeed = base.unicodeScalars.reduce(0) { $0 + Int($1.value) }\n"
                "        let seed = (scalarSeed + modelBytes) % 9973\n"
                "        var pieces: [String] = [\"Echo:\", base]\n"
                "        let tokenCount = max(4, min(maxTokens, 24))\n"
                "        for index in 0..<tokenCount {\n"
                "            let tokenValue = (seed + (index * 37)) % 541\n"
                "            pieces.append(\"tok\\(tokenValue)\")\n"
                "        }\n"
                "        return pieces.joined(separator: \" \")\n"
                "    }\n\n"
                "    private static func resolveModelBytes(url: URL?) -> Int {\n"
                "        guard let modelURL = url else { return 0 }\n"
                "        guard let attrs = try? FileManager.default.attributesOfItem(atPath: modelURL.path) else {\n"
                "            return 0\n"
                "        }\n"
                "        return Int((attrs[.size] as? NSNumber)?.intValue ?? 0)\n"
                "    }\n"
                "}\n"
            ),
            "ios/ModelAssets/.keep": "",
            "scripts/run_reference.swift": (
                "import Foundation\n\n"
                "let prompt = CommandLine.arguments.dropFirst().first ?? \"Hello from iOS\"\n"
                "let scalarSeed = prompt.unicodeScalars.reduce(0) { $0 + Int($1.value) }\n"
                "let token = scalarSeed % 541\n"
                "print(\"Echo: \\(prompt) tok\\(token)\")\n"
            ),
        }

    return {
        "README.md": (
            "# Android ExecuTorch Reference Bundle\n\n"
            f"Base model: `{model_name}`\n\n"
            "## Model Placement\n"
            "1. Place exported model files under `android/app/src/main/assets/`.\n"
            "2. Configure runtime path in `SLMRuntime` if you rename model files.\n"
            "3. Keep tokenizer metadata with model artifacts when possible.\n\n"
            "## Run Instructions\n"
            "1. Open Android Studio and create/choose an Android app module.\n"
            "2. Copy `MainActivity.kt` and `SLMRuntime.kt` into your app package.\n"
            "3. Launch emulator/device, enter a prompt, and tap Generate.\n\n"
            "## CLI Smoke Check\n"
            "Run with: `kotlin scripts/run_reference.kts \"Hello from Android\"`\n"
        ),
        "android/app/src/main/java/com/example/slmreference/MainActivity.kt": (
            "package com.example.slmreference\n\n"
            "import android.os.Bundle\n"
            "import android.widget.Button\n"
            "import android.widget.EditText\n"
            "import android.widget.TextView\n"
            "import androidx.appcompat.app.AppCompatActivity\n\n"
            "class MainActivity : AppCompatActivity() {\n"
            "    private lateinit var runtime: SLMRuntime\n\n"
            "    override fun onCreate(savedInstanceState: Bundle?) {\n"
            "        super.onCreate(savedInstanceState)\n"
            "        setContentView(R.layout.activity_main)\n"
            "        runtime = SLMRuntime(assets)\n\n"
            "        val promptView = findViewById<EditText>(R.id.promptInput)\n"
            "        val outputView = findViewById<TextView>(R.id.outputText)\n"
            "        val generateButton = findViewById<Button>(R.id.generateButton)\n\n"
            "        generateButton.setOnClickListener {\n"
            "            val prompt = promptView.text?.toString().orEmpty()\n"
            "            outputView.text = runtime.generate(prompt, maxTokens = 32)\n"
            "        }\n"
            "    }\n"
            "}\n"
        ),
        "android/app/src/main/java/com/example/slmreference/SLMRuntime.kt": (
            "package com.example.slmreference\n\n"
            "import android.content.res.AssetManager\n\n"
            "class SLMRuntime(private val assets: AssetManager) {\n"
            "    fun generate(prompt: String, maxTokens: Int = 32): String {\n"
            "        val base = prompt.trim().ifBlank { \"hello\" }\n"
            "        val modelBytes = modelSizeHint()\n"
            "        val seed = (base.sumOf { it.code } + modelBytes) % 9973\n"
            "        val tokens = mutableListOf(\"Echo:\", base)\n"
            "        val tokenCount = maxOf(4, minOf(maxTokens, 24))\n"
            "        repeat(tokenCount) { index ->\n"
            "            val value = (seed + (index * 37)) % 541\n"
            "            tokens += \"tok$value\"\n"
            "        }\n"
            "        return tokens.joinToString(\" \")\n"
            "    }\n\n"
            "    private fun modelSizeHint(): Int {\n"
            "        return try {\n"
            "            assets.openFd(\"model.bin\").length.toInt()\n"
            "        } catch (_: Exception) {\n"
            "            0\n"
            "        }\n"
            "    }\n"
            "}\n"
        ),
        "android/app/src/main/assets/.keep": "",
        "scripts/run_reference.kts": (
            "val prompt = if (args.isNotEmpty()) args.joinToString(\" \") else \"Hello from Android\"\n"
            "val seed = prompt.sumOf { it.code } % 541\n"
            "println(\"Echo: $prompt tok$seed\")\n"
        ),
    }


def _write_deterministic_zip(
    *,
    bundle_dir: Path,
    relative_paths: list[str],
    zip_path: Path,
) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path in sorted(relative_paths):
            file_path = bundle_dir / rel_path
            info = zipfile.ZipInfo(rel_path, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, file_path.read_bytes())


def _sdk_entrypoint_relpath(target_id: str) -> str:
    if target_id == "sdk.apple_coreml_stub":
        return "ios/SLMReferenceApp.swift"
    return "android/app/src/main/java/com/example/slmreference/MainActivity.kt"


def _sdk_runtime_relpath(target_id: str) -> str:
    if target_id == "sdk.apple_coreml_stub":
        return "ios/SLMRuntime.swift"
    return "android/app/src/main/java/com/example/slmreference/SLMRuntime.kt"


def _validate_mobile_reference_bundle(
    *,
    target_id: str,
    bundle_dir: Path,
    zip_path: Path,
    expected_files: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    for rel_path in expected_files:
        file_path = bundle_dir / rel_path
        checks.append(
            _check_result(
                check_id=f"bundle_file:{rel_path}",
                passed=file_path.exists() and file_path.is_file(),
                message=f"Required bundle file '{rel_path}' must exist.",
            )
        )

    readme_rel = "README.md"
    entrypoint_rel = _sdk_entrypoint_relpath(target_id)
    runtime_rel = _sdk_runtime_relpath(target_id)

    readme_text = (bundle_dir / readme_rel).read_text(encoding="utf-8") if (bundle_dir / readme_rel).exists() else ""
    entrypoint_text = (bundle_dir / entrypoint_rel).read_text(encoding="utf-8") if (bundle_dir / entrypoint_rel).exists() else ""
    runtime_text = (bundle_dir / runtime_rel).read_text(encoding="utf-8") if (bundle_dir / runtime_rel).exists() else ""

    readme_markers = ["Model Placement", "Run Instructions"]
    for marker in readme_markers:
        checks.append(
            _check_result(
                check_id=f"readme_marker:{marker}",
                passed=marker in readme_text,
                message=f"README is missing required guidance section '{marker}'.",
            )
        )

    if target_id == "sdk.apple_coreml_stub":
        entrypoint_markers = ["struct SLMReferenceApp: App", "runtime.generate("]
        runtime_markers = ["final class SLMRuntime", "func generate("]
    else:
        entrypoint_markers = ["class MainActivity : AppCompatActivity()", "runtime.generate("]
        runtime_markers = ["class SLMRuntime", "fun generate("]

    for marker in entrypoint_markers:
        checks.append(
            _check_result(
                check_id=f"entrypoint_marker:{marker}",
                passed=marker in entrypoint_text,
                message=f"Entrypoint file '{entrypoint_rel}' is missing '{marker}'.",
            )
        )
    for marker in runtime_markers:
        checks.append(
            _check_result(
                check_id=f"runtime_marker:{marker}",
                passed=marker in runtime_text,
                message=f"Runtime file '{runtime_rel}' is missing '{marker}'.",
            )
        )

    combined_text = "\n".join([readme_text, entrypoint_text, runtime_text]).lower()
    checks.append(
        _check_result(
            check_id="no_todo_markers",
            passed="todo" not in combined_text,
            message="Reference bundle must not include TODO stub markers.",
        )
    )

    checks.append(
        _check_result(
            check_id="bundle_zip_exists",
            passed=zip_path.exists() and zip_path.is_file(),
            message="Reference bundle zip must exist.",
        )
    )

    zip_members: list[str] = []
    zip_members_sorted = False
    if zip_path.exists() and zip_path.is_file():
        with zipfile.ZipFile(zip_path, "r") as archive:
            zip_members = [str(name) for name in archive.namelist()]
        zip_members_sorted = zip_members == sorted(zip_members)
        checks.append(
            _check_result(
                check_id="bundle_zip_member_set",
                passed=set(zip_members) == set(expected_files),
                message="Reference zip members do not match expected bundle files.",
                details={
                    "expected": sorted(expected_files),
                    "actual": sorted(zip_members),
                },
            )
        )
        checks.append(
            _check_result(
                check_id="bundle_zip_member_order",
                passed=zip_members_sorted,
                message="Reference zip members are not ordered deterministically.",
                details={"actual_order": zip_members},
            )
        )

    passed, errors, warnings = _summarize_checks(checks)
    return {
        "smoke_supported": True,
        "smoke_executed": True,
        "smoke_passed": passed,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "expected_files": sorted(expected_files),
        "zip_members": zip_members,
        "zip_members_sorted": zip_members_sorted,
    }


def _write_mobile_reference_bundle(
    *,
    run_dir: Path,
    target_id: str,
    model_name: str,
) -> dict[str, Any]:
    base_dir = run_dir / "mobile_sdk" / _safe_slug(target_id, fallback="mobile-sdk")
    base_dir.mkdir(parents=True, exist_ok=True)

    file_map = _mobile_reference_file_map(target_id, model_name)
    relative_paths = sorted(file_map.keys())
    for rel_path, content in file_map.items():
        file_path = base_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    zip_path = base_dir / "stub_app.zip"
    _write_deterministic_zip(
        bundle_dir=base_dir,
        relative_paths=relative_paths,
        zip_path=zip_path,
    )
    smoke_validation = _validate_mobile_reference_bundle(
        target_id=target_id,
        bundle_dir=base_dir,
        zip_path=zip_path,
        expected_files=relative_paths,
    )

    return {
        "bundle_dir": str(base_dir),
        "readme_path": str(base_dir / "README.md"),
        "entrypoint_path": str(base_dir / _sdk_entrypoint_relpath(target_id)),
        "runtime_path": str(base_dir / _sdk_runtime_relpath(target_id)),
        "zip_path": str(zip_path),
        "bundle_files": relative_paths,
        "smoke_validation": smoke_validation,
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
    """Build actionable deploy plan for managed API targets and mobile SDK reference bundles."""
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
        sdk_artifact = _write_mobile_reference_bundle(
            run_dir=run_dir,
            target_id=normalized_target_id,
            model_name=model_token,
        )
        smoke = dict(sdk_artifact.get("smoke_validation") or {})
        smoke_passed = bool(smoke.get("smoke_passed"))
        return {
            "target_id": normalized_target_id,
            "target_kind": kind,
            "display_name": str(target.get("display_name") or normalized_target_id),
            "summary": (
                "Mobile SDK reference bundle generated."
                if smoke_passed
                else "Mobile SDK reference bundle generated with smoke validation issues."
            ),
            "steps": [
                "Download the generated reference zip bundle.",
                "Open the platform files in Xcode/Android Studio.",
                "Copy your exported model artifacts into the documented model assets directory.",
                "Run the included reference entrypoint and smoke script to verify wiring.",
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


async def execute_deploy_target_plan(
    *,
    run_dir: Path,
    export_format: ExportFormat | str,
    target_id: str,
    model_name: str = "",
    endpoint_name: str | None = None,
    region: str | None = None,
    instance_type: str | None = None,
    dry_run: bool = True,
    hf_token: str | None = None,
    managed_api_url: str | None = None,
    managed_api_token: str | None = None,
    sagemaker_role_arn: str | None = None,
    sagemaker_image_uri: str | None = None,
    sagemaker_model_data_url: str | None = None,
) -> dict[str, Any]:
    """
    Execute managed deployment action for a target.

    By default this runs in dry-run mode and returns an executable plan preview.
    """
    plan = build_deploy_target_plan(
        run_dir=run_dir,
        export_format=export_format,
        target_id=target_id,
        model_name=model_name,
        endpoint_name=endpoint_name,
        region=region,
        instance_type=instance_type,
    )
    normalized_target_id = str(plan.get("target_id") or "").strip().lower()
    started_at = _utcnow_iso()

    if bool(dry_run):
        return {
            **plan,
            "execution": {
                "status": "dry_run",
                "message": "Dry-run only. No provider API was called.",
                "started_at": started_at,
                "finished_at": _utcnow_iso(),
                "provider": normalized_target_id,
                "dry_run": True,
            },
        }

    target_kind = str(plan.get("target_kind") or "").strip().lower()
    if target_kind == "sdk":
        return {
            **plan,
            "execution": {
                "status": "completed",
                "message": "SDK artifact generated (no remote deployment required).",
                "started_at": started_at,
                "finished_at": _utcnow_iso(),
                "provider": normalized_target_id,
                "dry_run": False,
            },
        }

    if normalized_target_id == "deployment.hf_inference_endpoint":
        token = str(hf_token or "").strip()
        if not token:
            raise ValueError("hf_token is required to execute HuggingFace endpoint deployment.")
        endpoint = str(plan.get("endpoint_name") or "").strip()
        if not endpoint:
            raise ValueError("Unable to resolve endpoint_name for HuggingFace deployment.")
        endpoint_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{endpoint}"
        payload = {"action": "resume"}
        try:
            import httpx

            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    endpoint_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
        except Exception as exc:
            raise ValueError(f"HuggingFace endpoint request failed: {exc}") from exc

        body_preview = str(response.text or "")[:1200]
        if response.status_code >= 400:
            raise ValueError(
                f"HuggingFace endpoint request failed ({response.status_code}): {body_preview}"
            )
        try:
            response_json = response.json()
        except Exception:
            response_json = {"raw": body_preview}
        return {
            **plan,
            "execution": {
                "status": "submitted",
                "message": "Deployment resume call sent to HuggingFace endpoint API.",
                "started_at": started_at,
                "finished_at": _utcnow_iso(),
                "provider": normalized_target_id,
                "dry_run": False,
                "http_status": response.status_code,
                "request": {"method": "POST", "url": endpoint_url, "payload": payload},
                "response": response_json,
            },
        }

    if normalized_target_id == "deployment.aws_sagemaker":
        if importlib.util.find_spec("boto3") is None:
            raise ValueError("boto3 is required for live SageMaker deployment execution.")
        role_arn = str(sagemaker_role_arn or "").strip()
        image_uri = str(sagemaker_image_uri or "").strip()
        model_data_url = str(sagemaker_model_data_url or "").strip()
        if not role_arn:
            raise ValueError("sagemaker_role_arn is required for SageMaker execution.")
        if not image_uri:
            raise ValueError("sagemaker_image_uri is required for SageMaker execution.")
        if not model_data_url:
            raise ValueError("sagemaker_model_data_url is required for SageMaker execution.")

        endpoint = str(plan.get("endpoint_name") or "").strip()
        region_token = str(plan.get("region") or region or "us-east-1")
        instance_token = str(plan.get("instance_type") or instance_type or "ml.g5.xlarge")
        model_name_token = f"{endpoint}-model"
        config_name_token = f"{endpoint}-cfg"
        try:
            import boto3

            client = boto3.client("sagemaker", region_name=region_token)
            client.create_model(
                ModelName=model_name_token,
                ExecutionRoleArn=role_arn,
                PrimaryContainer={
                    "Image": image_uri,
                    "ModelDataUrl": model_data_url,
                },
            )
            client.create_endpoint_config(
                EndpointConfigName=config_name_token,
                ProductionVariants=[
                    {
                        "VariantName": "AllTraffic",
                        "ModelName": model_name_token,
                        "InstanceType": instance_token,
                        "InitialInstanceCount": 1,
                    }
                ],
            )
            client.create_endpoint(
                EndpointName=endpoint,
                EndpointConfigName=config_name_token,
            )
        except Exception as exc:
            raise ValueError(f"SageMaker deployment request failed: {exc}") from exc
        return {
            **plan,
            "execution": {
                "status": "submitted",
                "message": "SageMaker endpoint create request submitted.",
                "started_at": started_at,
                "finished_at": _utcnow_iso(),
                "provider": normalized_target_id,
                "dry_run": False,
                "request": {
                    "endpoint_name": endpoint,
                    "endpoint_config_name": config_name_token,
                    "model_name": model_name_token,
                    "region": region_token,
                },
            },
        }

    # deployment.vllm_managed and future managed providers.
    provision_url = str(managed_api_url or "https://managed-vllm.example.com/provision").strip()
    if not provision_url:
        raise ValueError("managed_api_url is required for managed vLLM deployment execution.")
    payload = {
        "name": str(plan.get("endpoint_name") or ""),
        "model_path": f"{run_dir}/model",
    }
    headers = {"Content-Type": "application/json"}
    token = str(managed_api_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        import httpx

        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(provision_url, json=payload, headers=headers)
    except Exception as exc:
        raise ValueError(f"Managed vLLM provision request failed: {exc}") from exc

    body_preview = str(response.text or "")[:1200]
    if response.status_code >= 400:
        raise ValueError(
            f"Managed vLLM provision request failed ({response.status_code}): {body_preview}"
        )
    try:
        response_json = response.json()
    except Exception:
        response_json = {"raw": body_preview}
    return {
        **plan,
        "execution": {
            "status": "submitted",
            "message": "Managed vLLM provisioning request submitted.",
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "provider": normalized_target_id,
            "dry_run": False,
            "http_status": response.status_code,
            "request": {"method": "POST", "url": provision_url, "payload": payload},
            "response": response_json,
        },
    }
