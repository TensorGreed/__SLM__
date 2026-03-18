import re
from typing import Any
from pydantic import BaseModel

from app.services.model_introspection_service import introspect_hf_model

class TargetConstraint(BaseModel):
    max_parameters_billions: float | None = None
    min_vram_gb: float | None = None
    preferred_formats: list[str] = []
    required_capabilities: list[str] = []

class TargetProfile(BaseModel):
    id: str
    name: str
    description: str
    device_class: str = "laptop"
    constraints: TargetConstraint
    inference_runner_default: str | None = None

_TARGET_PROFILES = [
    TargetProfile(
        id="vllm_server",
        name="vLLM Server",
        description="High-throughput GPU server using vLLM.",
        device_class="server",
        constraints=TargetConstraint(
            min_vram_gb=16.0,
            preferred_formats=["huggingface"],
        ),
        inference_runner_default="runner.vllm"
    ),
    TargetProfile(
        id="mobile_cpu",
        name="Mobile (CPU)",
        description="On-device inference using mobile CPU (llama.cpp/GGUF).",
        device_class="mobile",
        constraints=TargetConstraint(
            max_parameters_billions=4.0,
            preferred_formats=["gguf"],
        ),
        inference_runner_default="runner.ollama"
    ),
    TargetProfile(
        id="edge_gpu",
        name="Edge GPU (NVIDIA Jetson/Desktop)",
        description="Inference on edge devices with NVIDIA GPUs (TensorRT).",
        device_class="laptop",
        constraints=TargetConstraint(
            min_vram_gb=4.0,
            preferred_formats=["tensorrt", "onnx"],
        ),
        inference_runner_default="exporter.tensorrt"
    ),
    TargetProfile(
        id="browser_webgpu",
        name="Browser (WebGPU)",
        description="In-browser inference using WebGPU and ONNX.",
        device_class="mobile",
        constraints=TargetConstraint(
            max_parameters_billions=2.0,
            preferred_formats=["onnx"],
        ),
        inference_runner_default="exporter.onnx"
    )
]

def list_targets() -> list[TargetProfile]:
    return _TARGET_PROFILES

def get_target_by_id(target_id: str) -> TargetProfile | None:
    for t in _TARGET_PROFILES:
        if t.id == target_id:
            return t
    return None


def resolve_target_device(
    target_id: str | None,
    *,
    fallback: str | None = None,
) -> str:
    token = str(target_id or "").strip().lower()
    target = get_target_by_id(token) if token else None
    if target is not None:
        device = str(target.device_class or "").strip().lower()
        if device in {"mobile", "laptop", "server"}:
            return device

    fallback_device = str(fallback or "").strip().lower()
    if fallback_device in {"mobile", "laptop", "server"}:
        return fallback_device
    return "laptop"


def _coerce_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return round(parsed, 4)


def _fallback_params_from_name(model_name: str) -> float | None:
    token = str(model_name or "").strip().lower()
    if not token:
        return None
    # Common model-size hints: "1b", "1.5b", "7b", "70b".
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", token)
    if match is None:
        return None
    try:
        parsed = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return round(parsed, 4)


def _fallback_memory_from_params(params_b: float | None) -> float | None:
    if params_b is None:
        return None
    # Coarse 4-bit-ish inference footprint heuristic.
    memory_gb = (float(params_b) * 0.7) + 1.0
    if memory_gb <= 0:
        return None
    return round(memory_gb, 2)


def _resolve_model_metadata(model_name: str) -> dict[str, Any]:
    introspection = introspect_hf_model(
        model_id=str(model_name or "").strip(),
        allow_network=True,
        timeout_seconds=1.8,
    )
    params_b = _coerce_positive_float(introspection.get("params_estimate_b"))
    params_source = "introspection"
    if params_b is None:
        params_b = _fallback_params_from_name(model_name)
        if params_b is not None:
            params_source = "name_hint"
        else:
            params_source = "unknown"

    memory_profile = dict(introspection.get("memory_profile") or {})
    estimated_min_vram = _coerce_positive_float(memory_profile.get("estimated_min_vram_gb"))
    estimated_ideal_vram = _coerce_positive_float(memory_profile.get("estimated_ideal_vram_gb"))
    if estimated_min_vram is None:
        estimated_min_vram = _fallback_memory_from_params(params_b)
    if estimated_ideal_vram is None and estimated_min_vram is not None:
        estimated_ideal_vram = round(max(estimated_min_vram + 1.0, estimated_min_vram * 1.25), 2)

    return {
        "model_id": str(model_name or "").strip(),
        "parameters_billions": params_b,
        "parameters_source": params_source,
        "estimated_min_vram_gb": estimated_min_vram,
        "estimated_ideal_vram_gb": estimated_ideal_vram,
        "architecture": str(introspection.get("architecture") or "").strip() or None,
        "context_length": introspection.get("context_length"),
        "license": str(introspection.get("license") or "").strip() or None,
        "source": str(introspection.get("source") or "none").strip() or "none",
        "resolved": bool(introspection.get("resolved", False)),
        "introspection": introspection,
    }


def check_compatibility(model_name: str, target_id: str) -> dict[str, Any]:
    target = get_target_by_id(target_id)
    if not target:
        return {
            "compatible": False,
            "reason": "Target profile not found",
            "reasons": ["Target profile not found."],
            "warnings": [],
            "target": None,
            "model_metadata": _resolve_model_metadata(model_name),
        }

    metadata = _resolve_model_metadata(model_name)
    parameters_billions = _coerce_positive_float(metadata.get("parameters_billions"))
    estimated_min_vram_gb = _coerce_positive_float(metadata.get("estimated_min_vram_gb"))

    reasons: list[str] = []
    warnings: list[str] = []

    max_parameters = _coerce_positive_float(target.constraints.max_parameters_billions)
    if max_parameters is not None:
        if parameters_billions is None:
            reasons.append(
                (
                    "Model parameter size could not be inferred; unable to validate "
                    f"max_parameters limit ({max_parameters:g}B) for target '{target.id}'."
                )
            )
        elif parameters_billions > max_parameters:
            reasons.append(
                (
                    f"Model size ({parameters_billions:g}B) exceeds target limit "
                    f"({max_parameters:g}B)."
                )
            )

    min_target_vram = _coerce_positive_float(target.constraints.min_vram_gb)
    if min_target_vram is not None and estimated_min_vram_gb is not None and estimated_min_vram_gb > min_target_vram:
        warnings.append(
            (
                f"Estimated minimum VRAM ({estimated_min_vram_gb:g} GB) is above target baseline "
                f"({min_target_vram:g} GB)."
            )
        )

    return {
        "compatible": len(reasons) == 0,
        "reasons": reasons,
        "warnings": warnings,
        "target": target.model_dump(),
        "model_metadata": metadata,
    }

def estimate_metrics(model_name: str, target_id: str) -> dict[str, Any]:
    metadata = _resolve_model_metadata(model_name)
    params_b = _coerce_positive_float(metadata.get("parameters_billions")) or 7.0
    memory_gb = _coerce_positive_float(metadata.get("estimated_min_vram_gb")) or _fallback_memory_from_params(params_b) or 5.9

    # Latency estimation (tokens/sec) with coarse target scaling.
    tps = 50.0
    if target_id == "mobile_cpu":
        tps = 5.0
    elif target_id == "browser_webgpu":
        tps = 10.0
    elif target_id == "edge_gpu":
        tps = 25.0

    # Smaller models generally decode faster on the same target.
    tps = tps * (7.0 / max(0.25, float(params_b)))

    return {
        "estimated_memory_gb": round(float(memory_gb), 2),
        "estimated_latency_tps": round(tps, 2),
        "target_id": target_id,
        "model_name": model_name,
        "model_metadata": metadata,
    }
