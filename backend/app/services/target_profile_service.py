from typing import Any
from pydantic import BaseModel

class TargetConstraint(BaseModel):
    max_parameters_billions: float | None = None
    min_vram_gb: float | None = None
    preferred_formats: list[str] = []
    required_capabilities: list[str] = []

class TargetProfile(BaseModel):
    id: str
    name: str
    description: str
    constraints: TargetConstraint
    inference_runner_default: str | None = None

_TARGET_PROFILES = [
    TargetProfile(
        id="vllm_server",
        name="vLLM Server",
        description="High-throughput GPU server using vLLM.",
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

def check_compatibility(model_name: str, target_id: str) -> dict[str, Any]:
    target = get_target_by_id(target_id)
    if not target:
        return {"compatible": False, "reason": "Target profile not found"}

    # Mock model metadata for now - in real life this would call model_introspection_service
    # Let's assume some heuristics based on model name
    model_size_b = 7.0
    if "1b" in model_name.lower(): model_size_b = 1.0
    elif "3b" in model_name.lower(): model_size_b = 3.0
    elif "8b" in model_name.lower(): model_size_b = 8.0
    elif "70b" in model_name.lower(): model_size_b = 70.0

    reasons = []
    compatible = True

    if target.constraints.max_parameters_billions and model_size_b > target.constraints.max_parameters_billions:
        compatible = False
        reasons.append(f"Model size ({model_size_b}B) exceeds target limit ({target.constraints.max_parameters_billions}B)")

    return {
        "compatible": compatible,
        "reasons": reasons,
        "target": target.model_dump(),
        "model_metadata": {"parameters_billions": model_size_b}
    }

def estimate_metrics(model_name: str, target_id: str) -> dict[str, Any]:
    # Heuristics for memory and latency
    model_size_b = 7.0
    if "1b" in model_name.lower(): model_size_b = 1.0
    elif "3b" in model_name.lower(): model_size_b = 3.0
    
    # Memory estimation (very rough)
    # 4-bit quantization roughly 0.7GB per 1B params + 1GB overhead
    memory_gb = model_size_b * 0.7 + 1.0
    
    # Latency estimation (tokens/sec)
    tps = 50.0 # base
    if target_id == "mobile_cpu": tps = 5.0
    elif target_id == "browser_webgpu": tps = 10.0
    elif target_id == "edge_gpu": tps = 25.0
    
    # Adjust by model size
    tps = tps * (7.0 / model_size_b)

    return {
        "estimated_memory_gb": round(memory_gb, 2),
        "estimated_latency_tps": round(tps, 2),
        "target_id": target_id,
        "model_name": model_name
    }
