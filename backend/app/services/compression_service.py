"""Compression engine service — quantization, LoRA merge, benchmarking."""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def _compression_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "compressed"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def quantize_model(
    project_id: int,
    model_path: str,
    bits: int = 4,
    output_format: str = "gguf",
) -> dict:
    """Quantize a model (placeholder for actual quantization calls)."""
    output_dir = _compression_dir(project_id)

    # In production, this would call:
    # - llama.cpp's convert/quantize for GGUF
    # - bitsandbytes for 4/8-bit
    # - AutoGPTQ/AutoAWQ for advanced quantization

    result = {
        "project_id": project_id,
        "source_model": model_path,
        "quantization": f"{bits}-bit",
        "output_format": output_format,
        "output_dir": str(output_dir),
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save compression config
    config_path = output_dir / "compression_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


async def merge_lora(
    project_id: int,
    base_model_path: str,
    lora_adapter_path: str,
) -> dict:
    """Merge LoRA adapter with base model."""
    output_dir = _compression_dir(project_id) / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    # In production:
    # from peft import PeftModel
    # model = AutoModelForCausalLM.from_pretrained(base_model_path)
    # model = PeftModel.from_pretrained(model, lora_adapter_path)
    # model = model.merge_and_unload()
    # model.save_pretrained(output_dir)

    return {
        "project_id": project_id,
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "output_dir": str(output_dir),
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


async def benchmark_model(
    project_id: int,
    model_path: str,
    num_samples: int = 100,
) -> dict:
    """Benchmark model performance (size, latency)."""
    model_dir = Path(model_path)

    # Calculate model size
    total_size = 0
    if model_dir.is_dir():
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    elif model_dir.is_file():
        total_size = model_dir.stat().st_size

    return {
        "project_id": project_id,
        "model_path": model_path,
        "model_size_bytes": total_size,
        "model_size_mb": round(total_size / (1024 * 1024), 2),
        "benchmark_samples": num_samples,
        "status": "queued",
        "message": "Full latency benchmarking requires model inference runtime",
    }
