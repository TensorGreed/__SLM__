"""Compression engine service — quantization, LoRA merge, benchmarking."""

from __future__ import annotations

import asyncio
import json
from asyncio.subprocess import PIPE
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.config import settings

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def _compression_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "compressed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_backend() -> str:
    backend = settings.COMPRESSION_BACKEND.strip().lower()
    if backend not in {"external", "stub"}:
        raise ValueError(
            f"Unsupported COMPRESSION_BACKEND '{settings.COMPRESSION_BACKEND}'. "
            "Use 'external' or 'stub'."
        )
    return backend


def _render_external_command(template: str, placeholders: dict[str, str | int]) -> str:
    try:
        return template.format(**placeholders)
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"Compression command template missing placeholder: {missing}")


async def _run_external_command(command: str, cwd: Path) -> dict:
    started = perf_counter()
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=PIPE,
            stderr=PIPE,
            cwd=str(cwd),
        )
    except Exception as e:
        raise ValueError(f"Failed to start external compression command: {e}")

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise ValueError(
            f"External command timed out after {settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS} seconds"
        )

    duration = round(perf_counter() - started, 2)
    return {
        "command": command,
        "returncode": process.returncode,
        "duration_seconds": duration,
        "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
        "stderr": stderr.decode("utf-8", errors="replace") if stderr else "",
    }


async def quantize_model(
    project_id: int,
    model_path: str,
    bits: int = 4,
    output_format: str = "gguf",
) -> dict:
    """Quantize a model using either external runtime or explicit stub mode."""
    output_dir = _compression_dir(project_id)
    backend = _resolve_backend()
    created_at = datetime.now(timezone.utc).isoformat()

    if backend == "stub":
        if not settings.ALLOW_STUB_COMPRESSION:
            raise ValueError(
                "Stub compression backend is disabled. Configure COMPRESSION_BACKEND=external "
                "with QUANTIZE_EXTERNAL_CMD or set ALLOW_STUB_COMPRESSION=true for demos."
            )
        result = {
            "project_id": project_id,
            "source_model": model_path,
            "quantization": f"{bits}-bit",
            "output_format": output_format,
            "output_dir": str(output_dir),
            "status": "simulated",
            "created_at": created_at,
            "message": "No external compression backend configured; returning simulated result.",
        }
        (output_dir / "compression_config.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        return result

    template = settings.QUANTIZE_EXTERNAL_CMD.strip()
    if not template:
        raise ValueError("QUANTIZE_EXTERNAL_CMD is required when COMPRESSION_BACKEND=external")

    output_model_path = output_dir / f"quantized_{bits}bit.{output_format}"
    command = _render_external_command(
        template,
        {
            "project_id": project_id,
            "model_path": model_path,
            "bits": bits,
            "output_format": output_format,
            "output_dir": str(output_dir),
            "output_model_path": str(output_model_path),
            "backend_dir": str(BACKEND_DIR),
        },
    )
    execution = await _run_external_command(command, cwd=BACKEND_DIR)
    report_path = output_dir / "quantize_report.json"
    report_path.write_text(json.dumps(execution, indent=2), encoding="utf-8")

    if execution["returncode"] != 0:
        stderr_tail = execution["stderr"][-500:]
        raise ValueError(f"Quantization command failed (exit {execution['returncode']}): {stderr_tail}")

    return {
        "project_id": project_id,
        "source_model": model_path,
        "quantization": f"{bits}-bit",
        "output_format": output_format,
        "output_dir": str(output_dir),
        "output_model_path": str(output_model_path),
        "status": "completed",
        "created_at": created_at,
        "report_path": str(report_path),
        "duration_seconds": execution["duration_seconds"],
    }


async def merge_lora(
    project_id: int,
    base_model_path: str,
    lora_adapter_path: str,
) -> dict:
    """Merge LoRA adapter using external runtime or explicit stub mode."""
    output_dir = _compression_dir(project_id) / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = _resolve_backend()
    created_at = datetime.now(timezone.utc).isoformat()

    if backend == "stub":
        if not settings.ALLOW_STUB_COMPRESSION:
            raise ValueError(
                "Stub compression backend is disabled. Configure COMPRESSION_BACKEND=external "
                "with MERGE_LORA_EXTERNAL_CMD or set ALLOW_STUB_COMPRESSION=true for demos."
            )
        return {
            "project_id": project_id,
            "base_model": base_model_path,
            "lora_adapter": lora_adapter_path,
            "output_dir": str(output_dir),
            "status": "simulated",
            "created_at": created_at,
            "message": "No external merge backend configured; returning simulated result.",
        }

    template = settings.MERGE_LORA_EXTERNAL_CMD.strip()
    if not template:
        raise ValueError("MERGE_LORA_EXTERNAL_CMD is required when COMPRESSION_BACKEND=external")

    output_model_path = output_dir / "merged_model"
    command = _render_external_command(
        template,
        {
            "project_id": project_id,
            "base_model_path": base_model_path,
            "lora_adapter_path": lora_adapter_path,
            "output_dir": str(output_dir),
            "output_model_path": str(output_model_path),
            "backend_dir": str(BACKEND_DIR),
        },
    )
    execution = await _run_external_command(command, cwd=BACKEND_DIR)
    report_path = output_dir / "merge_report.json"
    report_path.write_text(json.dumps(execution, indent=2), encoding="utf-8")

    if execution["returncode"] != 0:
        stderr_tail = execution["stderr"][-500:]
        raise ValueError(f"LoRA merge command failed (exit {execution['returncode']}): {stderr_tail}")

    return {
        "project_id": project_id,
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "output_dir": str(output_dir),
        "output_model_path": str(output_model_path),
        "status": "completed",
        "created_at": created_at,
        "report_path": str(report_path),
        "duration_seconds": execution["duration_seconds"],
    }


async def benchmark_model(
    project_id: int,
    model_path: str,
    num_samples: int = 100,
) -> dict:
    """Benchmark model artifacts with optional external latency run."""
    model_dir = Path(model_path)
    backend = _resolve_backend()

    total_size = 0
    if model_dir.is_dir():
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    elif model_dir.is_file():
        total_size = model_dir.stat().st_size

    result = {
        "project_id": project_id,
        "model_path": model_path,
        "model_size_bytes": total_size,
        "model_size_mb": round(total_size / (1024 * 1024), 2),
        "benchmark_samples": num_samples,
    }

    if backend == "stub":
        if not settings.ALLOW_STUB_COMPRESSION:
            raise ValueError(
                "Stub compression backend is disabled. Configure COMPRESSION_BACKEND=external "
                "with BENCHMARK_EXTERNAL_CMD or set ALLOW_STUB_COMPRESSION=true for demos."
            )
        result.update(
            {
                "status": "simulated",
                "message": "Size-only benchmark in stub mode. Configure external backend for latency metrics.",
            }
        )
        return result

    template = settings.BENCHMARK_EXTERNAL_CMD.strip()
    if not template:
        raise ValueError("BENCHMARK_EXTERNAL_CMD is required when COMPRESSION_BACKEND=external")

    output_dir = _compression_dir(project_id)
    command = _render_external_command(
        template,
        {
            "project_id": project_id,
            "model_path": model_path,
            "num_samples": num_samples,
            "output_dir": str(output_dir),
            "backend_dir": str(BACKEND_DIR),
        },
    )
    execution = await _run_external_command(command, cwd=BACKEND_DIR)
    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(execution, indent=2), encoding="utf-8")

    if execution["returncode"] != 0:
        stderr_tail = execution["stderr"][-500:]
        raise ValueError(f"Benchmark command failed (exit {execution['returncode']}): {stderr_tail}")

    result.update(
        {
            "status": "completed",
            "report_path": str(report_path),
            "duration_seconds": execution["duration_seconds"],
        }
    )
    return result
