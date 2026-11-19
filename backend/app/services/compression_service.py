"""Compression engine service — quantization, LoRA merge, benchmarking."""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import sys
from hashlib import sha256
from asyncio.subprocess import PIPE
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.config import settings
from app.exceptions import StrictExecutionError

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def _compression_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "compressed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def experiment_id_for_model_path(project_id: int, model_path: str | None) -> int | None:
    """The experiment id when ``model_path`` lives under this project's
    ``experiments/<id>/`` tree (e.g. ``…/experiments/7/model``) or is one of
    that run's own compression outputs (``compressed/exp-<id>/…`` — the
    merged model a quantize starts from), so a merge → quantize chain stays
    under the run that export packages from."""
    if not model_path:
        return None
    try:
        path = Path(model_path).expanduser().resolve()
    except (ValueError, OSError):
        return None
    project_dir = (settings.DATA_DIR / "projects" / str(project_id)).resolve()
    for tree, prefix in (("experiments", ""), ("compressed", "exp-")):
        try:
            relative = path.relative_to(project_dir / tree)
        except ValueError:
            continue
        head = relative.parts[0] if relative.parts else ""
        if head.startswith(prefix) and head[len(prefix):].isdigit():
            return int(head[len(prefix):])
    return None


def experiment_compression_dir(project_id: int, experiment_id: int) -> Path:
    """Per-experiment compression outputs — export only ever packages
    artifacts from the exported run's own directory."""
    d = _compression_dir(project_id) / f"exp-{int(experiment_id)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_dir_for_source(project_id: int, model_path: str | None) -> Path:
    experiment_id = experiment_id_for_model_path(project_id, model_path)
    if experiment_id is None:
        return _compression_dir(project_id)
    return experiment_compression_dir(project_id, experiment_id)


def _resolve_report_path(project_id: int, report_path: str) -> Path:
    base = _compression_dir(project_id).resolve()
    requested = Path(report_path).expanduser()
    if not requested.is_absolute():
        requested = (base / requested).resolve()
    else:
        requested = requested.resolve()
    if requested != base and base not in requested.parents:
        raise ValueError("report_path must be inside the project compression directory")
    return requested


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


def _normalize_external_python_command(command: str) -> tuple[str, str | None]:
    """Normalize bare `python` launcher to an explicit interpreter path."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return command, None
    if not parts:
        return command, None
    if parts[0] != "python":
        return command, None
    preferred_python = str(Path(sys.executable).expanduser())
    if preferred_python and Path(preferred_python).exists():
        parts[0] = preferred_python
        return (
            shlex.join(parts),
            f"normalized python launcher to '{preferred_python}'",
        )
    python3_path = shutil.which("python3")
    if not python3_path:
        return command, None
    parts[0] = python3_path
    return (
        shlex.join(parts),
        f"python launcher not resolved from runtime; falling back to '{python3_path}'",
    )


def _prepend_quantize_runtime_env(command: str) -> tuple[str, str | None]:
    """Inject runtime env vars for external quantization scripts."""
    env_pairs: list[tuple[str, str]] = []

    def _add(key: str, value: str | None) -> None:
        raw = str(value or "").strip()
        if raw:
            env_pairs.append((key, raw))

    _add("LLAMA_CPP_DIR", settings.LLAMA_CPP_DIR)
    _add("LLAMA_CPP_CONVERT_SCRIPT", settings.LLAMA_CPP_CONVERT_SCRIPT)
    _add("LLAMA_CPP_QUANTIZE_BIN", settings.LLAMA_CPP_QUANTIZE_BIN)
    _add("ONNX_EXPORT_TASK", settings.ONNX_EXPORT_TASK)
    _add("PYTHON_EXECUTABLE", settings.PYTHON_EXECUTABLE)

    if not env_pairs:
        return command, None

    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_pairs)
    note = "applied runtime env overrides: " + ", ".join(key for key, _ in env_pairs)
    return f"{prefix} {command}", note


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
    output_dir = _output_dir_for_source(project_id, model_path)
    backend = _resolve_backend()
    created_at = datetime.now(timezone.utc).isoformat()

    if backend == "stub":
        if settings.STRICT_EXECUTION_MODE:
            raise StrictExecutionError("compression", "Stub compression is blocked because STRICT_EXECUTION_MODE is enabled.")
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

    normalized_format = str(output_format or "gguf").strip().lower()
    if normalized_format not in {"gguf", "onnx"}:
        raise ValueError(
            f"Unsupported quantize output_format '{output_format}'. Supported values: gguf, onnx."
        )
    if normalized_format == "onnx" and int(bits) != 8:
        raise ValueError("ONNX quantization currently supports bits=8 only.")

    output_model_path = output_dir / f"quantized_{bits}bit.{normalized_format}"
    report_path = output_dir / "quantize_report.json"
    # The worker runs the command into a private temp folder under the folder's cross-process lock and
    # publishes the file with ``finish_conversion`` (os.replace + success marker) only on exit 0.
    tmp_dir, tmp_path = conversion_tmp_paths(output_dir, output_model_path)
    command = _render_external_command(
        template,
        {
            "project_id": project_id,
            "model_path": model_path,
            "bits": bits,
            "output_format": normalized_format,
            "output_dir": str(tmp_dir),
            "output_model_path": str(tmp_path),
            "backend_dir": str(BACKEND_DIR),
        },
    )
    command, command_note = _normalize_external_python_command(command)
    command, env_note = _prepend_quantize_runtime_env(command)
    if env_note:
        command_note = f"{command_note}; {env_note}" if command_note else env_note
    from app.worker import celery_app
    task = celery_app.send_task(
        "run_quantization_job",
        kwargs={
            "command": command,
            "report_path": str(report_path),
            "project_id": project_id,
            "job_type": "quantize",
            "finalize": {
                "output_dir": str(output_dir), "tmp_dir": str(tmp_dir), "tmp_path": str(tmp_path),
                "final_path": str(output_model_path),
            },
        }
    )

    return {
        "project_id": project_id,
        "source_model": model_path,
        "quantization": f"{bits}-bit",
        "output_format": normalized_format,
        "output_dir": str(output_dir),
        "output_model_path": str(output_model_path),
        "status": "queued",
        "created_at": created_at,
        "report_path": str(report_path),
        "task_id": task.id,
        "command_note": command_note,
    }


# The GGUF types scripts/quantize.py writes per bit width (its ``BITS_TO_GGUF_TYPE``).
GGUF_TYPE_BY_BITS: dict[int, str] = {2: "Q2_K", 3: "Q3_K_M", 4: "Q4_K_M", 5: "Q5_K_M", 6: "Q6_K", 8: "Q8_0", 16: "F16"}
# Temp folders of an in-progress conversion (never packaged: ``export_service._collect_compressed_files``).
CONVERT_TMP_PREFIX = ".convert-"


class QuantizationLabelError(ValueError):
    """An export quantization label that names no conversion BrewSLM runs (e.g. ``Q4_0``)."""


def quantization_plan(quantization: str | None, output_format: str) -> dict:
    """``{bits, label}`` for an export's quantization label, mapped explicitly: GGUF ``"4-bit"`` / ``"4bit"`` /
    ``"q4"`` / ``"int4"`` / ``"Q4_K_M"`` -> 4 (``Q4_K_M``), no label -> 16 (``F16``, unquantized); ONNX: no
    label / ``"8-bit"`` / ``"int8"`` -> 8 (``int8`` dynamic). Raises ``QuantizationLabelError`` for any other
    label (``Q4_0``, ``Q4_K_S``, ``7-bit``, ...) instead of converting to a nearby type silently."""
    import re

    fmt = str(output_format or "").strip().lower()
    raw = str(quantization or "").strip()
    norm = raw.lower().replace(" ", "")
    if fmt == "onnx":
        if norm in {"", "none", "null", "false", "off", "no", "8-bit", "8bit", "int8", "q8", "8"}:
            return {"bits": 8, "label": "int8 (dynamic)"}
        raise QuantizationLabelError(
            f"Quantization {raw!r} is not one BrewSLM converts ONNX to: ONNX export is int8 (8-bit) only."
        )
    if norm in {"", "none", "null", "false", "off", "no"}:
        return {"bits": 16, "label": "F16 (unquantized)"}
    for bits, gguf_type in GGUF_TYPE_BY_BITS.items():
        if norm == gguf_type.lower():
            return {"bits": bits, "label": gguf_type}
    match = re.fullmatch(r"(?:(\d+)-?bits?|q(\d+)|int(\d+)|(\d+))", norm)
    bits = int(next(g for g in match.groups() if g)) if match else None
    if bits in GGUF_TYPE_BY_BITS:
        return {"bits": bits, "label": GGUF_TYPE_BY_BITS[bits]}
    supported = ", ".join(f"{b}-bit ({t})" for b, t in GGUF_TYPE_BY_BITS.items())
    raise QuantizationLabelError(
        f"Quantization {raw!r} is not one BrewSLM converts GGUF to. Pick one of: {supported}."
    )


def quantization_bits(quantization: str | None, output_format: str) -> int | None:
    """The bit width of ``quantization_plan``, or None for a label it refuses."""
    try:
        return int(quantization_plan(quantization, output_format)["bits"])
    except QuantizationLabelError:
        return None


def can_convert_inline() -> bool:
    """Whether a run can be converted right now without a queue (the external quantize command is set)."""
    try:
        return _resolve_backend() == "external" and bool(settings.QUANTIZE_EXTERNAL_CMD.strip())
    except ValueError:
        return False


# Cross-process lock file per output folder (the API process's inline export conversion and the Celery
# worker's queued quantize both take it), and the success marker written next to a finished conversion.
CONVERT_LOCK_NAME = ".convert.lock"
CONVERT_OK_SUFFIX = ".ok"


def conversion_marker(path: Path) -> Path:
    return path.with_name(path.name + CONVERT_OK_SUFFIX)


def is_complete_conversion(path: Path) -> bool:
    """Whether ``path`` (a ``quantized_<bits>bit.<fmt>`` file) is a finished conversion: its success marker
    (``<file>.ok``, written after the move) records this file's size; or -- a file converted before markers
    existed -- the folder's ``quantize_report.json`` says the command that wrote this very path exited 0 and
    was written after the file. A truncated / in-progress file has neither."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    marker = conversion_marker(path)
    if marker.is_file():
        try:
            return int(json.loads(marker.read_text(encoding="utf-8")).get("size", -1)) == size
        except (OSError, ValueError, AttributeError):
            return False
    report = path.parent / "quantize_report.json"
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        return (
            isinstance(data, dict) and data.get("returncode") == 0 and str(path) in str(data.get("command") or "")
            and report.stat().st_mtime >= path.stat().st_mtime
        )
    except (OSError, ValueError):
        return False


def _lock_fd(output_dir: Path) -> int:
    import os

    output_dir.mkdir(parents=True, exist_ok=True)
    return os.open(output_dir / CONVERT_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o644)


class conversion_file_lock:
    """Blocking cross-process lock on one output folder (``flock``), for the Celery worker's quantize job."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.fd: int | None = None

    def __enter__(self):
        import fcntl

        self.fd = _lock_fd(self.output_dir)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        import os

        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
        return False


class run_conversion_lock:
    """Async form of ``conversion_file_lock`` on the run's ``compressed/exp-<id>/`` folder (non-blocking
    tries, so a cancelled waiter never leaves a lock behind; nothing is kept in memory): one conversion of a
    run at a time across exports and queued quantize jobs; a second export waits, then reuses the file."""

    def __init__(self, project_id: int, experiment_id: int):
        self.output_dir = experiment_compression_dir(project_id, experiment_id)
        self.fd: int | None = None

    async def __aenter__(self):
        import fcntl
        import os

        fd = _lock_fd(self.output_dir)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        return self

    async def __aexit__(self, *exc):
        import fcntl
        import os

        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
        return False


def conversion_tmp_paths(output_dir: Path, final_path: Path) -> tuple[Path, Path]:
    """A private temp folder (``.convert-<id>/``) and the model path inside it, for one conversion."""
    import uuid

    tmp_dir = Path(output_dir) / f"{CONVERT_TMP_PREFIX}{uuid.uuid4().hex[:12]}"
    return tmp_dir, tmp_dir / Path(final_path).name


def finish_conversion(output_dir: Path, tmp_dir: Path, tmp_path: Path, final_path: Path, succeeded: bool) -> bool:
    """Move a finished conversion into place and mark it complete; always remove the temp folder. Sidecars
    first (ONNX external data, tokenizer files), then the old marker is dropped, the model file replaced
    (``os.replace``) and the new marker written, so a reader never sees a marked partial file. Returns
    whether the file was published."""
    import os

    output_dir, tmp_dir, tmp_path, final_path = Path(output_dir), Path(tmp_dir), Path(tmp_path), Path(final_path)
    try:
        if not (succeeded and tmp_path.is_file()):
            return False
        for extra in sorted(p for p in tmp_dir.rglob("*") if p.is_file() and p != tmp_path):
            dest = output_dir / extra.relative_to(tmp_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(extra, dest)
        marker = conversion_marker(final_path)
        marker.unlink(missing_ok=True)
        os.replace(tmp_path, final_path)
        tmp_marker = tmp_dir / marker.name
        tmp_marker.write_text(json.dumps({
            "size": final_path.stat().st_size, "created_at": datetime.now(timezone.utc).isoformat(),
        }), encoding="utf-8")
        os.replace(tmp_marker, marker)
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def convert_run_for_export(
    project_id: int, experiment_id: int, model_path: str, bits: int, output_format: str,
) -> dict:
    """Convert a run's model to GGUF / ONNX in this process (no Celery queue), with the same command, output
    folder (``compressed/exp-<id>/``) and ``quantize_report.json`` as Compression -> Quantize, so an export
    can do the step itself. The command writes into a private temp folder; ``finish_conversion`` publishes it
    only when the command succeeded and removes the folder whatever happens (failure, timeout, cancel), so no
    partial file is ever packaged. Callers hold ``run_conversion_lock``. Returns the execution record
    (``returncode``, ``duration_seconds``, ``stdout`` / ``stderr``, ``output_model_path``). Raises
    ``ValueError`` when inline conversion is not configured."""
    if not can_convert_inline():
        raise ValueError("No external compression backend is configured (COMPRESSION_BACKEND=external with QUANTIZE_EXTERNAL_CMD).")
    normalized_format = str(output_format or "gguf").strip().lower()
    output_dir = experiment_compression_dir(project_id, experiment_id)
    final_path = output_dir / f"quantized_{bits}bit.{normalized_format}"
    tmp_dir, tmp_path = conversion_tmp_paths(output_dir, final_path)
    tmp_dir.mkdir(parents=True)
    execution: dict = {}
    try:
        command = _render_external_command(
            settings.QUANTIZE_EXTERNAL_CMD.strip(),
            {
                "project_id": project_id,
                "model_path": model_path,
                "bits": bits,
                "output_format": normalized_format,
                "output_dir": str(tmp_dir),
                "output_model_path": str(tmp_path),
                "backend_dir": str(BACKEND_DIR),
            },
        )
        command, _note = _normalize_external_python_command(command)
        command, _env_note = _prepend_quantize_runtime_env(command)
        execution = await _run_external_command(command, cwd=BACKEND_DIR)
    finally:
        finish_conversion(output_dir, tmp_dir, tmp_path, final_path, execution.get("returncode") == 0)
    (output_dir / "quantize_report.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
    return {**execution, "output_model_path": str(final_path), "bits": bits, "output_format": normalized_format}


async def merge_lora(
    project_id: int,
    base_model_path: str,
    lora_adapter_path: str,
) -> dict:
    """Merge LoRA adapter using external runtime or explicit stub mode."""
    output_dir = _output_dir_for_source(project_id, lora_adapter_path) / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = _resolve_backend()
    created_at = datetime.now(timezone.utc).isoformat()

    if backend == "stub":
        if settings.STRICT_EXECUTION_MODE:
            raise StrictExecutionError("compression", "Stub model merging is blocked because STRICT_EXECUTION_MODE is enabled.")
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
    report_path = output_dir / "merge_report.json"
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
    command, command_note = _normalize_external_python_command(command)
    command, env_note = _prepend_quantize_runtime_env(command)
    if env_note:
        command_note = f"{command_note}; {env_note}" if command_note else env_note
    from app.worker import celery_app
    task = celery_app.send_task(
        "run_quantization_job",  # using same worker task, as merge is just quantization config
        kwargs={
            "command": command,
            "report_path": str(report_path),
            "project_id": project_id,
            "job_type": "merge",
        }
    )

    return {
        "project_id": project_id,
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "output_dir": str(output_dir),
        "output_model_path": str(output_model_path),
        "status": "queued",
        "created_at": created_at,
        "report_path": str(report_path),
        "task_id": task.id,
        "command_note": command_note,
    }


async def merge_models(
    project_id: int,
    model_paths: list[str],
    *,
    merge_method: str = "ties",
    weights: list[float] | None = None,
    ties_density: float = 0.2,
) -> dict:
    """Queue multi-model merge using TIES/DEX style model soup command."""
    normalized_paths = [str(item or "").strip() for item in model_paths if str(item or "").strip()]
    if len(normalized_paths) < 2:
        raise ValueError("At least two model paths are required for model merge.")

    method = str(merge_method or "ties").strip().lower()
    if method not in {"ties", "dex"}:
        raise ValueError("merge_method must be one of: ties, dex.")

    normalized_weights: list[float] = []
    if weights is not None:
        for raw in list(weights):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                normalized_weights.append(value)
        if normalized_weights and len(normalized_weights) != len(normalized_paths):
            raise ValueError(
                "weights length must match model_paths length when provided."
            )

    safe_ties_density = max(0.01, min(float(ties_density), 1.0))
    output_dir = _compression_dir(project_id) / f"model_soup_{method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = _resolve_backend()
    created_at = datetime.now(timezone.utc).isoformat()

    if backend == "stub":
        if settings.STRICT_EXECUTION_MODE:
            raise StrictExecutionError("compression", "Stub model merging (multi) is blocked because STRICT_EXECUTION_MODE is enabled.")
        if not settings.ALLOW_STUB_COMPRESSION:
            raise ValueError(
                "Stub compression backend is disabled. Configure COMPRESSION_BACKEND=external "
                "with MERGE_MODELS_EXTERNAL_CMD or set ALLOW_STUB_COMPRESSION=true for demos."
            )
        return {
            "project_id": project_id,
            "merge_method": method,
            "model_paths": normalized_paths,
            "weights": normalized_weights or None,
            "ties_density": safe_ties_density,
            "output_dir": str(output_dir),
            "status": "simulated",
            "created_at": created_at,
            "message": "No external model merge backend configured; returning simulated result.",
        }

    template = settings.MERGE_MODELS_EXTERNAL_CMD.strip()
    if not template:
        raise ValueError("MERGE_MODELS_EXTERNAL_CMD is required when COMPRESSION_BACKEND=external")

    models_file_path = output_dir / "models.json"
    models_file_path.write_text(
        json.dumps(
            {
                "project_id": project_id,
                "method": method,
                "model_paths": normalized_paths,
                "weights": normalized_weights,
                "ties_density": safe_ties_density,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    output_model_path = output_dir / "merged_model"
    report_path = output_dir / "merge_models_report.json"
    weights_csv = ",".join(f"{item:.8f}" for item in normalized_weights) if normalized_weights else ""
    command = _render_external_command(
        template,
        {
            "project_id": project_id,
            "models_file_path": str(models_file_path),
            "merge_method": method,
            "output_dir": str(output_dir),
            "output_model_path": str(output_model_path),
            "weights_csv": weights_csv,
            "ties_density": safe_ties_density,
            "backend_dir": str(BACKEND_DIR),
        },
    )
    command, command_note = _normalize_external_python_command(command)
    command, env_note = _prepend_quantize_runtime_env(command)
    if env_note:
        command_note = f"{command_note}; {env_note}" if command_note else env_note

    from app.worker import celery_app

    task = celery_app.send_task(
        "run_quantization_job",
        kwargs={
            "command": command,
            "report_path": str(report_path),
            "project_id": project_id,
            "job_type": f"model_merge_{method}",
        },
    )

    return {
        "project_id": project_id,
        "merge_method": method,
        "model_paths": normalized_paths,
        "weights": normalized_weights or None,
        "ties_density": safe_ties_density,
        "output_dir": str(output_dir),
        "output_model_path": str(output_model_path),
        "status": "queued",
        "created_at": created_at,
        "models_file_path": str(models_file_path),
        "report_path": str(report_path),
        "task_id": task.id,
        "command_note": command_note,
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
        if settings.STRICT_EXECUTION_MODE:
            raise StrictExecutionError("compression", "Stub benchmarking is blocked because STRICT_EXECUTION_MODE is enabled.")
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
    benchmark_output_path = output_dir / "benchmark_results.json"
    job_report_path = output_dir / "benchmark_report.json"
    command = _render_external_command(
        template,
        {
            "project_id": project_id,
            "model_path": model_path,
            "num_samples": num_samples,
            "output_dir": str(output_dir),
            "report_path": str(benchmark_output_path),
            "benchmark_output_path": str(benchmark_output_path),
            "backend_dir": str(BACKEND_DIR),
        },
    )
    command, command_note = _normalize_external_python_command(command)
    command, env_note = _prepend_quantize_runtime_env(command)
    if env_note:
        command_note = f"{command_note}; {env_note}" if command_note else env_note

    from app.worker import celery_app
    task = celery_app.send_task(
        "run_benchmark_job",
        kwargs={
            "command": command,
            "report_path": str(job_report_path),
            "benchmark_output_path": str(benchmark_output_path),
            "project_id": project_id,
        }
    )

    result.update(
        {
            "status": "queued",
            "report_path": str(job_report_path),
            "benchmark_report_path": str(benchmark_output_path),
            "task_id": task.id,
            "command_note": command_note,
        }
    )
    return result


def get_compression_job_status(project_id: int, report_path: str) -> dict:
    """Inspect queued/completed status from a worker report path."""
    safe_report_path = _resolve_report_path(project_id, report_path)
    if not safe_report_path.exists():
        return {
            "project_id": project_id,
            "status": "running",
            "report_path": str(safe_report_path),
        }

    try:
        payload = json.loads(safe_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse report file {safe_report_path}: {e}")
    returncode = payload.get("returncode")
    stdout = payload.get("stdout", "")
    stderr = payload.get("stderr", "")
    digest = sha256(f"{stdout}\n{stderr}".encode("utf-8")).hexdigest()
    return {
        "project_id": project_id,
        "status": "completed" if returncode == 0 else "failed",
        "report_path": str(safe_report_path),
        "returncode": returncode,
        "duration_seconds": payload.get("duration_seconds"),
        "command": payload.get("command"),
        "stdout": stdout,
        "stderr": stderr,
        "output_digest": digest,
        "benchmark_report_path": payload.get("benchmark_report_path"),
        "benchmark": payload.get("benchmark"),
    }
