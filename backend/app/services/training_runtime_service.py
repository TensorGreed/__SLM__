"""Training runtime plugin SDK and registry.

This service provides a pluggable runtime layer for experiment execution so
training launch logic can be extended beyond built-in runtime backends.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import shlex
import shutil
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import settings


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

LEGACY_SIMULATE_BACKEND = "simulate"
LEGACY_EXTERNAL_BACKEND = "external"

BUILTIN_SIMULATE_RUNTIME_ID = "builtin.simulate"
BUILTIN_EXTERNAL_CELERY_RUNTIME_ID = "builtin.external_celery"


@dataclass(frozen=True)
class TrainingRuntimeSpec:
    """Declarative metadata for a runtime plugin."""

    runtime_id: str
    label: str
    description: str
    execution_backend: str
    required_dependencies: list[str] = field(default_factory=list)
    supports_task_tracking: bool = False
    supports_cancellation: bool = True
    is_builtin: bool = False


@dataclass(frozen=True)
class TrainingRuntimeStartContext:
    """Inputs provided to runtime plugin start handlers."""

    project_id: int
    experiment_id: int
    base_model: str
    config: dict[str, Any]
    output_dir: Path
    config_path: Path
    prepared_dir: Path
    train_file: Path
    val_file: Path
    simulate_runner: Callable[[int, dict[str, Any]], Awaitable[None]] | None = None


@dataclass(frozen=True)
class TrainingRuntimeStartResult:
    """Result returned by runtime plugin start handlers."""

    message: str
    task_id: str | None = None
    runtime_updates: dict[str, Any] = field(default_factory=dict)


RuntimeValidateFn = Callable[[], list[str]]
RuntimeStartFn = Callable[[TrainingRuntimeStartContext], Awaitable[TrainingRuntimeStartResult]]


@dataclass(frozen=True)
class _RuntimePlugin:
    spec: TrainingRuntimeSpec
    validate: RuntimeValidateFn
    start: RuntimeStartFn


_registry_lock = threading.Lock()
_runtime_plugins: dict[str, _RuntimePlugin] = {}
_plugins_loaded = False

_LEGACY_RUNTIME_ALIASES = {
    LEGACY_SIMULATE_BACKEND: BUILTIN_SIMULATE_RUNTIME_ID,
    LEGACY_EXTERNAL_BACKEND: BUILTIN_EXTERNAL_CELERY_RUNTIME_ID,
}


def _normalize_runtime_id(value: str | None) -> str:
    return str(value or "").strip().lower()


def _register_plugin(plugin: _RuntimePlugin) -> None:
    runtime_id = _normalize_runtime_id(plugin.spec.runtime_id)
    if not runtime_id:
        raise ValueError("runtime_id must be non-empty")
    _runtime_plugins[runtime_id] = plugin


def register_training_runtime_plugin(
    *,
    runtime_id: str,
    label: str,
    description: str,
    execution_backend: str,
    validate: RuntimeValidateFn,
    start: RuntimeStartFn,
    required_dependencies: list[str] | None = None,
    supports_task_tracking: bool = False,
    supports_cancellation: bool = True,
    is_builtin: bool = False,
) -> None:
    """Public plugin SDK entry-point for custom runtime modules."""

    runtime_token = _normalize_runtime_id(runtime_id)
    if not runtime_token:
        raise ValueError("runtime_id must be non-empty")
    backend_token = str(execution_backend or "").strip().lower()
    if backend_token not in {"local", "celery", "external"}:
        raise ValueError(
            f"execution_backend '{execution_backend}' is invalid; expected local|celery|external"
        )
    if not callable(validate):
        raise ValueError("validate callback is required")
    if not callable(start):
        raise ValueError("start callback is required")

    deps: list[str] = []
    for item in list(required_dependencies or []):
        token = str(item or "").strip()
        if token and token not in deps:
            deps.append(token)

    spec = TrainingRuntimeSpec(
        runtime_id=runtime_token,
        label=str(label or runtime_token),
        description=str(description or "").strip() or "Custom training runtime plugin.",
        execution_backend=backend_token,
        required_dependencies=deps,
        supports_task_tracking=bool(supports_task_tracking),
        supports_cancellation=bool(supports_cancellation),
        is_builtin=bool(is_builtin),
    )
    _register_plugin(
        _RuntimePlugin(
            spec=spec,
            validate=validate,
            start=start,
        )
    )


def _render_external_command(template: str, placeholders: dict[str, str | int]) -> str:
    try:
        return template.format(**placeholders)
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"TRAINING_EXTERNAL_CMD missing placeholder value: {missing}")


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


def _validate_builtin_simulate() -> list[str]:
    if settings.ALLOW_SIMULATED_TRAINING:
        return []
    return [
        (
            "Simulated training backend is disabled. "
            "Set ALLOW_SIMULATED_TRAINING=true for demos or configure TRAINING_BACKEND=external."
        )
    ]


async def _start_builtin_simulate(ctx: TrainingRuntimeStartContext) -> TrainingRuntimeStartResult:
    if ctx.simulate_runner is None:
        raise ValueError("simulate runtime is not available in this execution context")
    asyncio.create_task(ctx.simulate_runner(ctx.experiment_id, dict(ctx.config or {})))
    return TrainingRuntimeStartResult(
        message="Simulated training started. Connecting to telemetry stream...",
        task_id=None,
        runtime_updates={
            "backend": LEGACY_SIMULATE_BACKEND,
            "runtime_kind": "simulate",
        },
    )


def _validate_builtin_external_celery() -> list[str]:
    errors: list[str] = []
    if not settings.TRAINING_EXTERNAL_CMD.strip():
        errors.append("TRAINING_EXTERNAL_CMD is required when TRAINING_BACKEND=external")
    return errors


async def _start_builtin_external_celery(ctx: TrainingRuntimeStartContext) -> TrainingRuntimeStartResult:
    command_template = settings.TRAINING_EXTERNAL_CMD.strip()
    if not command_template:
        raise ValueError("TRAINING_EXTERNAL_CMD is required when TRAINING_BACKEND=external")

    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    external_command = _render_external_command(
        command_template,
        {
            "project_id": ctx.project_id,
            "experiment_id": ctx.experiment_id,
            "output_dir": str(output_dir),
            "base_model": ctx.base_model,
            "backend_dir": str(BACKEND_DIR),
            "config_path": str(ctx.config_path),
            "data_dir": str(settings.DATA_DIR),
            "prepared_dir": str(ctx.prepared_dir),
            "train_file": str(ctx.train_file),
            "val_file": str(ctx.val_file),
        },
    )
    external_command, command_note = _normalize_external_python_command(external_command)

    from app.worker import celery_app

    task = celery_app.send_task(
        "run_training_job",
        kwargs={
            "experiment_id": ctx.experiment_id,
            "command": external_command,
            "log_path": str(output_dir / "external_training.log"),
            "output_dir": str(output_dir),
        },
    )
    updates: dict[str, Any] = {
        "backend": LEGACY_EXTERNAL_BACKEND,
        "runtime_kind": "external_celery",
        "command": external_command,
        "log_path": str(output_dir / "external_training.log"),
        "config_path": str(ctx.config_path),
        "train_file": str(ctx.train_file),
        "val_file": str(ctx.val_file),
        "task_id": task.id,
    }
    if command_note:
        updates["command_note"] = command_note
    return TrainingRuntimeStartResult(
        message="External training command started.",
        task_id=task.id,
        runtime_updates=updates,
    )


def _load_runtime_plugins_from_settings() -> None:
    for module_path in settings.TRAINING_RUNTIME_PLUGIN_MODULES:
        path = str(module_path or "").strip()
        if not path:
            continue
        module = importlib.import_module(path)
        register_fn = getattr(module, "register_training_runtime_plugins", None)
        if register_fn is None:
            continue
        if not callable(register_fn):
            raise ValueError(
                f"Training runtime plugin module '{path}' has non-callable "
                "'register_training_runtime_plugins'."
            )
        signature = inspect.signature(register_fn)
        if len(signature.parameters) == 0:
            register_fn()
            continue
        register_fn(register_training_runtime_plugin)


def _ensure_plugins_loaded() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    with _registry_lock:
        if _plugins_loaded:
            return
        register_training_runtime_plugin(
            runtime_id=BUILTIN_SIMULATE_RUNTIME_ID,
            label="Built-in Simulate",
            description="Local simulated training loop with synthetic telemetry.",
            execution_backend="local",
            validate=_validate_builtin_simulate,
            start=_start_builtin_simulate,
            required_dependencies=[],
            supports_task_tracking=False,
            supports_cancellation=True,
            is_builtin=True,
        )
        register_training_runtime_plugin(
            runtime_id=BUILTIN_EXTERNAL_CELERY_RUNTIME_ID,
            label="External Command via Celery",
            description=(
                "Dispatches training command template to Celery worker and streams telemetry "
                "from worker logs."
            ),
            execution_backend="celery",
            validate=_validate_builtin_external_celery,
            start=_start_builtin_external_celery,
            required_dependencies=["torch", "transformers", "datasets", "accelerate"],
            supports_task_tracking=True,
            supports_cancellation=True,
            is_builtin=True,
        )
        _load_runtime_plugins_from_settings()
        _plugins_loaded = True


def resolve_default_training_runtime_id() -> str:
    """Resolve server default runtime from legacy TRAINING_BACKEND setting."""
    backend = _normalize_runtime_id(settings.TRAINING_BACKEND)
    if backend in _LEGACY_RUNTIME_ALIASES:
        return _LEGACY_RUNTIME_ALIASES[backend]
    return BUILTIN_EXTERNAL_CELERY_RUNTIME_ID


def resolve_training_runtime_id(config: dict[str, Any] | None) -> tuple[str, str]:
    """Resolve runtime id for an experiment config.

    Returns:
        tuple(runtime_id, source) where source is one of {"config", "settings_default"}.
    """

    _ensure_plugins_loaded()
    cfg = dict(config or {})
    requested = _normalize_runtime_id(
        cfg.get("training_runtime_id")
        or cfg.get("training_runtime")
    )
    if requested and requested != "auto":
        runtime_id = _LEGACY_RUNTIME_ALIASES.get(requested, requested)
        if runtime_id not in _runtime_plugins:
            available = ", ".join(sorted(_runtime_plugins))
            raise ValueError(
                f"Unknown training_runtime_id '{requested}'. Available runtimes: {available}"
            )
        return runtime_id, "config"

    runtime_id = resolve_default_training_runtime_id()
    if runtime_id not in _runtime_plugins:
        available = ", ".join(sorted(_runtime_plugins))
        raise ValueError(
            (
                f"Default runtime '{runtime_id}' (from TRAINING_BACKEND={settings.TRAINING_BACKEND}) "
                f"is not registered. Available runtimes: {available}"
            )
        )
    return runtime_id, "settings_default"


def get_runtime_spec(runtime_id: str) -> TrainingRuntimeSpec:
    _ensure_plugins_loaded()
    token = _normalize_runtime_id(runtime_id)
    plugin = _runtime_plugins.get(token)
    if plugin is None:
        raise ValueError(f"Unknown training runtime '{runtime_id}'")
    return plugin.spec


def validate_runtime(runtime_id: str) -> list[str]:
    _ensure_plugins_loaded()
    token = _normalize_runtime_id(runtime_id)
    plugin = _runtime_plugins.get(token)
    if plugin is None:
        return [f"Unknown training runtime '{runtime_id}'"]
    errors = plugin.validate()
    return [str(item) for item in errors if str(item).strip()]


def list_runtime_specs() -> list[TrainingRuntimeSpec]:
    _ensure_plugins_loaded()
    items = sorted(
        _runtime_plugins.values(),
        key=lambda plugin: (
            0 if plugin.spec.is_builtin else 1,
            plugin.spec.runtime_id,
        ),
    )
    return [item.spec for item in items]


def list_runtime_catalog() -> dict[str, Any]:
    _ensure_plugins_loaded()
    default_runtime_id = resolve_default_training_runtime_id()
    runtimes: list[dict[str, Any]] = []
    for spec in list_runtime_specs():
        runtimes.append(
            {
                "runtime_id": spec.runtime_id,
                "label": spec.label,
                "description": spec.description,
                "execution_backend": spec.execution_backend,
                "required_dependencies": list(spec.required_dependencies),
                "supports_task_tracking": spec.supports_task_tracking,
                "supports_cancellation": spec.supports_cancellation,
                "is_builtin": spec.is_builtin,
            }
        )
    return {
        "default_runtime_id": default_runtime_id,
        "runtime_count": len(runtimes),
        "legacy_aliases": dict(_LEGACY_RUNTIME_ALIASES),
        "runtimes": runtimes,
    }


async def start_runtime(runtime_id: str, ctx: TrainingRuntimeStartContext) -> TrainingRuntimeStartResult:
    _ensure_plugins_loaded()
    token = _normalize_runtime_id(runtime_id)
    plugin = _runtime_plugins.get(token)
    if plugin is None:
        raise ValueError(f"Unknown training runtime '{runtime_id}'")
    return await plugin.start(ctx)

