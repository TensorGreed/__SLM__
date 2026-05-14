"""Plugin loader for custom dataset-import mappers (Phase H).

Lets users drop a Python module under ``DATASET_MAPPER_PLUGIN_MODULES``
that registers extra :class:`TargetMapper` subclasses alongside the
built-ins. Mirrors the data-adapter plugin pattern (see
``data_adapter_service.load_data_adapter_plugins``) so plugin authors
who've shipped a data adapter can ship a mapper with the same mental
model:

  - Module exports ``register_dataset_mappers(register)`` (preferred)
    where ``register(mapper_id, factory)`` adds one mapper to the
    registry, OR
  - Module exports a top-level ``DATASET_MAPPERS`` dict mapping
    ``mapper_id`` → factory callable.

The factory must return an object that satisfies the
:class:`TargetMapper` protocol (``mapper_id`` attr, ``declared_target()``
method, ``transform()`` iterable). The loader doesn't instantiate the
mapper until a caller resolves the id — so a plugin module is cheap to
import.

Failures during a single plugin import are caught + recorded; the rest
of the modules in the list still load. The loader returns a
diagnostic dict the API can surface so a misconfigured plugin doesn't
silently disappear.

Boot-time entry point: ``load_dataset_mapper_plugins_from_settings()``
runs from ``app.main.lifespan``. Tests call ``load_dataset_mapper_plugins``
with an explicit list to keep the suite hermetic.
"""

from __future__ import annotations

import importlib
import inspect
import threading
from typing import Any, Callable

from app.services.dataset_import.protocols import TargetMapper
from app.services.dataset_import.registry import register_mapper


# Track which modules have loaded so a hot-reload (or repeated boot
# call) doesn't re-register the same mappers. ``_PLUGIN_ERRORS`` keeps
# the last failure message per module so the diagnostic surface can
# show the operator why a plugin didn't appear in the registry.
_LOADED_PLUGIN_MODULES: set[str] = set()
_PLUGIN_ERRORS: dict[str, str] = {}
_LOAD_LOCK = threading.RLock()


def _registered_plugins() -> set[str]:
    return set(_LOADED_PLUGIN_MODULES)


def _plugin_errors() -> dict[str, str]:
    return dict(_PLUGIN_ERRORS)


def _register_one(mapper_id: str, factory: Callable[[], TargetMapper]) -> None:
    """Same shape as the built-in mappers — delegate straight to the
    registry. Raises if the factory isn't callable or the id is empty."""

    if not isinstance(mapper_id, str) or not mapper_id.strip():
        raise ValueError("plugin tried to register an empty mapper_id")
    if not callable(factory):
        raise ValueError(
            f"plugin mapper '{mapper_id}' factory is not callable"
        )
    register_mapper(mapper_id, factory)


def load_dataset_mapper_plugins(
    module_paths: list[str],
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Import each module + invoke its registration hook.

    Returns a diagnostic dict::

        {
            "requested_modules": [...],
            "loaded_modules": [...],
            "failed_modules": {module: error_message},
            "registered_mappers": int,
        }

    ``force_reload=True`` re-imports already-loaded modules (handy for
    iterating on a plugin during development).
    """

    loaded: list[str] = []
    failed: dict[str, str] = {}
    registered = 0

    with _LOAD_LOCK:
        for raw in module_paths:
            module_name = str(raw or "").strip()
            if not module_name:
                continue
            if module_name in _LOADED_PLUGIN_MODULES and not force_reload:
                loaded.append(module_name)
                continue

            try:
                module = importlib.import_module(module_name)
                if force_reload:
                    module = importlib.reload(module)

                count_before = _registered_count()
                hook = getattr(module, "register_dataset_mappers", None)
                if callable(hook):
                    params = inspect.signature(hook).parameters
                    if len(params) != 1:
                        raise ValueError(
                            "register_dataset_mappers(register) must accept "
                            "exactly one argument"
                        )
                    hook(_register_one)

                mapping = getattr(module, "DATASET_MAPPERS", None)
                if isinstance(mapping, dict):
                    for mapper_id, factory in mapping.items():
                        _register_one(str(mapper_id), factory)

                if not callable(hook) and not isinstance(mapping, dict):
                    raise ValueError(
                        "module must export either "
                        "register_dataset_mappers(register) or a "
                        "DATASET_MAPPERS dict"
                    )

                _LOADED_PLUGIN_MODULES.add(module_name)
                _PLUGIN_ERRORS.pop(module_name, None)
                loaded.append(module_name)
                registered += max(0, _registered_count() - count_before)
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                _PLUGIN_ERRORS[module_name] = message
                failed[module_name] = message

    return {
        "requested_modules": [str(m).strip() for m in module_paths if str(m).strip()],
        "loaded_modules": sorted(set(loaded)),
        "failed_modules": failed,
        "registered_mappers": registered,
    }


def _registered_count() -> int:
    """Snapshot of the global registry size — used to count how many
    mappers a plugin module added across both registration hooks."""

    from app.services.dataset_import.registry import _MAPPERS

    return len(_MAPPERS)


def load_dataset_mapper_plugins_from_settings(
    *, force_reload: bool = False
) -> dict[str, Any]:
    """Read ``settings.DATASET_MAPPER_PLUGIN_MODULES`` + delegate.

    Returns the same diagnostic dict as :func:`load_dataset_mapper_plugins`
    plus a ``status`` field when no modules are configured (the boot path
    calls this unconditionally; an empty list is a normal state).
    """

    from app.config import settings

    raw_list = getattr(settings, "DATASET_MAPPER_PLUGIN_MODULES", None) or []
    modules = [str(item).strip() for item in raw_list if str(item).strip()]
    if not modules:
        return {
            "requested_modules": [],
            "loaded_modules": sorted(_LOADED_PLUGIN_MODULES),
            "failed_modules": dict(_PLUGIN_ERRORS),
            "registered_mappers": 0,
            "status": "no_plugin_modules_configured",
        }
    return load_dataset_mapper_plugins(modules, force_reload=force_reload)


def get_plugin_diagnostics() -> dict[str, Any]:
    """Read-only snapshot for the API's diagnostic endpoint."""

    return {
        "loaded_modules": sorted(_LOADED_PLUGIN_MODULES),
        "failed_modules": dict(_PLUGIN_ERRORS),
    }
