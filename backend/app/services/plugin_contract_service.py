"""Plugin contract orchestration service (priority.md P37, Wave H).

Glue layer between the pure contract validators in
:mod:`app.services.plugin_contracts` and the live registries owned by
the per-kind services. Provides:

- :func:`validate_extension` — import a module (optionally reloading
  it) and run the contract suite. Read-only with respect to the live
  registries.
- :func:`list_extensions` — combined catalog of extension state across
  every plugin kind, suitable for ``GET /api/extensions``.
- :func:`reload_extensions` — safe reload entry point that re-imports
  plugin modules for the kinds that have a live loader. Returns a
  per-kind status report.

Extension events are administrative (no project context), so the
service deliberately does **not** emit RunEvents — the canonical
``run_events`` table is project-scoped. The
:data:`EXTENSION_LOAD_FAILED` / :data:`EXTENSION_CONTRACT_INVALID`
reason codes are reserved here for future project-scoped emits
(e.g. autopilot triggering a plugin reload).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from app.config import settings
from app.services.data_adapter_service import (
    clear_plugin_data_adapters,
    list_data_adapter_catalog,
    load_data_adapter_plugins_from_settings,
)
from app.services.plugin_contracts import (
    KIND_HAS_MODULE_LOADER,
    KIND_RECOGNIZED_EXPORTS,
    KIND_SUPPORTS_SAFE_RELOAD,
    KNOWN_PLUGIN_KINDS,
    PLUGIN_CONTRACT_VERSIONS,
    PluginContractReport,
    PluginKind,
    normalize_plugin_kind,
    validate_plugin_module,
)
from app.services.training_runtime_service import (
    list_runtime_catalog,
    reload_runtime_plugins_from_settings,
    runtime_plugin_status,
)


# Mapping from kind to the settings key that lists plugin modules to
# import on startup. ``None`` for kinds whose loader hasn't shipped yet.
_KIND_SETTINGS_KEY: dict[PluginKind, str | None] = {
    "data_adapter": "DATA_ADAPTER_PLUGIN_MODULES",
    "training_runtime": "TRAINING_RUNTIME_PLUGIN_MODULES",
    "domain_pack": None,
    "eval_pack": None,
}


def _configured_modules_for_kind(kind: PluginKind) -> list[str]:
    key = _KIND_SETTINGS_KEY[kind]
    if key is None:
        return []
    raw = getattr(settings, key, None) or []
    return [str(item).strip() for item in raw if str(item).strip()]


# ----------------------------------------------------------------------
# Validate
# ----------------------------------------------------------------------


def validate_extension(
    kind: str,
    module_path: str,
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Import ``module_path`` and run contract checks.

    The module is imported (or reloaded if ``force_reload`` and the
    module is already in :data:`sys.modules`) but **not** registered
    into any live registry — this entry point is for pre-flight checks.

    Returns the report dict from :meth:`PluginContractReport.to_dict`
    augmented with ``import_error`` when the import itself failed.

    Raises :class:`ValueError` for unknown kinds; callers map that to
    HTTP 400.
    """

    normalized_kind = normalize_plugin_kind(kind)
    module_name = str(module_path or "").strip()
    if not module_name:
        raise ValueError(
            "extension_module_required:module_path must be a non-empty string"
        )

    try:
        if force_reload and module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "kind": normalized_kind,
            "module": module_name,
            "contract_version": PLUGIN_CONTRACT_VERSIONS[normalized_kind],
            "ok": False,
            "checks": [
                {
                    "name": "module_importable",
                    "ok": False,
                    "message": f"Import failed: {exc}",
                }
            ],
            "declared_ids": [],
            "declared_version": None,
            "import_error": str(exc),
        }

    report: PluginContractReport = validate_plugin_module(module, normalized_kind)
    payload = report.to_dict()
    payload["import_error"] = None
    return payload


# ----------------------------------------------------------------------
# List
# ----------------------------------------------------------------------


def _data_adapter_extension_view() -> dict[str, Any]:
    catalog = list_data_adapter_catalog()
    # Catalog includes the synthetic ``auto`` entry alongside real adapters;
    # subtract it so ``registered_count`` reflects only the loadable set.
    adapter_total = max(0, len(catalog.get("adapters") or {}) - 1)
    return {
        "kind": "data_adapter",
        "contract_version": PLUGIN_CONTRACT_VERSIONS["data_adapter"],
        "supports_safe_reload": KIND_SUPPORTS_SAFE_RELOAD["data_adapter"],
        "has_module_loader": KIND_HAS_MODULE_LOADER["data_adapter"],
        "settings_key": _KIND_SETTINGS_KEY["data_adapter"],
        "configured_modules": _configured_modules_for_kind("data_adapter"),
        "loaded_modules": list(catalog.get("loaded_plugin_modules") or []),
        "load_errors": dict(catalog.get("plugin_load_errors") or {}),
        "registered_count": adapter_total,
        "recognized_exports": list(KIND_RECOGNIZED_EXPORTS["data_adapter"]),
    }


def _training_runtime_extension_view() -> dict[str, Any]:
    status = runtime_plugin_status()
    catalog = list_runtime_catalog()
    plugin_runtimes = [
        item for item in catalog.get("runtimes") or [] if not item.get("is_builtin")
    ]
    return {
        "kind": "training_runtime",
        "contract_version": PLUGIN_CONTRACT_VERSIONS["training_runtime"],
        "supports_safe_reload": KIND_SUPPORTS_SAFE_RELOAD["training_runtime"],
        "has_module_loader": KIND_HAS_MODULE_LOADER["training_runtime"],
        "settings_key": _KIND_SETTINGS_KEY["training_runtime"],
        "configured_modules": _configured_modules_for_kind("training_runtime"),
        "loaded_modules": list(status.get("loaded_modules") or []),
        "load_errors": dict(status.get("failed_modules") or {}),
        "registered_count": len(plugin_runtimes),
        "recognized_exports": list(KIND_RECOGNIZED_EXPORTS["training_runtime"]),
    }


def _declarative_kind_view(kind: PluginKind) -> dict[str, Any]:
    return {
        "kind": kind,
        "contract_version": PLUGIN_CONTRACT_VERSIONS[kind],
        "supports_safe_reload": KIND_SUPPORTS_SAFE_RELOAD[kind],
        "has_module_loader": KIND_HAS_MODULE_LOADER[kind],
        "settings_key": _KIND_SETTINGS_KEY[kind],
        "configured_modules": [],
        "loaded_modules": [],
        "load_errors": {},
        "registered_count": 0,
        "recognized_exports": list(KIND_RECOGNIZED_EXPORTS[kind]),
        "note": (
            "Module loader for this kind is planned for P38 "
            "(Wave H scaffold generator)."
        ),
    }


_KIND_VIEW_BUILDERS = {
    "data_adapter": _data_adapter_extension_view,
    "training_runtime": _training_runtime_extension_view,
    "domain_pack": lambda: _declarative_kind_view("domain_pack"),
    "eval_pack": lambda: _declarative_kind_view("eval_pack"),
}


def list_extensions() -> dict[str, Any]:
    """Return the combined extensions catalog for ``GET /api/extensions``."""

    return {
        "kinds": [_KIND_VIEW_BUILDERS[kind]() for kind in KNOWN_PLUGIN_KINDS],
    }


# ----------------------------------------------------------------------
# Reload
# ----------------------------------------------------------------------


def _reload_data_adapter_kind() -> dict[str, Any]:
    clear_plugin_data_adapters()
    status = load_data_adapter_plugins_from_settings(force_reload=True)
    return {
        "kind": "data_adapter",
        "requested_modules": list(status.get("requested_modules") or []),
        "loaded_modules": list(status.get("loaded_modules") or []),
        "failed_modules": dict(status.get("failed_modules") or {}),
        "registered_count": int(status.get("registered_adapters") or 0),
    }


def _reload_training_runtime_kind() -> dict[str, Any]:
    result = reload_runtime_plugins_from_settings()
    reload_status = result.get("reload") or {}
    return {
        "kind": "training_runtime",
        "requested_modules": list(reload_status.get("requested_modules") or []),
        "loaded_modules": list(reload_status.get("loaded_modules") or []),
        "failed_modules": dict(reload_status.get("failed_modules") or {}),
        "registered_count": int(reload_status.get("registered_runtime_count") or 0),
    }


_KIND_RELOADERS = {
    "data_adapter": _reload_data_adapter_kind,
    "training_runtime": _reload_training_runtime_kind,
}


def reload_extensions(kind: str | None = None) -> dict[str, Any]:
    """Re-import plugin modules listed in settings for the requested kind(s).

    ``kind=None`` reloads every kind that has a live loader. Kinds
    without a loader return ``status="not_supported"``.
    """

    if kind is None:
        kinds: list[PluginKind] = list(KNOWN_PLUGIN_KINDS)
    else:
        kinds = [normalize_plugin_kind(kind)]

    results: list[dict[str, Any]] = []
    for plugin_kind in kinds:
        reloader = _KIND_RELOADERS.get(plugin_kind)
        if reloader is None:
            results.append(
                {
                    "kind": plugin_kind,
                    "status": "not_supported",
                    "message": (
                        "Reload not implemented for this kind yet "
                        "(P38 will add the loader)."
                    ),
                }
            )
            continue

        try:
            payload = reloader()
        except Exception as exc:
            results.append(
                {
                    "kind": plugin_kind,
                    "status": "error",
                    "message": str(exc),
                }
            )
            continue

        payload["status"] = "ok" if not payload.get("failed_modules") else "partial"
        results.append(payload)

    return {"results": results}


__all__ = [
    "validate_extension",
    "list_extensions",
    "reload_extensions",
]
