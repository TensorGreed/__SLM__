"""Plugin contract definitions for BrewSLM extension modules (priority.md P37, Wave H).

This module is the single authoritative source of truth for what a
third-party extension module must export so the core can load it
without surprises. P37 covers four plugin kinds:

- ``data_adapter`` — row-shape adapter plugins (loader already exists
  in :mod:`app.services.data_adapter_service`).
- ``training_runtime`` — runtime backends for the trainer (loader
  already exists in :mod:`app.services.training_runtime_service`).
- ``domain_pack`` — domain overlay packs (no module loader today;
  contract specified so P38 can scaffold + load them).
- ``eval_pack`` — evaluation pack manifests (no module loader today;
  contract specified so P38 can scaffold + load them).

For each kind we record:

- ``CONTRACT_VERSION`` — the schema version the runtime expects.
- A :class:`Protocol` describing the canonical module interface (used
  for documentation + static type-check hints).
- A pure validator returning :class:`PluginContractCheck` rows that the
  service layer surfaces to operators / CLI.

The validators are deliberately **pure** and side-effect free: they
import nothing, mutate nothing. The loader in
:mod:`app.services.plugin_contract_service` is responsible for the
``importlib.import_module`` / ``importlib.reload`` calls and for
threading the resulting module object through these checks.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Literal, Protocol, runtime_checkable


PluginKind = Literal["data_adapter", "training_runtime", "domain_pack", "eval_pack"]

KNOWN_PLUGIN_KINDS: tuple[PluginKind, ...] = (
    "data_adapter",
    "training_runtime",
    "domain_pack",
    "eval_pack",
)


# ----------------------------------------------------------------------
# Contract versions
# ----------------------------------------------------------------------

# Mirrors the runtime constants. Kept here so a plugin author has one
# place to grep, and so :func:`_check_version` can compare without
# importing the runtime services (which would create cycles).
PLUGIN_CONTRACT_VERSIONS: dict[PluginKind, str] = {
    "data_adapter": "slm.data_adapter/v3",
    "training_runtime": "slm.training_runtime/v1",
    "domain_pack": "slm.domain-pack/v1",
    "eval_pack": "slm.evaluation-pack/v2",
}


# Whether the kind has a working hot-reload path in the core today.
# ``False`` means the kind is declared (so scaffolds can be generated
# in P38) but the live registry doesn't accept module-reload yet.
KIND_SUPPORTS_SAFE_RELOAD: dict[PluginKind, bool] = {
    "data_adapter": True,
    "training_runtime": True,
    "domain_pack": False,
    "eval_pack": False,
}


# Whether the runtime currently has a loader that imports plugin
# modules listed in settings. ``True`` for the two kinds wired to
# pluggable loaders, ``False`` for the two declarative kinds whose
# loader is planned for P38 (Wave H scaffold generator).
KIND_HAS_MODULE_LOADER: dict[PluginKind, bool] = {
    "data_adapter": True,
    "training_runtime": True,
    "domain_pack": False,
    "eval_pack": False,
}


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------


@dataclass
class PluginContractCheck:
    """Single check row inside a :class:`PluginContractReport`.

    ``name`` is a short stable token so callers can grep / pin assertions;
    ``message`` is the human-readable detail.
    """

    name: str
    ok: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "message": self.message}


@dataclass
class PluginContractReport:
    kind: PluginKind
    module: str
    contract_version: str
    checks: list[PluginContractCheck] = field(default_factory=list)
    declared_ids: list[str] = field(default_factory=list)
    declared_version: str | None = None

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "module": self.module,
            "contract_version": self.contract_version,
            "declared_version": self.declared_version,
            "declared_ids": list(self.declared_ids),
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
        }


# ----------------------------------------------------------------------
# Protocol interfaces (documentation + structural typing)
# ----------------------------------------------------------------------


@runtime_checkable
class DataAdapterPluginModule(Protocol):
    """A data-adapter plugin module.

    Must export either:
    - ``register_data_adapters(register)`` — preferred, register is the
      callable injected by the loader (see
      :func:`app.services.data_adapter_service.load_data_adapter_plugins`).
    - ``get_data_adapters() -> dict[str, dict | callable]`` — return a
      mapping of adapter id → adapter payload / map_row callable.
    - ``DATA_ADAPTERS`` — module-level constant of the same shape as
      ``get_data_adapters()``'s return value.

    Optional: ``CONTRACT_VERSION`` (string, must match
    :data:`PLUGIN_CONTRACT_VERSIONS["data_adapter"]`) and
    ``__plugin_version__`` (free-form module version string).
    """

    def register_data_adapters(self, register: Callable[..., None]) -> None: ...


@runtime_checkable
class TrainingRuntimePluginModule(Protocol):
    """A training-runtime plugin module.

    Must export:
    - ``register_training_runtime_plugins(register=None)`` — invoked by
      :func:`app.services.training_runtime_service._load_runtime_plugins_from_settings`.
      Accepts either no args (module calls
      ``register_training_runtime_plugin(...)`` directly via the public
      SDK) or a single ``register`` argument (the injected SDK function).
    """

    def register_training_runtime_plugins(self) -> None: ...


@runtime_checkable
class DomainPackPluginModule(Protocol):
    """A domain-pack plugin module.

    Must export either:
    - ``register_domain_packs(register)`` — register receives a
      :class:`DomainPackContract`-shaped dict and persists the pack.
    - ``DOMAIN_PACKS`` — module-level list of
      :class:`DomainPackContract`-shaped dicts.
    - ``get_domain_packs() -> list[dict]`` — same shape, computed at
      load time.

    Each pack dict must include ``pack_id``, ``display_name``, and
    ``$schema``/``schema_ref`` matching
    :data:`PLUGIN_CONTRACT_VERSIONS["domain_pack"]`.

    Note: P37 only specifies the contract; the loader lands in P38.
    """

    def register_domain_packs(self, register: Callable[..., None]) -> None: ...


@runtime_checkable
class EvalPackPluginModule(Protocol):
    """An evaluation-pack plugin module.

    Must export either:
    - ``register_evaluation_packs(register)`` — register receives a
      pack-manifest dict and adds it to the runtime catalog.
    - ``EVALUATION_PACKS`` — module-level list of pack-manifest dicts.
    - ``get_evaluation_packs() -> list[dict]`` — same shape, computed.

    Each pack must include ``pack_id``, ``display_name``,
    ``task_specs``, and ``contract_version`` matching
    :data:`PLUGIN_CONTRACT_VERSIONS["eval_pack"]`.

    Note: P37 only specifies the contract; the loader lands in P38.
    """

    def register_evaluation_packs(self, register: Callable[..., None]) -> None: ...


# Mapping from kind to its expected Protocol (for documentation / docs surface).
KIND_PROTOCOLS: dict[PluginKind, type[Protocol]] = {
    "data_adapter": DataAdapterPluginModule,
    "training_runtime": TrainingRuntimePluginModule,
    "domain_pack": DomainPackPluginModule,
    "eval_pack": EvalPackPluginModule,
}


# Mapping from kind to the set of recognized hook function / constant
# names a module may export. Used in error messages so an author who
# forgot the hook sees the exact list of names that would have matched.
KIND_RECOGNIZED_EXPORTS: dict[PluginKind, tuple[str, ...]] = {
    "data_adapter": (
        "register_data_adapters",
        "get_data_adapters",
        "DATA_ADAPTERS",
    ),
    "training_runtime": (
        "register_training_runtime_plugins",
    ),
    "domain_pack": (
        "register_domain_packs",
        "get_domain_packs",
        "DOMAIN_PACKS",
    ),
    "eval_pack": (
        "register_evaluation_packs",
        "get_evaluation_packs",
        "EVALUATION_PACKS",
    ),
}


# ----------------------------------------------------------------------
# Validators
# ----------------------------------------------------------------------


def _module_attr(module: ModuleType, name: str) -> Any:
    return getattr(module, name, None)


def _has_callable(module: ModuleType, name: str) -> bool:
    obj = _module_attr(module, name)
    return obj is not None and callable(obj)


def _signature_param_count(func: Callable[..., Any]) -> int:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return -1
    return sum(
        1
        for param in signature.parameters.values()
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )


def _check_module_basics(
    module: ModuleType, kind: PluginKind
) -> list[PluginContractCheck]:
    """Common checks applied to every plugin kind."""

    checks: list[PluginContractCheck] = []
    module_name = getattr(module, "__name__", "<anonymous>")
    checks.append(
        PluginContractCheck(
            name="module_importable",
            ok=True,
            message=f"Imported module '{module_name}'.",
        )
    )

    recognized = KIND_RECOGNIZED_EXPORTS[kind]
    present = [name for name in recognized if _module_attr(module, name) is not None]
    if not present:
        checks.append(
            PluginContractCheck(
                name="module_interface",
                ok=False,
                message=(
                    "Module exports none of the recognised plugin hooks for kind "
                    f"'{kind}'. Expected one of: {', '.join(recognized)}."
                ),
            )
        )
    else:
        checks.append(
            PluginContractCheck(
                name="module_interface",
                ok=True,
                message=f"Module exports: {', '.join(present)}.",
            )
        )

    return checks


def _check_version(
    module: ModuleType, kind: PluginKind, report: PluginContractReport
) -> PluginContractCheck:
    expected = PLUGIN_CONTRACT_VERSIONS[kind]
    declared = _module_attr(module, "CONTRACT_VERSION") or _module_attr(
        module, "__plugin_contract_version__"
    )
    plugin_version = _module_attr(module, "__plugin_version__")
    if plugin_version is not None:
        report.declared_version = str(plugin_version)

    if declared is None:
        return PluginContractCheck(
            name="version_metadata",
            ok=True,
            message=(
                f"No CONTRACT_VERSION declared; assuming '{expected}'. "
                "Declare CONTRACT_VERSION for forward-compat guarantees."
            ),
        )

    declared_token = str(declared).strip()
    if declared_token != expected:
        return PluginContractCheck(
            name="version_metadata",
            ok=False,
            message=(
                f"Declared CONTRACT_VERSION '{declared_token}' does not match "
                f"runtime expectation '{expected}'."
            ),
        )

    return PluginContractCheck(
        name="version_metadata",
        ok=True,
        message=f"Declared CONTRACT_VERSION='{declared_token}' matches runtime.",
    )


# -- data_adapter ------------------------------------------------------


def _validate_data_adapter_interface(
    module: ModuleType, report: PluginContractReport
) -> None:
    register = _module_attr(module, "register_data_adapters")
    get_adapters = _module_attr(module, "get_data_adapters")
    constant = _module_attr(module, "DATA_ADAPTERS")

    if register is not None:
        if not callable(register):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message="'register_data_adapters' must be callable.",
                )
            )
            return
        params = _signature_param_count(register)
        if params != 1:
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=(
                        "'register_data_adapters(register)' must take exactly "
                        f"one positional parameter (found {params})."
                    ),
                )
            )
            return
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=True,
                message="register_data_adapters(register) has the expected signature.",
            )
        )
        return

    payload: dict[str, Any] | None = None
    if get_adapters is not None:
        if not callable(get_adapters):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message="'get_data_adapters' must be callable.",
                )
            )
            return
        try:
            payload = get_adapters()
        except Exception as exc:
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=f"get_data_adapters() raised: {exc}",
                )
            )
            return
    elif isinstance(constant, dict):
        payload = constant

    if payload is None:
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=False,
                message="DATA_ADAPTERS must be a non-empty dict of adapter id → payload.",
            )
        )
        return

    if not isinstance(payload, dict) or not payload:
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=False,
                message="DATA_ADAPTERS/get_data_adapters must return a non-empty dict.",
            )
        )
        return

    declared_ids: list[str] = []
    for adapter_id, entry in payload.items():
        if not isinstance(adapter_id, str) or not adapter_id.strip():
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=f"Adapter id must be a non-empty string (got {adapter_id!r}).",
                )
            )
            return
        if callable(entry):
            declared_ids.append(adapter_id)
            continue
        if not isinstance(entry, dict):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=(
                        f"Adapter '{adapter_id}' payload must be a dict or a callable "
                        f"(got {type(entry).__name__})."
                    ),
                )
            )
            return
        map_row = entry.get("map_row")
        if map_row is not None and not callable(map_row):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=f"Adapter '{adapter_id}' has non-callable 'map_row'.",
                )
            )
            return
        declared_ids.append(adapter_id)

    report.declared_ids = declared_ids
    report.checks.append(
        PluginContractCheck(
            name="schema_compliance",
            ok=True,
            message=f"Found {len(declared_ids)} adapter id(s): {', '.join(declared_ids)}.",
        )
    )


# -- training_runtime --------------------------------------------------


def _validate_training_runtime_interface(
    module: ModuleType, report: PluginContractReport
) -> None:
    register = _module_attr(module, "register_training_runtime_plugins")
    if register is None:
        # Already reported by basic check.
        return
    if not callable(register):
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=False,
                message="'register_training_runtime_plugins' must be callable.",
            )
        )
        return
    params = _signature_param_count(register)
    if params not in (0, 1):
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=False,
                message=(
                    "'register_training_runtime_plugins' must take 0 or 1 "
                    f"positional parameters (found {params})."
                ),
            )
        )
        return
    report.checks.append(
        PluginContractCheck(
            name="schema_compliance",
            ok=True,
            message=f"register_training_runtime_plugins has {params}-arg signature.",
        )
    )


# -- domain_pack -------------------------------------------------------


_DOMAIN_PACK_REQUIRED_KEYS: tuple[str, ...] = ("pack_id", "display_name")


def _normalize_pack_list(payload: Any) -> list[dict[str, Any]] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        # Allow single-pack convenience form.
        return [payload]
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                return None
            out.append(entry)
        return out
    return None


def _validate_pack_dict(
    pack: dict[str, Any], required_keys: tuple[str, ...]
) -> str | None:
    for key in required_keys:
        value = pack.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"Missing required field '{key}'."
    return None


def _validate_domain_pack_interface(
    module: ModuleType, report: PluginContractReport
) -> None:
    register = _module_attr(module, "register_domain_packs")
    if register is not None:
        if not callable(register):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message="'register_domain_packs' must be callable.",
                )
            )
            return
        params = _signature_param_count(register)
        if params != 1:
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=(
                        "'register_domain_packs(register)' must take exactly "
                        f"one positional parameter (found {params})."
                    ),
                )
            )
            return
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=True,
                message="register_domain_packs(register) has the expected signature.",
            )
        )
        return

    payload: Any
    getter = _module_attr(module, "get_domain_packs")
    if getter is not None:
        if not callable(getter):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message="'get_domain_packs' must be callable.",
                )
            )
            return
        try:
            payload = getter()
        except Exception as exc:
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=f"get_domain_packs() raised: {exc}",
                )
            )
            return
    else:
        payload = _module_attr(module, "DOMAIN_PACKS")

    packs = _normalize_pack_list(payload)
    if packs is None or not packs:
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=False,
                message=(
                    "DOMAIN_PACKS / get_domain_packs() must yield a non-empty list of "
                    "pack-manifest dicts."
                ),
            )
        )
        return

    declared_ids: list[str] = []
    for pack in packs:
        err = _validate_pack_dict(pack, _DOMAIN_PACK_REQUIRED_KEYS)
        if err is not None:
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=err,
                )
            )
            return
        declared_ids.append(str(pack["pack_id"]))

    report.declared_ids = declared_ids
    report.checks.append(
        PluginContractCheck(
            name="schema_compliance",
            ok=True,
            message=f"Found {len(declared_ids)} domain pack(s): {', '.join(declared_ids)}.",
        )
    )


# -- eval_pack ---------------------------------------------------------


_EVAL_PACK_REQUIRED_KEYS: tuple[str, ...] = ("pack_id", "display_name")


def _validate_eval_pack_interface(
    module: ModuleType, report: PluginContractReport
) -> None:
    register = _module_attr(module, "register_evaluation_packs")
    if register is not None:
        if not callable(register):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message="'register_evaluation_packs' must be callable.",
                )
            )
            return
        params = _signature_param_count(register)
        if params != 1:
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=(
                        "'register_evaluation_packs(register)' must take exactly "
                        f"one positional parameter (found {params})."
                    ),
                )
            )
            return
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=True,
                message="register_evaluation_packs(register) has the expected signature.",
            )
        )
        return

    payload: Any
    getter = _module_attr(module, "get_evaluation_packs")
    if getter is not None:
        if not callable(getter):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message="'get_evaluation_packs' must be callable.",
                )
            )
            return
        try:
            payload = getter()
        except Exception as exc:
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=f"get_evaluation_packs() raised: {exc}",
                )
            )
            return
    else:
        payload = _module_attr(module, "EVALUATION_PACKS")

    packs = _normalize_pack_list(payload)
    if packs is None or not packs:
        report.checks.append(
            PluginContractCheck(
                name="schema_compliance",
                ok=False,
                message=(
                    "EVALUATION_PACKS / get_evaluation_packs() must yield a non-empty "
                    "list of pack-manifest dicts."
                ),
            )
        )
        return

    declared_ids: list[str] = []
    for pack in packs:
        err = _validate_pack_dict(pack, _EVAL_PACK_REQUIRED_KEYS)
        if err is not None:
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=err,
                )
            )
            return
        task_specs = pack.get("task_specs")
        if task_specs is not None and not isinstance(task_specs, list):
            report.checks.append(
                PluginContractCheck(
                    name="schema_compliance",
                    ok=False,
                    message=(
                        f"Pack '{pack['pack_id']}' field 'task_specs' must be a list "
                        f"(got {type(task_specs).__name__})."
                    ),
                )
            )
            return
        declared_ids.append(str(pack["pack_id"]))

    report.declared_ids = declared_ids
    report.checks.append(
        PluginContractCheck(
            name="schema_compliance",
            ok=True,
            message=f"Found {len(declared_ids)} eval pack(s): {', '.join(declared_ids)}.",
        )
    )


# ----------------------------------------------------------------------
# Public validator
# ----------------------------------------------------------------------


_KIND_INTERFACE_VALIDATORS: dict[
    PluginKind, Callable[[ModuleType, PluginContractReport], None]
] = {
    "data_adapter": _validate_data_adapter_interface,
    "training_runtime": _validate_training_runtime_interface,
    "domain_pack": _validate_domain_pack_interface,
    "eval_pack": _validate_eval_pack_interface,
}


def normalize_plugin_kind(value: str | None) -> PluginKind:
    """Normalize an inbound kind string; raise :class:`ValueError` on miss."""

    token = str(value or "").strip().lower().replace("-", "_")
    if token not in KNOWN_PLUGIN_KINDS:
        raise ValueError(
            f"unknown_plugin_kind:{value!r}. Expected one of: "
            f"{', '.join(KNOWN_PLUGIN_KINDS)}"
        )
    return token  # type: ignore[return-value]


def validate_plugin_module(
    module: ModuleType, kind: PluginKind
) -> PluginContractReport:
    """Run the full contract check suite against ``module``.

    The caller is responsible for the ``importlib`` calls. This function
    is pure and side-effect free.
    """

    normalized_kind = normalize_plugin_kind(kind)
    report = PluginContractReport(
        kind=normalized_kind,
        module=getattr(module, "__name__", "<anonymous>"),
        contract_version=PLUGIN_CONTRACT_VERSIONS[normalized_kind],
    )
    report.checks.extend(_check_module_basics(module, normalized_kind))

    # Only run schema check if the module exposed at least one recognised
    # hook (otherwise the schema check would be noise on top of the
    # already-failing module_interface check).
    if any(check.name == "module_interface" and check.ok for check in report.checks):
        _KIND_INTERFACE_VALIDATORS[normalized_kind](module, report)

    report.checks.append(_check_version(module, normalized_kind, report))

    report.checks.append(
        PluginContractCheck(
            name="safe_reload",
            ok=True,
            message=(
                "Kind supports hot reload via the core loader."
                if KIND_SUPPORTS_SAFE_RELOAD[normalized_kind]
                else "Reload not implemented for this kind yet (P38 will add it)."
            ),
        )
    )

    return report


__all__ = [
    "PluginKind",
    "KNOWN_PLUGIN_KINDS",
    "PLUGIN_CONTRACT_VERSIONS",
    "KIND_SUPPORTS_SAFE_RELOAD",
    "KIND_HAS_MODULE_LOADER",
    "KIND_RECOGNIZED_EXPORTS",
    "KIND_PROTOCOLS",
    "PluginContractCheck",
    "PluginContractReport",
    "DataAdapterPluginModule",
    "TrainingRuntimePluginModule",
    "DomainPackPluginModule",
    "EvalPackPluginModule",
    "normalize_plugin_kind",
    "validate_plugin_module",
]
