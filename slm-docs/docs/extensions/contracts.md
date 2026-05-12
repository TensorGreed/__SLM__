---
sidebar_position: 1
title: Plugin contracts
---

# Plugin contracts

BrewSLM has formal **Protocol contracts** for four plugin kinds. Any module that follows a kind's contract can be dropped into a settings list and loaded with no app changes. Contracts are version-pinned so the runtime can reject a stale plugin instead of crashing inside it.

## The four kinds

| Kind | What it does | Contract version | Loader state |
|---|---|---|---|
| **`data_adapter`** | Maps raw rows into BrewSLM's canonical record shape. | `slm.data_adapter/v3` | Live |
| **`training_runtime`** | Launches a training run (local, Celery, your own). | `slm.training_runtime/v1` | Live |
| **`domain_pack`** | Reusable training + eval overlay for a domain. | `slm.domain-pack/v1` | Contract-only (loader lands in a follow-up) |
| **`eval_pack`** | Task-aware metric schemas + gate policies. | `slm.evaluation-pack/v2` | Contract-only (loader lands in a follow-up) |

"Live" means the module path is auto-imported on startup. "Contract-only" means the contract is defined and validated, but the loader that registers the plugin into the runtime catalog hasn't shipped yet — generate a scaffold today, wire to the runtime later.

## What every contract specifies

Each kind has its own Protocol class (in `app/services/plugin_contracts.py`), but they share five validators:

1. **`module_importable`** — `importlib.import_module(path)` doesn't raise.
2. **`module_interface`** — the module exposes one of the recognised hook names for this kind.
3. **`schema_compliance`** — the hook has the right signature, declared id format, etc.
4. **`version_metadata`** — declared `CONTRACT_VERSION` matches the runtime's expected version.
5. **`safe_reload`** — the loader supports `force_reload=True` (live kinds) or N/A (contract-only).

The validator is **pure**: it imports the module, runs the checks, returns a structured report. It never registers the module into the live registry. That's [validate-and-reload](validate-and-reload.md)'s job.

## Recognised hooks per kind

### Data adapter

A module is a valid data adapter if it exports one of:

```python
def register_data_adapters(register):
    register(
        adapter_id="my-adapter",
        map_row=lambda record, config: {"text": record.get("body", "")},
        description="…",
        task_profiles=["instruction_sft"],
        preferred_training_tasks=["causal_lm"],
        output_contract={"required_fields": ["text"], "optional_fields": []},
    )
```

Or:

```python
def get_data_adapters() -> dict[str, dict | callable]:
    return {"my-adapter": {"map_row": _my_map_row, ...}}
```

Or a module-level constant:

```python
DATA_ADAPTERS = {"my-adapter": {"map_row": _my_map_row, ...}}
```

### Training runtime

```python
from app.services.training_runtime_service import (
    TrainingRuntimeStartContext,
    TrainingRuntimeStartResult,
)

CONTRACT_VERSION = "slm.training_runtime/v1"
__plugin_version__ = "0.1.0"
RUNTIME_ID = "my-runtime"


def validate() -> list[str]:
    """Return a list of human-readable errors, or [] if ready."""
    return []


async def start(ctx: TrainingRuntimeStartContext) -> TrainingRuntimeStartResult:
    # … launch a training run …
    return TrainingRuntimeStartResult(
        message="Started",
        task_id="my-task-id",
        runtime_updates={},
    )


def register_training_runtime_plugins(register) -> None:
    register(
        runtime_id=RUNTIME_ID,
        label="My Runtime",
        description="…",
        execution_backend="local",
        validate=validate,
        start=start,
        required_dependencies=[],
        supported_modalities=["text"],
        supports_task_tracking=False,
        supports_cancellation=True,
    )
```

The `register_training_runtime_plugins` hook can also take 0 args — in that case it's expected to call the public SDK function `register_training_runtime_plugin(...)` directly.

### Domain pack

A constant or a register hook. Each pack manifest is a dict with `pack_id`, `display_name`, plus the standard overlay sections (`dataset_split`, `training_defaults`, `registry_gates`, `data_quality`, `normalization`, `tools`, `evaluation`, `audit`) and hook ids (normalizer / validator / evaluator).

```python
DOMAIN_PACKS = [
    {
        "$schema": "slm.domain-pack/v1",
        "pack_id": "my-pack",
        "version": "1.0.0",
        "display_name": "My Pack",
        "owner": "alice",
        "status": "active",
        "overlay": {...},
        "hooks": {
            "normalizer": {"id": "default-normalizer", "config": {}},
            "validator":  {"id": "default-validator",  "config": {}},
            "evaluator":  {"id": "default-evaluator",  "config": {}},
        },
    }
]


def register_domain_packs(register) -> None:
    for pack in DOMAIN_PACKS:
        register(pack)
```

### Eval pack

```python
CONTRACT_VERSION = "slm.evaluation-pack/v2"
__plugin_version__ = "0.1.0"

EVALUATION_PACKS = [
    {
        "pack_id": "my-eval",
        "display_name": "My Eval Pack",
        "version": "1.0.0",
        "owner": "alice",
        "contract_version": CONTRACT_VERSION,
        "default_task_profile": "instruction_sft",
        "task_specs": [
            {
                "task_profile": "instruction_sft",
                "required_metric_ids": ["exact_match"],
                "gates": [{"gate_id": "exact_match_min", "metric_id": "exact_match", "threshold": 0.5, "required": True}],
                "metric_schema": {"exact_match": {"description": "…", "expected_range": [0.0, 1.0]}},
            }
        ],
    }
]


def register_evaluation_packs(register) -> None:
    for pack in EVALUATION_PACKS:
        register(pack)
```

## Version pinning

The runtime expects an exact `CONTRACT_VERSION` match. A plugin declaring `slm.data_adapter/v2` won't load against a `slm.data_adapter/v3` runtime. This is intentional: contracts evolve, plugins should track them explicitly. The validator surfaces the mismatch as a `version_metadata: fail` check so you can see it immediately.

## Why "Protocol", not "ABC"?

Protocols are **structural**: a module just needs to expose the right hook names with the right signatures. No subclassing, no decorators. This keeps plugin modules tiny — most of mine are 30–50 lines.

## Next

- [Scaffold a plugin](scaffold.md) — generate a contract-valid starter module.
- [Validate + reload](validate-and-reload.md) — load a module + run the checks.
- [Extension Studio](extension-studio.md) — the in-app UI for all of the above.
- [CLI](cli.md) — `brewslm scaffold` and `brewslm extensions`.
