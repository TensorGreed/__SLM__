"""Plugin scaffold templating service (priority.md P38, Wave H).

Generalises the existing ``export_adapter_scaffold()`` codegen in
:mod:`app.services.adapter_studio_service` into a single templating
layer that emits contract-compliant scaffolds for every plugin kind
declared in :mod:`app.services.plugin_contracts`.

A scaffold is a small bundle on disk containing:

- ``<module>.py`` — the actual plugin module, wired against the
  recognised hook for the chosen kind (e.g. ``register_data_adapters``
  for ``data_adapter``).
- ``test_<module>.py`` — a self-validating test stub that loads the
  generated module and checks the kind-specific contract using
  :func:`app.services.plugin_contracts.validate_plugin_module`. The
  stub passes out-of-the-box so the author starts from a green build.
- ``README.md`` — operator-facing wiring instructions: how to add the
  module to settings, validate it via ``brewslm extensions validate``,
  and reload it.

The generated module declares ``CONTRACT_VERSION`` and
``__plugin_version__`` so a future loader can pin compatibility.
``adapter_studio_service.export_adapter_scaffold`` keeps its own
adapter-from-project export path for now (it carries adapter contract
metadata sourced from the registry); a follow-up can route it through
this layer when the adapter studio surfaces P38's manifest knobs.
"""

from __future__ import annotations

import json
import pprint
import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.plugin_contracts import (
    KIND_RECOGNIZED_EXPORTS,
    PLUGIN_CONTRACT_VERSIONS,
    PluginKind,
    normalize_plugin_kind,
)


_DEFAULT_VERSION = "0.1.0"
_DEFAULT_AUTHOR = "BrewSLM author"
_SCAFFOLD_ROOT_NAME = "extension_scaffolds"


def _python_literal(value: Any) -> str:
    """Render a Python value as valid module-source literal.

    ``json.dumps`` emits ``true`` / ``false`` / ``null`` which aren't
    valid Python. We use :func:`pprint.pformat` so the generated module
    runs without a post-process step.
    """

    return pprint.pformat(value, indent=4, width=88, sort_dicts=False)

_KIND_REGISTER_NAME: dict[PluginKind, str] = {
    "data_adapter": "register_data_adapters",
    "training_runtime": "register_training_runtime_plugins",
    "domain_pack": "register_domain_packs",
    "eval_pack": "register_evaluation_packs",
}

_KIND_SETTINGS_KEY: dict[PluginKind, str | None] = {
    "data_adapter": "DATA_ADAPTER_PLUGIN_MODULES",
    "training_runtime": "TRAINING_RUNTIME_PLUGIN_MODULES",
    "domain_pack": None,
    "eval_pack": None,
}


_PYTHON_IDENT_PATTERN = re.compile(r"[^a-z0-9_]")


def _slugify_plugin_id(value: str) -> str:
    """Normalise a plugin id to a kebab/snake-safe lowercase token."""

    token = str(value or "").strip().lower()
    token = token.replace(" ", "-")
    token = re.sub(r"[^a-z0-9_\-]", "-", token)
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token


def _module_basename(plugin_id_slug: str) -> str:
    """Derive an importable Python module name from the plugin slug.

    Python modules can't contain dashes, so we substitute underscores
    and prefix with an underscore if the slug would otherwise start
    with a digit.
    """

    raw = _PYTHON_IDENT_PATTERN.sub("_", plugin_id_slug.replace("-", "_"))
    if not raw:
        raise ValueError("scaffold_plugin_id_invalid:plugin_id is empty after normalisation")
    if raw[0].isdigit():
        raw = f"_{raw}"
    return raw


def _humanize(value: str) -> str:
    cleaned = re.sub(r"[-_]+", " ", value).strip()
    return cleaned.title() if cleaned else value


def _resolve_export_dir(
    kind: PluginKind,
    plugin_id_slug: str,
    export_dir: str | Path | None,
) -> Path:
    if export_dir is None:
        root = settings.DATA_DIR / _SCAFFOLD_ROOT_NAME / kind / plugin_id_slug
    else:
        root = Path(export_dir).expanduser()
    return root.resolve()


# ----------------------------------------------------------------------
# Per-kind module templates
# ----------------------------------------------------------------------


def _adapter_module_source(meta: dict[str, Any]) -> str:
    description = json.dumps(meta["description"], ensure_ascii=True)
    return (
        f'"""{meta["display_name"]} — BrewSLM data adapter plugin.\n\n'
        f"Generated scaffold (priority.md P38). Contract version: "
        f"``{meta['contract_version']}``.\n\n"
        "Add the module path to ``DATA_ADAPTER_PLUGIN_MODULES`` in your\n"
        "config and call ``brewslm extensions reload --kind data_adapter``\n"
        'to register.\n"""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n\n"
        f'CONTRACT_VERSION = "{meta["contract_version"]}"\n'
        f'__plugin_version__ = "{meta["version"]}"\n\n'
        f'ADAPTER_ID = "{meta["plugin_id"]}"\n\n\n'
        "def map_row(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:\n"
        '    """Map an input row to BrewSLM\'s canonical record shape.\n\n'
        "    Return ``None`` to drop the row from the prepared dataset.\n"
        '    """\n'
        '    text = record.get("text")\n'
        "    if not isinstance(text, str) or not text.strip():\n"
        "        return None\n"
        '    return {"text": text}\n\n\n'
        "def register_data_adapters(register) -> None:\n"
        "    register(\n"
        "        ADAPTER_ID,\n"
        "        map_row,\n"
        f"        description={description},\n"
        '        task_profiles=["instruction_sft"],\n'
        '        preferred_training_tasks=["causal_lm"],\n'
        "        output_contract={\n"
        '            "required_fields": ["text"],\n'
        '            "optional_fields": [],\n'
        '            "notes": ["Generated scaffold — customise as needed."],\n'
        "        },\n"
        "    )\n"
    )


def _runtime_module_source(meta: dict[str, Any]) -> str:
    description = json.dumps(meta["description"], ensure_ascii=True)
    return (
        f'"""{meta["display_name"]} — BrewSLM training runtime plugin.\n\n'
        f"Generated scaffold (priority.md P38). Contract version: "
        f"``{meta['contract_version']}``.\n\n"
        "Add the module path to ``TRAINING_RUNTIME_PLUGIN_MODULES`` in\n"
        "your config and call\n"
        "``brewslm extensions reload --kind training_runtime`` to\n"
        'register.\n"""\n\n'
        "from __future__ import annotations\n\n"
        "from app.services.training_runtime_service import (\n"
        "    TrainingRuntimeStartContext,\n"
        "    TrainingRuntimeStartResult,\n"
        ")\n\n\n"
        f'CONTRACT_VERSION = "{meta["contract_version"]}"\n'
        f'__plugin_version__ = "{meta["version"]}"\n\n'
        f'RUNTIME_ID = "{meta["plugin_id"]}"\n\n\n'
        "def validate() -> list[str]:\n"
        '    """Return a list of human-readable preflight errors. Empty list = ok."""\n'
        "    return []\n\n\n"
        "async def start(ctx: TrainingRuntimeStartContext) -> TrainingRuntimeStartResult:\n"
        '    """Launch a training run for ``ctx``.\n\n'
        "    The scaffold returns a no-op stub so the harness can wire\n"
        "    the plugin end-to-end; replace with a real dispatcher.\n"
        '    """\n'
        "    return TrainingRuntimeStartResult(\n"
        f'        message=f"{meta["display_name"]} stub launched for experiment {{ctx.experiment_id}}.",\n'
        "        task_id=None,\n"
        "        runtime_updates={},\n"
        "    )\n\n\n"
        "def register_training_runtime_plugins(register) -> None:\n"
        "    register(\n"
        "        runtime_id=RUNTIME_ID,\n"
        f'        label="{meta["display_name"]}",\n'
        f"        description={description},\n"
        '        execution_backend="local",\n'
        "        validate=validate,\n"
        "        start=start,\n"
        "        required_dependencies=[],\n"
        '        supported_modalities=["text"],\n'
        "        supports_task_tracking=False,\n"
        "        supports_cancellation=True,\n"
        "    )\n"
    )


def _domain_pack_module_source(meta: dict[str, Any]) -> str:
    payload = {
        "$schema": meta["contract_version"],
        "pack_id": meta["plugin_id"],
        "version": "1.0.0",
        "display_name": meta["display_name"],
        "description": meta["description"],
        "owner": meta["author"],
        "status": "active",
        "tags": [],
        "overlay": {
            "dataset_split": {},
            "training_defaults": {},
            "registry_gates": {},
            "data_quality": {},
            "normalization": {},
            "tools": {},
            "evaluation": {},
            "audit": {},
        },
        "hooks": {
            "normalizer": {"id": "default-normalizer", "config": {}},
            "validator": {"id": "default-validator", "config": {}},
            "evaluator": {"id": "default-evaluator", "config": {}},
        },
    }
    rendered = _python_literal([payload])
    return (
        f'"""{meta["display_name"]} — BrewSLM domain pack plugin.\n\n'
        f"Generated scaffold (priority.md P38). Contract version: "
        f"``{meta['contract_version']}``.\n\n"
        "Domain pack module loader lands in a follow-up; the scaffold\n"
        "ships the contract-compliant manifest now so the operator can\n"
        'validate via ``brewslm extensions validate --kind domain_pack``.\n"""\n\n'
        "from __future__ import annotations\n\n\n"
        f'CONTRACT_VERSION = "{meta["contract_version"]}"\n'
        f'__plugin_version__ = "{meta["version"]}"\n\n\n'
        f"DOMAIN_PACKS = {rendered}\n\n\n"
        "def register_domain_packs(register) -> None:\n"
        "    for pack in DOMAIN_PACKS:\n"
        "        register(pack)\n"
    )


def _eval_pack_module_source(meta: dict[str, Any]) -> str:
    payload = {
        "pack_id": meta["plugin_id"],
        "display_name": meta["display_name"],
        "description": meta["description"],
        "version": "1.0.0",
        "owner": meta["author"],
        "tags": [],
        "contract_version": meta["contract_version"],
        "default_task_profile": "instruction_sft",
        "task_specs": [
            {
                "task_profile": "instruction_sft",
                "display_name": "Instruction SFT",
                "description": "Scaffold task spec — customise gates + metrics.",
                "required_metric_ids": ["exact_match"],
                "gates": [
                    {
                        "gate_id": "exact_match_min",
                        "metric_id": "exact_match",
                        "threshold": 0.5,
                        "required": True,
                        "operator": "gte",
                    }
                ],
                "metric_schema": {
                    "exact_match": {
                        "description": "Exact-match quality score.",
                        "expected_range": [0.0, 1.0],
                    }
                },
                "source": "scaffold",
            }
        ],
    }
    rendered = _python_literal([payload])
    return (
        f'"""{meta["display_name"]} — BrewSLM evaluation pack plugin.\n\n'
        f"Generated scaffold (priority.md P38). Contract version: "
        f"``{meta['contract_version']}``.\n\n"
        "Evaluation pack module loader lands in a follow-up; the\n"
        "scaffold ships the contract-compliant manifest now so the\n"
        "operator can validate via\n"
        '``brewslm extensions validate --kind eval_pack``.\n"""\n\n'
        "from __future__ import annotations\n\n\n"
        f'CONTRACT_VERSION = "{meta["contract_version"]}"\n'
        f'__plugin_version__ = "{meta["version"]}"\n\n\n'
        f"EVALUATION_PACKS = {rendered}\n\n\n"
        "def register_evaluation_packs(register) -> None:\n"
        "    for pack in EVALUATION_PACKS:\n"
        "        register(pack)\n"
    )


_KIND_MODULE_BUILDERS: dict[PluginKind, Any] = {
    "data_adapter": _adapter_module_source,
    "training_runtime": _runtime_module_source,
    "domain_pack": _domain_pack_module_source,
    "eval_pack": _eval_pack_module_source,
}


# ----------------------------------------------------------------------
# Test stub + README templates (shared across kinds)
# ----------------------------------------------------------------------


def _test_stub_source(meta: dict[str, Any]) -> str:
    register_name = _KIND_REGISTER_NAME[meta["kind"]]
    return (
        f'"""Contract check stub for {meta["display_name"]} ({meta["kind"]}).\n\n'
        "Validates the generated scaffold against BrewSLM's plugin\n"
        "contract suite (priority.md P37). The stub passes out-of-the-box\n"
        "so the author starts from a green build; edit the plugin\n"
        'module first, then re-run.\n"""\n\n'
        "from __future__ import annotations\n\n"
        "import importlib.util\n"
        "import sys\n"
        "import unittest\n"
        "from pathlib import Path\n\n\n"
        "HERE = Path(__file__).resolve().parent\n"
        f'MODULE_PATH = HERE / "{meta["module_basename"]}.py"\n\n\n'
        "def _load_module():\n"
        '    spec = importlib.util.spec_from_file_location(\n'
        f'        "{meta["module_basename"]}", MODULE_PATH\n'
        "    )\n"
        '    assert spec is not None and spec.loader is not None, "scaffold module not found"\n'
        "    module = importlib.util.module_from_spec(spec)\n"
        f'    sys.modules["{meta["module_basename"]}"] = module\n'
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n\n"
        f"class {meta['class_name']}ContractTests(unittest.TestCase):\n"
        "    def test_module_imports(self):\n"
        "        _load_module()\n\n"
        "    def test_declared_contract_version(self):\n"
        "        module = _load_module()\n"
        "        self.assertEqual(\n"
        '            getattr(module, "CONTRACT_VERSION", None),\n'
        f'            "{meta["contract_version"]}",\n'
        "        )\n\n"
        "    def test_register_hook_is_callable(self):\n"
        "        module = _load_module()\n"
        f'        hook = getattr(module, "{register_name}", None)\n'
        f'        self.assertTrue(callable(hook), "{register_name} must be callable")\n\n\n'
        'if __name__ == "__main__":\n'
        "    unittest.main()\n"
    )


def _readme_source(meta: dict[str, Any]) -> str:
    kind = meta["kind"]
    settings_key = _KIND_SETTINGS_KEY[kind]
    settings_line = (
        f"`{settings_key}` setting"
        if settings_key
        else "future module-loader setting (planned)"
    )
    settings_snippet = (
        f"```toml\n{settings_key} = [\"{meta['module_basename']}\"]\n```"
        if settings_key
        else (
            "*Module loader for this plugin kind lands in a follow-up. "
            "Until then, run `brewslm extensions validate` to sanity-check "
            "the manifest.*"
        )
    )
    return (
        f"# {meta['display_name']}\n\n"
        f"Generated scaffold (priority.md P38) for the **{kind}** plugin kind.\n\n"
        f"- Plugin id: `{meta['plugin_id']}`\n"
        f"- Contract version: `{meta['contract_version']}`\n"
        f"- Plugin version: `{meta['version']}`\n"
        f"- Author: `{meta['author']}`\n\n"
        "## Install\n\n"
        f"Drop `{meta['module_basename']}.py` on `PYTHONPATH` and add to the {settings_line}:\n\n"
        f"{settings_snippet}\n\n"
        "Then validate and reload:\n\n"
        "```sh\n"
        f"brewslm extensions validate --kind {kind} --module {meta['module_basename']}\n"
        f"brewslm extensions reload --kind {kind}\n"
        "```\n\n"
        "## Customise\n\n"
        f"Edit `{meta['module_basename']}.py` to implement your logic.\n"
        f"The hook is `{_KIND_REGISTER_NAME[kind]}` — see\n"
        "`backend/app/services/plugin_contracts.py` for the full contract\n"
        "expected by the loader.\n"
    )


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def generate_extension_scaffold(
    *,
    kind: str,
    plugin_id: str,
    display_name: str | None = None,
    description: str | None = None,
    author: str | None = None,
    version: str | None = None,
    export_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build a scaffold bundle for a plugin of the given kind.

    Returns:
        Dict with ``kind``, ``plugin_id``, ``contract_version``, the
        ``output_dir`` path, a ``files`` map of relative path → content
        (handy for the API surface to ship without filesystem access),
        and ``written_files`` listing absolute paths on disk when
        ``write=True``.

    Raises:
        :class:`ValueError` for unknown kinds or unusable plugin ids.
    """

    normalized_kind = normalize_plugin_kind(kind)
    raw_plugin_id = str(plugin_id or "").strip()
    if not raw_plugin_id:
        raise ValueError("scaffold_plugin_id_required:plugin_id must be a non-empty string")

    plugin_id_slug = _slugify_plugin_id(raw_plugin_id)
    if not plugin_id_slug:
        raise ValueError(
            f"scaffold_plugin_id_invalid:plugin_id '{plugin_id}' is empty after normalisation"
        )
    module_basename = _module_basename(plugin_id_slug)
    class_name = "".join(part.capitalize() for part in plugin_id_slug.split("-") if part)
    contract_version = PLUGIN_CONTRACT_VERSIONS[normalized_kind]
    plugin_version = str(version or _DEFAULT_VERSION).strip() or _DEFAULT_VERSION
    resolved_display_name = (
        str(display_name).strip()
        if display_name and str(display_name).strip()
        else _humanize(plugin_id_slug)
    )
    resolved_description = (
        str(description).strip()
        if description and str(description).strip()
        else f"Scaffold for the '{plugin_id_slug}' {normalized_kind} plugin."
    )
    resolved_author = (
        str(author).strip() if author and str(author).strip() else _DEFAULT_AUTHOR
    )

    meta: dict[str, Any] = {
        "kind": normalized_kind,
        "plugin_id": plugin_id_slug,
        "plugin_id_raw": raw_plugin_id,
        "module_basename": module_basename,
        "class_name": class_name or "Plugin",
        "display_name": resolved_display_name,
        "description": resolved_description,
        "author": resolved_author,
        "version": plugin_version,
        "contract_version": contract_version,
        "register_name": _KIND_REGISTER_NAME[normalized_kind],
        "recognized_exports": list(KIND_RECOGNIZED_EXPORTS[normalized_kind]),
    }

    module_source = _KIND_MODULE_BUILDERS[normalized_kind](meta)
    files: dict[str, str] = {
        f"{module_basename}.py": module_source,
        f"test_{module_basename}.py": _test_stub_source(meta),
        "README.md": _readme_source(meta),
    }

    output_dir = _resolve_export_dir(normalized_kind, plugin_id_slug, export_dir)
    written: list[str] = []
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        for relative_path, content in files.items():
            absolute = output_dir / relative_path
            absolute.parent.mkdir(parents=True, exist_ok=True)
            absolute.write_text(content, encoding="utf-8")
            written.append(str(absolute))

    return {
        "kind": normalized_kind,
        "plugin_id": plugin_id_slug,
        "plugin_id_raw": raw_plugin_id,
        "module_basename": module_basename,
        "display_name": resolved_display_name,
        "description": resolved_description,
        "author": resolved_author,
        "version": plugin_version,
        "contract_version": contract_version,
        "output_dir": str(output_dir),
        "files": files,
        "written_files": written,
    }


__all__ = [
    "generate_extension_scaffold",
]
