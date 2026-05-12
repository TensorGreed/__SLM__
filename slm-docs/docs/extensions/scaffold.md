---
sidebar_position: 2
title: Scaffold a plugin
---

# Scaffold a plugin

The scaffold generator emits a **contract-valid starter** for any of the four plugin kinds — a module file, a self-validating test stub, and a README — in a single command. Every scaffold round-trips through its own contract suite, so the generated module passes [`validate_plugin_module`](validate-and-reload.md) the moment it's written.

## What you get

For `kind=data_adapter`, `plugin_id="my-csv-adapter"`:

```
DATA_DIR/extension_scaffolds/data_adapter/my-csv-adapter/
├── my_csv_adapter.py       # the plugin module (~30 lines)
├── test_my_csv_adapter.py  # contract check stub
└── README.md               # wiring instructions
```

The module file already calls `register_data_adapters(register)` with sensible defaults; you replace the `map_row` body with your real logic.

## Generate

### UI

**Training rail → Extension Studio → Generate scaffold** card.

Pick a kind (the four buttons across the top), fill in:

- **Plugin id** (required, kebab-case).
- **Display name** (optional, defaults to title-cased plugin id).
- **Description** (optional).
- **Author**, **Version** (optional).
- **Write files to DATA_DIR** (toggle; off if you only want the preview).

Click **Generate scaffold**. The preview pane fills with each file's contents. Click a tab to switch files, or **Download all files** to grab the bundle.

### CLI

```sh
brewslm scaffold adapter \
  --plugin-id my-csv-adapter \
  --description "Maps a domain CSV to canonical text rows."
```

Aliases: `adapter` → `data_adapter`, `runtime` → `training_runtime`, `domain-pack` → `domain_pack`, `eval-pack` → `eval_pack`.

Flags:

| Flag | Effect |
|---|---|
| `--plugin-id` | Required. The new plugin's id. |
| `--display-name` | Optional, defaults to title-cased plugin id. |
| `--description` | Optional, lands in the module docstring + README. |
| `--author` | Optional, lands in README. |
| `--version` | Optional, defaults to `0.1.0`. |
| `--export-dir` | Optional, defaults to `DATA_DIR/extension_scaffolds/<kind>/<plugin_id>/`. |
| `--no-write` | Don't touch disk; return file contents inline. |

### API

```sh
curl -X POST http://localhost:8000/api/extensions/scaffold \
  -H "Content-Type: application/json" \
  -d '{
    "kind": "data_adapter",
    "plugin_id": "my-csv-adapter",
    "description": "Maps a domain CSV to canonical text rows.",
    "write": true
  }'
```

Returns the full payload:

```json
{
  "kind": "data_adapter",
  "plugin_id": "my-csv-adapter",
  "module_basename": "my_csv_adapter",
  "contract_version": "slm.data_adapter/v3",
  "output_dir": "/data/extension_scaffolds/data_adapter/my-csv-adapter",
  "written_files": ["/data/.../my_csv_adapter.py", "..."],
  "files": {
    "my_csv_adapter.py": "...",
    "test_my_csv_adapter.py": "...",
    "README.md": "..."
  }
}
```

## Plugin id rules

The id you pass gets normalised:

- Lowercased.
- Spaces → dashes.
- Anything not in `[a-z0-9_-]` → dash.
- Leading digits → underscored prefix on the **module basename** only (since Python idents can't start with a digit).

So `Phase87 Weird_Mix` becomes plugin_id `phase87-weird_mix` and module basename `phase87_weird_mix`. The slug is reported back in the response.

## Edit the scaffold

After generation, edit the module file. The standard moves:

1. Replace the `map_row` / `start` / pack manifest body with your real logic.
2. Update `task_profiles` / `output_contract` / `task_specs` for what you actually emit.
3. Bump `__plugin_version__` as you iterate.
4. Run `pytest test_<basename>.py` — the generated test stub validates the contract independently of BrewSLM.

## Wire the plugin into settings

Once the module file lives somewhere on Python's import path, add it to the right settings list:

```sh
# .env
DATA_ADAPTER_PLUGIN_MODULES="my_company.adapters.my_csv_adapter"
TRAINING_RUNTIME_PLUGIN_MODULES="my_company.runtimes.my_runtime"
```

Then reload:

```sh
brewslm extensions reload --kind adapter
```

…or restart the backend.

For `domain_pack` and `eval_pack`, the scaffold ships now but the live loader lands in a follow-up — you can `extensions validate` them today, but they won't auto-register into the runtime catalog yet.

## Next

- [Validate + reload](validate-and-reload.md) — sanity-check before wiring.
- [Extension Studio](extension-studio.md) — UI for the same flow.
- [CLI](cli.md) — every flag for every subcommand.
