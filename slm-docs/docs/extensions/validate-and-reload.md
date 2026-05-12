---
sidebar_position: 3
title: Validate + reload
---

# Validate + reload

`validate` imports a plugin module **without registering it** and runs the contract suite. `reload` re-imports the modules listed in settings for one or all live-loader kinds. Together they're the safe iteration loop for plugin development.

## Validate

Each kind runs the five-check suite — see [Contracts](contracts.md). The output is a structured report you can read at a glance or pipe into CI.

### UI

**Extension Studio → Validate module** card.

1. Pick the kind in the left sidebar (Data adapter / Training runtime / Domain pack / Eval pack).
2. Type the importable module path: `my_company.adapters.my_csv_adapter`.
3. (Optional) tick **Force reload if already imported** — useful while editing.
4. Click **Validate**.

Each check renders as a row: green `pass` or red `fail`, the check name, the message. Declared adapter ids (if any) appear above the check list.

### CLI

```sh
brewslm extensions validate \
  --kind adapter \
  --module my_company.adapters.my_csv_adapter
```

Exit code:

- `0` — every check passed.
- `1` — at least one check failed.

The JSON output goes to stdout, so you can wire this as a CI gate:

```sh
brewslm extensions validate --kind runtime --module my.runtime || exit 1
```

### API

```sh
curl -X POST http://localhost:8000/api/extensions/validate \
  -H "Content-Type: application/json" \
  -d '{
    "kind": "data_adapter",
    "module": "my_company.adapters.my_csv_adapter",
    "force_reload": false
  }'
```

Returns:

```json
{
  "kind": "data_adapter",
  "module": "my_company.adapters.my_csv_adapter",
  "contract_version": "slm.data_adapter/v3",
  "declared_version": "0.1.0",
  "declared_ids": ["my-csv-adapter"],
  "ok": true,
  "import_error": null,
  "checks": [
    {"name": "module_importable", "ok": true, "message": "Imported module 'my_company.adapters.my_csv_adapter'."},
    {"name": "module_interface",  "ok": true, "message": "Module exports: register_data_adapters."},
    {"name": "schema_compliance", "ok": true, "message": "register_data_adapters(register) has the expected signature."},
    {"name": "version_metadata",  "ok": true, "message": "Declared CONTRACT_VERSION='slm.data_adapter/v3' matches runtime."},
    {"name": "safe_reload",       "ok": true, "message": "Kind supports hot reload via the core loader."}
  ]
}
```

If the import itself failed:

```json
{
  "ok": false,
  "import_error": "No module named 'my_company.adapters.my_csv_adapter'",
  "checks": [
    {"name": "module_importable", "ok": false, "message": "Import failed: …"}
  ]
}
```

## Reload

Reload re-imports every module listed in this kind's settings key and re-runs `register`. State is fully replaced — old registrations are cleared before new ones are read. Idempotent: running it twice with no code changes returns identical catalogs.

### UI

**Extension Studio**:

- The kind list (left column) shows per-kind reload badges after a reload call: `ok` / `partial` / `error` / `not_supported`.
- Per-kind **Reload kind** button in the detail card.
- Top-of-list **Reload all** button.

### CLI

```sh
# All kinds at once
brewslm extensions reload

# Just one kind
brewslm extensions reload --kind adapter
```

Exit code:

- `0` — every kind succeeded, OR was `not_supported` (domain/eval pack today).
- `1` — at least one kind came back `partial` or `error`.

### API

```sh
curl -X POST http://localhost:8000/api/extensions/reload \
  -H "Content-Type: application/json" \
  -d '{"kind": "training_runtime"}'  # omit "kind" for all
```

Returns:

```json
{
  "results": [
    {
      "kind": "data_adapter",
      "status": "ok",
      "requested_modules": ["my_company.adapters.my_csv_adapter"],
      "loaded_modules":    ["my_company.adapters.my_csv_adapter"],
      "failed_modules":    {},
      "registered_count":  4
    },
    {
      "kind": "training_runtime",
      "status": "partial",
      "requested_modules": ["good.runtime", "bad.runtime"],
      "loaded_modules":    ["good.runtime"],
      "failed_modules":    {"bad.runtime": "AttributeError: ..."},
      "registered_count":  1
    },
    {"kind": "domain_pack", "status": "not_supported", "message": "Reload not implemented for this kind yet (loader lands in a follow-up)."},
    {"kind": "eval_pack",   "status": "not_supported", "message": "..."}
  ]
}
```

## List configured extensions

The catalog endpoint shows what each kind has loaded right now.

### UI

The Extension Studio's left sidebar **is** this catalog — kind, contract version, configured modules, load errors, count of registered plugins.

### CLI

```sh
brewslm extensions list
```

### API

```sh
curl http://localhost:8000/api/extensions
```

Returns:

```json
{
  "known_kinds": ["data_adapter", "training_runtime", "domain_pack", "eval_pack"],
  "kinds": [
    {
      "kind": "data_adapter",
      "contract_version": "slm.data_adapter/v3",
      "supports_safe_reload": true,
      "has_module_loader": true,
      "settings_key": "DATA_ADAPTER_PLUGIN_MODULES",
      "configured_modules": ["my_company.adapters.my_csv_adapter"],
      "loaded_modules":    ["my_company.adapters.my_csv_adapter"],
      "load_errors":       {},
      "registered_count":  4,
      "recognized_exports": ["register_data_adapters", "get_data_adapters", "DATA_ADAPTERS"]
    },
    {
      "kind": "domain_pack",
      "supports_safe_reload": false,
      "has_module_loader": false,
      "note": "Module loader for this kind is planned for P38 (Wave H scaffold generator).",
      "..."
    }
  ]
}
```

## Iterate quickly

The standard inner loop:

1. Edit `my_csv_adapter.py`.
2. `brewslm extensions validate --kind adapter --module my_csv_adapter` (catches structural mistakes before runtime).
3. `brewslm extensions reload --kind adapter` (re-imports + re-registers).
4. Use the plugin (e.g. run the pipeline's adapter preview tab).
5. Goto 1.

Or, in the Extension Studio UI: edit → Validate → Reload, all in one tab.

## Next

- [Extension Studio](extension-studio.md) — the in-app UI for all of the above.
- [CLI](cli.md) — every flag.
- [Contracts](contracts.md) — what each check is actually verifying.
