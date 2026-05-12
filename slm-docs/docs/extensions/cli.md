---
sidebar_position: 5
title: Extensions CLI
---

# Extensions CLI

The `brewslm` CLI exposes everything the [Extension Studio](extension-studio.md) does. Use it when you're scripting plugin generation, gating a build on contract validity, or reloading plugins after a deploy.

Two top-level commands: `brewslm scaffold` and `brewslm extensions`.

## `brewslm scaffold <kind>`

Generate a contract-valid starter for one plugin kind. Aliases:

| Alias | Canonical kind |
|---|---|
| `adapter` | `data_adapter` |
| `runtime` | `training_runtime` |
| `domain-pack` | `domain_pack` |
| `eval-pack` | `eval_pack` |

### Common flags

| Flag | Required | Default |
|---|---|---|
| `--plugin-id` | yes | — |
| `--display-name` | no | title-cased plugin id |
| `--description` | no | "Scaffold for the '\<id\>' \<kind\> plugin." |
| `--author` | no | "BrewSLM author" |
| `--version` | no | `0.1.0` |
| `--export-dir` | no | `DATA_DIR/extension_scaffolds/{kind}/{plugin_id}/` |
| `--no-write` | no | write to disk |

### Examples

```sh
# Data adapter
brewslm scaffold adapter \
  --plugin-id support-tickets-v1 \
  --description "Maps Zendesk export to canonical text rows."

# Training runtime
brewslm scaffold runtime \
  --plugin-id my-cluster \
  --display-name "On-prem cluster runtime" \
  --description "Submits to internal Kubernetes via job CRD."

# Domain pack (preview only — loader lands in a follow-up)
brewslm scaffold domain-pack \
  --plugin-id support-pack-v1 \
  --no-write

# Eval pack
brewslm scaffold eval-pack \
  --plugin-id strict-qa-eval \
  --version 1.0.0
```

Output is a JSON object printed to stdout. The shape mirrors the API response from `POST /api/extensions/scaffold`.

## `brewslm extensions list`

Print every plugin kind with its current status (configured modules, loaded modules, errors, registered count).

```sh
brewslm extensions list
```

The JSON output is identical to `GET /api/extensions`. Pipe to `jq` to slice:

```sh
brewslm extensions list | jq '.kinds[] | {kind, registered_count, load_errors}'
```

## `brewslm extensions validate`

Import a module and run the contract suite without registering it.

### Flags

| Flag | Required |
|---|---|
| `--kind` | yes (alias or canonical) |
| `--module` | yes (importable Python module path) |
| `--force-reload` | no — re-imports if already in `sys.modules` |

### Examples

```sh
# Validate a freshly-edited adapter
brewslm extensions validate \
  --kind adapter \
  --module my_company.adapters.support_tickets_v1

# Inside a CI step — fail the job if it can't load
brewslm extensions validate --kind runtime --module my.runtime || exit 1
```

Exit code:

- `0` — every check passed.
- `1` — at least one check failed.

The JSON report (same shape as the API) lands on stdout regardless.

## `brewslm extensions reload`

Re-import the modules configured in settings + re-run their register hooks. Idempotent.

### Flags

| Flag | Required |
|---|---|
| `--kind` | optional — single kind (or omit for all live kinds) |

### Examples

```sh
# Reload everything that has a live loader
brewslm extensions reload

# Just one kind
brewslm extensions reload --kind adapter
```

Exit code:

- `0` — every kind came back `ok` or `not_supported` (the latter for declarative kinds).
- `1` — at least one kind returned `partial` or `error`.

## End-to-end example

A scratchpad workflow from "fresh idea" to "registered plugin":

```sh
# 1. Generate the scaffold + write to disk.
brewslm scaffold adapter \
  --plugin-id support-tickets-v1 \
  --description "Maps Zendesk export to canonical text rows."

# Generated under DATA_DIR/extension_scaffolds/data_adapter/support-tickets-v1/

# 2. Symlink into your project's import path.
ln -s $DATA_DIR/extension_scaffolds/data_adapter/support-tickets-v1/support_tickets_v1.py \
      ./my_company/adapters/support_tickets_v1.py

# 3. Edit map_row to do real work.
$EDITOR my_company/adapters/support_tickets_v1.py

# 4. Validate — catches signature mistakes before runtime.
brewslm extensions validate \
  --kind adapter \
  --module my_company.adapters.support_tickets_v1

# 5. Add to settings.
echo 'DATA_ADAPTER_PLUGIN_MODULES="my_company.adapters.support_tickets_v1"' >> .env

# 6. Reload — the runtime catalog now includes "support-tickets-v1".
brewslm extensions reload --kind adapter

# 7. Verify.
brewslm extensions list | jq '.kinds[0]'
```

Total time first pass: ~5 minutes. After that the loop is edit → validate → reload, which takes seconds.

## Next

- [Contracts](contracts.md) — what every check is verifying.
- [Scaffold](scaffold.md) — the full set of options.
- [Extension Studio](extension-studio.md) — the UI version of all of the above.
- [CLI reference](../reference/cli.md) — every BrewSLM CLI command.
