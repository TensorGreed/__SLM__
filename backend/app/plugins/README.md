# Plugin Ecosystem Guide

This project supports third-party plugins for:

- `Data Adapters` (dataset normalization/mapping)
- `Training Runtimes` (experiment launch backends)

## Versioning and Compatibility

Keep plugin IDs stable and semantic:

- Data adapter id example: `acme-support-ticket-v1`
- Runtime id example: `acme.cuda-cluster-v1`

Recommended compatibility policy:

- Increment patch version for non-breaking metadata updates.
- Increment minor version for backward-compatible behavior improvements.
- Increment major version when contracts or required fields change.

## Runtime Discovery and Validation

Data adapters:

- Configure `DATA_ADAPTER_PLUGIN_MODULES` with module paths.
- Use:
  - `GET /api/projects/{project_id}/dataset/adapters/catalog`
  - `POST /api/projects/{project_id}/dataset/adapters/reload`

Training runtimes:

- Configure `TRAINING_RUNTIME_PLUGIN_MODULES` with module paths.
- Use:
  - `GET /api/projects/{project_id}/training/runtimes`
  - `GET /api/projects/{project_id}/training/runtimes/plugins/status`
  - `POST /api/projects/{project_id}/training/runtimes/plugins/reload`

## Authoring Templates

Start from:

- `app/plugins/data_adapters/template_adapter.py`
- `app/plugins/training_runtimes/template_runtime.py`

Then copy to your own module path and add it to runtime settings.

## Safety Checklist

- Validate plugin config inputs before use.
- Keep file access project-scoped where possible.
- Fail fast with clear error messages for missing dependencies.
- Emit structured logs with actionable context for operator debugging.

