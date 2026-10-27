---
sidebar_position: 65
title: Tool Catalog
---

<!-- Hand-written in the generator's format (G7). Re-running slm-docs/scripts/generate_api_ref.py will regenerate it. -->

# Tool Catalog

Reference for the **Tool Catalog** API. 7 endpoint(s). Narrative: [Tool catalog](../../extensions/tool-catalog.md).

Errors: `422 INVALID_TOOL_CATALOG` (`metadata.errors`), `403 MCP_STDIO_DISABLED` (binary not allow-listed; the message names `MCP_STDIO_ALLOWED_BINARIES`), `502 MCP_SERVER_ERROR` (redacted message), `404 TOOL_CATALOG_VERSION_NOT_FOUND`.

### `GET /api/projects/{project_id}/tool-catalog`

**Get Tool Catalog**

Returns `{catalog, history}`. `catalog` is `null` without a catalog, else `{tools, version, flags_version, source, server, imported_at}`; each tool carries `name`, `description`, `input_schema`, `annotations`, `read_only`, `destructive`, `allowed`, `read_only_effective`, `mutating`, `signature` (and `schema_error` when its schema cannot be validated). `history` entries: `{version, imported_at, source_kind, tool_count, current, restorable}`.

| Status | Schema | Description |
|---|---|---|
| `200` | `object` | Successful Response |

### `POST /api/projects/{project_id}/tool-catalog/import/mcp`

**Import Mcp**

Connect to an MCP server and replace the catalog. `?async_job=true` runs it as a `tool_catalog_import` Job (202); the Job result is the catalog status. `mcp_stdio` needs an allow-listed binary and, with auth on, the admin role.

**Request body** (required) — `McpImportBody`

| Field | Type | Required | Description |
|---|---|---|---|
| `kind` | `string` | yes | `mcp_stdio` or `mcp_http` |
| `label` | `string \| null` | no | Names the secret provider `mcp_<slug>` |
| `command` | `string[] \| null` | stdio | Program then arguments (no shell) |
| `env` | `object \| null` | no | stdio: name to value; `null` / blank keeps the stored secret |
| `url` | `string \| null` | http | http(s) URL |
| `headers` | `object \| null` | no | http: name to value; `null` / blank keeps the stored secret |

| Status | Schema | Description |
|---|---|---|
| `200` | `object` | Catalog status (plus `warnings` when tools were skipped) |
| `202` | `object` | Job (with `async_job=true`) |
| `403` | `object` | `MCP_STDIO_DISABLED` |
| `422` | `object` | `INVALID_TOOL_CATALOG` |
| `502` | `object` | `MCP_SERVER_ERROR` |

### `POST /api/projects/{project_id}/tool-catalog/import/openapi`

**Import Openapi**

**Request body** (required) — `OpenApiImportBody`: `{"filename": string | null, "content": string}` (JSON or YAML text, at most 5 MB).

| Status | Schema | Description |
|---|---|---|
| `200` | `object` | Catalog status |
| `422` | `object` | `INVALID_TOOL_CATALOG` |

### `PUT /api/projects/{project_id}/tool-catalog/tools`

**Put Tools**

Replace the catalog with `{"tools": [...]}` (source `manual`). Names must match `[A-Za-z0-9_.-]{1,64}` and be unique.

| Status | Schema | Description |
|---|---|---|
| `200` | `object` | Catalog status |
| `422` | `object` | `INVALID_TOOL_CATALOG` |

### `PATCH /api/projects/{project_id}/tool-catalog/flags`

**Patch Flags**

`{"flags": {name: {"allowed"?: bool, "read_only"?: bool | null}}}`. `read_only: null` clears the override; omitted keys stay. A destructive tool cannot be made read-only (422).

| Status | Schema | Description |
|---|---|---|
| `200` | `object` | Catalog status |
| `422` | `object` | `INVALID_TOOL_CATALOG` |

### `POST /api/projects/{project_id}/tool-catalog/restore`

**Restore**

`{"version": "<12-hex>"}`. 404 `TOOL_CATALOG_VERSION_NOT_FOUND` when the version is unknown or its content was trimmed.

| Status | Schema | Description |
|---|---|---|
| `200` | `object` | Catalog status |
| `404` | `object` | `TOOL_CATALOG_VERSION_NOT_FOUND` |

### `DELETE /api/projects/{project_id}/tool-catalog`

**Delete Catalog**

Removes the catalog (history is kept; stored secrets are kept).

| Status | Schema | Description |
|---|---|---|
| `200` | `object` | Catalog status (`catalog: null`) |
