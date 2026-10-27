---
sidebar_position: 8
title: Tool catalog
---

# Tool catalog

A **tool catalog** is the list of tools a project's model is allowed to call: each tool has a name, a description, an argument schema and read-only / destructive hints. The catalog is generic. It feeds the prompt (a compact list of the allowed tools) and tool-call scoring.

It lives in the project settings (`runtime_config.tool_catalog`). Every change that alters the tools produces a new **version** (a 12-character hash of the tools); flags (allowed, read-only override) have their own `flags_version`. The newest 10 versions are remembered and the newest 5 can be restored.

## Import sources

| Source | How | Notes |
|---|---|---|
| MCP server over **stdio** | command + arguments, optional environment values | Launches a process. Disabled until an admin allow-lists the binary (see below). |
| MCP server over **streamable HTTP** | URL + optional header values | One POST per message; JSON or event-stream replies. |
| **OpenAPI** 3.0 / 3.1 | upload JSON or YAML | Each operation becomes a tool. |
| Manual | paste tools as JSON | `name`, `description`, `inputSchema`, `annotations`. |
| Domain pack | the pack's `tool_catalog` | Items may use `id` as an alias for `name`. |

The older two-endpoint HTTP+SSE transport is **not supported**; use streamable HTTP or stdio.

Re-importing the same tools keeps the version and adds no history entry; your allow / read-only flags are kept for tools that still exist.

### OpenAPI conversion

The tool name is the `operationId`, or `<method>_<path slug>`, sanitised to `A-Za-z0-9_.-` (at most 64 characters) and de-duplicated with `_2`. Path, query and header parameters become properties (path ones required) and the JSON request body becomes `body`. Referenced component schemas are copied into the tool's `$defs`. GET / HEAD are read-only, DELETE is destructive, the rest are not read-only. At most 1000 operations and 5 MB per document; a tool whose schema cannot be validated is skipped with a warning.

## Security gates

MCP stdio runs a program chosen by a user, so it is locked down:

- **Off by default.** Set `MCP_STDIO_ALLOWED_BINARIES` (comma list). A bare name matches a bare command name; an absolute path matches by real path. Otherwise the import answers `403 MCP_STDIO_DISABLED`.
- With auth on, importing from stdio needs the **admin** role.
- No shell; its own process group (killed on exit); a temporary working directory; a timeout; a 4 MB line cap.
- **Minimal environment**: `PATH`, `HOME`, locale and temp variables, plus the names in `MCP_STDIO_ENV_PASSTHROUGH`, plus the values you declare for this source. Declared names that change how a program loads or runs code are refused (`422`, and again when the process starts): `LD_*`, `DYLD_*`, `PATH`, `PYTHON*`, `NODE_OPTIONS`, `NODE_PATH`, `PERL5*`, `PERLLIB`, `RUBY*`, `BASH_ENV`, `ENV`, `HOME`, `IFS`, `SHELL`, `TMPDIR`, `GCONV_PATH`, `LOCPATH`, `MALLOC_*`, `GLIBC_TUNABLES`, `JAVA_TOOL_OPTIONS`, `_JAVA_OPTIONS`, `CLASSPATH`, `GIT_*`, and every name of the base environment. Your values can never override those.
- **Do not allow-list an interpreter or a launcher.** Allowing `python3`, `node`, `bash`, `sh`, `env`, `npx`, `uvx` (and similar) is the same as allowing arbitrary code, because the command is `python3 -c "..."`. Allow-list an **absolute path to a dedicated server binary** instead.
- HTTP: http(s) only, redirects are never followed, responses are capped at 8 MB, timeout `MCP_HTTP_TIMEOUT_S` (30 s). `MCP_HTTP_ALLOW_PRIVATE_HOSTS` defaults to true because local lab servers are the main use; set it to false to refuse loopback, private and link-local addresses.

| Setting | Default | Meaning |
|---|---|---|
| `MCP_STDIO_ALLOWED_BINARIES` | empty (stdio off) | Allow-list for stdio commands |
| `MCP_STDIO_ENV_PASSTHROUGH` | empty | Extra environment names forwarded to the process |
| `MCP_HTTP_TIMEOUT_S` | 30 | HTTP timeout |
| `MCP_HTTP_ALLOW_PRIVATE_HOSTS` | true | Allow private / loopback servers |
| `MCP_DRY_RUN_ENABLED` | false | Allow dry-run calls (below) |

Known limits: the private-host check resolves the name once before connecting, so a hostile DNS server could still answer differently for the connection itself (DNS rebinding); run lab servers you do not trust behind a network policy. A dry run trusts the annotations **stored** in the catalog: if the server later changes what a tool does, nothing re-checks it until you import again. A catalog is capped at 4 MB of tool definitions and a tool schema at 64 levels of nesting (deeper ones are kept but marked unusable).

## Secrets

Environment values and header values are stored only in the project's **secrets**, under provider `mcp_stdio_<label slug>` or `mcp_http_<label slug>` and the lower-cased name as key. The catalog keeps only the **names** (`env_names`, `header_names`) and the **endpoint** a secret was entered for. API responses, Job parameters and logs show names only. Leave a value blank (or send `null`) to reuse the stored one: that works only when the kind (stdio or HTTP) and the endpoint (the URL's scheme, host, port and path, or the exact command) are the same as the stored source's. Otherwise the import answers `422 missing_secret` ("re-enter the value: the endpoint changed") and nothing is sent. Values are at most 4096 characters. Error messages and the tail of a server's stderr are redacted: every known secret value and any `NAME=value` pair whose name looks secret becomes `***`. Secrets are only saved after a successful import.

## Read-only rules

A tool is *effectively read-only* only when its server says `readOnlyHint: true` (or you override it) **and** it is not destructive. A tool with `destructiveHint: true` can never be made read-only (`422`). Unknown hints count as mutating. A tool whose annotations say `readOnlyHint: false` and do not say `destructiveHint: false` is treated as **destructive** (the MCP default), so it cannot be overridden to read-only either; a tool with no annotations at all can still be marked read-only by you. You can mark tools **not allowed**: they are left out of the prompt and fail scoring.

The only code path that ever sends `tools/call` is the optional **dry run**: it needs `MCP_DRY_RUN_ENABLED=true`, an MCP source, and a tool that is allowed and effectively read-only. Anything else is refused and never executed.

## Prompt block

The allowed tools are rendered as one signature per line, sorted by name:

```text
Available tools (call only these):
- get_service(name: string) — Fetch one service. [read-only]
- create_service(name: string, tier: free|pro|enterprise) — Create a service. [mutating]
```

Long blocks are cut with a visible "truncated" line.
