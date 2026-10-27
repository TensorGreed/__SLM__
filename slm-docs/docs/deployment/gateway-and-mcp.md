---
sidebar_position: 4
title: Serving gateway and MCP server
---

# Serving gateway and MCP server

The gateway puts a project's trained models behind one **OpenAI-compatible** API (`/v1`). Any OpenAI SDK, `curl`, an editor plugin or an agent can call them, with optional retrieval (answers come back with citations), schema-constrained JSON, scoped keys, rate limits and request metrics. A small **MCP server** exposes the same models as tools for Claude Code and other MCP clients.

In the UI, open **Ship** → **Deploy** → **Use from Claude Code** for keys, settings and copy-paste snippets.

## Keys

`/v1` always requires a gateway key, even when `AUTH_ENABLED=false`. A key is scoped to a list of projects and has a per-minute rate limit.

```bash
curl -sS -X POST http://localhost:8000/api/gateway/keys \
  -H 'Content-Type: application/json' \
  -d '{"name": "ci", "project_ids": [12], "rate_limit_per_minute": 60}'
```

The response carries the plaintext `key` (`bsk_...`) **once**; only its hash is stored. `GET /api/gateway/keys` lists keys without hashes, `DELETE /api/gateway/keys/{id}` revokes one. With auth enabled, creating a key needs the editor role on every listed project, and a key is listed and revocable by its creator, an admin, or someone who owns every project in its scope. A key keeps working after its creator loses access to a project until it is revoked. Deleting a project removes it from every key's scope (a key left with no project is revoked) and deletes that project's request log, so a reused project id is never covered by an old key.

Send the key as `Authorization: Bearer <key>` or `X-API-Key: <key>` (if you send both and they differ, the request is a `401`). A query-string token is never accepted. Over the limit, the gateway answers `429` with a `Retry-After` header (a sliding 60 second window per key).

## Model ids

A project's id is its **slug**: the name lowercased, runs of other characters turned into `-`, at most 48 characters. If another project already has that slug, `-<id>` is appended (and a counter if even that is taken). `p<id>` also works. Slugs derive from names, so renaming or deleting a same-named project can change a slug; module names baked into a running multi-LoRA vLLM then need a restart.

| Model id | Serves |
|---|---|
| `<slug>` | The project's served run: `default_run_id` from the gateway settings, else the latest completed run with saved weights. A RAG-first project with no run falls back to its base model with retrieval on. A `default_run_id` that is no longer servable is a `400` |
| `<slug>:run-<id>` | That run (must belong to the project and have weights) |
| `<slug>:base` | The project's base model, no adapter |

`GET /v1/models` lists every id the key can use. Each entry says where it comes from (`brewslm.project_id`, `project_name`, `run_id`, `base_model`) and whether retrieval is on by default and a JSON schema is set. An unknown id is a `404`, a project outside the key's scope a `403`.

## Calling it

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ${BREWSLM_API_KEY}" -H "Content-Type: application/json" \
  -d '{"model": "my-project", "messages": [{"role": "user", "content": "Hello"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=BREWSLM_API_KEY)
resp = client.chat.completions.create(
    model="my-project",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"brewslm": {"retrieval": True, "k": 4}},
)
```

Decoding matches evaluation: temperature `0` unless you set it, `max_tokens` defaults to 512 (1 to 4096). `stream: true` returns the finished answer as server-sent events (one content chunk, a final chunk with the finish reason, citations and `brewslm` block, then `data: [DONE]`). Errors use the OpenAI shape `{"error": {"message", "type", "code"}}`.

By default a model runs **in process** (the project's checkpoint on this machine). Under **Gateway settings** a project can instead forward to an OpenAI-compatible server (`upstream: openai_compatible`, `base_url`, and `upstream_kind` of `vllm`, `ollama` or `openai`). `upstream_model_map` renames a model id to the upstream's model name; without a map the upstream receives `<slug>:run-<id>` or `<slug>:base`, which is exactly what the multi-LoRA plan below serves.

## Retrieval and citations

Retrieval is on by default for runs trained with passages, off otherwise; override per request with `brewslm.retrieval` and `brewslm.k` (1 to 10). It uses the project's retrieval setting over its document passages. The preamble is put in front of the question exactly as passage training does it; if you send a system message, it follows the preamble inside the same system message.

The response carries `citations` (`n` matches the `[n]` numbering in the preamble, plus `document`, `source_url` when the document has one, a snippet and the score) and `brewslm.retrieval` (`applied`, `k`, `reranker`, `pool`, `chunk_size`, `source`, `count`). Without an index the request still succeeds, with `applied: false` and a `reason`.

## Guided JSON

If the project has a JSON schema (the structured-output hint), the gateway uses it unless you send `brewslm: {"guided_json": false}`. Every upstream kind (vLLM, Ollama, OpenAI-style) gets the same request shape, `response_format: {"type": "json_schema", "json_schema": {"name": "output", "schema": ..., "strict": true}}`, because Ollama's `/v1` ignores `format` and recent vLLM releases removed `guided_json`. The gateway never assumes the reply is valid: it validates it against the schema and reports the result.

| Case | What happens | `brewslm.guided_json` |
|---|---|---|
| Upstream (any kind) | `response_format` is sent; the reply is validated | `{"mode": "enforced", "via": "response_format", "valid": true, "errors": []}` |
| In process | cannot be enforced; the reply is validated | `{"mode": "validated_only", "valid": true, "errors": []}` |
| You send your own `response_format` / `guided_json` | yours is forwarded untouched, nothing is validated | `{"mode": "client", "via": ...}` |

`valid: false` comes with up to five error messages. `upstream_kind` is only a hint now (snippets and the multi-LoRA plan use it); it does not change what is sent. An upstream failure is a `502` with the message `upstream returned HTTP <status>`; the upstream's body is never echoed.

## Named tools

A project can publish **named tools**: a saved prompt over one of its models, with a JSON Schema for the input. Each is a tool in the MCP server and a call at `POST /v1/brewslm/tools/{name}/call`. Define them in the gateway settings:

```json
{
  "name": "detect_issues",
  "description": "Find issues in a code chunk",
  "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
  "model": "",
  "prompt_template": "@recipe",
  "retrieval": false,
  "max_tokens": 512
}
```

`prompt_template` substitutes `{field}` placeholders from the arguments; nothing else is interpreted (`{x.attr}` stays literal text). The special value `@recipe` means *use this project's training prompt format* (recipes that register a prompt builder; today: code-findings, which takes `code` plus optional `path`, `language`, `start_line`). Other recipes have no builder, so write a template. Arguments are validated against `input_schema` before the model runs. If two projects behind one key use the same tool name, the MCP-visible names become `<name>__<slug>`; a name colliding even then is skipped with a logged warning. `list_models`, `run_model` and `retrieve` are reserved and rejected in the settings. In the editor, the retrieval setting of a tool is *model default / always on / always off*. For `@recipe`, the labels follow training: the adapter config's explicit `finding_labels`, else the prepared manifest's `prompt_labels`, else the project taxonomy.

## Multi-LoRA serving

Several LoRA runs that share a base model can be served by one vLLM process. `POST /api/projects/{id}/serve/multi-lora-plan` with `{"experiment_ids": [..]}` returns a launch template (`--enable-lora`, `--max-loras`, `--max-lora-rank` rounded up to 8, 16, 32, 64, 128 or 256, one `--lora-modules <slug>:run-<id>=<adapter dir>` per run, the base as `<slug>:base`) and a `gateway_settings_hint` to paste into the gateway settings. Runs on different bases, full fine-tunes, runs without saved weights and adapters above rank 256 are refused with a `400` naming the run. The plan binds whatever `host` you give it and vLLM has no authentication, so the default stays `127.0.0.1`; only widen it behind your own access control. `POST .../serve/multi-lora-runs/start` launches the same plan.

## MCP server

```json
{
  "mcpServers": {
    "brewslm": {
      "type": "stdio",
      "command": "/path/to/python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/path/to/backend",
      "env": {"BREWSLM_GATEWAY_URL": "http://localhost:8000", "BREWSLM_API_KEY": "${BREWSLM_API_KEY}"}
    }
  }
}
```

The Ship tab generates this block with the right paths. Tools: `list_models`, `run_model {model, input, max_tokens?, system?, retrieval?}`, `retrieve {project, query, k?}` and every named tool. All are marked read-only. Results are text (with a `Sources:` list when citations exist) plus `structuredContent`; gateway failures come back as `isError` results with the gateway's message. The server only calls the gateway over HTTP with your key; it never opens the platform database. `python -m app.mcp_server --http --port 8765` serves the same protocol over HTTP at `POST /mcp` (bound to 127.0.0.1). The HTTP mode rejects a present `Origin` that is not a loopback http(s) origin and a `Host` that is not this server's loopback address (`403`), and requires `Content-Type: application/json` (`415`). A non-loopback `--host` is refused unless you pass `--allow-remote` (a warning is printed); the server has no authentication of its own and uses the key from its environment. Unknown tool names return JSON-RPC `-32602`; an internal failure returns `-32603`.

## What is logged

For each request the gateway stores the project, key, model id, run, time, latency, status, token counts, whether retrieval applied and the guided-JSON mode. **Prompt and reply text are never stored.** `GET /api/projects/{id}/gateway/telemetry` returns per-model request counts, error rate and p50 / p95 latency plus the 50 most recent requests; the Ship tab shows the same.
