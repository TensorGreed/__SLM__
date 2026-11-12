---
sidebar_position: 3
title: Environment
---

# Environment

Every BrewSLM setting reads from one of three sources, in order:

1. Environment variable (`export VAR=...`).
2. A `.env` file next to `backend/` (loaded by `pydantic-settings`).
3. The default in `backend/app/config.py`.

You don't need to set anything for a first dev run — SQLite + simulated training + no auth is the default. The variables below matter as you scale up.

## Runtime variables

| Variable | Default | Effect |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/slm_platform.db` (relative to where uvicorn starts) | Where the DB lives. Use Postgres for prod. |
| `DATA_DIR` | `./data` | Filesystem root for projects, exports, support bundles, extension scaffolds. |
| `DEBUG` | `false` | Enables verbose SQL logging and reload. |
| `DB_AUTO_CREATE` | `false` | Auto-create schema if missing. |
| `ALLOW_SQLITE_AUTOCREATE` | `true` | Auto-create schema specifically for SQLite. Convenient in dev. |
| `DB_REQUIRE_ALEMBIC_HEAD` | `true` in prod | Refuse boot if DB isn't at the migration head. |

## Synthetic-data CLI backend + API budget

| Variable | Default | Effect |
|---|---|---|
| `SYNTH_COMMAND_ALLOWED_BINARIES` | empty | Allow-list for the command (local CLI) synth backend; empty disables it. |
| `SYNTH_COMMAND_ARGV` | `["claude","-p","--output-format","json","--model","{model}"]` | argv template (JSON list). |
| `SYNTH_COMMAND_PROMPT_MODE` | `stdin` | `stdin` or `arg`. |
| `SYNTH_COMMAND_OUTPUT` | `.result` | `raw` or a dotted JSON path into the CLI's reply. |
| `SYNTH_COMMAND_DEFAULT_MODEL` | `sonnet` | Value for `{model}`. |
| `SYNTH_COMMAND_CONCURRENCY` | `1` | Parallel CLI processes. |
| `SYNTH_COMMAND_TIMEOUT_SECONDS` | `600` | Per-call timeout. |
| `SYNTH_COMMAND_MAX_RETRIES` | `2` | Retries for non-usage-limit failures. |
| `SYNTH_COMMAND_BACKOFF_SECONDS` | `5.0` | First backoff (doubles). |
| `SYNTH_COMMAND_BATCH_SIZE` | `8` | Requests packed per CLI call. |
| `SYNTH_COMMAND_USAGE_LIMIT_REGEX` | `usage limit\|rate limit\|429` | What marks a usage-limit failure. |
| `SYNTH_COMMAND_TRANSIENT_API_ERRORS` | `connection_failed,…,service_unavailable` | `api_error` values in a failed CLI reply that mean the network or the provider failed (retry later), not the request. |
| `SYNTH_COMMAND_TRANSIENT_REGEX` | `EAI_AGAIN\|ENOTFOUND\|…\|timed out\|overloaded` | The same, matched against the failure text only (stderr / the reply's `result`), never the whole JSON. |
| `SYNTH_COMMAND_COOLDOWN_SECONDS` | `900` | Pause after a usage-limit failure. |
| `SYNTH_COMMAND_MAX_PAUSE_SECONDS` | `28800` | Total pause per Job before failing. |
| `SYNTH_COMMAND_ENV_PASSTHROUGH` | empty | Extra env var names forwarded to the CLI. |
| `SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL` | `25` | Rows per CLI call in a synthetic playbook. |
| `SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL` | `10` | Rows per cloud call in a synthetic playbook (larger targets run as checkpointed chunks). |
| `SYNTH_CLOUD_EFFORT` | `medium` | Effort (`low` / `medium` / `high` / `xhigh` / `max`, or empty for the model default) sent to Anthropic models that take it. |
| `LLM_MONTHLY_BUDGET_USD` | unset | Account-wide hard cap on metered cloud-API spend per calendar month (UTC), all projects. |

Details: [Synthetic-data backends](../extensions/synth-backends.md#command-local-cli-backend).

### Judge resilience (LLM judges in evals)

When the judge model that scores eval answers is unreachable, a held-out eval retries the failed call, then stops calling it and leaves the remaining rows **unjudged** (every answer is still saved). Each eval starts fresh: an outage during an earlier eval never stops the next one before it has tried. A project can change the breaker, the mode and the longest wait under **Settings → Judge model → When the judge is unreachable**.

| Variable | Default | Effect |
|---|---|---|
| `JUDGE_BREAKER_ENABLED` | `true` | `false` asks the judge about every row once, with no retries and no stop, however long each failure takes (the behaviour before the breaker; the command backend still retries a failed CLI call itself, as it always did). Unjudged rows still get their reason. |
| `JUDGE_BREAKER_CONSECUTIVE` | `3` | Rows in a row whose judge call still failed after its retries (connection, timeout, 408 / 425 / 429 / 5xx) that stop the judge for the rest of that eval. A refused request in between resets the count. |
| `JUDGE_BREAKER_WINDOW` / `JUDGE_BREAKER_MIN_CALLS` / `JUDGE_BREAKER_ERROR_RATE` | `20` / `10` / `0.5` | Also stop when more than this share of the last calls failed. |
| `JUDGE_BREAKER_COOLDOWN_SECONDS` | `300` | In `wait` mode, how long a stopped judge rests before a normal call is tried again. |
| `JUDGE_TRANSIENT_RETRIES` | `2` | Extra attempts for a failed call (none for the command backend, which retries itself, and none with the breaker off). |
| `JUDGE_RETRY_BACKOFF_SECONDS` | `10` | First wait between attempts (then ×3, ±20 %). |
| `JUDGE_OUTAGE_MODE` | `stop` | `stop` leaves the remaining rows unjudged; `wait` checks the judge again every `JUDGE_OUTAGE_PROBE_SECONDS` and resumes when it answers. Only an eval running as a background job waits; any other eval stops. |
| `JUDGE_OUTAGE_PROBE_SECONDS` | `300` | Time between checks in `wait` mode. |
| `JUDGE_OUTAGE_MAX_WAIT_SECONDS` | `3600` | Longest wait in `wait` mode before stopping: wall time, the checks' own calls included, one budget for the whole job (both legs of a lift check share it; the call under way finishes, so it can run over by one call). Capped at half of `CELERY_VISIBILITY_TIMEOUT_SECONDS` (6 h by default). |

## Auth variables

| Variable | Default | Effect |
|---|---|---|
| `AUTH_ENABLED` | `true` | Master switch for the auth system. With it on, local login is always available. |
| `API_KEY` | `sk-mock-admin-key` | The local-login password. Change it outside a dev machine. |
| `AUTH_BOOTSTRAP_API_KEY` | empty (`backend/.env.example` sets `sk-mock-admin-key`) | When set, startup creates the bootstrap admin user. When empty, no admin is created and every local login is an engineer with no project access. |
| `AUTH_BOOTSTRAP_USERNAME` | `admin` | Bootstrap admin's username. |
| `AUTH_BOOTSTRAP_ROLE` | `admin` | Bootstrap user's role. |
| `OIDC_CLIENT_ID` | — | OIDC client id (SSO). |
| `OIDC_CLIENT_SECRET` | — | OIDC client secret (SSO). |
| `OIDC_DISCOVERY_URL` | — | OIDC discovery URL (`…/.well-known/openid-configuration`). |
| `AUDIT_LOG_ENABLED` | `false` | Write every API request to `audit_logs`. |

## Training runtime

| Variable | Default | Effect |
|---|---|---|
| `TRAINING_BACKEND` | `simulate` | Legacy alias. `simulate` → `builtin.simulate`; `external` → `builtin.external_celery`. |
| `CELERY_BROKER_URL` | — | RabbitMQ / Redis URL. Only needed for `external` backend. |
| `CELERY_RESULT_BACKEND` | — | Result store URL. Same caveat. |
| `EXTERNAL_TRAINING_COMMAND` | — | Shell template invoked by the external_celery runtime. |

See [Training runtime contract](../extensions/contracts.md) if you're writing your own runtime.

## Plugin module paths

These tell the loaders where to look for installed plugin modules. Each accepts a comma-separated list of importable Python module paths:

| Variable | Plugin kind | Loader |
|---|---|---|
| `DATA_ADAPTER_PLUGIN_MODULES` | data adapter | live |
| `TRAINING_RUNTIME_PLUGIN_MODULES` | training runtime | live |
| `DOMAIN_HOOK_PLUGIN_MODULES` | domain hooks | live |
| `TARGET_PROFILE_PLUGIN_MODULES` | target profile | live |
| `MODEL_CATALOG_PLUGIN_MODULES` | model catalog | live |
| `STARTER_PACK_PLUGIN_MODULES` | starter pack | live |

The domain pack and eval pack loaders ship in a follow-up — the contracts are defined now (see [Extensions → Contracts](../extensions/contracts.md)).

## Frontend variables

`frontend/.env` or `frontend/.env.local`:

| Variable | Default | Effect |
|---|---|---|
| `VITE_API_BASE` | `/api` (proxied) | Where axios sends requests. Set if your backend isn't co-located. |
| `VITE_DOCS_URL` | `http://localhost:3001/docs/getting-started/quickstart` | The `?` icon's `href` in the TopBar. Point to your prod docs site here. |

## Where these are read

| Setting | Code |
|---|---|
| Backend config | `backend/app/config.py` (Pydantic `Settings`) |
| Auth flags | `backend/app/api/auth.py` + `backend/app/middleware/auth.py` |
| Plugin loaders | `backend/app/services/*_service.py` for each kind |
| Frontend | `frontend/src/vite-env.d.ts` + `frontend/src/api/client.ts` |

## Inspecting current settings

### UI

TopBar → user menu → **Runtime settings** (requires `admin` role when auth is enabled). The modal shows every editable field with its current source: `env` (read from environment) or `override` (saved via this page). Restart-required fields are flagged.

### CLI

```sh
brewslm settings list
brewslm settings get DATABASE_URL
```

### API

```sh
curl http://localhost:8000/api/settings/runtime
```

Returns the same payload the UI modal renders.

## Next

- [Install + boot](install.md)
- [Auth + SSO](auth-and-sso.md)
- [Quickstart](../getting-started/quickstart.md)
