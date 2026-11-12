---
sidebar_position: 2
title: Auth + SSO
---

# Auth + SSO

BrewSLM supports three auth modes, all driven by a single setting (`AUTH_ENABLED`).

## Modes

| Mode | When | How to enable |
|---|---|---|
| **None** | Local single-user. No login screen, no audit principal. | `AUTH_ENABLED=false`. |
| **Local login** (default) | Multi-user but no SSO. The password is the server's `API_KEY`; unknown usernames are created as engineers with no project access. Log in as the bootstrap admin (`AUTH_BOOTSTRAP_USERNAME`, default `admin`), which exists only when `AUTH_BOOTSTRAP_API_KEY` is set. | `AUTH_ENABLED=true` (default). |
| **SSO (OIDC)** | Production with corporate identity. | `AUTH_ENABLED=true` + `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` / `OIDC_DISCOVERY_URL` set. |

You can run **both Local and SSO at the same time** — the login screen offers both flows. Useful when a few admin accounts pre-exist before SSO is wired.

## Local login flow

### UI

Visit any protected URL. You're redirected to `/login`. The page shows a **Username + Password** form and (if SSO is on) a "Continue with SSO" button. Submit; you land on the project list.

### CLI

```sh
brewslm auth login --username alice --password '...'
# Stores the JWT in ~/.brewslm/token (configurable via BREWSLM_TOKEN_PATH).
```

After login the token is auto-attached on every CLI call.

You can also skip login and pass the token explicitly:

```sh
brewslm --token $YOUR_TOKEN projects list
```

Or set `BREWSLM_TOKEN` in your shell.

### API

```sh
# Get a token
curl -X POST http://localhost:8000/api/auth/local/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "..."}'
# → { "token": "..." }   (a JWT valid for 7 days; the password is the server's API_KEY)

# Use the token
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/auth/me
```

## SSO flow

Set the OIDC variables before starting the backend:

```sh
export AUTH_ENABLED=true
export OIDC_DISCOVERY_URL="https://auth.example.com/.well-known/openid-configuration"
export OIDC_CLIENT_ID="brewslm-app"
export OIDC_CLIENT_SECRET="..."
```

On the login page, **Continue with SSO** redirects to the provider, then back. The local DB `users` table gets a new row keyed on the OIDC `sub` claim on first login.

## API keys

Long-lived tokens for scripts + CI. Roles + sharing land in Wave I (priority.md P41–P45). For now the API accepts a static bearer token under `BREWSLM_TOKEN` for service-to-service calls.

When Wave I ships you'll be able to create API keys per user, scoped to specific projects.

## Audit

When `AUDIT_LOG_ENABLED=true` (default in production), every API write captures:

- `request_id`, `method`, `path`, `status_code`, `duration_ms`.
- `principal` (user id + role) if auth is on.
- `request_body_size`, `response_body_size`.

Persisted into the `audit_logs` table; surfaced via:

- `GET /api/audit/recent` — paginated read.
- The (planned) Audit Explorer UI in Project Settings (priority.md P44).

## Sample dev setup

A quick "everything on, one local user" config:

```sh
export AUTH_ENABLED=true
export API_KEY='StrongPass!23'                   # the local-login password
export AUTH_BOOTSTRAP_API_KEY='sk-your-admin-key' # creates the `admin` user on startup
export AUDIT_LOG_ENABLED=true
# (SSO off: leave OIDC_* unset)
```

Log in as `admin` with the `API_KEY`. Without `AUTH_BOOTSTRAP_API_KEY` no admin user is created, and `admin`, like any new username, becomes an engineer with no project access.

You can then log in via the UI at `/login`.

## Next

- [Install + boot](install.md) — getting the servers running.
- [Environment](environment.md) — every variable + its default.
