---
sidebar_position: 4
title: Support bundles
---

# Support bundles

A **support bundle** is a single zip file that packages everything someone else would need to triage your project's recent activity, with **two-layer redaction** so you can safely hand it off without leaking secrets.

## What's in a bundle

| Section | Contents |
|---|---|
| `manifest.json` | Bundle metadata: uid, project id, generated_at, expiry, included sections. |
| `project.json` | Project record (without secrets). |
| `run_events.jsonl` | Recent RunEvents (default: last 7 days). |
| `failure_clusters.json` | Current cluster snapshot. |
| `deployment_versions.json` | All deployment versions for the project. |
| `deployment_telemetry.json` | Last telemetry window per active deployment. |
| `drift_checks.json` | Last N drift checks per deployment. |
| `experiments.json` | Recent experiments + training manifest pointer. |
| `autopilot_decisions.json` | Recent autopilot planning + repair decisions. |
| `support_bundle.txt` | Cover letter — what failed, what was redacted, who generated it. |

## Two-layer redaction

Before write, every section runs through two filters:

### 1. Key blocklist

Any field whose **name** matches a sensitive pattern is replaced with `"<redacted>"`. Examples: `password`, `secret`, `api_key`, `token`, `private_key`, `client_secret`.

### 2. Value pattern matching

Any field whose **value** matches a known secret pattern is replaced. Patterns include:

| Pattern | Catches |
|---|---|
| `hf_token` | HuggingFace tokens (`hf_…`). |
| `openai_key` | OpenAI `sk-` keys. |
| `anthropic_key` | Anthropic `sk-ant-…` keys. |
| `aws_access_key` | `AKIA…` (AWS access keys, 20 chars total). |
| `bearer_token` | `Bearer …` headers in payloads. |
| `jwt` | Three-segment dotted base64 tokens. |
| `url_with_credentials` | `https://user:pass@host/...`. |
| `ssh_private_key` | `-----BEGIN ...PRIVATE KEY-----`. |

Each redaction is **counted** and surfaced in the bundle metadata so you can verify before forwarding.

## Generate a bundle

### UI

**Observability page → Support bundle** card.

1. Click **Generate bundle**.
2. The card shows live progress, then surfaces the new bundle:
   - **UID** (unguessable hex).
   - **Size** + **expiry** (default 24h).
   - **Section counts** (rows per section).
   - **Redactions applied** (per-section totals + breakdown by reason).
   - **Download zip** link → opens in a new tab.

The link embeds the download token as a query parameter. Constant-time compared on the server. Tokens expire by `expires_at`; expired downloads return `410 Gone`.

### CLI

```sh
# Generate + auto-download to ./<uid>.zip
brewslm support-bundle create --project 7 --download

# Generate without downloading
brewslm support-bundle create --project 7 --ttl-seconds 86400

# List recent bundles
brewslm support-bundle list --project 7

# Download an existing bundle
brewslm support-bundle download --bundle-uid abc1234… --token tok…
```

### API

```sh
# Create
curl -X POST http://localhost:8000/api/projects/7/support-bundle \
  -H "Content-Type: application/json" \
  -d '{"actor": "alice@example.com", "ttl_seconds": 86400}'
```

Returns:

```json
{
  "bundle_uid": "abc1234567890def",
  "project_id": 7,
  "size_bytes": 4096,
  "sha256": "...",
  "section_counts": {
    "project": 1,
    "run_events": 184,
    "failure_clusters": 24,
    "deployment_versions": 3,
    "experiments": 12
  },
  "redactions_applied": {
    "run_events": {"total": 5, "by_reason": {"hf_token": 5}},
    "project":     {"total": 0, "by_reason": {}}
  },
  "expires_at": "2026-05-13T11:30:00Z",
  "created_at": "2026-05-12T11:30:00Z",
  "download_url": "/api/support-bundles/abc1234.../download?token=tok...",
  "download_token": "tok..."
}
```

Download:

```sh
curl -o bundle.zip "http://localhost:8000/api/support-bundles/abc1234.../download?token=tok..."
```

List for a project:

```sh
curl "http://localhost:8000/api/projects/7/support-bundles?limit=50"
```

## Reading a bundle

Unzip it; every section is a single JSON or JSONL file. A teammate can replay your timeline locally:

```sh
unzip abc1234567890def.zip -d bundle
jq '.[] | select(.severity == "error")' bundle/run_events.jsonl | head
```

Or import into another BrewSLM install (planned, not yet shipped).

## Storage + retention

Bundles live under `DATA_DIR/support_bundles/{project_id}/`. They're git-ignored. There's no automatic cleanup yet — the file persists past `expires_at` so you can still inspect it; only the **download URL** is invalidated. Manual delete is fine:

```sh
rm DATA_DIR/support_bundles/7/abc1234567890def.zip
```

## Stable reason codes

The API uses these `detail` codes for client retries:

| Reason | Status | Meaning |
|---|---|---|
| `project_not_found` | 404 | Bad project id. |
| `support_bundle_not_found` | 404 | Bundle uid doesn't exist (or expired record was pruned). |
| `support_bundle_invalid_token` | 403 | Token didn't match (constant-time compared). |
| `support_bundle_expired` | 410 | Past `expires_at`. |

## Next

- [Run timeline](timeline.md) — the events that get packed.
- [Failure clusters](failure-clusters.md) — the cluster snapshot in the bundle.
- [Rollback + score](../deployment/rollback-and-score.md) — when to generate a bundle (right before / after).
