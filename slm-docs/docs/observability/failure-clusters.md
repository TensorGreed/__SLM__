---
sidebar_position: 3
title: Failure clusters
---

# Failure clusters

A failure cluster is **a group of similar error events**, computed by folding the [RunEvent log](run-events.md) on the tuple `(stage, reason_code, signature)`. It tells you: *out of 184 events this week, only 4 distinct things actually went wrong, and here's how often each one happened*.

## What's a "signature"

The signature is a 12-char SHA1 hash of a **normalised** version of the event's summary text. Normalisation strips:

- ISO timestamps
- Hex tokens (request ids, run ids embedded in the message)
- Long digit runs (step numbers, byte counts)
- Whitespace runs

So these three error summaries:

```
Training failed: CUDA OOM at step 4321 (request abc123)
Training failed: CUDA OOM at step 4501 (request def456)
Training failed: CUDA OOM at step 4612 (request 9af0a2)
```

…all share the same signature → they fold into a single cluster. Two with the same `reason_code` but distinguishably different messages stay separate.

## Reason code taxonomy

There are 27 canonical reason codes across 9 stages. The full set lives in `app/models/reason_codes.py`. A few highlights:

| Stage | Reason code | When |
|---|---|---|
| `ingestion` | `ingest_unsupported_format` | File extension isn't allowed. |
| `cleaning` | `cleaning_pii_block` | PII scan blocked the dataset from advancing. |
| `adapter` | `adapter_schema_mismatch` | Adapter couldn't match its declared schema to the data. |
| `training` | `training_oom` | GPU out-of-memory during training. |
| `training` | `training_timeout` | Run exceeded wallclock budget. |
| `eval` | `eval_judge_unavailable` | LLM judge call failed. |
| `export` | `export_quantization_failed` | Quantization step exited non-zero. |
| `deployment` | `deployment_drift_detected` | Drift check found pass-rate delta beyond tolerance. |
| `deployment` | `deployment_smoke_failed` | Post-deploy smoke check failed. |
| `autopilot` | `autopilot_repair_blocked` | Strict mode refused an auto-repair. |
| `system` | `extension_load_failed` | Plugin module import / register raised. |

Adding a new code is a small change to `reason_codes.py` + a hook in the emitting service.

## Compute clusters

Clustering is **idempotent**: running it again on the same event log produces the same set of cluster rows with the same counts. The service `compute_failure_clusters` upserts on the 4-tuple key — never duplicates.

### UI

**Observability page → Failure clusters** card.

- Each row shows: red count badge, stage, reason_code, signature suffix, last-seen timestamp.
- **Show exemplars** expands to 3 representative events with their run id + summary.
- **View events** opens the per-run drilldown drawer (same as the timeline).
- **Recompute** button (top right) runs the full recompute and surfaces a summary toast: *"Recompute scanned 184 event(s); 2 created, 6 updated (24 total)."*

### CLI

```sh
brewslm doctor --project 7 --deep
```

The `--deep` flag includes the top clusters in the doctor output (compact, top 5 by count). For the full list:

```sh
brewslm logs clusters --project 7
```

### API

```sh
# List clusters, ordered by failure_count DESC
curl "http://localhost:8000/api/projects/7/failure-clusters?limit=100"

# Recompute (idempotent)
curl -X POST http://localhost:8000/api/projects/7/failure-clusters/recompute \
  -H "Content-Type: application/json" \
  -d '{"since": "2026-05-01", "until": null}'
```

The recompute response summary:

```json
{
  "events_considered": 184,
  "clusters_created": 2,
  "clusters_updated": 6,
  "clusters_total": 24
}
```

## Cluster row anatomy

```json
{
  "id": 142,
  "project_id": 7,
  "stage": "training",
  "reason_code": "training_oom",
  "signature": "a3f9c0b14d22",
  "failure_count": 8,
  "first_seen_at": "2026-05-09T10:23:00Z",
  "last_seen_at": "2026-05-12T11:08:00Z",
  "exemplar_event_ids": [987654, 987721, 987802],
  "exemplar_run_ids": ["exp-42", "exp-43", "exp-44"],
  "exemplar_summaries": [
    "Training failed: CUDA OOM at step 4321",
    "Training failed: CUDA OOM at step 4501",
    "Training failed: CUDA OOM at step 4612"
  ]
}
```

Exemplars are capped at 3 per cluster — enough to spot the pattern, small enough to render in a sidebar card.

## Workflow: from cluster to fix

A typical incident loop:

1. Notice a red count badge in the cluster list.
2. Expand exemplars → click **View events** on the most recent.
3. The drilldown drawer shows the full `payload` (e.g., `batch_size=32, seq_len=2048`).
4. Cross-reference with the [run timeline](timeline.md) for context (was autopilot running? did a deployment change?).
5. Fix the underlying issue.
6. The cluster's `last_seen_at` stops advancing — you can tell from the dashboard the fix took.

## Clusters in the support bundle

Every [support bundle](support-bundles.md) includes the current cluster snapshot. Useful when a teammate has to triage offline — they get the same "this is what's broken" view you saw.

## Next

- [Support bundles](support-bundles.md) — package clusters + events for hand-off.
- [Run timeline](timeline.md) — drill into individual events.
- [Reason codes glossary](../reference/glossary.md) — every code + meaning.
