---
sidebar_position: 1
title: Run events
---

# Run events

A **RunEvent** is the unit of observability in BrewSLM. Every pipeline stage, autopilot decision, deployment action, and system-level event emits at least one RunEvent. The [timeline](timeline.md), [failure clusters](failure-clusters.md), and [support bundles](support-bundles.md) all read from this one table — there's no parallel logging system.

## Schema

```python
class RunEvent:
    id: int                              # primary key
    project_id: int                      # FK; every event is project-scoped
    run_id: str                          # e.g. "exp-42", "deploy-17", "autopilot-abc123"
    parent_run_id: str | None            # parent's run_id; used to build the tree
    stage: str                           # one of: ingestion, cleaning, adapter, training,
                                         #         eval, export, deployment, autopilot, system
    severity: str                        # info | warning | error | critical
    reason_code: str | None              # required on error/critical; see below
    actor: str                           # "system" | "user:<id>" | "agent:<name>"
    summary: str | None                  # short human-readable line
    payload: dict                        # structured details, free-form
    ts: datetime                         # when it happened
    created_at: datetime                 # when we wrote it
```

## What `run_id` looks like

The convention is `<stage>-<id>`:

| run_id | Emitted by |
|---|---|
| `exp-42` | training service, for experiment 42 |
| `deploy-17` | deployments router, for deployment 17 |
| `autopilot-{hex}` | autopilot service, one per planning session |
| `ingest-{hex}` | ingestion service, one per ingest job |
| `system-{hex}` | startup / plugin reload / config validation |

`parent_run_id` is set when one op started another. E.g. an autopilot-launched training run has `parent_run_id="autopilot-abc"` and its own `run_id="exp-42"`. The [timeline](timeline.md) walks this pointer to build the tree.

## Stages + severities + reason codes

The full canonical lists:

```python
# app/models/run_event.py
STAGES = frozenset({
    "ingestion", "cleaning", "adapter", "training",
    "eval", "export", "deployment", "autopilot", "system",
})

SEVERITIES = frozenset({"info", "warning", "error", "critical"})
```

Reason codes are a closed taxonomy. The emit service **rejects** any `severity in {error, critical}` event that's missing one, or that has a value outside the registered set. See [Failure clusters](failure-clusters.md#reason-code-taxonomy) for the full list (27 codes across 9 stages).

## Emit an event

Stages emit via `app/services/run_event_service.emit_event`. The contract is a best-effort wrapper: the emit call is wrapped in `try / except` so an observability bug never breaks the stage that's reporting on itself.

```python
# Inside a stage service
from app.services.run_event_service import emit_event
from app.models.run_event import STAGE_TRAINING, SEVERITY_ERROR
from app.models.reason_codes import TRAINING_OOM

try:
    await emit_event(
        db,
        project_id=project_id,
        run_id=f"exp-{experiment_id}",
        parent_run_id=parent_run_id,  # optional
        stage=STAGE_TRAINING,
        severity=SEVERITY_ERROR,
        reason_code=TRAINING_OOM,
        summary="CUDA OOM at step 4321 — batch_size=32, seq=2048",
        payload={"step": 4321, "batch_size": 32, "seq_len": 2048},
    )
except Exception:
    pass  # never break the action
```

## Read events

### UI

The [Run Timeline](timeline.md) page is the primary read surface. It supports filters:

- Stage (one or more).
- Severity (one or more).
- `run_id` (anchor on a specific run).
- `since` / `until` (window).
- `limit` (default 500, max 2000).

Plus a per-event drill-in drawer showing the full `payload`.

### CLI

```sh
# Tail events for a specific run
brewslm logs tail --project 7 --run-id exp-42

# Or filter the timeline
brewslm logs timeline --project 7 --stage training --severity error --since "2026-05-01"
```

### API

```sh
# Paginated list
curl "http://localhost:8000/api/projects/7/run-events?stage=training&severity=error&limit=200"

# Events for one run (no project_id needed)
curl http://localhost:8000/api/run-events/run/exp-42

# Tree-ordered timeline (walks parent_run_id)
curl "http://localhost:8000/api/projects/7/timeline?since=2026-05-01"
```

## When *not* to emit a RunEvent

RunEvents are **operationally interesting** facts. They're not a metric, not a debug log, not a hot-path counter.

| Use RunEvent for | Use something else for |
|---|---|
| "Training run started / failed / completed" | Loss curves (those are eval metrics) |
| "Export to vLLM target finished" | Token-by-token decoding traces |
| "Deployment promoted / rolled back" | Request-level inference logs (those are telemetry samples) |
| "Plugin failed to load" | Per-request timing (those are telemetry) |

Roughly: emit one event per state transition, not one per row processed. A 50,000-row ingestion job emits 1–3 events total (start, optional progress, end), not 50,000.

## Storage + retention

Events live in `run_events`. There's no automatic retention — they're cheap (tens of kB per project per day at typical activity) and the failure-cluster service depends on history. If your project's table grows past comfort, the [support bundle](support-bundles.md) flow can archive a snapshot before pruning.

## Next

- [Timeline](timeline.md) — the tree-ordered read surface.
- [Failure clusters](failure-clusters.md) — RunEvents folded by reason_code + signature.
- [Support bundles](support-bundles.md) — redacted export of recent events for hand-off.
