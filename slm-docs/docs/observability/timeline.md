---
sidebar_position: 2
title: Run timeline
---

# Run timeline

The Run Timeline is a tree-ordered read surface over the [RunEvent log](run-events.md). It joins events by `project_id` and walks the `parent_run_id` pointer to reconstruct *which op started which*. The result is the page that answers "what actually happened in this project this week?"

## What you see

```
exp-42 (training, info, 11:03)
├── exp-42 (training, error, 11:08, reason_code=training_oom)
└── exp-42 (training, info, 11:08, cancelled)

autopilot-abc (autopilot, info, 11:02)
├── exp-42 (started by autopilot)  ← same tree above
└── deploy-17 (deployment, info, 11:15)
    └── deploy-17 (deployment, error, 11:22, reason_code=deployment_drift_detected)
```

Each tree node shows:

- The run id + stage + highest severity in the run.
- A one-line summary.
- Expand button → child events.
- Click to deep-link into the event drill-in drawer.

## Read the timeline

### UI

**Training rail → Observability** → top section.

Filter bar:

- **Stage** — single-select.
- **Severity** — single-select.
- **Run id anchor** — type a `run_id` to highlight just that branch.
- **Since / Until** — ISO timestamps or relative (`1h ago`, `2026-05-01`).
- **Limit** — default 500.

Each tree row has:

- **Severity badge** (info / warning / error / critical).
- **Stage badge**.
- **Summary** + timestamp.
- **Expand** to see children.
- **View events** button → opens the per-run drilldown drawer (shows every event for this run with full payload).

Truncation: the service caps at `_MAX_EVENTS_PER_TIMELINE = 2000`. When that hits, a `truncated` badge appears in the header. Narrow your filters to see more.

### CLI

`brewslm doctor --deep` shells out to this surface. Compact view:

```sh
brewslm doctor --project 7 --deep
```

Prints a flat list of recent error/critical events, grouped by run. Useful for spotting "what's broken right now?"

For a full timeline as JSON:

```sh
brewslm logs timeline \
  --project 7 \
  --stage training \
  --since "2026-05-01" \
  --limit 200 \
  --json
```

### API

```sh
curl "http://localhost:8000/api/projects/7/timeline?\
stage=training&\
severity=error&\
since=2026-05-01T00:00:00Z&\
limit=200"
```

Returns:

```json
{
  "project_id": 7,
  "total_runs": 12,
  "total_events": 184,
  "orphaned_count": 0,
  "truncated": false,
  "tree": [
    {
      "run_id": "exp-42",
      "stage": "training",
      "highest_severity": "error",
      "summary": "Training failed: CUDA OOM at step 4321",
      "started_at": "2026-05-12T11:03:00Z",
      "duration_ms": 282000,
      "severity_counts": {"info": 5, "error": 2},
      "stages_present": ["training"],
      "latest_reason_code": "training_oom",
      "children": [...]
    }
  ]
}
```

## Orphans

When a child event's `parent_run_id` references a run that isn't in the current window (e.g. parent happened last month, child is fresh), the child becomes an **orphan**. Orphans render at the root with a small "orphan" badge. They're not a bug — just a hint that you may need to widen `since` to see the parent.

## Window math

`since` and `until` are inclusive on `since`, exclusive on `until`. Default window is the most recent 24 hours of events. The largest practical window is 30 days; beyond that the tree gets too big to render quickly.

The `--since` CLI flag accepts:

- ISO timestamps: `2026-05-01T00:00:00Z`.
- Relative durations: `30m`, `2h`, `7d`.
- Calendar dates: `2026-05-01` (interpreted as 00:00 UTC).

## Deep links

Every cluster exemplar (see [Failure clusters](failure-clusters.md)) and every drift-check failure includes an `event_id`. Clicking them in the UI scrolls + highlights the matching event in the timeline. Programmatically:

```
/api/projects/7/timeline?event_id=987654
```

…returns a tree narrowly scoped around that event.

## Storage

The timeline service is **read-only**: it doesn't write. All state is in `run_events`. The service is idempotent — calling it twice returns the same tree.

## Next

- [Failure clusters](failure-clusters.md) — events folded by `(stage, reason_code, signature)`.
- [Support bundles](support-bundles.md) — export a redacted snapshot of the timeline.
