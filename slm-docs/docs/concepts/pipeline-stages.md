---
sidebar_position: 3
title: Pipeline stages
---

# Pipeline stages

BrewSLM's pipeline has **11 canonical stages**, defined once in `app/models/run_event.py` and consumed by the readiness service, the timeline, the workflow runner, and the workspace pages. Each stage corresponds to:

- One service module (`app/services/*_service.py`).
- One API resource (`/api/projects/{id}/<stage>/...`).
- A step on one of the workspace pages (**Data**, **Train**, **Results**, **Ship**).
- A `stage` value on every `RunEvent` it emits.

## The 11 stages

| # | Stage | Service | Produces |
|---|---|---|---|
| 1 | `ingestion` | `ingestion_service` | `RawDocument` rows + raw artifacts in `DATA_DIR/projects/{id}/raw/`. |
| 2 | `cleaning` | `cleaning_service` | A cleaned `Dataset` version with PII/dup/outlier stats. |
| 3 | `gold_set` | `gold_service` / `gold_workbench_service` | `GoldSetVersion` + `GoldSetRow` (the ground-truth eval set). |
| 4 | `synthetic` | `synthetic_service` | Generated Q&A pairs joined to a dataset. |
| 5 | `dataset_prep` | `dataset_service` | Normalised, deduped, split prepared records. |
| 6 | `data_adapter_preview` | `data_adapter_service` | An adapter contract (which fields map where). |
| 7 | `tokenization` | `tokenization_service` | Tokenized dataset under `prepared_dir`. |
| 8 | `training` | `training_service` + a runtime plugin | Checkpoints + an immutable manifest. |
| 9 | `evaluation` | `evaluation_service` | `EvalResult` rows per gate / metric set. |
| 10 | `compression` | `compression_service` | Quantized / pruned model weights. |
| 11 | `export` | `export_service` | A deployable artifact for a target profile. |

Plus three cross-cutting "virtual" stages used by RunEvents only:

- `deployment` — every action on `/api/deployments/...`.
- `autopilot` — every plan / repair / snapshot action.
- `system` — startup, plugin reloads, config validation, support bundles.

## Stage ordering + unlocks

Stages become runnable progressively (no page is ever locked). The readiness service (`readiness_service.py`) decides what's runnable based on what artifacts exist:

```mermaid
flowchart LR
  raw[Has raw docs?] -- no --> n0[Stage 1 only]
  raw -- yes --> clean[Has cleaned dataset?]
  clean -- no --> n1[Stages 1-3]
  clean -- yes --> prep[Has prepared rows?]
  prep -- no --> n2[Stages 1-7]
  prep -- yes --> trained[Has trained checkpoint?]
  trained -- no --> n3[Stages 1-8]
  trained -- yes --> n4[All stages]
```

The Coach bar points at the next step that can run; pages are never disabled.

## Emit pattern

Every stage service wraps its work in a best-effort RunEvent emit. The pattern (intentionally minimal so observability bugs never break the stage):

```python
async def some_stage_action(...):
    try:
        await emit_event(
            db,
            project_id=project_id,
            run_id=run_id,
            stage=STAGE_TRAINING,
            severity=SEVERITY_INFO,
            summary=f"Started training experiment {experiment_id}",
        )
    except Exception:
        pass  # observability never breaks the action it reports on

    # ... actual work ...

    try:
        await emit_event(
            db,
            project_id=project_id,
            run_id=run_id,
            stage=STAGE_TRAINING,
            severity=SEVERITY_ERROR,
            reason_code=TRAINING_RUNTIME_ERROR,
            summary=f"Training failed: {exc}",
        )
    except Exception:
        pass
```

Notice the **`reason_code` is required** on `error` / `critical` severities. The taxonomy lives in `app/models/reason_codes.py`. See [Failure clusters](../observability/failure-clusters.md) for the full list.

## Stage → page in the UI

| Stage | Where |
|---|---|
| Stage 1 Ingestion | **Data** → **Sources** |
| Stage 2 Cleaning | **Data** → **Prepare** |
| Stage 3 Answer key | **Data** → **Answer key** |
| Stage 4 Synthetic | **Data** → **Generate** |
| Stage 5 Dataset prep | **Data** → **Split** |
| Stage 6 Adapter preview | **Data** → **Split** (adapter preview) |
| Stage 7 Tokenization | **Train** → **Tokenization** |
| Stage 8 Training | **Train** → **Training** (header links: Training configuration, Base model registry, Task type); Adapter Studio and Autopilot Planner under **Settings** |
| Stage 9 Evaluation | **Results** (header links: Playground, Compare runs) |
| Stage 10 Compression | **Ship** → **Compression** |
| Stage 11 Export | **Ship** → **Export** |
| `deployment` | **Ship** (header links: Deployments, Observability) |
| `autopilot` | **Settings** → **Autopilot Planner** |
| `system` | **Ship** → **Observability** (timeline filter `stage=system`) |

## Next

- [Beginner mode](beginner-mode.md) — what the flag changes.
- [Pipeline overview](../workflows/pipeline-overview.md) — per-stage walkthroughs.
- [Run events](../observability/run-events.md) — the canonical log every stage writes to.
