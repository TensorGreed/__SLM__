---
sidebar_position: 2
title: Common blockers
---

# Common blockers

A field guide to the errors a new BrewSLM user hits most often. Every entry maps a symptom → a reason_code → a fix, indexed against the [reason-code taxonomy](../observability/failure-clusters.md#reason-code-taxonomy).

When in doubt, run:

```sh
brewslm doctor --project <id> --deep
```

The deep doctor inspects readiness, recent timeline, and the cluster table — usually surfaces the right page below.

## Boot / setup

### Alembic head mismatch on backend startup

**Symptom**: `RuntimeError: Database revision mismatch. Current: ...; expected head: ...`

**Why**: Local DB is older than current migration head.

**Fixes** (pick one):

```sh
# Dev: nuke + auto-recreate
rm backend/data/brewslm.db
# Restart the backend — SQLite auto-create kicks in.

# Or run migrations explicitly
cd backend && alembic -c alembic.ini upgrade head
```

### `ModuleNotFoundError: app` running CLI

**Symptom**: CLI fails before any HTTP call.

**Fix**: activate the backend venv. `source backend/.venv/bin/activate`. The CLI shells out to backend modules.

### Invalid env values at startup

**Symptom**: backend doesn't start / boots but acts weird.

**Fix**: use `true` / `false` (lowercase) for booleans. Numeric env vars must be valid integers. The full list is in [Setup → Environment](../setup/environment.md).

### Frontend `ENOSPC` watcher limit (Linux)

**Symptom**: `npm run dev` fails with file watcher exhaustion.

**Fix**:

```sh
sudo sysctl fs.inotify.max_user_watches=524288
sudo sysctl -p
```

## Ingestion + cleaning

### `ingest_unsupported_format`

**Symptom**: file upload returns `400` with `detail="ingest_unsupported_format:<ext>"`.

**Fix**: convert to a supported format (CSV / JSONL / Parquet / TXT) OR add the extension to the connector. List the supported set with `brewslm dataset profile --help`.

### `cleaning_pii_block`

**Symptom**: cleaning stage blocks the dataset with PII findings.

**Fix**: choose one —

- **Redact at source** — strip PII from your dataset before re-uploading.
- **Relax the block** — edit the active domain pack's `data_quality.pii_policy` to `warn` instead of `block`. (Hidden in beginner mode; see [Domain packs](../workflows/pipeline-overview.md).)
- **Synthetic substitution** — let the cleaning hook replace PII tokens with placeholders.

### `cleaning_outlier_threshold_exceeded`

**Symptom**: outlier removal would drop too many rows.

**Fix**: relax the threshold in the domain pack, OR bring more data — small datasets trigger this easily.

## Adapter / dataset prep

### `adapter_schema_mismatch`

**Symptom**: prep stage fails with the wrong adapter selected.

**Fix**:

```sh
# Profile the data first
brewslm dataset profile --project 1 --source-type csv --source-ref ./data.csv

# Then let auto pick
brewslm adapter preview --project 1 --source-type csv --source-ref ./data.csv --adapter-id auto
```

If `auto` doesn't fit either, [scaffold a custom adapter](../extensions/scaffold.md).

### `adapter_field_resolution_failed`

**Symptom**: a required field is missing in the data.

**Fix**: check the dataset profile output for what columns actually exist, then either rename in source or pass `--field-mapping '{"text":"body"}'` to the prep call.

## Training

### `training_dispatch_error`

**Symptom**: training start returns 500; runner never picks up the job.

**Fix**:

```sh
# Default (simulate): no Celery needed
export TRAINING_BACKEND=simulate

# External: Celery + your training command
export TRAINING_BACKEND=external
export CELERY_BROKER_URL=redis://localhost:6379/0
celery -A app.celery_app worker -l info
```

### `training_oom`

**Symptom**: training crashes with CUDA OOM at step N.

**Fix**: the runner's CUDA OOM auto-retry planner usually halves the batch size and retries. If it gives up:

- Drop batch_size more (the recipe knob).
- Reduce context length.
- Switch base model to a smaller variant.
- Move to a larger target (server vs edge).

The [Autopilot decision log](../workflows/newbie-autopilot.md#decision-log) shows every retry path it tried.

### `training_timeout`

**Symptom**: run hits the wallclock budget and is cancelled.

**Fix**: raise the budget in the recipe (`max_minutes`), reduce dataset size for the first iteration, or use a smaller model.

### VRAM blocker at preflight

**Symptom**: preflight blocks before training starts, citing VRAM > target.

**Fix**:

- Smaller base model.
- Higher-memory target profile.
- Stronger quantisation (LoRA + 4-bit).

## Evaluation

### `eval_judge_unavailable`

**Symptom**: an eval pack with LLM-judge metrics fails partway.

**Fix**: check the LLM provider's status / quota / API key. The eval falls back to non-judge metrics; see the decision log for what gates ran without the judge.

### `eval_dataset_missing`

**Symptom**: eval pack referenced a gold set that no longer exists.

**Fix**: re-create the gold set OR pick a different eval pack.

## Export / deployment

### `export_quantization_failed`

**Symptom**: quantisation step exits non-zero.

**Fix**: usually an unsupported activation type. Check the compression job log; switching method (GGUF-Q4 → GGUF-Q8 → ONNX-INT8) is the most common fix.

### `deployment_smoke_failed`

**Symptom**: post-deploy smoke check failed.

**Fix**: read the failing prompt rows. Common causes:

- Tokenizer mismatch between checkpoint + target runtime.
- Compression hurt accuracy more than expected.
- Live runtime is using a stale model file (restart it).

### `deployment_drift_detected`

**Symptom**: drift check returns `verdict="drift_detected"`.

**Fix**: see [Drift checks](../deployment/drift-checks.md). Usually upstream data changed; rollback or retrain.

## Observability

### Estimated metrics never become measured

**Symptom**: cost estimator always says `provenance="estimated"`.

**Fix**: see [Measured vs estimated](measured-vs-estimated.md). The path:

1. Run one tiny calibration experiment (1 epoch, small batch, real backend).
2. Verify it appears in `experiments` with `status=COMPLETED` + both `started_at` and `completed_at` set.
3. Next similar config should report `provenance="measured"`.

### Plugin load errors

**Symptom**: extension page shows red `load_errors` rows for a kind.

**Fix**: pass the module path to validate first:

```sh
brewslm extensions validate --kind adapter --module my.adapter
```

The check report tells you exactly which contract step failed.

## Debugging rule of thumb

When blocked, capture (in this order):

1. The exact API endpoint / CLI command that failed + its response body.
2. The reason code (from the error or the linked RunEvent).
3. The relevant project / experiment / deployment id.
4. The last 10 RunEvents for the same `run_id` (timeline drilldown).

That context turns most triage from hours to minutes. If you need to forward to someone else, the [support bundle](../observability/support-bundles.md) packages all of it (with redaction) into one zip.

## Next

- [Measured vs estimated](measured-vs-estimated.md) — provenance principles.
- [Failure clusters](../observability/failure-clusters.md) — the cross-stage cluster view.
- [Reason-code glossary](../reference/glossary.md) — every code + meaning.
