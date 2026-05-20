# Video 09 — Training Run · Recording Plan

Status: **shipped 2026-05-20**. First runtime-dependent video in
the series. The training experiment in this recording is a real
Celery-dispatched run on the support-faq sample (16 train rows × 2
epochs = 16 steps, finishes in ~12s on GB10).

## Goal

Show that the platform's training loop actually runs end-to-end on
a seeded demo: experiment created, queued, executed by Celery,
checkpoint written to disk, status updated in the UI. The point is
**proof of the loop**, not impressive metrics — the model is tiny
and the dataset has 16 rows, so the final loss will be high.

## Audience

Intermediate. Assumes the viewer watched Videos 02 and 03 and is
already familiar with the Training tab and Training Config page.

## Final length

**1:26** (target was 4–6 min; the tightened narration came in well
under).

## Prerequisites

All runtime services up. Verified before recording:

```bash
docker ps --filter "name=slm_redis"          # → slm_redis Up
pgrep -fa "celery.*worker"                    # → worker on .venv
curl -sS http://localhost:8000/api/health    # → {"status":"ok"}
curl -sS http://localhost:11434/api/version  # → Ollama version
```

Required env in `backend/.env`:
- `TRAINING_BACKEND=external`
- `ALLOW_SIMULATED_TRAINING=false` (the default; recordings must be
  real)

## Training configuration used

| Field | Value | Why |
|---|---|---|
| `base_model` | `HuggingFaceTB/SmolLM2-135M-Instruct` | Smallest open instruction-tuned model that completes the loop quickly on GB10. Ungated. |
| `num_epochs` | `2` | Per Q17 in `12-runtime-decisions-2026-05-19.md`. |
| `batch_size` | `2` | Keeps memory low; sample has only 16 train rows. |
| `gradient_accumulation_steps` | `1` | No need to accumulate at this size. |
| `optimizer` | `adamw_torch` | Overrides the default `paged_adamw_8bit` because `bitsandbytes` isn't installed on this aarch64 host. |
| `max_seq_length` | `512` | Short context fits support-faq's `{question, answer}` rows. |
| `sequence_packing` | `false` | Pointless at 16 rows; would just confuse the timing math. |
| `warmup_ratio` | `0.0` | Run is too short to warm up. |

All other fields take the platform defaults. The
`/api/projects/{id}/training/experiments/effective-config` endpoint
resolves the full resolved config including LoRA defaults (rank 16,
target modules `q_proj, v_proj`).

## Exact starting state

1. Backend + frontend running.
2. Logged in as **admin**.
3. Seeded support-faq project (id varies per DB; pre-roll captures
   the id from the URL).
4. Training tab loaded with **no in-progress experiments**. The
   Playwright spec deletes prior dry-run experiments before
   recording, so the runs list starts empty.

## Recording arc (6 sections, audio durations from
`tts/audio/v09-durations.json`)

| # | Section | Audio (s) | What happens on screen |
|---|---|---:|---|
| 1 | Cold open | 17.15 | Training tab, empty runs list visible |
| 2 | Config recap | 17.58 | Navigate to `/training-config`; click Advanced |
| 3 | Kickoff | 12.12 | Back to Training tab; spec POSTs the experiment + start endpoint; page reloads to show "running" |
| 4 | Watching | 12.80 | Status visible; spec polls API every 2s until `completed`; UI re-renders to completed status |
| 5 | Results | 17.24 | Completed experiment row visible with final eval loss |
| 6 | Wrap | 8.87 | Hand-off to Video 10 (evaluation) |

Total audio: **85.8s**. Final muxed video: **1:25.76**.

## Why API kickoff instead of UI form

The Playwright spec creates and starts the experiment by POSTing
directly to `/api/projects/<id>/training/experiments` and
`/training/experiments/{id}/start`, using the JWT acquired from the
same local-login endpoint the UI uses. The TrainingPanel UI form is
not used.

This is deliberate. The form has dozens of fields and dynamic
validation that would either need every field selectored, or have
to rely on default values that change across releases. The API path
is the canonical entry point used by Autopilot, by the CLI, and by
external integrations, so it's a more durable selector for the
recording than the form's DOM.

The TrainingPanel UI polls the experiments list (every ~3s) and
re-renders to show the new experiment, then transitions through
running → completed. The viewer sees the same lifecycle they'd see
if they'd clicked through the form themselves.

## What's visible on the "completed" frame

| Element | Verified |
|---|---|
| `training-experiment-item` row with name `v09-narrated-run` | ✓ |
| Status badge "COMPLETED" | ✓ |
| Base model `HuggingFaceTB/SmolLM2-135M-Instruct` in row meta | ✓ |
| Auto-Gate panel showing **FAIL** with "Missing required: memory: train_metrics_..." | ✓ — Story 1.5 data-shape gate detects we haven't run eval yet; narration mentions this is normal |
| Pipeline Progress at training stage 60% | ✓ |
| "Training Complete" footer banner | ✓ |

## Things to not say

- Don't claim the model is "ready for production." It isn't — final
  loss is 4.84 because there are 16 training rows.
- Don't read the literal env var `TRAINING_BACKEND` aloud.
- Don't promise a specific number for total_steps if the dataset
  size ever changes. The current value is 16 (= 16 rows × 2 epochs
  ÷ batch_size 2). If the seeder's split ratio changes, this number
  drifts.

## What happens to the experiment after recording

The spec leaves the completed experiment in the DB. Video 10
(Evaluation) will pick it up to score against the gold set. If you
re-run this spec, the existing experiment doesn't conflict — the
new run gets a fresh id.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Spec times out at `expect(finalStatus).toBe('completed')` | Celery worker not running, or model download blocked | Verify `pgrep -fa celery` and check `~/.cache/huggingface/hub/` |
| `paged_adamw_8bit` requires bitsandbytes | Default config picked the 8bit optimizer | Spec hard-codes `adamw_torch` override; if removed, install `bitsandbytes` |
| Run finishes in <8s | GB10 is faster than dataset size expects | Increase `num_epochs` or `max_seq_length` to land the loop on a slightly slower budget |
| Disk full on `data/projects/{id}/experiments/` | Old experiments accumulating | API supports `DELETE /api/projects/{id}/training/experiments/{exp_id}` |

## Open questions parked for later

- **Q9** — Which eval handler dispatches for `support-faq`'s
  `instruction_sft` task profile? The completed experiment from this
  video will go through Video 10's eval. The handler resolution
  happens at score-time and will be answered when Video 10 records.
- **Q20** — Tokenizer download for the Tokenization tab. Not
  relevant here (we never opened that tab in this video) but worth
  resolving before any video that does open it for real.
