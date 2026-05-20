# Training Run — Narration

Status: **synced** with the actual narrated take produced by
`tts/generate_v09_narration.py` (Orpheus voice "leo") on 2026-05-20.

The **Python script** at
[tts/generate_v09_narration.py](../../../tts/generate_v09_narration.py)
is the **authoritative source** of the spoken text. This file mirrors
the same text plus stage directions / Playwright cues. Edit the
script first.

Total runtime: **1:26** (matches
`docs-demo/recordings/raw/09-training-run-narrated.mp4`).
Section timings come from `tts/audio/v09-durations.json`.

Companion to:
[docs-demo/videos/09-training-run/recording-plan.md](../../videos/09-training-run/recording-plan.md).

First runtime-dependent video in the series. Actually trains an
experiment on the support-faq sample using the real Celery worker
and `builtin.external_celery` runtime.

---

## Pre-roll (not narrated)

Playwright logs in as admin, opens the **Demo · Support FAQ**
project, clicks the **Training** tab. ~5 seconds before narration
starts.

## Section 1 — Cold open (0:00–0:17)

**On screen**: Training tab, empty runs list.

> "Now we actually train. The first four videos walked the
> surfaces. This one launches a real training run on the support
> FAQ sample. Small model — a hundred and thirty-five million
> parameters — two epochs over sixteen prepared rows. Real Celery
> worker, real loss curve, real artifact on disk at the end."

## Section 2 — Config recap (0:17–0:35)

**On screen**: click **Open Training Config →** → land on
`/training-config` → click **Advanced** on the config-mode switch.

> "Quick recap of the Training Config page. Essentials view covers
> what you'd touch first — base model, epochs, batch size, learning
> rate. Flip to Advanced and you get the parameter-efficient
> training controls: low-rank adaptation rank, target modules,
> optimizer. Defaults are tuned for this hardware."

## Section 3 — Kickoff (0:35–0:47)

**On screen**: navigate back to `/pipeline/training`. Spec POSTs
the experiment create + start endpoints via the backend API. Page
reload shows the new experiment row in "running" status.

> "Back to the Training tab. I'm creating a new experiment and
> starting it. The Playwright spec uses the API for the
> create-and-start sequence so the recording stays deterministic.
> Either way, the worker queues the job and the runtime takes
> over."

## Section 4 — Watching (0:47–1:00)

**On screen**: experiment row shows status; spec polls the status
endpoint every 2s. Training runs to completion (~12s on GB10).
Page reloads to land the final state.

> "Status is running. Sixteen training steps total — each step does
> a forward pass, a backward pass, an optimizer step. The loss
> should drop across the run. On this hardware the whole thing
> finishes in about twelve seconds. Refresh the table."

## Section 5 — Results (1:00–1:17)

**On screen**: completed experiment row with metrics; Auto-Gate
panel showing the eval-schema check fail (expected — eval hasn't
run yet); "Training Complete" footer banner visible.

> "Completed. Two epochs, sixteen steps, final evaluation loss
> around five. The loss number is high because the model is tiny
> and the dataset has sixteen rows. The point isn't the loss
> number — the point is the loop completed end to end, and we now
> have a checkpoint on disk ready for evaluation."

## Section 6 — Wrap (1:17–1:26)

**On screen**: hold on the completed Training tab.

> "That's the training loop. Next video scores this experiment
> against the two-hundred-row gold set and tells us how often the
> model actually got the answer right."

---

## Things to **not** say

- Don't claim the model is "ready for production." It isn't.
- Don't read literal env var names or REST paths.
- Don't promise specific step counts if the dataset ever changes.
- Don't conflate `final_eval_loss = 4.84` with model quality — it's
  a perplexity-like metric on a tiny held-out set, dominated by
  variance.

## Optional technical notes (background; not spoken)

- The runtime is `builtin.external_celery`, dispatched from the
  backend's training service to a Celery worker connected to local
  Redis (`slm_redis` Docker container on :6379).
- The default optimizer `paged_adamw_8bit` requires `bitsandbytes`,
  which doesn't install cleanly on this aarch64 host. The spec
  overrides to `adamw_torch`. If you re-create the spec for an
  x86_64 host or with `bitsandbytes` installed, the override is
  optional but harmless.
- Checkpoint output dir: `data/projects/{project_id}/experiments/{exp_id}/`.
  Inspect with `tree` if you want to see the LoRA adapter shards.
- The Auto-Gate panel's "Missing required: memory: train_metrics_…"
  message is Story 1.5's data-shape gate detecting that eval hasn't
  run yet. Expected for this video. Video 10 unblocks it.

## Why the dry-run mattered

Before recording, the same training config was run once via the
backend API to verify it actually completes on this hardware. That
dry run hit a stub: `paged_adamw_8bit` blew up because
`bitsandbytes` isn't installed. The recording's spec hard-codes
the `adamw_torch` override based on that dry-run finding. If the
recorded spec is re-used as a template for Video 09's PII or
sentiment variants, keep the optimizer override.
