---
sidebar_position: 4
title: Training
---

# Training

Stage 8 of the [pipeline](pipeline-overview.md). BrewSLM treats training as a **first-class resumable + reproducible primitive**: every launch snapshots an immutable manifest, every checkpoint is browsable + promotable, every run is rerun-able from a single id.

## Three ways to start a run

| Want | Surface |
|---|---|
| Pick recipe + model interactively, see resolved defaults | UI → Training rail → **Configurations** |
| Reproduce a known-good run | UI / CLI → `train rerun --experiment N` |
| Brief in plain English, accept the plan | UI / CLI → Autopilot → [Newbie autopilot](newbie-autopilot.md) |
| Tweak one config knob | UI → Training Configurations → edit + **Apply + start** |

## Start a fresh run

### UI

Training rail → **Configurations**.

1. Pick a **Recipe** from the dropdown. Recipes are pre-tuned config templates: `safe-balanced-sft`, `lora-fast`, `classification`, `seq2seq`, etc. The picker shows their key knobs at a glance.
2. **Base model** — defaults to the project default; override here.
3. **Training mode** — `sft` / `dpo` / `orpo` / `classification` / `seq2seq` / `distillation` (filtered to what the recipe + model support). `distillation` trains against captured teacher logits — see [Distillation](distillation.md).
4. **Resolved defaults panel** below shows every field that will be applied with provenance (`recipe` / `domain_pack` / `model_metadata` / `default`).
5. **Cost estimate card** — gpu_hours, USD, CO2, provenance, confidence band. Pulled from real history when available; estimated otherwise. See [Measured vs estimated](../reliability/measured-vs-estimated.md).
6. Click **Preflight**. If green, click **Start training**.

### CLI

```sh
# Use a named recipe
brewslm train start --project 1 \
  --recipe safe-balanced-sft \
  --base-model 12

# Or pass an explicit config
brewslm train start --project 1 \
  --base-model 12 \
  --training-mode sft \
  --learning-rate 2e-4 \
  --num-epochs 3 \
  --batch-size 8

# Autopilot path
brewslm train start --project 1 --autopilot --one-click \
  --intent "Support FAQ, deploy on vLLM."
```

### API

```sh
# Preflight
curl -X POST http://localhost:8000/api/projects/1/training/preflight \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": 12,
    "recipe": "safe-balanced-sft",
    "training_mode": "sft"
  }'

# Start
curl -X POST http://localhost:8000/api/projects/1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": 12,
    "recipe": "safe-balanced-sft",
    "training_mode": "sft",
    "config": {"learning_rate": 2e-4, "num_epochs": 3}
  }'
```

## Warm-start checkpoints

A recipe can recommend a **pre-fine-tuned warm-start checkpoint** — a base model already task-pretuned on open corpora, so your rows only teach the *delta* (~3–5× fewer rows for the same quality). Recipes carry the recommendation as `recommended_starting_checkpoint`; the task-shaped offline-KD recipes already point at the planned task bases:

| Recipe | Recommended checkpoint |
| --- | --- |
| `recipe.kd.classification` | `classifier-base-135m` |
| `recipe.kd.qa` | `qa-base-135m` |
| `recipe.kd.span_extraction` | `ner-base-135m` |

Checkpoints live in a local registry at `backend/data/pretrained_checkpoints/<name>/manifest.json` (only the small manifests are tracked; the ~200 MB weights are produced on first use). At launch, training resolves the recommended checkpoint to its local weights **only when** the checkpoint is registered, architecture-compatible with the chosen base model, and its weights exist on disk — otherwise it **falls back to the base model** and records the reason (`checkpoint_planned`, `checkpoint_base_model_mismatch`, `checkpoint_artifact_missing`, …) under `_runtime.warm_start` on the experiment.

**Where you see it.** The resolution is surfaced as a **Starting weights** line so you always know which weights a run used:

- *Before launch* — applying a recipe (or running preflight) shows a Starting-weights chip in the Training Config setup, and a **Starting Weights** row in the advanced **Resolved Defaults** panel. The `/recipes/resolve`, `/experiments/effective-config`, and `/experiments/preflight` responses all carry a `warm_start` preview block.
- *After launch* — the **Why this plan** panel's Strategy section shows it for the active run, and the captured **run manifest** records it under `warm_start` (lifted out of the transient `_runtime` block because *which weights a run used* is reproducibility-relevant provenance).

> **Status:** the registry, recipe field, resolution, and UI/manifest surfacing are wired and tested. The four planned task bases (ClassifierBase / NERBase / QABase / SQLBase, all on the `SmolLM2-135M-Instruct` line) ship as `status: "planned"` manifests, so every run currently falls back to a clean base-model cold start. Training the actual checkpoints (~32 GPU-hours on a GB10, published to `TensorGreed/`) is follow-up work.

## Trainability forecast

The **trainability forecast** runs *before* preflight, on the Training Config page. It looks at the project's recipe + gold set + base model and predicts whether the upcoming run is likely to clear the default Auto-Gates. Advisory only — it never blocks the run; if the verdict is amber/red the Train button just relabels to "Train anyway".

### Recipe-agnostic signals (always run)

| Signal | Fires when |
|---|---|
| `row_count_below_minimum` | Labeled-corpus size is below the recipe's `min_rows_recommended` (block) or below 1.5× (warn). |
| `goldset_diversity_low` | Mean pairwise token-Jaccard over gold rows is above 0.40 — rows look too similar to each other. |
| `gate_pass_probability` | The overall heuristic; combines row count, base-model capacity, recipe difficulty, diversity, and (for classification) class entropy. |

### Per-recipe signals

Dispatched by the recipe's `task_profile`. A non-classification project never sees the classification signals and vice versa — the forecast was previously qa-sft-flavored and now adapts per recipe.

| Recipe | Signal | Fires when |
|---|---|---|
| `classification` | `class_imbalance` | Shannon entropy of the label distribution is low (warn at `<1.0`, block at `<0.5`). |
| `classification` | `per_class_minimum_unmet` | Any class has fewer than 5 examples (warn) or fewer than 2 (block). The corpus-wide minimum doesn't catch per-class starvation. |
| `classification` | `label_vocab_fragmented` | Two or more labels collapse to the same canonical (lowercased + stripped) form — `"positive"` vs `"Positive"`. Same drift class the gold-set add form already warns about. |
| `classification` | `single_class_dominance` | Any one class is more than 80% of the gold set. The model defaults to that class regardless of other signals. |
| `span-extraction` | `format_inconsistency` | Some gold rows have missing/invalid span structures (non-dict, non-int offsets, `start > end`). |
| `span-extraction` | `entity_type_coverage_thin` | Fewer than 3 distinct entity types across the gold set (warn). Single-type tasks block. |
| `span-extraction` | `span_offset_invalid` | `text[start:end]` doesn't match `span.text` on some rows — silent offset rot that tanks exact-match scoring. Block when more than 10% of rows are bad. |
| `span-extraction` | `negative_examples_missing` | No rows have an empty entities list. Without negatives the model learns "always extract something" and over-fires. |
| `summarization` | `summary_doc_ratio_outliers` | Rows where the summary is more than 70% of the document length — usually a mislabeled paraphrase or the wrong column loaded into the summary slot. |

Suggested actions on every signal map to one of `synth_augment` / `synth_balance` / `synth_diversify` / `fix_gold_rows`, surfaced as a one-click button next to the signal row. Each actionable signal also carries a `cost_estimate: {time_minutes, llm_cost_usd | null, confidence}` payload — the panel renders it as a chip ("~25 min · $0.01" or "~6 min · no $" for manual fixes) and, when ≥2 signals carry actions, surfaces a "Cheapest fix first" hint ranking by wall-clock then LLM cost. The estimator is heuristic ("rough" confidence); it'll move to "calibrated" once enough T5 telemetry lands to retune the per-row constants.

### Snapshot history + sparkline

Every cache-miss compute writes one row to `training_forecast_snapshots`. The Training Config panel reads `GET /api/projects/{id}/training/forecast/history?limit=10` and renders a sparkline above the signal list — `confidence_pct` on the y-axis (fixed [0, 100] so the shape is comparable across sessions) with a coloured dot per snapshot (green = `likely_pass`, amber = `borderline`, red = `likely_fail`). Hover a dot to see its verdict + signal severities at the time of compute.

### Calibration (admin)

`GET /api/admin/forecast/calibration?recipe=<id>` exposes forecast-vs-reality calibration: every experiment is paired with the user's most-recent forecast snapshot at creation time, and resolved against the actual gate-pass verdict when `evaluate_experiment_auto_gates` runs. The response buckets resolved observations into 10%-confidence bands so per-recipe calibration drift (e.g. predicted 70-80% but actually passing 40% of the time) is visible without leaving the JSON. Used for retuning the heuristic coefficients in `trainability_forecast_service.estimate_gate_pass_prob`. No UI in v1 — admin-only endpoint.

Next to the sparkline a three-chip strip shows the last three verdict deltas (`▼ -24%`, `· 0%`, `▲ +12%`). The user can pin down whether a gold-set edit or synth run actually moved the needle without re-reading the signal list.

Cache hits do not add to history — only true recomputes do, so the sparkline reflects iteration, not idle polling. Snapshots older than 60 days are pruned on insert.

### API

```sh
curl http://localhost:8000/api/projects/1/training/forecast
# Cached by default on `Project.training_forecast_cache`. Recipe + dataset + base-model changes invalidate.
curl http://localhost:8000/api/projects/1/training/forecast?refresh=true

# Recent snapshots (newest-first). ``limit`` is clamped to [1, 100].
curl http://localhost:8000/api/projects/1/training/forecast/history?limit=10
```

The forecast reads the project's `selected_recipe.recipe_id` and dispatches signals through a per-recipe builder — see [`backend/app/services/trainability_forecast_service.py`](https://github.com/anugram/__SLM__/blob/main/backend/app/services/trainability_forecast_service.py).

## Preflight blockers

The preflight endpoint catches the common "this won't work" cases **before** the runner starts burning compute.

| Reason code | Means |
|---|---|
| `training_dispatch_error` | The training runtime backend wasn't selectable (Celery down, external command missing). |
| `training_runtime_error` (preflight) | Tokenizer / chat template / adapter incompatible with the chosen model. |
| Capability check fail | Model doesn't support the requested training mode (e.g. `dpo` on a base model with no chat template). |
| VRAM over budget | Model + batch size + context exceeds estimated VRAM for the target. |

The UI's Resolved Defaults panel surfaces each blocker with the actionable fix.

## Reproducibility — the manifest

Every successful training launch writes an **immutable** training manifest to `manifests/exp-<id>.json`:

```json
{
  "experiment_id": 42,
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "training_mode": "sft",
  "recipe": "safe-balanced-sft",
  "config": {"learning_rate": 2e-4, "num_epochs": 3, "batch_size": 8},
  "dataset_version_id": 17,
  "tokenized_dir": "DATA_DIR/projects/1/prepared/tokenized_v3",
  "adapter": {"id": "qa-pair", "version": 2},
  "domain_pack": {"pack_id": "support-pack-v1", "version": 1},
  "target_profile": "vllm_server",
  "seed": 42,
  "git_sha": "f3dccd8…",
  "captured_at": "2026-05-12T10:23:00Z"
}
```

Once written it's append-only. Modifying it doesn't change the run; rerun reads from this file.

## Rerun an experiment

The single fastest path to a reproducible run.

### UI

Pipeline → **Pipeline Runs** → click an experiment row → **Rerun**. The manifest is replayed verbatim into a new experiment id.

### CLI

```sh
brewslm train rerun --experiment 42
brewslm train rerun --experiment 42 --run-name "phase-a-rerun"
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/experiments/rerun \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": 42, "run_name": "phase-a-rerun"}'
```

## Clone with overrides

When you want "the same run but with one knob changed":

### CLI

```sh
brewslm train clone --experiment 42 \
  --name "lr-3e-4" \
  --config-overrides '{"learning_rate": 3e-4}'
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/experiments/clone \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": 42,
    "name": "lr-3e-4",
    "config_overrides": {"learning_rate": 3e-4}
  }'
```

The clone gets its own manifest. Both parent + clone manifests are queryable.

## Live monitoring

While a run is in flight:

- **UI** — Training Configurations page swaps to live mode: loss curves, per-step metrics, GPU memory, telemetry, stop / pause / resume buttons.
- **CLI** — `brewslm logs tail --project 1 --run-id exp-42` streams events.
- **API** — `GET /api/run-events/run/exp-42`.

## Pause + resume + cancel

Long runs can be paused (writes a checkpoint, releases GPU) and resumed from the same step later.

### UI

Live training page → **Pause**. Later, **Resume** picks up at the next step.

### CLI

```sh
brewslm train pause --project 1 --experiment 42
brewslm train resume --project 1 --experiment 42
brewslm train cancel --project 1 --experiment 42 --reason "wrong recipe"
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/experiments/42/pause
curl -X POST http://localhost:8000/api/projects/1/experiments/42/resume
curl -X POST http://localhost:8000/api/projects/1/experiments/42/cancel \
  -d '{"reason": "wrong recipe"}'
```

## Checkpoint browser

Every N steps the runtime saves a checkpoint. The browser lets you promote a non-final one (useful when training overfit late) or resume from any past step.

### UI

Training rail → **Models** → click the experiment row → **Checkpoints** drawer. Each row has actions:

- **Promote** — mark this checkpoint as the run's `final` (replaces what eval will use).
- **Resume from** — kick off a continuation from this step.

### CLI

```sh
brewslm train checkpoints --project 1 --experiment 42
brewslm train checkpoints --project 1 --experiment 42 --promote-step 200
brewslm train checkpoints --project 1 --experiment 42 --resume-from-step 150
```

### API

```sh
curl "http://localhost:8000/api/projects/1/experiments/42/checkpoints"
curl -X POST http://localhost:8000/api/projects/1/experiments/42/checkpoints/200/promote
curl -X POST http://localhost:8000/api/projects/1/experiments/42/checkpoints/150/resume
```

## Reason codes you might hit

| Code | Means |
|---|---|
| `training_dispatch_error` | Runtime backend couldn't launch (Celery down, external command missing). |
| `training_runtime_error` | Generic runtime failure inside the training loop. |
| `training_oom` | GPU out of memory. The CUDA OOM auto-retry planner may have already tried smaller batch sizes; see the decision log. |
| `training_timeout` | Wallclock budget exhausted. |
| `training_cancelled` | Operator action (UI / CLI / autopilot strict-mode refusal). |

For walk-throughs of each, see [Common blockers](../reliability/common-blockers.md).

## Next

- [Evaluation + remediation](evaluation-and-remediation.md) — what to do with the trained checkpoint.
- [Export + deployment](export-and-deployment.md) — shipping it.
- [Measured vs estimated](../reliability/measured-vs-estimated.md) — reading the cost estimate.
