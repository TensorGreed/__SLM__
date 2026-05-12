---
sidebar_position: 3
title: Build your first project
---

# Build your first project

A narrated walkthrough that picks up where the [Quickstart](quickstart.md) left off. We take one realistic example — a support-ticket FAQ assistant — from a CSV file to a deployable artifact, with the UI / CLI / API alternatives at each step.

## Scenario

You have a CSV of resolved support tickets (~500 rows) with columns `question` and `answer`. You want a small instruction-tuned model that responds to FAQs in the same style. Target: a vLLM server.

## Step 1 — Create the project

### UI

Click **New Project** on the project list. Name it `Support FAQ`, template `support`, beginner mode on. Submit.

### CLI

```sh
brewslm project create --name "Support FAQ" --template support
```

### API

```sh
curl -X POST http://localhost:8000/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Support FAQ", "template": "support"}'
```

The template pre-fills sensible defaults for an FAQ-style assistant: starter eval pack, conservative training recipe, vLLM as the default target profile. You can override any of these later.

## Step 2 — Ingest your dataset

Aim for 100–500 high-quality rows for a first iteration. Tiny datasets are fine — you can grow once you've seen the loop work.

### UI

Pipeline rail → **Data** → **Add source** → **Upload CSV**. Pick `tickets.csv`. The Dataset Structure Explorer auto-profiles the file: row count, columns, sample values. Click **Continue**.

### CLI

```sh
brewslm dataset upload --project 1 \
  --source-type csv --source-ref ./tickets.csv \
  --name "tickets_v1"
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/datasets/upload \
  -F file=@tickets.csv \
  -F source_type=csv \
  -F name=tickets_v1
```

You should see the dataset appear in the Pipeline → Data tab with `status=ingested`.

## Step 3 — Clean

The cleaning stage normalises text, dedups, and runs a PII scan. The defaults from the `support` template are conservative — strict PII blocking, light dedup.

### UI

Pipeline rail → **Cleaning** → **Run cleaning**. The page shows row counts before / after, the PII findings (if any), and the dedup ratio.

### CLI

```sh
brewslm dataset clean --project 1 --dataset tickets_v1
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/datasets/clean \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "tickets_v1"}'
```

If the PII scan blocks the run with `reason_code=cleaning_pii_block`, see [Common blockers](../reliability/common-blockers.md).

## Step 4 — Build a gold set

The gold set is the ground-truth eval set. The workbench helps you label 50–100 rows by sampling intelligently.

### UI

Pipeline rail → **Gold set** → **Sample 100 rows (stratified)**. Each row has a textarea — paste the gold answer and approve. Submit; the gold version locks (draft → locked).

### CLI

```sh
brewslm eval gold-set sample --project 1 --strategy stratified --count 100
# … label rows in the UI …
brewslm eval gold-set submit --project 1 --version 1
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/gold-sets/sample \
  -H "Content-Type: application/json" \
  -d '{"strategy": "stratified", "count": 100}'
```

## Step 5 — Pick a base model

The Base Model Registry lists candidates with **measured compatibility** for your project's target + license.

### UI

Training rail → **Base Model Registry**. Filter `family=qwen`, `context >= 4096`, `license=permissive`. Click **Validate for project** on a candidate — surfaces tokenizer / chat-template / runtime warnings. Click **Set as default** when you find one you like (e.g. `Qwen2.5-1.5B-Instruct`).

### CLI

```sh
brewslm models list --family qwen --hardware-fit server --json
brewslm models validate --project 1 --model 12 --json
brewslm models set-default --project 1 --model 12
```

### API

```sh
curl "http://localhost:8000/api/projects/1/models/recommend?limit=5"
curl -X POST http://localhost:8000/api/projects/1/models/12/validate
```

## Step 6 — Train

For a first project, let the Autopilot pick the recipe. It chooses between safe-SFT, LoRA-fast, and a small full-fine-tune based on dataset size + base model.

### UI

Training rail → **Autopilot Planner** → describe the goal in plain English: *"Support FAQ tone, concise answers, no hallucinations beyond the dataset."* → **Plan** → **One-click run**. The page swaps to live mode.

### CLI

```sh
brewslm train start --project 1 --autopilot --one-click \
  --intent "Support FAQ tone, concise answers, no hallucinations beyond the dataset."
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/autopilot/plan \
  -H "Content-Type: application/json" \
  -d '{"intent": "Support FAQ tone, concise answers, no hallucinations beyond the dataset."}'

curl -X POST http://localhost:8000/api/projects/1/autopilot/run \
  -H "Content-Type: application/json" \
  -d '{"plan_id": "auto_..."}'
```

Watch the loss curve. Cost + provenance show in the Resolved Defaults panel — see [Measured vs estimated](../reliability/measured-vs-estimated.md).

## Step 7 — Evaluate

Once training finishes, the Eval stage kicks off automatically (Autopilot path) or you trigger it.

### UI

Pipeline rail → **Eval** → **Run evaluation**. Each gate (exact match, LLM-judge, safety) lands as a row with pass / fail and a per-gate metric. The **Failure Clusters** card below groups errors by reason code — click one to see exemplars.

### CLI

```sh
brewslm eval run --project 1 --experiment 1 --pack support-default
brewslm eval clusters --project 1
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/eval/run \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": 1, "pack_id": "support-default"}'
```

If any gate fails, the **Remediation** panel suggests concrete fixes (data, hyperparameters, prompt template). Apply, re-run training, re-evaluate.

## Step 8 — Export

When eval passes, export for the target.

### UI

Pipeline rail → **Export** → **New export** → target `vllm_server` → format `huggingface`. Click **Run export**. Artifact lands under `DATA_DIR/exports/`.

### CLI

```sh
brewslm export --project 1 --experiment 1 --target vllm_server --format huggingface
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/export \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": 1, "target_profile": "vllm_server", "format": "huggingface"}'
```

## Step 9 — Plan + smoke + promote deployment

Once you have an export, the [deployment loop](../deployment/plan.md) takes over. Plan → smoke → promote → telemetry. The full deployment workflow lives in the [Deployment](../deployment/plan.md) section.

## Definition of done (first iteration)

- [ ] One full project from ingest → export, all stages green.
- [ ] At least one failure cluster looked at, one remediation applied.
- [ ] One deployment promoted (against the simulator if no real serving env yet).
- [ ] You can re-run the whole thing from the project's training manifest — `brewslm train rerun --experiment 1`.

If you can check these four boxes, you've used every layer of BrewSLM.

## What to read next

- [Pipeline overview](../workflows/pipeline-overview.md) — every stage in more depth.
- [Newbie autopilot](../workflows/newbie-autopilot.md) — when autopilot helps and when to override.
- [Failure clusters](../observability/failure-clusters.md) — making sense of eval failures.
- [Measured vs estimated](../reliability/measured-vs-estimated.md) — reading provenance labels.
