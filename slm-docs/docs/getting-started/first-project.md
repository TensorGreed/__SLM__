---
sidebar_position: 3
title: Build your first project
---

# Build your first project

A narrated walkthrough that picks up where the [Quickstart](quickstart.md) left off. We take one realistic example — a support-ticket FAQ assistant — from a CSV file to a deployable artifact, with the UI / CLI / API alternatives at each step.

**Before you start** (same setup as the Quickstart):

- The CLI is the `./brewslm` script at the repo root; run it from there (it is not on your `PATH`). Log in once: `./brewslm auth login --username admin --save`.
- The examples assume the default ports (API `8000`, UI `5173`). If you moved the backend, set `BREWSLM_API_BASE=http://localhost:<port>` for the CLI and change the port in the `curl` lines.
- Auth is on by default, so every API call needs a token:

  ```sh
  TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/local/login \
    -H "Content-Type: application/json" \
    -d '{"username": "admin", "password": "sk-mock-admin-key"}' | python3 -c 'import sys, json; print(json.load(sys.stdin)["token"])')
  ```

  The `curl` lines below pass it as `-H "Authorization: Bearer $TOKEN"`. Full request shapes are in the [API reference](../reference/api/projects.md).

## Scenario

You have a CSV of resolved support tickets (~500 rows) with columns `question` and `answer`. You want a small instruction-tuned model that responds to FAQs in the same style. Target: a vLLM server.

## Step 1 — Create the project

### UI

Click **+ New Project** on the project list. In the **New BrewSLM Project** dialog, optionally set **Project name** to `Support FAQ` (left empty, the name is taken from the brief and shown under the field) and, under **What problem do you want this model to solve?**, describe the goal in a sentence or two (for example *"Answer customer support FAQs from our resolved tickets, in the same tone."*). BrewSLM reads that brief to pick defaults: base model, task profile and output schema. **Advanced options** holds sample inputs / outputs, the base model, the deployment target and a **Domain preset**. Click **Create project**. A new project has no data yet, so it opens on **Data › Sources** (the **Overview** is in the sidebar).

### CLI

```sh
./brewslm project create --name "Support FAQ" --template support
```

### API

```sh
curl -X POST http://localhost:8000/api/projects \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "Support FAQ", "brief_text": "Answer customer support FAQs from our resolved tickets, in the same tone."}'
```

The brief (UI and API) or the CLI's `--template` pre-fills sensible defaults for an FAQ-style assistant. You can override any of these later.

## Step 2 — Ingest your dataset

Aim for 100–500 high-quality rows for a first iteration. Tiny datasets are fine — you can grow once you've seen the loop work.

### UI

**Data** → **Sources** → **Add source** → **Upload CSV**. Pick `tickets.csv`. The Dataset Structure Explorer (Details → *Exploratory analysis*) auto-profiles the file: row count, columns, sample values. Click **Continue**.

### CLI

```sh
./brewslm dataset import --project 1 --file ./tickets.csv --source tickets_v1
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/ingestion/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@tickets.csv \
  -F source=tickets_v1
```

You should see the dataset appear under **Data** → **Sources** with `status=ingested`.

## Step 3 — Clean

The cleaning stage normalises text, dedups, and runs a PII scan. The defaults from the `support` template are conservative — strict PII blocking, light dedup.

### UI

**Data** → **Prepare** → **Clean All Documents**. The *Cleaning results* Details section shows row counts before / after, the PII findings (if any), and the dedup ratio.

### CLI

There is no cleaning subcommand; `./brewslm dataset import` processes the file on upload (skip that with `--no-process`). Use the UI or the API to re-clean.

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/cleaning/clean-batch \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"document_ids": [1], "redact_pii": true}'
```

If the PII scan blocks the run with `reason_code=cleaning_pii_block`, see [Common blockers](../reliability/common-blockers.md).

## Step 4 — Build an answer key

The answer key (API name: *gold set*) is the ground-truth eval set. The workbench helps you label 50–100 rows by sampling intelligently.

### UI

**Data** → **Labels** → Details → *Answer-key workbench* → **Sample 100 rows (stratified)**. Each row has a textarea — paste the correct answer and approve. Submit; the answer-key version locks (draft → locked).

### CLI

```sh
# Sample 100 rows from source dataset 1 into the answer key (the result prints the gold-set id)
./brewslm eval gold-set add-row --project 1 --from-source-dataset-id 1 --target-count 100 --strategy stratified
# … label rows in the UI …
./brewslm eval gold-set submit --gold-set-id <gold-set id>
```

### API

The answer-key endpoints (`/api/projects/1/gold/...`: add, import, review, lock) are in the [Gold dataset API reference](../reference/api/gold-dataset.md).

## Step 5 — Pick a base model

The Base Model Registry lists candidates with **measured compatibility** for your project's target + license.

A base model set under **Advanced options** in the **+ New Project** dialog is kept: the task recipe applied from your brief only fills the base model when you left it empty.

### UI

**Train** → **More options** → *Base model registry* (or **Browse** next to the base model field). Filter `family=qwen`, `context >= 4096`, `license=permissive`. Click **Validate for project** on a candidate — surfaces tokenizer / chat-template / runtime warnings. Click **Set as default** when you find one you like (e.g. `Qwen2.5-1.5B-Instruct`).

### CLI

```sh
./brewslm models list --family qwen --hardware-fit server --json
./brewslm models recommend --project 1 --limit 5
./brewslm models validate --project 1 --model 12 --json
```

The CLI has no set-default command: set the base model in the UI or with `./brewslm train start --base-model …`.

### API

```sh
curl -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/projects/1/models/compatible?limit=5"
curl -X POST http://localhost:8000/api/projects/1/models/validate \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model_id": 12}'
```

## Step 6 — Train

For a first project, let the Autopilot Planner pick the training preset. It chooses between safe-SFT, LoRA-fast, and a small full-fine-tune based on dataset size + base model. It needs training data first: on a project with no data it stops with "No training data yet" and links to the Data step.

### UI

**Settings** → **Advanced** → **Tools** → **Autopilot Planner** → describe the goal in plain English: *"Support FAQ tone, concise answers, no hallucinations beyond the dataset."* → **Preview Plan** (shows what it would change; nothing runs yet) → **Apply Plan**. **Run Directly** skips the preview.

To train by hand instead, open **Train**: confirm the **Base model** and **Task**, choose *One run* or *Three seeds*, then click **Train** and **Launch**. The readiness line tells you what to fix first.

### CLI

```sh
./brewslm train start --project 1 --autopilot --one-click \
  --intent "Support FAQ tone, concise answers, no hallucinations beyond the dataset."
```

### API

```sh
# Preview (always a dry run): returns a plan_token
curl -X POST http://localhost:8000/api/autopilot/repair-preview \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"project_id": 1, "intent": "Support FAQ tone, concise answers, no hallucinations beyond the dataset."}'

# Apply that plan
curl -X POST http://localhost:8000/api/autopilot/repair-apply \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"plan_token": "<plan_token from the preview>"}'
```

Watch the loss curve. Cost + provenance show in the Resolved Defaults panel — see [Measured vs estimated](../reliability/measured-vs-estimated.md).

## Step 7 — Evaluate

Once a real training run finishes, an automatic check evaluates it against its base model on the test split (watch the bell). A simulated run trains no weights, so it is not evaluated; see the [Quickstart](quickstart.md).

### UI

**Results** → **Verdict**. If the run has no result yet, click **Evaluate this run**. Each pass/fail rule (exact match, LLM-judge, safety) lands as a row with pass / fail and its metric. The **Failures** tab groups errors by reason code — click one to see exemplars.

### CLI

```sh
./brewslm eval run --project 1 --experiment-id 1
./brewslm eval clusters --project 1 --eval-result-id <id from the eval run>
```

### API

```sh
curl -X POST "http://localhost:8000/api/projects/1/evaluation/run-heldout?async_job=true" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"experiment_id": 1}'
```

If any pass/fail rule fails, the **Remediation** panel suggests concrete fixes (data, hyperparameters, prompt template). Apply, re-run training, re-evaluate.

## Step 8 — Export

When eval passes, export for the target.

### UI

**Ship** → **Package** → **New export** → target `vllm_server` → format `huggingface`. Click **Run export**. Artifact lands under `DATA_DIR/exports/`.

### CLI

```sh
./brewslm export --project 1 --experiment 1 --target vllm_server --format huggingface
```

### API

```sh
# Create the export, then run it with the id it returns
curl -X POST http://localhost:8000/api/projects/1/export/create \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"experiment_id": 1, "export_format": "huggingface"}'
curl -X POST http://localhost:8000/api/projects/1/export/<export id>/run \
  -H "Authorization: Bearer $TOKEN"
```

## Step 9 — Plan + smoke + promote deployment

Once you have an export, the [deployment loop](../deployment/plan.md) takes over. Plan → smoke → promote → telemetry. The full deployment workflow lives in the [Deployment](../deployment/plan.md) section.

## Definition of done (first iteration)

- [ ] One full project from ingest → export, all stages green.
- [ ] At least one failure cluster looked at, one remediation applied.
- [ ] One deployment promoted (against the simulator if no real serving env yet).
- [ ] You can re-run the whole thing from the project's training manifest — `./brewslm train rerun --project 1 --experiment-id 1`.

If you can check these four boxes, you've used every layer of BrewSLM.

## What to read next

- [Pipeline overview](../workflows/pipeline-overview.md) — every stage in more depth.
- [Newbie autopilot](../workflows/newbie-autopilot.md) — when autopilot helps and when to override.
- [Failure clusters](../observability/failure-clusters.md) — making sense of eval failures.
- [Measured vs estimated](../reliability/measured-vs-estimated.md) — reading provenance labels.
