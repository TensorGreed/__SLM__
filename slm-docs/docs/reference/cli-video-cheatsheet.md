---
sidebar_position: 2
title: CLI ↔ video cheatsheet
---

# `brewslm` ↔ video walkthrough cheatsheet

Every flow demonstrated in the 11-part YouTube series has a CLI
equivalent. Engineers will script before they click — this page is
the parallel surface to the videos.

## Configure once

```bash
export BREWSLM_API_BASE='http://127.0.0.1:8000/api'   # default
export BREWSLM_TOKEN="$(brewslm auth login --username admin --password $PW)"
# Or persist (writes ~/.brewslm/token with 0600 on POSIX):
brewslm auth login --username admin --save
```

When `AUTH_ENABLED=false` (local dev), the token isn't required — every
command runs as the implicit admin principal.

---

## Video-by-video

### Video 1 — SLM 101 (concept) — *no command*

### Video 2 — BrewSLM Quickstart

Equivalent of clicking the **Demo · Support FAQ** tile:

```bash
brewslm project list-demos                     # show what's bundled
brewslm project create-demo --slug support-faq # seed a fresh demo project
```

Or the project-template gallery (8 cloneable starting kits):

```bash
brewslm template list
brewslm template instantiate ticket-router --name "Acme Tickets"
```

### Video 3 — Support FAQ Pipeline (full walkthrough)

After seeding the demo project (Video 2), drive the pipeline:

```bash
brewslm pipeline run --project $PID --to-stage training
```

Step-by-step is the same command with `--stage cleaning`,
`--stage gold_set`, `--stage dataset_prep`, `--stage tokenization`.

### Video 4 — PII Detector Pipeline

Same pipeline command on the PII demo slug:

```bash
brewslm project create-demo --slug pii-detector
brewslm pipeline run --project $PID --to-stage training
```

### Video 5 — Sentiment Classifier (mobile CPU)

```bash
brewslm project create-demo --slug sentiment-classifier
brewslm pipeline run --project $PID --to-stage training
```

### Video 6 — BYO Custom Samples

```bash
brewslm project create \
  --name "Coffee FAQ" \
  --description "Train an SLM on my coffee-shop FAQ CSV"

brewslm dataset import --project $PID \
  --file ./coffee-faq.csv \
  --dataset-type raw
```

### Video 7 — Real Training Run

```bash
brewslm train start --project $PID \
  --intent "Train an instruction-following assistant on the support FAQ data" \
  --target-device gpu_consumer
```

### Video 8 — Evaluation against a 200-row gold set

```bash
brewslm eval run --project $PID --experiment-id $EXP_ID

# Side-by-side compare:
brewslm eval compare --project $PID --baseline $BASE_EXP --candidate $TRAINED_EXP
```

### Video 9 — Compression + Export (GGUF)

```bash
# Score candidate quantization plans:
brewslm optimize --project $PID --experiment-id $EXP_ID

# Run the chosen export:
brewslm export create --project $PID --experiment-id $EXP_ID --target ollama
brewslm export run    --project $PID --export-id $EXPORT_ID
```

### Video 10 — Serve with Ollama (close the loop)

```bash
brewslm serve plan  --project $PID --export-id $EXPORT_ID
brewslm serve start --project $PID --export-id $EXPORT_ID --template-id ollama_local
brewslm serve get   --project $PID --run-id $RUN_ID --logs-tail 200
brewslm serve stop  --project $PID --run-id $RUN_ID
```

### Video 11 — Architecture (concept) — *no command*

---

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `BREWSLM_API_BASE` | `http://127.0.0.1:8000/api` | Base URL the CLI talks to. |
| `BREWSLM_TOKEN` | `""` | JWT used as `Authorization: Bearer` + `X-API-Key`. |
| `BREWSLM_TOKEN_FILE` | `~/.brewslm/token` | Where `auth login --save` writes the token. |
| `BREWSLM_PASSWORD` | `""` | Fallback for `auth login` when `--password` is omitted. |
| `BREWSLM_TIMEOUT_SECONDS` | `60` | HTTP timeout per request. |

All flags can also be passed inline:
`brewslm --api-base https://prod.example.com/api --token $TOK ...`.

---

## JSON output

Every subcommand has a `--json` flag that emits the raw API response.
Pipe into `jq` for scripted flows:

```bash
PID=$(brewslm project create --name "Auto" --json | jq -r '.id')
EXP=$(brewslm train start --project $PID --json | jq -r '.experiment_id')
brewslm eval run --project $PID --experiment-id $EXP --json | jq '.metrics'
```

See [CLI reference](cli.md) for full command tables, or
[full auto-generated reference](cli-full.md) for every flag of every
subcommand.
