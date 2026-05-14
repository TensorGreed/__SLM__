---
sidebar_position: 2
title: Data ingestion + cleaning + prep
---

# Data ingestion + cleaning + prep

Stages 1, 2, 4, and 5 of the [pipeline](pipeline-overview.md) all live under the same Pipeline rail tabs. This page narrates the upstream flow from raw files to prepared, deduped, split records.

## Step 1 — Ingestion

### What it does

Pulls raw data into the project's `RawDocument` table + persists files under `DATA_DIR/projects/{id}/raw/`. Supports CSV, TSV, JSON, JSONL, Parquet, Hugging Face datasets, URL pulls, and document corpora (PDF/DOCX/MD).

### UI

Pipeline → **Data** → **Add source**. Pick the source type. For CSV/JSONL, drag the file; for HuggingFace, paste `org/dataset`; for URLs, enter a fetchable URL. The Dataset Structure Explorer profiles the file inline.

### CLI

```sh
# File upload
brewslm dataset upload --project 1 \
  --source-type csv --source-ref ./tickets.csv \
  --name tickets_v1

# HuggingFace pull
brewslm ingest --project 1 \
  --source-type huggingface \
  --source-ref "Anthropic/hh-rlhf" \
  --name hh_rlhf

# URL pull (queued; status via doctor)
brewslm ingest --project 1 \
  --source-type url \
  --source-ref "https://example.com/data.jsonl"
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/datasets/upload \
  -F file=@tickets.csv \
  -F source_type=csv \
  -F name=tickets_v1

curl -X POST http://localhost:8000/api/projects/1/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type":"huggingface","source_ref":"Anthropic/hh-rlhf","name":"hh_rlhf"}'
```

### Reason codes you might hit

| Code | Means |
|---|---|
| `ingest_unsupported_format` | File extension isn't in the allowed list. Add it via the connector or convert first. |
| `ingest_io_error` | Couldn't read the file (disk full, network blip on URL pull). |
| `ingest_validation_failed` | Content didn't match schema (e.g., CSV has fewer columns than declared). |

### Generic dataset-import pipeline (sources × mappers)

The ingest helpers above know about *file formats*. For datasets that already carry task structure (BIO-tagged tokens, classification labels, preference pairs, chat threads, …) BrewSLM also ships a generic three-layer pipeline that introspects the shape and proposes a mapping straight into the project's synthetic dataset — no per-domain converter needed.

```sh
# Sniff a JSONL file and print the proposed mapping (no writes).
python -m app.cli.dataset_import introspect \
  --locator jsonl:./train.json

# Run with --auto: the introspector picks the mapper + field_map
# when confidence ≥ 0.8. Falls back to --force below the threshold.
python -m app.cli.dataset_import run \
  --locator jsonl:./train.json \
  --project 1 --auto

# Override one field without losing the rest of the auto suggestion.
python -m app.cli.dataset_import run \
  --locator jsonl:./train.json --project 1 --auto \
  --map-json '{"entity_type_map": {"NAME_STUDENT": "person_name"}}'
```

See [Schema introspection](../reference/glossary.md#schema-introspection) and the [PII demo `--auto` walkthrough](../demos/pii-detector.md#skip-the-converter-with---auto).

## Step 2 — Clean

### What it does

Normalises text, deduplicates rows, runs a PII scan, applies your domain pack's data-quality overlay. Emits a new `DatasetVersion` rather than mutating the ingested rows.

### UI

Pipeline → **Cleaning** → **Run cleaning**. The result card shows rows before / after, dedup ratio, and PII findings.

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

### Reason codes you might hit

| Code | Means |
|---|---|
| `cleaning_outlier_threshold_exceeded` | Outlier removal would drop more rows than the threshold allows. Lower the threshold in the domain pack overlay or bring more data. |
| `cleaning_pii_block` | PII scan blocked the dataset. Review the findings; either redact at source or relax the block via the domain pack. |

## Optional — Synthetic augmentation

If your dataset is small (< a few hundred rows), the synthetic stage can use an LLM judge to fabricate plausible additional Q&A pairs. Only useful for question / extraction tasks; bad for chat-style data.

### UI

Pipeline → **Synthetic** → **Generate N rows**. Pick the seed dataset, target count, and LLM (defaults to an inexpensive small model). Review samples before merging.

### CLI

```sh
brewslm synthetic generate --project 1 \
  --seed-dataset tickets_v1 \
  --count 200 \
  --strategy q_and_a
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{"seed_dataset":"tickets_v1","count":200,"strategy":"q_and_a"}'
```

## Step 3 — Dataset prep

### What it does

Joins cleaned + synthetic rows (if any), runs the chosen adapter, splits train/val/test, and writes the prepared records to `DATA_DIR/projects/{id}/prepared/`.

### UI

Pipeline → **Data prep** → **Run prep**. Settings:

- **Adapter** — defaults to the project default (`auto` if you haven't pinned one). See [Adapter examples](../getting-started/adapter-studio-examples.md).
- **Train / val / test split** — default 80/10/10.
- **Seed** — for reproducibility.
- **Resolved Defaults panel** below shows every applied setting + its source (`config` / `recipe` / `domain_pack` / `default`).

### CLI

```sh
brewslm dataset prepare --project 1 \
  --dataset tickets_v1 \
  --adapter-id auto \
  --train 0.8 --val 0.1 --test 0.1 \
  --seed 42
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/datasets/prepare \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name":"tickets_v1",
    "adapter_id":"auto",
    "split":{"train":0.8,"val":0.1,"test":0.1},
    "seed":42
  }'
```

### Reason codes you might hit

| Code | Means |
|---|---|
| `adapter_schema_mismatch` | The picked adapter couldn't match the data shape. Run `adapter infer` to find a better fit. |
| `adapter_field_resolution_failed` | A required field is missing. Run `dataset profile` to see what's actually there. |

## What's in the prepared dir

```
prepared/
├── train.jsonl          # canonical records, one per line
├── val.jsonl
├── test.jsonl
├── manifest.json        # adapter id, split seed, source dataset version
└── stats.json           # row counts, token estimates, length quantiles
```

`manifest.json` is what `train rerun` reads to reproduce the prep step exactly.

## Tokenization

Stage 7. Runs the base model's tokenizer over the prepared records and writes the result back under `prepared/`. Triggered automatically by the training stage, so most users never touch it directly.

If you do need to inspect:

```sh
brewslm dataset tokenize --project 1 --dataset tickets_v1 --base-model 12
```

## Quality checklist

Before kicking off training, verify on the Dataset Prep tab:

1. No empty prompts/targets (the prep stage rejects these; the count appears in `stats.json`).
2. Train / val splits are non-overlapping (the seeded shuffle guarantees this).
3. Class balance — if you're doing classification, the per-label histogram in `stats.json` should not be hugely skewed.
4. Sensitive content handled per policy — the PII block from cleaning is visible in the audit log.

## Common upstream mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Training on noisy scraped text | Eval bounces around, lots of `adapter_schema_mismatch`. | Stricter cleaning + dedup. |
| Mixing two task shapes in one dataset | Adapter `auto` chooses something weird; some rows fail validation. | Split into two projects or pick a multi-task adapter. |
| No validation split | Training "passes" but real eval is bad. | Use the 80/10/10 default. |
| Ignoring class imbalance for classification | Model collapses to majority class. | Resample or class-weight in the recipe. |

Fixing these wins more than hyperparameter tuning.

## Next

- [Newbie autopilot](newbie-autopilot.md) — let the planner pick every step above.
- [Training](training.md) — what to do with prepared records.
- [Adapter examples](../getting-started/adapter-studio-examples.md) — common adapter mappings.
