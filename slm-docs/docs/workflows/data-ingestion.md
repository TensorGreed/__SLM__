---
sidebar_position: 2
title: Data ingestion + cleaning + prep
---

# Data ingestion + cleaning + prep

Stages 1, 2, 4, and 5 of the [pipeline](pipeline-overview.md) all live under the same Pipeline rail tabs. This page narrates the upstream flow from raw files to prepared, deduped, split records.

## Step 1 — Ingestion

:::important Two import paths — pick the right one
There are **two different things** on the Data tab, and they land your rows in **different places**:

- **Upload / Add source** (this section) → your rows become **raw training data** (`RawDocument`, `dataset_type=RAW`). **This is the path for your own labelled dataset** (e.g. a `{text, label}` JSONL for a classifier). It advances the pipeline so the Cleaning / Gold Set / Dataset Prep / Training tabs unlock.
- **Import dataset (auto-mapping)** wizard (covered [below](#generic-dataset-import-pipeline-sources--mappers)) → mapped rows go to the project's **synthetic** dataset (pending review). It's for **augmenting** an existing dataset with an external/HF source, **not** for your primary training data.

If you run your own data through the *Import dataset* wizard, it lands in *synthetic* (not RAW), no raw source exists, and the downstream tabs stay disabled. Use **Upload** for your training data. (The guided "Start" button auto-opens the *Import* wizard — close it and use **Upload** if you're bringing your own labelled file.)
:::

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

:::note Lands in *synthetic*, not RAW
This wizard writes accepted rows to the project's **synthetic** dataset (pending review) — it's for **augmenting** with an external/HF dataset, not for your primary training data. For your own labelled file, use **Upload / Add source** above. See the callout at the top of this page.
:::

#### Via the UI

Pipeline → **Data** → **Import dataset (auto-mapping)** opens a 3-step wizard:

1. **Source** — pick `jsonl` / `csv` / `hf` / `kaggle` from a dropdown and enter the locator. Source-specific helper text shows the format; gated-dataset auth gets a banner reminding you to set env vars or save secrets under Project → Secrets first.
2. **Map** — column signatures table (detected type + confidence per column), ranked-hypothesis dropdown pre-selected to the top proposal, free-form rationale, and a JSON field-map editor you can override before previewing. A red banner blocks low-confidence proposals until you tick "I've reviewed the proposal — proceed anyway."
3. **Preview & Confirm** — accepted-row sample + rejected breakdown grouped by reason; tick reasons to bulk-drop them before commit (counts are preserved in the result for audit). Final *Import* button writes the rows to the project's synthetic dataset and surfaces the `written_path`.

A "Save this mapping" card on the Preview step persists the current locator + mapper + field_map + drop_reasons under a name you choose. Saved mappings appear at the top of the Data tab in a **Saved mappings** panel; each row gets a **Re-run** button that re-imports against the (potentially refreshed) source without going back through the wizard, plus a **Delete** action. Re-runs bump `last_run_at` and `last_run_accepted` so the panel shows the latest yield at a glance.

Every dataset-import run — interactive or re-run from a saved mapping — emits a [run event](../reference/glossary.md#run-event) with `stage=ingestion` and `reason_code=dataset_import_run` (`dataset_import_failed` on errors), carrying the source, locator, mapper, accepted/rejected row counts, `written_path`, and `config_id` if it came from a saved mapping. The Observability page surfaces these automatically — the audit log is just RunEvents you can already filter.

#### Extensibility (Phase H)

- **[Mapper plugins](../reference/glossary.md#mapper-plugin)** — drop a Python module under `DATASET_MAPPER_PLUGIN_MODULES` to register custom `TargetMapper` classes. Two registration shapes (`register_dataset_mappers(register)` hook or top-level `DATASET_MAPPERS` dict). Loaded at boot alongside the data-adapter / training-runtime plugins. A broken plugin module never blocks the rest of the list — failures surface in the loader's diagnostic dict.
- **[LLM-assisted mapping](../reference/glossary.md#llm-assisted-mapping)** — optional fallback when the deterministic column sniffer can't form a high-confidence hypothesis. Set `DATASET_IMPORT_LLM_ASSIST_ENABLED=true` plus a `TEACHER_MODEL_API_URL`, then pass `--llm-assist` on the CLI (or `llm_assist: true` on the `/introspect` API body). The teacher's proposal joins the ranked hypothesis list with a `proposal-source: llm-assist` warning tag; same confidence gate applies, hallucinated mapper ids are rejected at the registry boundary.

```sh
# Opt in per call — falls through silently when no teacher is configured.
python -m app.cli.dataset_import introspect \
  --locator jsonl:./unknown-shape.jsonl \
  --llm-assist
```

Source locators today:

- `jsonl:/path/to/file.jsonl`
- `csv:/path/to/file.csv`
- `hf:<dataset_id>[:<split>[:<revision>]]` — fetches directly from the HuggingFace Hub via the `datasets` library (set `HF_TOKEN` for gated datasets).
- `kaggle:competition:<slug>` / `kaggle:dataset:<owner/slug>` — downloads + extracts via the Kaggle API (set `KAGGLE_USERNAME` + `KAGGLE_KEY` or drop a `~/.kaggle/kaggle.json`). Append `?file=<path>` to disambiguate multi-file archives.

#### Via the CLI

```sh
# Sniff a JSONL file and print the proposed mapping (no writes).
python -m app.cli.dataset_import introspect \
  --locator jsonl:./train.json

# Same, but from HuggingFace — streams the first 20 rows for sniffing.
python -m app.cli.dataset_import introspect \
  --locator hf:imdb:train

# Or from Kaggle — downloads + extracts on first run, cached for re-runs.
python -m app.cli.dataset_import introspect \
  --locator 'kaggle:competition:pii-detection-removal-from-educational-data'

# Run with --auto: the introspector picks the mapper + field_map
# when confidence ≥ 0.8. Falls back to --force below the threshold.
python -m app.cli.dataset_import run \
  --locator hf:Anthropic/hh-rlhf:train \
  --project 1 --auto --limit 5000

# Override one field without losing the rest of the auto suggestion.
python -m app.cli.dataset_import run \
  --locator jsonl:./train.json --project 1 --auto \
  --map-json '{"entity_type_map": {"NAME_STUDENT": "person_name"}}'
```

The introspector auto-detects every mapper in the catalog except `kv_to_structured` (which needs an explicit `fields` config). Current built-ins:

- `bio_to_spans` — BIO-tagged tokens → entity spans (NER / PII).
- `label_to_classification` — `{text, label}` for sentiment / intent.
- `text_only` — single text column for plain LM training.
- `qa_pair_passthrough` — `{question, answer}` for short-answer QA.
- `chat_messages_passthrough` — `messages` list for chat SFT.
- `preference_pair` — `{prompt, chosen, rejected}` for DPO / ORPO.
- `rag_passthrough` — `{question, context, answer}` for grounded QA.
- `kv_to_structured` — flat key-value extractions for invoices / forms (manual `--map-json '{"fields":[...]}'`).

See [Schema introspection](../reference/glossary.md#schema-introspection), [Target mapper](../reference/glossary.md#target-mapper-dataset-import), and the [PII demo `--auto` walkthrough](../demos/pii-detector.md#skip-the-converter-with---auto).

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

The split form **hydrates from the active prepared version** on open: if the project has already been prepared, the stratify/disjoint field, ratios, seed, and chat template default to the config that produced the current splits (read from `prepared/manifest.json`) rather than empty fields — so a re-run reproduces the active split unless you change something. A "♻️ Reusing the active prepared version's split config" hint names which fields were filled. Edit any field to override.

:::note Canonical split-config key
`prepared/manifest.json` carries the split config under **`resolved_split_config`** (`{train_ratio, val_ratio, test_ratio, seed, chat_template}`) — the same shape the split API response returns. This is the canonical key readers should use (the split-form hydration and the "Re-split with dedup" inheritance both read it). The older top-level `seed`/`chat_template` and `ratios: {train, val, test}` fields are still written for back-compat; manifests prepared before `resolved_split_config` was persisted fall back to those and pick up the canonical key on their next Prepare. The manifest also reflects the **activated** version, not just the latest run — activating an older version via the Dataset Versions panel restores its `manifest.json` over the active file.
:::

### Data Health Report (D1+D2 of the data-quality arc)

The Data Prep tab opens with an aggregated **Data Health Report** at the top — a single panel that pulls every data-quality signal scattered across the platform (ingestion, cleaning, shape vs recipe, classification balance) into one traffic-light scorecard. Backed by `GET /api/projects/{id}/data-health`.

Each signal row carries:

- **Severity badge** — `ok` / `warn` / `block`, matching the Coach Mode + trainability-forecast palette so the same red means the same thing across panels.
- **Plain-English summary** — the actual problem, in words a non-technical user can act on. Sits above the technical headline (which is still rendered for users who want the numbers).
- **"Why this matters" expander** — closed by default, one-click expand. The point is to teach the consequence at training time, not to wall users with text.
- **Suggested action chip** — informational, navigates to the relevant tab.
- **Preview button** (D3/D4, where applicable) — green button next to the action chip when the platform can resolve the signal. Opens a per-item diff modal; the destructive change only lands after the user clicks Apply in the modal. See "Auto-fixes (preview-then-apply)" below.

The top of the panel is an overall verdict: `ok` ("all clear"), `warn` ("warnings to address"), or `block` ("training won't produce reliable results until these are fixed"), with severity counts. The overall is the worst of all signals — any block bubbles up.

Groups, in order:

| Group | What it covers |
|---|---|
| Ingestion | Document count, parse-failure rate (warn at 10%, block at 25%). |
| Cleaning | PII findings + redaction status, low-quality fraction (block at 30%), duplicate-document share via `text_hash` (block at 30%). |
| Shape vs recipe | Recipe selected? Train/val/test prepared? Corpus size above the recipe minimum? |
| Class balance | Classification-only: delegated to the trainability forecast's existing signals (`class_imbalance`, `per_class_minimum_unmet`, `label_vocab_fragmented`, `single_class_dominance`) so the report and Coach Mode share one source of truth for the thresholds. |

Empty groups (e.g. Balance for non-classification recipes) are silently skipped.

#### Auto-fixes (preview-then-apply) — D3 + D4

Every signal that carries an `autofix_kind` hint surfaces a green **Preview: …** button on the row. Clicking opens a modal that shows the **per-item diff** the fix would produce (filenames being dropped, keep-vs-drop pairs for dedup, label merge map for canonicalisation, PII finding counts) — no destructive change lands until the user clicks Apply inside that modal.

The contract is:

1. UI POSTs `/data-health/autofix/preview` with `{fix_kind}`. The server returns `{would_apply_count, summary, items, safe_to_apply, details}` without mutating anything.
2. The modal renders the `items` list. If `safe_to_apply: false`, the Apply button is disabled and a "Safety guard" callout explains why.
3. On Apply, UI POSTs `/data-health/autofix`. The endpoint runs the matching transform and returns `{applied_count, summary, details}` — the panel surfaces that as a toast and refreshes the report.

| Signal | Fix kind | What it does |
|---|---|---|
| `ingestion.parse_failure_rate` | `drop_failed_docs` | Deletes every `RawDocument` with `status=ERROR` plus its on-disk artefacts (raw file + `.extracted.txt` / `.cleaned.txt` / `.chunks.jsonl` sidecars). Failed parses have no extracted text — they were already useless. Preview lists each filename + the parse error string so the user can see *which* files would be dropped. |
| `cleaning.duplicate_chunks` | `dedupe_duplicate_docs` | Groups `ACCEPTED` docs by `metadata_.text_hash`; for each group of >1, keeps the lowest-id occurrence and deletes the rest. Pure dedup, no semantic change. Preview shows each duplicate set as a "keep ✓ / drop −" group so the user picks up which copy survives. |
| `cleaning.pii_unredacted` | `redact_pii` | Re-runs `clean_document(..., redact=True)` on every doc that has PII findings but `redact_pii` flag unset. Cleaning is itself idempotent — this just re-renders the cleaned text with PII replaced by `[REDACTED]`. Skips docs that aren't yet cleaned (would require running the full pipeline; the user should click Clean first). Preview lists each affected doc + its PII finding count. |
| `balance.label_vocab_fragmented` | `canonicalise_labels` (D4) | Classification-only. Groups labels by their normalised form (lowercase + collapsed whitespace), picks the most-common variant as canonical (ties broken alphabetically), and rewrites every gold-set JSONL row whose label sits in a non-canonical bucket. Idempotent (re-running on already-canonicalised gold is a no-op). Preview shows the merge map: `"Positive" (3) + "POSITIVE" (1) → "positive" (15 canonical)`. Refuses on non-classification recipes (`safe_to_apply: false`). |

**Recipe-aware PII guard**: for `structured_extraction` recipes (PII detection, NER, entity extraction) the source-document PII IS the training signal — auto-redacting it would destroy what the model needs to learn. For these projects, the data-health signal flips to `ok` severity with explanatory copy, the **Preview** button is hidden in the panel, the preview endpoint returns `safe_to_apply: false` with `details.blocked_reason = "span_extraction_needs_pii"` so the modal can't apply, and a direct call to `POST /data-health/autofix` with `fix_kind=redact_pii` returns 400. If you need redaction for a separate non-training use, do it manually on a copy of the cleaned outputs.

**Endpoints**:

- `POST /api/projects/{id}/data-health/autofix/preview` — `{fix_kind}` → `{would_apply_count, summary, items, safe_to_apply, details}`. Read-only.
- `POST /api/projects/{id}/data-health/autofix` — `{fix_kind}` → `{applied_count, summary, details}`. Mutating; UI is expected to call `/preview` first.
- `GET /api/projects/{id}/data-health/autofix/supported` — `{fix_kinds: [...]}`. Lists every kind the server understands.

Further fixes (drop-low-quality docs, truncation, row drops) follow the same preview-then-apply pattern and ship in D5+.

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

### Token-length distribution across splits (V3 ML-native viz)

The Tokenization tab lands on the **TokenLengthDistributionPanel** before the single-split deep-dive: a grouped histogram with one bar per `(bucket, split)` so train / validation / test render on the same axis. Backed by `POST /api/projects/{id}/tokenization/analyze-splits`, which orchestrates the existing single-split analyze across all three splits in one call.

Reading the overlay:

- **Grouped histogram** — the bucket axis is fixed (`0-256` → `256-512` → `512-1024` → `1024-2048` → `2048+`). Three bars per group, one per split, coloured train (blue) / validation (green) / test (orange).
- **Per-split percentile table** — `samples`, `p50`, `p95`, `p99`, `max`, `truncated`. The `truncated` column flags rows where the prepared sample exceeded the configured `max_seq_length` — those rows get cut at training time. Bigger projects often discover that the long tail of test exceeds train's coverage; this is where it surfaces.
- **Distribution-shift note** — when test's p95 is more than **+30%** above train's p95, a one-line honest beat fires: "test p95 = X tokens vs train p95 = Y; model trained with `max_seq_length=Z` will silently truncate longer test rows at eval". Below +30% the note stays silent — that gap is within sampling variance and warning on it would cry wolf.
- **Missing-splits chip** — when dataset prep hasn't materialised all three splits yet (train is first), the panel renders what it has and names the rest in a chip. The grouped histogram fills in as the missing splits land.

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
