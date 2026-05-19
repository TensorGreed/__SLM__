# Video 05 — Sentiment Classifier Sample · Full Pipeline · Recording Plan

Status: **partial** — every UI surface verified by selector pass on
2026-05-19.

## Goal

Walk the three-way sentiment-classifier sample end-to-end, with two
extra teaching beats specific to classification:

1. **Class balance**. Source CSV is exactly 10/10/10 across
   `positive/neutral/negative`. Gold is 70/65/65. Show why this
   matters.
2. **Mobile / CPU target**. Manifest's `target_profile = mobile_cpu`
   points at an ONNX-INT8-style export story — but no real ONNX
   artifact has been produced yet, so the export story is **partial**
   and must be marked as such.

## Audience

Beginner / early intermediate.

## Expected video length

8–10 minutes.

## Exact starting state

1. Backend + frontend running.
2. Logged in.
3. Seed via UI tile (`Demo · Sentiment classifier`) or curl
   `POST /api/demo-projects/sentiment-classifier`.
4. Browser at `/project/<id>/pipeline/data`.
5. Confirm 30 ingested documents on Data tab.

## Sample files involved

| Path | Role | Rows |
|---|---|---:|
| `backend/data/demo_samples/sentiment-classifier/manifest.json` | Task profile `classification`, target `mobile_cpu`, eval pack `evalpack.classification.default`. | n/a |
| `backend/data/demo_samples/sentiment-classifier/reviews.csv` | Source: `text`, `label`. | 30 |
| `backend/data/demo_samples/sentiment-classifier/gold.jsonl` | Gold. | 200 |
| `data/projects/<id>/prepared/train.jsonl` | Seeded 22 rows. | 22 |
| `data/projects/<id>/prepared/val.jsonl` | 4 rows. | 4 |
| `data/projects/<id>/prepared/test.jsonl` | 4 rows. | 4 |

## Label vocabulary

`positive`, `neutral`, `negative`.

CSV distribution: 10 / 10 / 10. Gold distribution: 70 / 65 / 65.

## UI route + selector sequence

| # | Route | Component | Focal selector | Status |
|---|---|---|---|---|
| 1 | `/project/<id>/pipeline/data` | `IngestionPanel` | `[data-testid="expand-doc-91"]` (concrete observed) | verified |
| 2 | `/project/<id>/pipeline/cleaning` | `CleaningPanel` | "Cleaning Configuration" | verified surface only |
| 3 | `/project/<id>/pipeline/goldset` | `GoldSetPanel` | "Entries 200"; label distribution callout | verified |
| 4 | `/project/<id>/pipeline/dataprep` | `DatasetPrepPanel` | Schema Profile shows `labels: positive, neutral, negative` | verified |
| 5 | `/project/<id>/pipeline/tokenization` | `TokenizationPanel` | Surface tour | verified surface only |
| 6 | `/project/<id>/training-config` | `ProjectTrainingConfigPage` | `target_profile=mobile_cpu` should influence recipe defaults | verified |
| 7 | `/project/<id>/pipeline/eval` | `EvalPanel` | Empty state; `evalpack.classification.default` callout | verified surface |
| 8 | `/project/<id>/pipeline/compression` | `CompressionPanel` | ONNX/INT8 callout; **do not run** | verified surface |
| 9 | `/project/<id>/pipeline/export` | `ExportPanel` | Show that `onnx` is in the format enum | verified surface |

## API calls observed

- `POST /api/demo-projects/sentiment-classifier`
- `GET /api/projects/<id>/ingestion/documents` (returns 30)
- `GET /api/projects/<id>/ingestion/documents/91/sample` (shape `text + label`)
- `GET /api/projects/<id>/gold/entries?dataset_type=gold_dev`
- `GET /api/projects/<id>/prepared-manifest` — confirms labels +
  adapter `classification-label` + task `classification`
- `GET /api/projects/<id>/evaluation/packs` — should list classification pack
- `GET /api/projects/<id>/export/deployment-targets?export_format=gguf`

## Pipeline stages to show

| Stage | Show | Skip / mark |
|---|---|---|
| Ingestion | Yes — 30 docs; expand doc 91; show `{text, label}` shape | n/a |
| Cleaning | Light surface tour | not run |
| Gold Set | Yes — 200 entries; call out 70/65/65 split | n/a |
| Synthetic | Skip OR very light tour (Q12 — classification-specific synthetic path unverified). Mention but do not click Generate. | runtime + unverified path |
| Dataset Prep | Yes — Schema Profile shows the three labels | n/a |
| Tokenization | Light surface tour | runtime |
| Training | Light surface tour + click into Training Config | Video 09 |
| Eval | Empty state; **say**: "the classification eval pack would emit accuracy + macro-F1 once an experiment exists" | Video 10 |
| Compression / Export | Light tour. Show `onnx` is in the export enum. **Do not run.** Explicitly say "ONNX-INT8 would be the natural target for mobile_cpu, but no successful run has been done in this pass" — open Q23/Q24. | partial |

## Step-by-step user journey

| Time | Action | What viewer sees |
|---|---|---|
| 00:00 – 00:30 | Recap + click Sentiment classifier tile | Project opens. |
| 00:30 – 02:00 | Data tab. 30 docs. Expand doc 91. Read `text` + `label`. | Raw row. |
| 02:00 – 02:45 | Gold Set. 200 entries. Distribution callout. | Gold table. |
| 02:45 – 04:00 | Dataset Prep → Schema Profile. Labels visible. | Schema panel. |
| 04:00 – 05:30 | Tokenization light tour. Mention max sequence length matters for mobile. | Token panel. |
| 05:30 – 07:00 | Training Config. Show that `target_profile=mobile_cpu` influences recipe defaults. | Training config. |
| 07:00 – 08:00 | Eval empty state. Mention classification pack and accuracy/macro-F1. | Eval empty. |
| 08:00 – 09:00 | Light tour of Compression + Export. Point at `onnx` in the format dropdown. Mark ONNX-INT8 story as future. | Export panel. |
| 09:00 – 09:30 | Wrap. | Title card. |

## Narration checkpoints

- **00:30** — "Sentiment classifier ships 30 source rows, balanced
  10 / 10 / 10 across positive, neutral, negative. Three-way
  classification is the simplest pipeline shape: each row is text and
  a single label."
- **02:00** — "Gold has 200 hand-labelled rows: 70 positive, 65
  neutral, 65 negative. Slightly skewed positive — typical of
  real-world reviews."
- **04:00** — "Dataset Prep applies the `classification-label`
  adapter, which canonicalizes every prepared row with `text` and
  `label` columns."
- **05:30** — "Training Config picks up the `mobile_cpu` target
  profile from the manifest. That hints at a smaller batch, shorter
  sequences, and an ONNX-INT8 export — but the export story for
  this sample is still partial."
- **08:00** — "The eval pack is `evalpack.classification.default`,
  which emits accuracy and macro-F1. Once we have an experiment, the
  per-class precision/recall lands here."

## Screenshots

| # | Filename | Action |
|---|---|---|
| 1 | `v05-data-tab.png` | Data tab loaded |
| 2 | `v05-expanded-row.png` | Doc 91 expanded |
| 3 | `v05-goldset.png` | Gold Set with 200 entries |
| 4 | `v05-schema-profile.png` | Dataset Prep showing labels |
| 5 | `v05-training-config-mobile.png` | Training Config with mobile_cpu callout |
| 6 | `v05-export-onnx.png` | Export tab showing ONNX in format dropdown |

(Selector-pass screenshots `selector-pass-sentiment-01..13.png`
reusable.)

## What to mark

| Item | Marker |
|---|---|
| Three-way classification | **verified** |
| Gold label distribution (70/65/65) | **verified** — measured from file |
| Manifest target `mobile_cpu` | **verified** |
| ONNX-INT8 export actually working | **partial** — never run on this sample |
| Synthetic data for classification | **unknown** — Q12 open |

## Recording feasibility

**Feasible now** for stages 1–7. ONNX/INT8 export demo (stage 8+)
should be marked clearly as "natural target, not yet validated."

## Open questions specific to this video

- (Q12) Does the synthetic generator have a classification-specific
  path? Selector pass observed no class-aware synthetic UI on this
  sample; if so, mark as "not yet supported for sentiment."
- (Q23) Which compression path is practical locally for the
  mobile_cpu target — full ONNX-INT8, benchmark-only, or stub? This
  video should park the question and let Video 11 answer.
