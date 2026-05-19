# Video 03 — Support FAQ Sample · Full Pipeline · Recording Plan

Status: **partial** — every UI surface verified by selector pass on
2026-05-19. Real training run is out of scope for this video unless
Q16/Q17 (runtime decision) resolves first.

## Goal

Walk the support-faq sample from seed to evaluation surface,
without running a real training job. Viewer leaves understanding the
shape of the data and the seven pipeline tabs that actually do work
on a seeded demo.

## Audience

Beginner / early intermediate.

## Expected video length

9–12 minutes.

## Exact starting state

Run the prerequisites from
`docs-demo/videos/02-brewslm-quickstart/recording-plan.md`, then:

1. Log in.
2. From `/`, click the **Demo · Support FAQ** tile (or hit
   `POST /api/demo-projects/support-faq` via curl if not recording
   the seed step itself).
3. Confirm browser at `/project/<id>/pipeline/data`.
4. Confirm `Entries 200` visible on Gold Set tab and 20 ingested
   documents on Data tab.

## Sample files involved

| Path | Role | Rows |
|---|---|---:|
| `backend/data/demo_samples/support-faq/manifest.json` | Sample manifest. Task profile `instruction_sft`. Target `vllm_server`. | n/a |
| `backend/data/demo_samples/support-faq/tickets.csv` | Source. Columns `question`, `answer`. | 20 |
| `backend/data/demo_samples/support-faq/gold.jsonl` | Gold eval set. | 200 |
| `data/projects/<id>/prepared/train.jsonl` (seeded) | 16 rows after 70/15/15 split | 16 |
| `data/projects/<id>/prepared/val.jsonl` (seeded) | 2 rows | 2 |
| `data/projects/<id>/prepared/test.jsonl` (seeded) | 2 rows | 2 |

## UI route + selector sequence

| # | Route | Component | Selector for the focal element | Status |
|---|---|---|---|---|
| 1 | `/project/<id>/pipeline/data` | `IngestionPanel` | `[data-testid^="expand-doc-"]` (first row id varies; see selector pass) | verified |
| 2 | `/project/<id>/pipeline/cleaning` | `CleaningPanel` | `button.tab[title="Cleaning"]` | verified |
| 3 | `/project/<id>/pipeline/goldset` | `GoldSetPanel` | `button.tab[title="Gold Set"]`; visible `Entries 200` | verified |
| 4 | `/project/<id>/pipeline/synthetic` | `SyntheticPanel` | `button.tab[title="Synthetic"]` | verified — but generation requires runtime |
| 5 | `/project/<id>/pipeline/dataprep` | `DatasetPrepPanel` | `button.tab[title="Dataset Prep"]`; visible "Dataset Preview" / "Schema Profile" | verified |
| 6 | `/project/<id>/pipeline/tokenization` | `TokenizationPanel` | `button.tab[title="Tokenization"]` | verified surface only |
| 7 | `/project/<id>/pipeline/training` | `TrainingPanel` | `button.tab[title="Training"]`; "No experiments yet" empty state | verified surface only |
| 8 | `/project/<id>/training-config` | `ProjectTrainingConfigPage` | `Essentials / Advanced` toggle (page header) | verified |
| 9 | `/project/<id>/pipeline/eval` | `EvalPanel` | `button.tab[title="Evaluation"]`; "No experiments to evaluate" empty state | verified surface only |

## API calls that fire during the walkthrough

Observed during the support-faq selector pass on 2026-05-19:

- `GET /api/demo-projects`
- `POST /api/demo-projects/support-faq` (only if recording the seed step)
- `GET /api/projects/<id>`
- `GET /api/projects/<id>/pipeline/status`
- `GET /api/projects/<id>/ingestion/documents`
- `GET /api/projects/<id>/ingestion/eda`
- `GET /api/projects/<id>/ingestion/documents/20/sample`
- `GET /api/projects/<id>/gold/entries?dataset_type=gold_dev`
- `GET /api/projects/<id>/prepared-manifest` — **highest-value
  request to call out in narration**; it returns the prepared split
  counts (16/2/2), the adapter id (`qa-pair`), the task profile
  (`instruction_sft`), and the field mapping (`question` →
  `answer`).
- `POST /api/projects/<id>/dataset/split/effective-config`
- `GET /api/projects/<id>/training/runtimes`
- `GET /api/projects/<id>/training/experiments`
- `GET /api/projects/<id>/evaluation/packs`

## Pipeline stages to show (and exactly how)

| Stage | Show | Skip / mark as | Reason |
|---|---|---|---|
| Ingestion | Yes — Data tab with 20 raw rows, expand one row to show `{question, answer}` | n/a | Verified surface. |
| Cleaning | Yes — open Cleaning tab; **explain** chunk-size + PII redaction options but do **not click Start Cleaning** | mark as "available but not run for this seeded sample" | Cleaning would re-chunk text the seeder already prepared; not necessary. |
| Gold Set | Yes — Gold Set tab showing `Entries 200`; pause for viewer to read | n/a | Verified surface. Highlight that gold is locked. |
| Synthetic | Yes — open Synthetic tab to **show modes** (Q&A / Conversation) | **mark as runtime-dependent** | Real generation requires `TEACHER_MODEL_API_URL`/key or `ALLOW_SYNTHETIC_DEMO_FALLBACK=true`. Codex's selector pass showed `WARN: missing TEACHER_MODEL_API_KEY`. |
| Dataset Prep | Yes — Dataset Prep tab; click into Schema Profile + Adapter Preview | n/a | Verified surface; shows the qa-pair adapter applied. |
| Tokenization | Yes — open tab; analyze a small tokenizer if Q20 resolved | **mark as runtime-dependent** otherwise | Requires `transformers` + a tokenizer download; ungated small tokenizer recommended. |
| Training | Show **config + preflight** on `/training-config` page, **do not** click Start | **mark training run as separate Video 09** | Per series plan, training run is Module 9. This module stops at preflight. |
| Evaluation | Show eval workbench + empty state | **mark as Video 10** | No experiments yet exists in this project. |
| Compression / Export / Final | **skip entirely** | **mark as Videos 11/12** | Out of scope for sample tour. |

## Step-by-step user journey + visual results

| Time | Action | What viewer sees |
|---|---|---|
| 00:00 – 00:30 | Recap quickstart: log in, click Support FAQ tile. | Repeat of Video 02 ending state. |
| 00:30 – 01:30 | Data tab. Show 20 docs. Expand row 20. Read out the `question` and `answer` columns. | Two-column raw row content. |
| 01:30 – 02:30 | Switch to Cleaning. Walk through chunk size, redaction toggles. **Do not click Start.** Explain what would happen. | Cleaning config form. |
| 02:30 – 04:00 | Gold Set tab. Show `Entries 200`. Scroll through a few rows. Explain locked/approved status. | Gold table. |
| 04:00 – 05:00 | Synthetic tab. Show the three modes. Point at the `WARN: TEACHER_MODEL_API_KEY missing` banner if visible. | Synthetic config + warning. |
| 05:00 – 06:30 | Dataset Prep tab. Open the prepared manifest preview (or call out the prepared-manifest API). Explain 16/2/2 split. | Prep summary panels. |
| 06:30 – 07:30 | Tokenization tab. Light surface tour. | Tokenizer panel. |
| 07:30 – 09:00 | Training tab → "No experiments yet". Click into Training Config. Walk Essentials → Advanced → Power Tools. | Training config form. |
| 09:00 – 10:00 | Evaluation tab → empty. Wrap. | Eval empty state. |
| 10:00 – 10:30 | Wrap. Hand off to Video 09 for actual training run. | Title card. |

## Narration checkpoints

- **00:30** — "The Support FAQ sample ships with 20 source rows of
  customer questions and answers. The seeder has already imported
  these as raw documents, created a 200-row locked gold set, and
  pre-built 16/2/2 train/val/test splits."
- **02:00** — "Cleaning is where you'd chunk, redact regex PII, and
  toxicity-mask. The seed didn't run cleaning, because this dataset
  is already small and clean. For a real-world support corpus you'd
  flip it on."
- **04:00** — "Synthetic generation is the lever that takes 20 raw
  rows to 2000 training rows. It needs a teacher model, like local
  Ollama. Right now the banner warns we don't have one configured.
  Video 4 covers this in detail."
- **05:30** — "Dataset Prep is where the magic of *contract* happens —
  the adapter (`qa-pair`) wraps each row into a `{question, answer}`
  shape, and the splits get written to JSONL."
- **07:30** — "Training Config supports an Essentials view by default
  and an Advanced toggle. The Advanced view exposes LoRA rank, target
  modules, optimizer choice — covered in Video 8."
- **09:30** — "We've intentionally stopped short of pressing Start.
  Real training is Video 9, because the runtime decisions deserve
  their own walkthrough."

## Screenshots to capture

| # | Filename | Action |
|---|---|---|
| 1 | `v03-data-tab.png` | Data tab loaded |
| 2 | `v03-expanded-row.png` | After expanding row 20 |
| 3 | `v03-cleaning-config.png` | Cleaning tab |
| 4 | `v03-goldset.png` | Gold Set tab with `Entries 200` visible |
| 5 | `v03-synthetic-warn.png` | Synthetic tab with the missing-teacher-key warning |
| 6 | `v03-dataprep.png` | Dataset Prep tab |
| 7 | `v03-training-config.png` | Training Config Power Tools view |

(Reuse existing `selector-pass-03..14.png` if recording resolution
matches; otherwise reshoot.)

## Pauses for viewer comprehension

- 1.5 sec after each tab switch (let the viewer's eye land).
- 3 sec on the expanded raw row (let them read the `question`).
- 2 sec on the Gold Set `Entries 200` badge.
- 2 sec on the Synthetic warning banner if visible.

## What to mark as conceptual / simulated / unknown

| Item | Marker |
|---|---|
| Synthetic generation actually running | **runtime-dependent** — needs `TEACHER_MODEL_API_KEY` or fallback flag |
| Tokenization analysis | **runtime-dependent** — needs `transformers` + a non-gated tokenizer |
| Training run | **next video (09)** |
| Evaluation metrics | **next video (10)** |
| Compression / Export | **next videos (11)** |
| Deployment / final use | **next video (12)** |
| The exact eval handler for `instruction_sft` | **unknown** — open Q9 in `10-open-questions.md` |

## What to skip if unsupported

- Do NOT show a cleaning *run* unless Q10 (does cleaning remove
  duplicates anywhere?) resolves; selector pass shows it computes
  hashes only.
- Do NOT click "Generate" in Synthetic unless a teacher endpoint is
  wired; the recording will show a network error.
- Do NOT click Start in Training; the empty state is the correct end
  for this video.

## Evidence files

- `docs-demo/evidence/02-demo-flows.md` Flow C
- `docs-demo/evidence/06-official-demo-samples-map.md` (support-faq block)
- `docs-demo/evidence/07-pipeline-step-evidence.md`
- `docs-demo/evidence/11-selector-route-evidence.md` (full disposable-pass record)
- Screenshots: `docs-demo/screenshots/selector-pass-01..14.png`
- Sample files: `backend/data/demo_samples/support-faq/`

## Recording feasibility

**Feasible now.** All selectors verified. All API responses
observed. No new instrumentation required. Single-take recording is
plausible at 9–12 min.

## Risk / blockers

- **Manifest text staleness**: support-faq's `manifest.json` says "6
  hand-labelled gold rows", but the file has 200. The narrator MUST
  say 200 (the file is the source of truth). Flag this in narration
  rather than ignoring it.
- **`PipelineStage.TRAINING` on seed**: the project's pipeline_stage
  jumps to `training` at 60% on seed, making earlier tabs feel
  "completed" even though we haven't really run them. Narration must
  call this out so viewers don't think they magically trained
  something.
- **Existing project name collision**: if a previous `Demo · Support
  FAQ` exists in the DB, the seed reuses it (idempotent). Recording
  should either start from a clean DB or accept reusing an existing
  project id.
