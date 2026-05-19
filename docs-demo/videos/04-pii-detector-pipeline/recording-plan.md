# Video 04 — PII Detector Sample · Full Pipeline · Recording Plan

Status: **partial** — every UI surface verified by selector pass on
2026-05-19. Real synthetic span generation requires a teacher
endpoint (open Q21).

## Goal

Walk the pii-detector sample end-to-end and **cleanly separate**
two product features that share the name "PII":

1. **Cleaning-time PII redaction** — regex in `cleaning_service.py`,
   used to mask PII in source text before training.
2. **PII Detector model task** — span_set extraction; the model
   learns to *find* entities, not redact them.

Conflating these two is the #1 confusion risk for this sample.

## Audience

Intermediate.

## Expected video length

10–12 minutes.

## Exact starting state

1. Backend + frontend running (see Video 02 prereqs).
2. Logged in.
3. Seed via UI tile (`Demo · PII / PCI Detector`) **or** curl
   `POST /api/demo-projects/pii-detector`.
4. Browser at `/project/<id>/pipeline/data`.
5. Confirm 61 ingested documents on Data tab.

## Sample files involved

| Path | Role | Rows |
|---|---|---:|
| `backend/data/demo_samples/pii-detector/manifest.json` | Task profile `structured_extraction`, `output_schema.scoring_mode=span_set`, 10 entity types. | n/a |
| `backend/data/demo_samples/pii-detector/pii_records.csv` | Source: `text`, `entities_json`. | 61 |
| `backend/data/demo_samples/pii-detector/gold.jsonl` | Gold eval (all 10 entity types covered). | 200 |
| `backend/data/demo_samples/pii-detector/_generate_bundle.py` | Author-side generator. **Not** part of the recording. | n/a |
| `backend/data/demo_samples/pii-detector/kaggle_pii_to_brewslm.py` | Kaggle converter. **Not** part of the recording. | n/a |
| `data/projects/<id>/prepared/train.jsonl` | Seeded 45 rows. | 45 |
| `data/projects/<id>/prepared/val.jsonl` | 8 rows. | 8 |
| `data/projects/<id>/prepared/test.jsonl` | 8 rows. | 8 |

## Entity type vocabulary (callout content)

From `manifest.json`:
`email, phone, ssn, credit_card, person_name, street_address,
date_of_birth, ip_address, api_key, bank_account` (10 types).

Counts in gold (per `06-official-demo-samples-map.md`):

| Type | Gold count |
|---|---:|
| person_name | 138 |
| email | 72 |
| phone | 47 |
| street_address | 42 |
| date_of_birth | 36 |
| ip_address | 34 |
| bank_account | 29 |
| ssn | 28 |
| credit_card | 26 |
| api_key | 21 |

## UI route + selector sequence

| # | Route | Component | Focal selector | Status |
|---|---|---|---|---|
| 1 | `/project/<id>/pipeline/data` | `IngestionPanel` | `[data-testid="expand-doc-61"]` (concrete observed) | verified |
| 2 | `/project/<id>/pipeline/cleaning` | `CleaningPanel` | `button.tab[title="Cleaning"]`; **show redaction toggles** | verified |
| 3 | `/project/<id>/pipeline/goldset` | `GoldSetPanel` | "Entries 200" | verified |
| 4 | `/project/<id>/pipeline/synthetic` | `SyntheticPanel` | Span mode auto-selected from prepared manifest | partial (runtime) |
| 5 | `/project/<id>/pipeline/dataprep` | `DatasetPrepPanel` | Schema profile shows `output_schema.scoring_mode=span_set` | verified |
| 6 | `/project/<id>/pipeline/training` | `TrainingPanel` | Empty state | verified surface |
| 7 | `/project/<id>/training-config` | `ProjectTrainingConfigPage` | Essentials → Advanced → Power Tools | verified |
| 8 | `/project/<id>/pipeline/eval` | `EvalPanel` | Empty state | verified surface |

## API calls observed during selector pass

- `POST /api/demo-projects/pii-detector`
- `GET /api/projects/<id>/ingestion/documents` (returns 61)
- `GET /api/projects/<id>/ingestion/documents/61/sample` (shape `text + entities_json`)
- `GET /api/projects/<id>/gold/entries?dataset_type=gold_dev`
- `GET /api/projects/<id>/prepared-manifest` — **headline API**;
  returns `output_schema.scoring_mode=span_set` and the 10 entity
  types.
- `GET /api/projects/<id>/export/deployment-targets?export_format=gguf`

## Pipeline stages to show

| Stage | Show | Skip / mark | Reason |
|---|---|---|---|
| Ingestion | Yes — Data tab; expand doc 61; show `entities_json` JSON inside the row | n/a | Verified. Critical: viewer sees that the raw row has TWO columns: the text + the structured entity ground truth. |
| Cleaning | Yes — open the tab; **highlight the regex PII redaction option**; explicitly contrast: "this is masking-at-cleaning, not the detector model." Do **NOT** run a cleaning batch. | n/a | This is the disambiguation moment — must be on-screen. |
| Gold Set | Yes — `Entries 200`; scroll one row to see the typed entity structure | n/a | Verified. |
| Synthetic | Yes — span_extraction mode visible; mention the `qwen2.5:7b/14b-instruct-q4_K_M` teacher recommendation from `slm-docs/docs/demos/pii-detector.md`. Do **NOT** click Generate unless a teacher is wired up. | mark runtime-dependent | Real generation needs teacher; selector pass observed `WARN: TEACHER_MODEL_API_KEY missing`. |
| Dataset Prep | Yes — click into Schema Profile; the `span_set` scoring mode is the prepared manifest's distinguishing field | n/a | Verified. |
| Tokenization | Light tour (open + close) | mark runtime-dependent | Q20 unresolved. |
| Training | Light tour to empty state; click into Training Config | mark Video 09 for actual run | Same as Support FAQ. |
| Eval | Empty state; **call out that the eval handler will be span_set with per-class precision/recall** | mark Video 10 for actual run | Verified handler dispatch via `eval_task_handler_service.py`. |

## Step-by-step user journey

| Time | Action | What viewer sees |
|---|---|---|
| 00:00 – 00:30 | Recap + click PII / PCI Detector tile. | Project opens at Data tab. |
| 00:30 – 02:00 | Data tab. Show 61 docs. Expand a row that has 3+ entity types. Read the `entities_json` payload. | Two-column row content with JSON entity array. |
| 02:00 – 03:30 | **Cleaning tab — disambiguation moment**. Walk through PII redaction toggle. State plainly: "this masks PII at cleaning time. The detector model is a totally separate thing that *finds* entities. We're showing both because the word 'PII' applies to both." | Cleaning config form. |
| 03:30 – 04:30 | Gold Set tab. Scroll one row showing the `entities` list inside `expected`. | Gold table. |
| 04:30 – 06:00 | Synthetic tab. Span mode is auto-selected. Walk through entity types field. Explicitly mention "we'd type our 10 PII types here and generate 2000 rows; today the teacher key is missing." | Synthetic span form. |
| 06:00 – 07:30 | Dataset Prep tab. Open Schema Profile. Point at `output_schema.scoring_mode=span_set` and the 10 entity types. | Prep summary. |
| 07:30 – 09:00 | Training Config Power Tools. Show LoRA defaults, `target_modules=q_proj,v_proj` (default), and the *recommendation* from the docs to bump to `q_proj, k_proj, v_proj, o_proj` for span tasks. | Training config form. |
| 09:00 – 10:30 | Training + Eval empty states. Wrap. | Empty states. |
| 10:30 – 11:00 | Wrap. Hand off to Video 09 (run) and Video 10 (eval per-class F1). | Title card. |

## Narration checkpoints

- **00:30** — "The PII detector sample ships 61 source rows. Each row
  is text plus a structured `entities_json` ground truth — `[{type,
  start, end, text}, …]`. The model's job is to learn that shape."
- **02:00** — "Watch out: 'PII' is used twice in this product. The
  Cleaning tab can redact regex-detected PII as `[REDACTED]` *before*
  training. The PII Detector model is the opposite — it *finds*
  entities and emits structured JSON. Same word, different feature."
- **04:30** — "Synthetic generation is how you'd grow 61 rows to 2k.
  The synthetic generator runs the teacher LLM over your cleaned
  chunks. We'd point it at our 10 entity types and let it sample."
- **06:00** — "Prepared manifest. Notice the
  `output_schema.scoring_mode = span_set`. This is the *contract*
  with eval: the model has to emit JSON with an `entities` array,
  and eval scores it per entity type."
- **07:30** — "Training Config defaults LoRA rank 8 with two target
  modules. For span tasks the docs recommend rank 16 with all four
  attention projections — see `slm-docs/docs/demos/pii-detector.md`."

## Screenshots

| # | Filename | Action |
|---|---|---|
| 1 | `v04-data-tab.png` | Data tab loaded |
| 2 | `v04-expanded-row.png` | Row with multiple entity types expanded |
| 3 | `v04-cleaning-redact.png` | Cleaning tab showing redaction toggle |
| 4 | `v04-goldset.png` | Gold Set with `Entries 200` |
| 5 | `v04-synthetic-span.png` | Synthetic tab in span_extraction mode |
| 6 | `v04-schema-profile.png` | Dataset Prep showing span_set schema |
| 7 | `v04-training-power-tools.png` | Advanced PEFT view in Training Config |

(Selector-pass screenshots `selector-pass-pii-01..13.png` can be
reused at matching resolution.)

## What to mark

| Item | Marker |
|---|---|
| Cleaning redaction (regex) | **verified** — runnable but we're not running it |
| Detector model task (span extraction) | **partial** — needs training run + eval run |
| Synthetic span generation | **runtime-dependent** — needs teacher |
| Per-class F1 / recall numbers | **unknown** — captured in Video 10 |
| Recommended LoRA tuning (rank 16, 4 attention projections) | **conceptual** — sourced from docs, not measured |

## Evidence files

- `docs-demo/evidence/02-demo-flows.md` Flow D
- `docs-demo/evidence/06-official-demo-samples-map.md` (pii-detector block)
- `docs-demo/evidence/11-selector-route-evidence.md` (PII section)
- `slm-docs/docs/demos/pii-detector.md` (operator-side troubleshooting + entity types)
- Screenshots: `docs-demo/screenshots/selector-pass-pii-01..13.png`

## Failure modes / risks

- **Confusing redaction with detection**. Mitigated by Section 2's
  scripted disambiguation; rehearse before recording.
- **Synthetic Generate button click**. The button looks inviting on
  the tab; do not click it without a teacher URL. The recording can
  hover but should NOT click.
- **Manifest prose mentions 60 snippets, CSV has 61**. Codex flagged
  this in Q1. Narration should say 61 (file truth).

## Recording feasibility

**Feasible now**, with the same caveats as Video 03: real
training/eval still belongs to Videos 09 and 10.

## Open questions specific to this video

- (Q15) What is the **exact UI path** for showing cleaning redaction
  separately from the detector model task? Selector pass confirmed
  both surfaces render; recording will need to lean on narration
  rather than UI affordance to disambiguate.
- (Q1) Do we update the manifest description text to say 61 instead
  of 60? (Docs-only change; recommend separate ticket.)
