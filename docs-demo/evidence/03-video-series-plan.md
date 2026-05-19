# Video Series Plan

Discovery date: 2026-05-19.

## Series Goals

- Teach SLM basics to a beginner without leaning on generic ML
  assumptions.
- Show what this repo actually does, backed by file paths and the
  selector-discovery pass already captured under
  `docs-demo/screenshots/`.
- Record real UI/API paths only after the relevant
  `docs-demo/evidence/*` row is at least `partial` and the recording
  plan has called out runtime prerequisites.
- Keep every product claim labeled verified, partial, simulated,
  estimated, conceptual, or unknown. Anything that is not in
  `docs-demo/evidence/` should not be in a narration script.

## Status legend per video

- **ready** — evidence + selectors + screenshots already exist; the
  narration just needs a polish pass and a recording session.
- **partial** — surfaces verified, but at least one runtime step
  (training, eval, compression, export, serve) still needs an
  evidence-backed decision before recording.
- **conceptual** — explicitly no product-specific claims; the slide
  deck does the talking.
- **blocked** — depends on a decision (data-testid additions, runtime
  choice, judge model availability) that has not been made yet.

## 12-Module Arc

Status updated 2026-05-19 after the runtime decisions in
`12-runtime-decisions-2026-05-19.md`. Modules 09–12 are no longer
blocked.

| # | Working title | Audience | Duration | Status | Primary evidence dependency |
|---|---|---|---:|---|---|
| 01 | SLM 101 | beginner | 5–7 min | conceptual | None; conceptual only. |
| 02 | BrewSLM Quickstart (login, demo seed, workspace) | beginner | 5–7 min | **ready** | `01`, `02`, `06` evidence + selector-pass screenshots. |
| 03 | Dataset Lifecycle Overview (concepts + map onto BrewSLM tabs) | beginner | 6–8 min | partial | `05`, `07`. |
| 04 | Gold Set + Synthetic Generation Workflow | intermediate | 7–9 min | **ready** | Q21 resolved (local Ollama teacher); `12-runtime-decisions`. |
| 05 | Support-FAQ Sample — full pipeline walkthrough | beginner / intermediate | 9–12 min | partial | `06` sample map; Q9 (eval handler) deferred to recording time. |
| 06 | PII-Detector Sample — full pipeline walkthrough | intermediate | 10–12 min | partial | `06` sample map; Q15 (cleaning vs detector) handled in narration. |
| 07 | Sentiment-Classifier Sample — full pipeline walkthrough | beginner / intermediate | 8–10 min | partial | `06` sample map; Q12 (classification synthetic) handled in narration. |
| 08 | Training Configuration + Runtime Choices | intermediate | 8–10 min | **ready** | Q16 resolved (real Celery + external runtime); `12-runtime-decisions`. |
| **09** | **Training Run, Logs, and Monitoring** | intermediate | 7–9 min | **ready** | **Q16 + Q17 resolved.** support-faq @ 2 epochs is the recording target. |
| 10 | Evaluation, Scorecards, Quality Gates | intermediate | 7–9 min | **ready** | Q22 resolved (Ollama qwen2.5:7b judge). |
| 11 | Compression, Export, Model Registry | intermediate / advanced | 6–8 min | **ready** | Q23 = GGUF via llama.cpp; Q24 = GGUF export. |
| 12 | Final Model Usage (Playground, Serve, API smoke) | intermediate | 8–10 min | **ready** | Q25 = Ollama; full loop reachable. Q26/Q27 (playground linkage) resolve at record time. |
| 13 | Bring-Your-Own Dataset / Custom Pipeline | intermediate | 8–10 min | partial | `08-outside-template-demo-ideas.md` + Flow F. |
| 14 | Architecture / Operator Deep Dive | advanced | 8–10 min | partial | `frontend/vite.config.ts`, `backend/app/main.py`, async-task patterns. |

**Net change after runtime decisions**: 7 of 13 modules now `ready`
(was 1 of 13 before today). Modules 03, 05–07, 13, 14 remain
`partial` for non-runtime reasons (narration nuances captured in
their respective recording-plan files); only Module 01 stays
conceptual by design.

13 modules in this version (one over the requested 10–15 lower bound;
under the 15 ceiling). Module 09 is split out from 08 because
training-run capture has a different recording posture (log
streaming + observability page) than the static training-config
walkthrough.

## Per-module detail

### 01 — SLM 101

- **Audience**: beginner.
- **Learning objective**: explain what an SLM is, what changes when
  you go smaller than an LLM, and what a usable lifecycle looks like
  without making product claims.
- **UI/demo steps**: none in app. Slide-only video.
- **Slide sections needed**: A (SLM 101) only.
- **Recording script**: optional — can run as a slide deck with
  voice-over.
- **Narration script**: `docs-demo/scripts/narration/01-slm-101.md`
  (already drafted).
- **Evidence**: none required; explicitly conceptual.
- **Expected duration**: 5–7 min.
- **Status**: conceptual.

### 02 — BrewSLM Quickstart

- **Audience**: beginner.
- **Learning objective**: get a viewer from a fresh repo to a seeded
  demo project in under five minutes, with confidence that they're
  on the right path.
- **UI/demo steps**: log in (`/login`) → land on project list →
  read the three demo tiles → click one (recommended: `support-faq`)
  → land on `/project/{id}/pipeline/data`.
- **Slide sections needed**: B (BrewSLM 101), C (Quickstart).
- **Recording script needed**: yes — see
  `docs-demo/videos/02-brewslm-quickstart/recording-plan.md`.
- **Narration script needed**: `docs-demo/scripts/narration/02-brewslm-quickstart.md`.
- **Evidence**:
  `docs-demo/evidence/01-feature-inventory.md` (auth + workspace rows),
  `docs-demo/evidence/02-demo-flows.md` (Flow 0, A),
  `docs-demo/evidence/06-official-demo-samples-map.md`,
  `docs-demo/screenshots/selector-pass-01-login.png`,
  `docs-demo/screenshots/selector-pass-02-demo-tiles.png`,
  `docs-demo/screenshots/selector-pass-03-support-faq-data-tab.png`.
- **Expected duration**: 5–7 min.
- **Status**: ready (selectors discovered; screenshots captured).

### 03 — Dataset Lifecycle Overview

- **Audience**: beginner / early intermediate.
- **Learning objective**: map the conceptual lifecycle (ingest →
  inspect → clean → normalize → gold → synthetic → prep → tokenize)
  onto the actual BrewSLM pipeline tabs without running any heavy
  step.
- **UI/demo steps**: walk through `/project/{id}/pipeline/data`,
  `cleaning`, `goldset`, `synthetic`, `dataprep`, `tokenization` —
  inspecting only, not running.
- **Slide sections needed**: D (Dataset Lifecycle).
- **Recording**: ready surface; selector-pass screenshots
  `selector-pass-03..09.png` already cover the tabs in support-faq.
- **Narration**: needed but not yet drafted; covered by 03 in
  scripts/narration when ready.
- **Evidence**: `05-dataset-and-pipeline-map.md`,
  `07-pipeline-step-evidence.md`, sample manifests.
- **Expected duration**: 6–8 min.
- **Status**: partial (synthetic + tokenization runtime decisions
  unresolved; tabs render but actual runs need Q21).

### 04 — Gold Set + Synthetic Generation Workflow

- **Audience**: intermediate.
- **Learning objective**: explain why a gold set is the spine of
  any honest eval, then show synthetic generation as one way to
  multiply training examples while keeping gold honest.
- **UI/demo steps**: open `/goldset` (show seeded 200-row gold,
  locked-approved status), then `/synthetic` (show modes: Q&A,
  conversation, PII span; explain teacher-vs-fallback toggle).
- **Slide sections needed**: D (Dataset Lifecycle), E (Sample
  Demos) for context.
- **Recording**: surface verified; actual synthetic generation
  needs either a teacher model URL or
  `ALLOW_SYNTHETIC_DEMO_FALLBACK=true`.
- **Narration**: not yet drafted.
- **Evidence**: `01-feature-inventory.md` Synthetic-generation row;
  `frontend/src/components/data/SyntheticPanel.tsx`;
  `backend/app/services/synthetic_service.py`.
- **Expected duration**: 7–9 min.
- **Status**: partial (Q21 unresolved).

### 05 — Support-FAQ Sample full pipeline

- **Audience**: beginner / intermediate.
- **Learning objective**: take the simplest sample (Q&A) from seed
  to evaluation, with explicit callouts for any step that requires
  external runtime.
- **UI/demo steps**: seed → data tab → goldset → dataprep (already
  seeded; show prepared manifest) → training-config (preflight) →
  eval workbench surface. Skip live training run unless Q16
  resolved.
- **Slide sections**: E (support-faq slide).
- **Recording**:
  `docs-demo/videos/03-support-faq-pipeline/recording-plan.md`.
- **Narration**: `docs-demo/scripts/narration/03-support-faq-pipeline.md`.
- **Evidence**: `06-official-demo-samples-map.md` (support-faq
  block), `07-pipeline-step-evidence.md`,
  `backend/data/demo_samples/support-faq/*`,
  `docs-demo/screenshots/selector-pass-03..14.png`.
- **Expected duration**: 9–12 min.
- **Status**: partial.

### 06 — PII-Detector Sample full pipeline

- **Audience**: intermediate.
- **Learning objective**: show a span-extraction task end-to-end and
  draw the contrast between cleaning-time PII redaction (regex in
  cleaning service) and a trained PII *detector* (model task).
- **UI/demo steps**: seed → data tab (raw `text + entities_json`) →
  cleaning tab (point at PII redaction option, do not conflate) →
  goldset (200 rows, all 10 entity types) → synthetic span mode
  (label as runtime-dependent) → dataprep (show
  `output_schema.scoring_mode=span_set`) → eval (per-class metrics).
- **Slide sections**: E (pii-detector slide).
- **Recording**:
  `docs-demo/videos/04-pii-detector-pipeline/recording-plan.md`.
- **Narration**: `docs-demo/scripts/narration/04-pii-detector-pipeline.md`.
- **Evidence**: `06` sample map (pii-detector block, including the
  10 entity types and counts), `slm-docs/docs/demos/pii-detector.md`,
  `docs-demo/screenshots/selector-pass-pii-01..13.png`.
- **Expected duration**: 10–12 min.
- **Status**: partial.

### 07 — Sentiment-Classifier Sample full pipeline

- **Audience**: beginner / intermediate.
- **Learning objective**: show a three-way classification flow with
  classification-specific adapter + eval pack, and use this sample
  to introduce mobile/CPU export framing (without claiming a real
  ONNX-INT8 was built).
- **UI/demo steps**: seed → data (raw `text + label`) → goldset (200
  rows; 70/65/65 split) → dataprep → training-config (mobile_cpu
  target profile) → eval (classification handler). Stop short of
  real ONNX export unless Q24 resolved.
- **Slide sections**: E (sentiment-classifier slide), G (Eval pack).
- **Recording**:
  `docs-demo/videos/05-sentiment-classifier-pipeline/recording-plan.md`.
- **Narration**: `docs-demo/scripts/narration/05-sentiment-classifier-pipeline.md`.
- **Evidence**: `06` sample map (sentiment block),
  `docs-demo/screenshots/selector-pass-sentiment-01..13.png`.
- **Expected duration**: 8–10 min.
- **Status**: partial.

### 08 — Training Configuration + Runtime Choices

- **Audience**: intermediate.
- **Learning objective**: explain the training-config page,
  Essentials vs Advanced toggle, LoRA controls (rank, alpha, target
  modules), runtime selector (`auto`, `builtin.external_celery`,
  etc), and the preflight gate from Story 1.5.
- **UI/demo steps**: `/project/{id}/training-config` Essentials tab
  → Advanced toggle → Config tab → Power Tools tab (PEFT) → Review.
- **Slide sections**: B (architecture), G (Training).
- **Recording**: separate plan (not yet drafted); should reuse
  selector-pass training-config screenshots.
- **Narration**: needed.
- **Evidence**: `07-pipeline-step-evidence.md` training-config row;
  `frontend/src/components/training/TrainingPanel.tsx`;
  `frontend/src/pages/ProjectTrainingConfigPage.tsx`.
- **Expected duration**: 8–10 min.
- **Status**: partial — preflight surface verified, but full
  training run launch is module 09's problem.

### 09 — Training Run, Logs, Monitoring

- **Audience**: intermediate.
- **Learning objective**: actually press Start, capture a training
  log stream, show the observability page during a run, demonstrate
  the Story 1.7 checkpoint-compat gate + Story 1.5 status reconciler
  if they fire.
- **UI/demo steps**: training tab → Start → switch to Observability
  + Run Events → wait for completion → checkpoint list.
- **Slide sections**: G (Training).
- **Recording**: not yet drafted; depends on Q16 (real vs simulated
  runtime decision) and Q17 (smallest reliable job per sample).
- **Narration**: needed.
- **Evidence**: `07` training-run row;
  `backend/scripts/train.py`; Story 1.7 commit `65a439a` for the
  compatibility-gate behavior.
- **Expected duration**: 7–9 min.
- **Status**: blocked on Q16 + Q17.

### 10 — Evaluation, Scorecards, Quality Gates

- **Audience**: intermediate.
- **Learning objective**: take a completed experiment (or one with
  predictions) and walk the eval workbench, scorecard, gates, and
  the Story 1.5 schema-mismatch banner.
- **UI/demo steps**: `/pipeline/eval` → run held-out eval (with the
  30-min axios timeout from Story 1.7) → scorecard panel → gates →
  failure-clusters / sample predictions card.
- **Slide sections**: G (Evaluation).
- **Recording**: not yet drafted; depends on module 09 (need an
  experiment to evaluate) or supplied predictions.
- **Narration**: needed.
- **Evidence**: `07` eval / scorecard / gates rows; Story 1.5 commit
  `92cf7a5` for the schema-mismatch banner.
- **Expected duration**: 7–9 min.
- **Status**: partial — handler dispatching for `instruction_sft`
  remains open (Q9).

### 11 — Compression, Export, Model Registry

- **Audience**: intermediate / advanced.
- **Learning objective**: show the post-training packaging story:
  optional compression, then export to one of `gguf` / `onnx` /
  `tensorrt` / `huggingface` / `docker`, then registry registration
  + promotion.
- **UI/demo steps**: `/pipeline/compression` → quantize/benchmark
  panel → `/pipeline/export` → format picker → export run → registry
  card.
- **Slide sections**: H (Evaluation/Compression/Export).
- **Recording**: not yet drafted; depends on Q23 (which compression
  path locally) + Q24 (canonical export format).
- **Narration**: needed.
- **Evidence**: `07` compression/export/registry rows;
  `backend/app/api/compression.py`,
  `backend/app/services/export_service.py`,
  `backend/scripts/quantize.py`.
- **Expected duration**: 6–8 min.
- **Status**: partial.

### 12 — Final Model Usage (Playground, Serve, API Smoke)

- **Audience**: intermediate.
- **Learning objective**: prove the loop closes — a real or
  intentionally-simulated trained model returns from a playground
  prompt or a local serve subprocess, with the curl smoke test
  visible.
- **UI/demo steps**: `/project/{id}/playground` → model selector →
  prompt → response → feedback log. Optionally: `/pipeline/export`
  → serve plan → start local serve.
- **Slide sections**: I (Final Model Usage).
- **Recording**: not yet drafted; depends on Q25 (which local serve
  runtime is installed).
- **Narration**: needed.
- **Evidence**: `09-final-model-usage-plan.md` in full.
- **Expected duration**: 8–10 min.
- **Status**: partial.

### 13 — Bring-Your-Own Dataset / Custom Pipeline

- **Audience**: intermediate.
- **Learning objective**: take a user-supplied dataset through the
  generic import wizard (Flow F), explain the eight built-in
  mappers, and explicitly flag the cases where no mapper applies
  (e.g. `ai4privacy/pii-masking-200k` inline character-span format).
- **UI/demo steps**: `/pipeline/data` → Import wizard → source
  picker → mapper picker → preview → run.
- **Slide sections**: F (Beyond Official Samples).
- **Recording**: not yet drafted.
- **Narration**: needed.
- **Evidence**: `08-outside-template-demo-ideas.md`, Flow F,
  `backend/app/services/dataset_import/mappers/`.
- **Expected duration**: 8–10 min.
- **Status**: partial.

### 14 — Architecture / Operator Deep Dive

- **Audience**: advanced.
- **Learning objective**: pull back from the per-task tour and show
  the service map: React/Vite frontend, FastAPI backend, async-task
  registry pattern (cleaning, synthetic, training, compression),
  Celery worker, audit/run-event spine, Lab Journal gamification,
  experiment recovery (Story 1.7), training data gate (`222bc5d`).
- **UI/demo steps**: light — mostly slides + occasional terminal
  for `brewslm experiment reset` / API curl demos.
- **Slide sections**: J (Advanced Topics).
- **Recording**: not yet drafted.
- **Narration**: needed.
- **Evidence**: `backend/app/main.py`, `frontend/vite.config.ts`,
  `backend/app/services/run_event_service.py`,
  `backend/app/services/training_data_gate.py`,
  `backend/app/services/experiment_recovery_service.py`,
  `backend/scripts/brewslm.py` (CLI).
- **Expected duration**: 8–10 min.
- **Status**: partial — most surfaces verified by recent commits
  shipped in Stories 1.5 / 1.6 / 1.7.

## Recording Order (rebuilt 2026-05-19 after runtime decisions)

With Q16–Q25 resolved (`12-runtime-decisions-2026-05-19.md`),
nothing is blocked except by upstream prerequisite videos. New
recommended order, one video per session:

1. **02 quickstart** — proves the recording pipeline works; smallest
   scope; every selector verified.
2. **03 dataset-lifecycle overview** — slide-heavy; no runtime
   needed; can record in parallel with #1.
3. **04 gold + synthetic** — first runtime-dependent video; tests
   the Ollama-teacher wiring before training.
4. **05 support-faq walkthrough** — first sample tour; inspect-only,
   no training yet.
5. **09 training run (support-faq)** — first real training capture;
   produces the artifact downstream videos depend on.
6. **10 evaluation** against #5's experiment — produces the first F1
   number captured by the series.
7. **11 compression + export** of #5's artifact → GGUF.
8. **12 final-model usage** — loop closes; Ollama serves the GGUF.
9. **06 PII walkthrough** + **07 sentiment walkthrough** — can
   record in parallel; independent of each other.
10. **08 training-config deep dive** — can record any time after
    #5; mostly slide + form walkthrough.
11. **13 BYO custom dataset** + **14 architecture deep dive** —
    deck-heavy; record last.
12. **01 SLM 101** — slide-only; record any time.

## What this plan deliberately does NOT include

- A polished thumbnail / branding pass.
- A YouTube/Vimeo upload plan.
- Final video lengths in seconds (rough min/max only).
- Any claim that a trained Qwen-PII-V6 model is in the recording set
  yet. Past experiments 9/10/11 failed, exp 12 completed at
  `final_eval_loss=0.61`, but **F1 / per-class metrics still need
  to be captured** (this is the work in module 10 once Q16+Q22
  resolve).
