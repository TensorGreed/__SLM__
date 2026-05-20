# Support FAQ Pipeline — Narration

Status: **synced** with the actual narrated take produced by
`tts/generate_v03_narration.py` (Orpheus voice "leo") on 2026-05-20.

The **Python script** at
[tts/generate_v03_narration.py](../../../tts/generate_v03_narration.py)
is the **authoritative source** of the spoken text. This file mirrors
the same text plus stage directions / Playwright cues. Edit the
script first.

Total runtime: **2:45** (matches
`docs-demo/recordings/raw/03-support-faq-pipeline-narrated.mp4`).
Section timings below come from `tts/audio/v03-durations.json`.

Companion to:
[docs-demo/videos/03-support-faq-pipeline/recording-plan.md](../../videos/03-support-faq-pipeline/recording-plan.md).

---

## Pre-roll (not narrated)

Playwright logs in, clicks the **Demo · Support FAQ** tile, lands on
the Data tab. ~5 seconds of seed-time before narration starts.

## Section 1 — Cold open (0:00–0:15)

**On screen**: Data tab loaded; 20 raw documents listed below.

> "Welcome to the Support FAQ pipeline walkthrough. We're taking the
> simplest of the three official samples — twenty customer tickets
> with hand-written answers — and walking it through every pipeline
> tab that does something useful on a seeded demo. No training, no
> synthetic generation, just inspection."

## Section 2 — Data tab + expand a raw row (0:15–0:35)

**On screen**: click `[data-testid^="expand-doc-"]` on the first row;
the `question`/`answer` pair expands inline.

> "Data tab. Twenty raw documents — one per source ticket. The
> seeder turned each row into a raw document record. Expand one. You
> see the shape: a question and an answer. This is what the model
> has to learn — the agent's writing style for these specific
> questions. Imagine pasting thousands of resolved tickets here and
> you've got the dataset for a real support assistant."

## Section 3 — Cleaning tab (0:35–0:56)

**On screen**: click the **Cleaning** tab; cleaning config form
visible.

> "Cleaning. Skip it for this sample — the corpus is already small
> and clean. For a messy real-world corpus, this is where you'd
> chunk long text, redact personal information, mask toxicity, and
> score quality. Same word as the next sample's PII Detector, but
> two completely different features — cleaning here is a regex
> pre-processing step, the detector is a trained model."

## Section 4 — Gold Set tab (0:56–1:12)

**On screen**: click **Gold Set**; "Entries 200" badge + locked
indicator visible. Viewport-only screenshot.

> "Gold Set. Two hundred entries. Locked. This is the evaluation
> ground truth — never trained against, only measured against. Each
> row has a question, an expected answer, and a rationale. The eval
> handler walks the entire two-hundred-row set after training and
> reports the fraction the model got right."

## Section 5 — Synthetic tab (1:12–1:30)

**On screen**: click **Synthetic**; mode toggles visible.

> "Synthetic. The lever that scales twenty source rows into two
> thousand training rows. It runs a teacher model — local Ollama on
> this machine — over your cleaned corpus, asking the teacher to
> generate matching question and answer pairs. Video Four is the
> full walkthrough; we're not running it here."

## Section 6 — Dataset Prep tab (1:30–1:48)

**On screen**: click **Dataset Prep**; Schema Profile + adapter
preview visible.

> "Dataset Prep. This is where the contract gets made. The adapter
> applied — question and answer pair — turns each row into the shape
> the trainer expects. Splits are already written: sixteen train,
> two validation, two test. That's the deterministic
> seventy-fifteen-fifteen split with a two-row floor on validation
> and test."

## Section 7 — Tokenization tab (1:48–2:04)

**On screen**: click **Tokenization**.

> "Tokenization. Runs a tokenizer over the prepared splits and
> reports the length distribution — how many tokens per row, what
> maximum sequence length you'd budget for. The actual analysis
> needs a tokenizer download, which is its own setup. Surface only
> for this video."

## Section 8 — Training tab → Training Config (2:04–2:28)

**On screen**: click **Training** → click **Open Training Config →**
button → land on `/project/<id>/training-config` → click **Advanced**
tab on the config-mode switch.

> "Training tab. No experiments yet — normal. Jumping into the
> Training Config page. Essentials view by default — base model,
> training mode, epochs, batch size, learning rate. Flip to Advanced
> and you unlock the parameter controls: low-rank adaptation rank,
> target modules, optimizer choice. The defaults work; the controls
> are there when you need them. Launching a run is Video Nine."

## Section 9 — Evaluation tab + wrap (2:28–2:45)

**On screen**: navigate back to `/project/<id>/pipeline/eval`;
"No experiments to evaluate" empty state.

> "Evaluation tab. Empty until we have a finished experiment. This
> is where accuracy, F1, gate pass and fail, and side-by-side
> predictions would land. That's the Support FAQ tour. We touched
> ten tabs without running anything heavy. Next video walks the
> same shape for the PII Detector sample."

---

## Things to **not** say

- Don't say "we trained a model" — we didn't.
- Don't say "the demo has 6 gold rows" — that's the stale manifest
  prose. Say 200.
- Don't say cleaning automatically removes duplicates — it computes
  hashes but row-removal is unverified (open Q10 in
  [10-open-questions.md](../../evidence/10-open-questions.md)).
- Don't read literal tech tokens aloud. The on-screen action shows
  them; the TTS engine will mispronounce them.

## Optional technical notes (background; not spoken)

- The `prepared-manifest` API at `GET /api/projects/<id>/prepared-manifest`
  is the headline endpoint for this walkthrough — returns adapter
  id, task profile, field mapping, output schema, and the
  train/val/test counts in one shot.
- The QA-pair adapter is one of eight registered data adapters; see
  [backend/app/services/data_adapter_service.py](../../../backend/app/services/data_adapter_service.py).
- The deterministic split uses `random.seed(42)` by default in the
  seeder; manual splits via `POST /api/projects/<id>/dataset/split`
  can override.

## Why no "missing teacher key" warning

Before today, the Synthetic tab rendered a `WARN: missing
TEACHER_MODEL_API_KEY` banner when the env var was unset. The
Section 5 narration in earlier drafts pointed at that banner. As of
2026-05-19 the runtime decision (Q21) wired a real Ollama teacher
into `backend/.env`, so the banner no longer appears. The narration
now says "local Ollama on this machine" instead of pointing at a
warning that isn't there.
