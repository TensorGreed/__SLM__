# Support FAQ Workspace Walkthrough — Narration (video 03)

Status: **rewritten 2026-10-18 for the Epic U workspace.** The walkthrough
is now Data (five steps) → Train (the form, then More options) → Results
(Verdict, Failures, Compare, Try it) → Ship (Package, Publish, Deploy,
Monitor) → wrap. One chapter per beat; the chapter list below is what
`frontend/src/data/tabVideos.ts` (v03) carries once the video is
re-recorded. Audio regenerated the same day by
`tts/generate_v03_narration.py` (Orpheus voice "leo", direct Ollama + SNAC
path).

The **Python script** at
[tts/generate_v03_narration.py](../../../tts/generate_v03_narration.py)
is the **authoritative source** of the spoken text. This file mirrors the
same text plus stage directions / Playwright cues. Edit the script first.

Total runtime: **3:48** (`tts/audio/v03-durations.json`; the spec
`tests/demo-recordings/03-support-faq-pipeline-narrated.spec.ts` pads each
section to these lengths).

Honesty rule for this video: the sample project has **no trained run**
when the recording starts, so every Results / Ship beat describes what the
page offers — never a result.

---

## Chapters

| # | Chapter | Starts |
|---|---------|--------|
| 1 | Intro (Overview) | 0:00 |
| 2 | Data · Sources | 0:18 |
| 3 | Data · Prepare | 0:32 |
| 4 | Data · Labels | 0:43 |
| 5 | Data · Generate | 0:59 |
| 6 | Data · Split | 1:16 |
| 7 | Train | 1:33 |
| 8 | Train · More options | 1:50 |
| 9 | Results · Verdict | 2:07 |
| 10 | Results · Failures | 2:19 |
| 11 | Results · Compare | 2:29 |
| 12 | Results · Try it | 2:41 |
| 13 | Ship · Package | 2:50 |
| 14 | Ship · Publish | 3:02 |
| 15 | Ship · Deploy | 3:10 |
| 16 | Ship · Monitor | 3:22 |
| 17 | Wrap | 3:34 |

---

## Pre-roll (not narrated)

Playwright logs in as `admin`, clicks the **Demo · Support FAQ** tile and
lands on the project Overview (`/project/<id>`). A few seconds of
seed-time before narration starts.

## Section 1 — Cold open, the Overview (0:00–0:18)

**On screen**: the Overview; spotlight the hero (where you are / next
step), then the four sidebar entries Data, Train, Results, Ship in turn.

> "Welcome to the Support FAQ walkthrough. We're taking the simplest of
> the three starter projects — twenty customer tickets with hand-written
> answers — and walking every page of the workspace: Data, Train,
> Results, Ship. No training, no generation, just a tour of what each
> page offers."

## Section 2 — Data · Sources (0:18–0:32)

**On screen**: sidebar **Data**; spotlight the five-step bar; open the
Details drawer's **Documents** section; click the first
`[data-testid^="expand-doc-"]`; spotlight the expanded row.

> "Data is five steps. Sources: twenty raw documents, one per ticket.
> Expand one — a question and the agent's answer. Imagine pasting
> thousands of resolved tickets here and you have the dataset for a real
> support assistant."

## Section 3 — Data · Prepare (0:32–0:43)

**On screen**: step **Prepare** (`stage-step-cleaning`); spotlight the form.

> "Prepare. For a messy corpus this is where you chunk long text, redact
> personal information, and score quality. This sample is already small
> and clean, so there is nothing to run."

## Section 4 — Data · Labels (0:43–0:59)

**On screen**: step **Labels** (`stage-step-goldset`); open the Details
section **Entries and human review**; spotlight the entries header (200
rows, locked), then the task-shape card.

> "Labels. The answer key: two hundred rows, locked. Each has a question,
> an expected answer, and a rationale. After training, the evaluation
> walks the whole key and reports the fraction the model got right. The
> task shape is confirmed here too."

## Section 5 — Data · Generate (0:59–1:16)

**On screen**: step **Generate** (`stage-step-synthetic`); spotlight the
generation form (provider = local Ollama). Nothing is run.

> "Generate. The lever that scales twenty source rows into thousands of
> training rows. A teacher model — local Ollama on this machine — reads
> your cleaned corpus and writes matching question and answer pairs. New
> rows land as pending until you review them. We're not running it
> here."

## Section 6 — Data · Split (1:16–1:33)

**On screen**: step **Split** (`stage-step-dataprep`); spotlight the
Train / Val / Test card, then the "Keep the current N test examples"
checkbox (`split-keep-test-label`).

> "Split. This is where the contract gets made. The adapter — question
> and answer pair — turns each row into the shape the trainer expects.
> The splits are written: sixteen train, two validation, two test. You
> can keep the test examples fixed when you add data, and the page
> watches for leakage between the splits."

## Section 7 — Train, the form (1:33–1:50)

**On screen**: sidebar **Train**; `[data-testid="train-form"]`. Spotlight
beats: base model + Browse → task + Change → Seeds radio → readiness line
→ Train button.

> "The Train page is one form. Base model, with Browse. The task shape,
> read-only, with Change. One run or three seeds — three picks the best
> by validation. A run name. The readiness line says whether the project
> can train right now, and the Train button starts it."

## Section 8 — Train · More options (1:50–2:07)

**On screen**: click **More options** (`train-form-more`); spotlight the
section list; open **LoRA and advanced training** (`train-section-lora`)
and spotlight it, then the **Seeds** and **Cloud burst** section headers.

> "Everything else is under More options. Open it and the sections
> appear: model and hyperparameters, LoRA and advanced training, seeds,
> checks, cloud burst, and training presets. The defaults work; the
> controls are there when you need them. Launching a run is a later
> video."

## Section 9 — Results · Verdict (2:07–2:19)

**On screen**: sidebar **Results**; tab **Verdict** (`results-tab-verdict`)
spotlighted, then the summary card ("nothing trained yet").

> "Results has four workflows. Verdict answers one question — did it
> work? With no run yet it says so. After a run it shows the lift over
> the base model, with evidence, and a button to evaluate the run."

## Section 10 — Results · Failures (2:19–2:29)

**On screen**: tab **Failures** (`results-tab-failures`); first content card.

> "Failures — where does it fail? The failing rows, the built-in checks,
> your custom checks, the pass/fail rules, and the failure clusters all
> live here."

## Section 11 — Results · Compare (2:29–2:41)

**On screen**: tab **Compare** (`results-tab-compare`); first content card
(external benchmarks / auto-RAG / student–teacher).

> "Compare — against other runs and other models: external benchmarks, a
> retrieval-only baseline, and a student-versus-teacher view. Pick two
> runs under Verdict and they appear side by side here."

## Section 12 — Results · Try it (2:41–2:50)

**On screen**: tab **Try it** (`results-tab-try`); the playground
(`results-playground`).

> "Try it — a chat playground to talk to the trained model, with the
> option to answer from your own documents instead. It stays empty
> until a run completes."

## Section 13 — Ship · Package (2:50–3:02)

**On screen**: sidebar **Ship**; stage **Package** (`ship-stage-package`);
the export form card.

> "Ship is four stages. Package: export the model — Hugging Face format,
> GGUF, ONNX — with a compress-first option and the project manifest.
> The releases list sits below it."

## Section 14 — Ship · Publish (3:02–3:10)

**On screen**: stage **Publish** (`ship-stage-publish`); the Hub-publish
card (empty — no export yet).

> "Publish: push a completed export to the Hugging Face Hub as a
> repository, model card included. Nothing to publish yet — no export
> exists."

## Section 15 — Ship · Deploy (3:10–3:22)

**On screen**: stage **Deploy** (`ship-stage-deploy`); serve / deploy plan
cards + deployments.

> "Deploy: serve plans and deployments — where the packaged model will
> run, and with what resources. The plans are drafted from the export,
> so there is nothing to deploy before one exists."

## Section 16 — Ship · Monitor (3:22–3:34)

**On screen**: stage **Monitor** (`ship-stage-monitor`); the observability
view (`ship-observability`).

> "Monitor: once a model is out, the timeline, production failures, and
> the bundles you shipped land here. What you learn feeds the next round
> of data — that closes the loop back to Data."

## Section 17 — Wrap (3:34–3:48)

**On screen**: sidebar **Overview**; spotlight the hero, then Data, Train,
Results, Ship in the sidebar as they are named.

> "That's the Support FAQ tour — four pages, nothing heavy run. Overview
> tells you where you are, Data gets the rows ready, Train is one form,
> Results says whether it worked, Ship gets it out. Next video walks the
> same shape for the PII Detector sample."

---

## Things to **not** say

- Don't say "we trained a model" — we didn't.
- Don't say "the demo has 6 gold rows" — that's the stale manifest
  prose. Say 200.
- Don't call the answer key a "gold set", the Results page a "tab", or
  mention the Coach bar / Dataset Prep / Tokenization / Training Config —
  they no longer exist in the UI.
- Don't read literal tech tokens aloud. The on-screen action shows
  them; the TTS engine will mispronounce them.

## Optional technical notes (background; not spoken)

- Workspace map: `frontend/src/utils/workspaceRoutes.ts`; Results tabs
  `?tab=verdict|failures|compare|try`; Ship stages
  `?stage=package|publish|deploy|monitor`.
- The `prepared-manifest` API at `GET /api/projects/<id>/prepared-manifest`
  returns adapter id, task profile, field mapping, output schema, and the
  train/val/test counts in one shot.
- The QA-pair adapter is one of the registered data adapters; see
  [backend/app/services/data_adapter_service.py](../../../backend/app/services/data_adapter_service.py).
- The deterministic split uses `random.seed(42)` by default in the
  seeder; manual splits via `POST /api/projects/<id>/dataset/split`
  can override.
