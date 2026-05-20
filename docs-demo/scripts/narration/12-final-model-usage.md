# Final Model Usage — Narration

Status: **synced** with the actual narrated take produced by
`tts/generate_v12_narration.py` (Orpheus voice "leo") on 2026-05-20.

The **Python script** at
[tts/generate_v12_narration.py](../../../tts/generate_v12_narration.py)
is the **authoritative source** of the spoken text. This file
mirrors the same text plus stage directions / Playwright cues.
Edit the script first.

Total runtime: **~1:36** (audio 96.4s; muxed close to that).
Section timings come from `tts/audio/v12-durations.json`.

Companion to:
[docs-demo/videos/12-final-model-usage/recording-plan.md](../../videos/12-final-model-usage/recording-plan.md).

Closes the runtime arc. The model we trained in V09, evaluated in
V10, and compressed in V11 finally gets served by Ollama and
responds to a prompt through the Playground UI.

---

## Pre-roll (not narrated)

The Playwright spec:
1. Asserts V11's `quantized_4bit.gguf` exists.
2. Stages it under `/tmp/v12-ollama/` with a one-line Modelfile.
3. Runs `ollama create slm-supportfaq -f Modelfile` via
   Node's `child_process` (~0.2s; Ollama indexes existing bytes).
4. Logs in as admin, opens the support-faq project, navigates to
   the **Playground** sidebar entry.

By the time narration starts, the Playground is loaded with an
empty messages area.

## Section 1 — Cold open (0:00–0:13)

**On screen**: Chat Playground panel; empty messages area.

> "Last step. We trained, we evaluated, we compressed. The GGUF
> artifact is on disk. Now we serve. Ollama loads the model, the
> Playground sends a prompt, and the trained model actually
> responds. Loop closed."

## Section 2 — Ollama register (0:13–0:30)

**On screen**: hold on the Playground form area. The narration
explains the `ollama create` step the spec ran in pre-roll.

> "First, register the artifact with Ollama. The Playwright spec
> runs the ollama create command in the background — it points at
> the GGUF file via a tiny Modelfile and publishes the model under
> a friendly alias. Takes a fraction of a second; Ollama is just
> indexing the bytes it already has on disk."

## Section 3 — Playground setup (0:30–0:46)

**On screen**: spec selects Provider dropdown → "OpenAI-Compatible
/ Ollama"; fills Model input with `slm-supportfaq`.

> "Open the Playground. Provider is OpenAI-Compatible, which
> covers Ollama's compatibility endpoint on port one one four
> three four. Model name is the alias we just created.
> Temperature low, max tokens enough for a short reply. That's
> all the configuration this needs."

## Section 4 — Send prompt (0:46–0:59)

**On screen**: prompt textarea filled with "How do I reset my
password?"; Send button clicked.

> "Now the prompt. I'll ask the model the kind of question the
> training set covered — how to reset a password. The trained
> model has seen sixteen rows of customer support tickets. Not
> enough to be excellent, but enough to produce the right shape
> of answer."

## Section 5 — Response (0:59–1:15)

**On screen**: assistant response streams into the messages area.
The dry-run captured 162 tokens in 13.5s, output:
> "To reset your password, you'll need to take the following
> steps:
> 1. Visit a website that allows users to reset their password.
> 2. Enter your new password and login credentials into the
>    website.
> 3. Follow the on-screen prompts or guide provided by the
>    website's administrator to confirm your actions."

> "And there it is. A coherent, numbered, support-ticket style
> answer. It's not factually grounded in this company's real
> password-reset flow — the model has never seen one. But the
> format is correct, the tone is correct, and the loop fired end
> to end on a hundred and five megabytes of quantized weights."

## Section 6 — Wrap (1:15–1:36)

**On screen**: hold on the response.

> "That's the full SLM platform demo. Eight videos. We started
> with raw customer tickets, walked the dataset pipeline, trained
> a tiny model with real Celery, scored it against gold,
> compressed the LoRA into GGUF, and served the result through
> Ollama. Same shape works on the PII detector and sentiment
> samples, scales up to real datasets, and runs entirely on
> local hardware."

---

## Things to **not** say

- Don't claim this is production-ready. 135M params + 16 train
  rows; the answer's format is correct, substance is generic.
- Don't read literal URLs aloud — say "OpenAI-compatible
  endpoint on port one one four three four" rather than
  `http://localhost:11434/v1/chat/completions`.
- Don't say Ollama trained anything. Ollama serves; Celery
  trained.
- Don't promise specific response text — the actual generation
  can vary slightly per re-run even with temperature=0.

## Why the response quality is what it is

The model is **HuggingFaceTB/SmolLM2-135M-Instruct** with a LoRA
adapter trained on **16 rows** of support tickets, for **2 epochs**.
That's enough to:
- Pick up the format conventions (numbered steps, support tone).
- Show that the SFT loop works end-to-end on this hardware.

It is NOT enough to:
- Produce factually correct, dataset-specific answers.
- Beat the base model on real benchmark sets.
- Anything resembling production-grade output.

Scaling up — different sample (PII / Sentiment), more rows
(synthetic generation in V04), bigger base (Qwen2.5-7B), more
epochs (configurable per Story 1.7 reset/resume) — works through
the same pipeline shown across V09–V12. That's the headline of
this whole eight-video arc.
