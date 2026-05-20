# Video 12 — Final Model Usage · Recording Plan

Status: **shipped 2026-05-20**. Closes the runtime arc. Loads V11's
GGUF artifact into Ollama and sends a real prompt through the
BrewSLM Playground UI. The trained model responds end-to-end.

## Goal

Prove the **full loop closes**. We have a checkpoint on disk
(V09), a measured evaluation score (V10), a compressed GGUF
artifact (V11) — and now an actual prediction from a served model
through the Playground UI. The viewer sees a coherent answer to
"How do I reset my password?", which is the kind of question the
trained checkpoint was optimized for.

## Final length

**~1:36** (audio 96.4s; muxed close to that).

## Prerequisites

In addition to V11's runtime services:

- Ollama daemon already running (used for the teacher model since
  V04; same process serves the trained model here).
- V11's `quantized_4bit.gguf` artifact exists at
  `data/projects/4/compressed/quantized_4bit.gguf`. Spec asserts
  this before running.
- No new environment changes required.

## What the spec actually does

The pre-roll stages the GGUF into `/tmp/v12-ollama/` and writes a
single-line Modelfile (`FROM ./model.gguf`), then runs `ollama
create slm-supportfaq -f Modelfile` via Node's `child_process`.
Ollama indexes the existing bytes (no copy, no quantization re-
run), so registration takes ~0.2s.

The recording itself drives the Playground UI:
- Provider: `OpenAI-Compatible / Ollama`
- Model: `slm-supportfaq` (the Ollama alias just created)
- Prompt: "How do I reset my password?"
- Click Send → wait for response → screenshot

## Pipeline timings (verified)

| Step | Wall time | Notes |
|---|---:|---|
| `ollama create` | **~0.2s** | Ollama indexes existing bytes |
| Playground form fill | ~2s | Provider + Model + temp |
| Send + first-token latency | ~5–10s | Model load on first invocation |
| Full response (~160 tokens) | ~13s | bf16-on-CUDA equivalent for GGUF |

The dry-run produced:

> "To reset your password, you'll need to take the following steps:
> 1. Visit a website that allows users to reset their password.
> 2. Enter your new password and login credentials into the website.
> 3. Follow the on-screen prompts or guide provided by the website's
>    administrator to confirm your actions."

162 tokens in 13.5s on GB10 via Ollama's OpenAI-compatible API.

## Recording arc (6 sections)

| # | Section | Audio (s) | On-screen |
|---|---|---:|---|
| 1 | Cold open | 12.97 | Playground tab, empty messages area |
| 2 | Ollama register | 16.90 | Hold on Playground; narration explains the `ollama create` step that the spec ran in pre-roll |
| 3 | Playground setup | 16.04 | Provider dropdown → OpenAI-Compatible; Model input → `slm-supportfaq` |
| 4 | Send prompt | 12.71 | Type "How do I reset my password?"; click Send |
| 5 | Response | 16.30 | Streamed response renders in messages area |
| 6 | Wrap | 21.50 | Series-closing summary; hand-off message |

## Honest about the response quality

The model is 135M parameters trained on **16 rows**. The answer is
generic — it doesn't reference any specific company's password
flow because the training data doesn't contain one. Format is
correct (numbered steps, support-ticket tone), substance is weak.
Narration acknowledges this directly: "not factually grounded in
this company's real password-reset flow — the model has never seen
one. But the format is correct, the tone is correct."

This is the honest take. Selling a 135M+16-row model as production-
ready would be a lie. Selling "we proved the loop fires end-to-end"
is the actual win.

## Things to not say

- Don't claim this is a working production model.
- Don't read literal API URLs aloud (`http://localhost:11434/v1/chat/completions`).
- Don't describe Ollama as something that "trained" the model — it
  serves; the training was done by Celery in V09.

## Cross-references

- V09 produced the LoRA adapter (`experiment_id=13`).
- V11 produced the GGUF (`quantized_4bit.gguf`, Q4_K_M, 105 MB).
- V12 produces the served-model prediction.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ollama: command not found` | Ollama not in PATH | `which ollama`; ensure systemd/launchd has it |
| `ollama create` succeeds but Playground shows no response | The Playground might be configured with a different `api_url`. Default is `http://localhost:11434/v1/chat/completions` (OpenAI-compatible). | Set Provider explicitly to `openai_compatible`; check Ollama API at `:11434/api/tags` |
| Response is gibberish | Q4_K_M quantization too aggressive for 135M weights | Re-export at Q8_0 (smaller compression ratio) |
| Response is too slow | First-token latency on first invocation includes model load | Acceptable for this take; subsequent prompts in the same session are faster |
