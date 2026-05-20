# Compression + Export — Narration

Status: **synced** with the actual narrated take produced by
`tts/generate_v11_narration.py` (Orpheus voice "leo") on 2026-05-20.

The **Python script** at
[tts/generate_v11_narration.py](../../../tts/generate_v11_narration.py)
is the **authoritative source** of the spoken text. This file
mirrors the same text plus stage directions / Playwright cues. Edit
the script first.

Total runtime: **1:43** (matches
`docs-demo/recordings/raw/11-compression-export-narrated.mp4`).
Section timings come from `tts/audio/v11-durations.json`.

Companion to:
[docs-demo/videos/11-compression-export/recording-plan.md](../../videos/11-compression-export/recording-plan.md).

Third runtime-dependent video. The first two (V09 + V10) produced a
trained adapter and an evaluation score; V11 turns that adapter into
a quantized, deployable artifact.

---

## Pre-roll (not narrated)

Playwright logs in as admin, opens the **Demo · Support FAQ**
project, clicks the **Compression** tab. The spec asserts V09's
`v09-narrated-run` experiment exists and is completed before
continuing.

## Section 1 — Cold open (0:00–0:16)

**On screen**: Compression tab loaded; empty form.

> "Now we ship. The training run produced a LoRA adapter — a small
> set of weight deltas that ride on top of the base model. To
> actually deploy, we merge those deltas back into the base, then
> quantize the merged weights down to four bits so the model is
> small and fast enough to serve on modest hardware."

## Section 2 — Compression setup (0:16–0:32)

**On screen**: Compression Engine form filled in:
- Model Path: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Quantization Bits: `4-bit`
- Output Format: `GGUF`
- LoRA Adapter Path: V09's model directory

> "Compression tab. Two settings matter here: quantization bits —
> four for a tight artifact, eight for higher quality — and output
> format. GGUF is the one we want because Ollama loads it natively
> in the next video. The LoRA adapter path is filled in from the
> experiment."

## Section 3 — Compress run (0:32–0:50)

**On screen**: spec fires `POST /compression/merge-lora` (~12s),
then `POST /compression/quantize` (~5s). The UI form stays visible
while the work happens in the background (the panel only renders
its Logs/Result cards for UI-driven runs).

> "I'm kicking off both steps. First, merge LoRA — the trained
> adapter folds into the base model, producing a full half-
> precision checkpoint. Then quantize — llama.cpp converts that
> checkpoint into GGUF and then crunches it down to four-bit. The
> whole pipeline takes about twenty seconds on this hardware."

## Section 4 — Compression result (0:50–1:04)

**On screen**: hold on the Compression tab. The narration carries
the size numbers; the artifact itself is on disk at
`data/projects/<id>/compressed/quantized_4bit.gguf`.

> "Done. The merged half-precision model was around two hundred
> and fifty megabytes. After quantization the GGUF is roughly one
> hundred megabytes — under half the size, ready to load on a
> phone-class CPU. The file is on disk in the project's compressed
> directory."

## Section 5 — Export create (1:04–1:19)

**On screen**: click the **Export** tab. The "Export and Registry"
panel renders with:
- Experiment dropdown
- Format (GGUF (CPU) selected)
- Quantization (4-bit selected)
- Deployment Targets grid — GGUF Exporter + Ollama Runner pre-selected

> "Now the Export tab. This is where we register the artifact
> against the experiment and pick deployment targets. Format: GGUF.
> Quantization: four-bit. The recommended deployment target for
> this combination is Ollama — that's Video Twelve."

## Section 6 — Export run (1:19–1:32)

**On screen**: spec fires `POST /export/create` then `POST
/export/{id}/run`. Page reloads, Export tab re-renders with the new
export row in the Export History table at the bottom.

> "Running the export. It validates the GGUF artifact, writes a
> manifest with the model hash and the deployment plan, and
> registers everything against the experiment. The result is a
> packaged export that downstream serving can pick up."

## Section 7 — Wrap (1:32–1:43)

**On screen**: hold on the Export tab with the registered export.

> "That's compression and export. We started with a LoRA adapter
> from training, and we end with a quantized GGUF file and a
> registered export manifest. Next video loads this artifact in
> Ollama and actually serves a prediction."

---

## Things to **not** say

- Don't claim production readiness. The 135M-param model trained
  on 16 rows isn't shipping anywhere serious.
- Don't read `Q4_K_M` as letters/numbers aloud. Say "four-bit" —
  the recording plan keeps the literal `Q4_K_M` only for technical
  context, not narration.
- Don't quote specific byte counts; the narration uses rounded
  numbers ("around 250 MB", "roughly 100 MB") because file sizes
  vary slightly per re-run.
- Don't say "Ollama is serving it now" — that's the next video.

## Optional technical notes (background; not spoken)

- The export pipeline isn't fully wired for the case where the
  GGUF artifact lives outside the export's own `model/` directory.
  The recording's export-run step may show a validation warning;
  for the narrative this is fine because the artifact + manifest
  are both on disk and the next video reads them directly.
- The actual quantization tool is `llama-quantize` from llama.cpp,
  invoked via `backend/scripts/quantize.py`. The script also runs
  `convert_hf_to_gguf.py` to produce the intermediate FP16 GGUF
  before quantizing.
- Why Q4_K_M specifically: it's the canonical 4-bit preset for
  most modern Llama-family GGUF quantizations, balancing size with
  reasonable inference quality. Q2_K is smaller but quality drops
  sharply at this scale.

## Two pre-recording fixes captured here for reproducibility

These tripped up the dry-run and are documented in the recording
plan's "Two subtle bugs" section, but worth flagging here too for
anyone running the spec on a fresh machine:

1. The `quantize.py` patch in this commit fixes a venv-symlink
   resolution bug. Without it, `PYTHON_EXECUTABLE` gets resolved to
   the system interpreter (`/usr/bin/python3.12`), which doesn't
   have the venv's `transformers` package.
2. `sentencepiece` needs to be installed in the backend venv
   (`backend/.venv/bin/pip install sentencepiece`). It's not in
   `requirements.txt` because the rest of the training stack
   doesn't import it — only `convert_hf_to_gguf.py` does, lazily,
   for the Llama tokenizer path.
