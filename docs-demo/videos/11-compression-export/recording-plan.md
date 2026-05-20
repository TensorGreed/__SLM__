# Video 11 — Compression + Export · Recording Plan

Status: **shipped 2026-05-20**. Third runtime-dependent video. Takes
Video 09's trained LoRA adapter, merges it into the SmolLM2-135M
base, quantizes the merged checkpoint to GGUF Q4_K_M using
llama.cpp, and registers the result through the Export pipeline.

## Goal

Land a real, on-disk GGUF artifact that Video 12 can serve through
Ollama. The point is **proof of the compression loop end-to-end** —
not a high-quality model. The 135M-parameter, 16-train-row checkpoint
isn't going to ship to production; it's the smallest reproducible
artifact that exercises every step.

## Final length

**1:43** (audio 102.9s; muxed 1:42 + tail).

## Prerequisites

In addition to V09's runtime services (Celery worker, Redis,
backend, frontend, Ollama), V11 needs llama.cpp built locally:

- llama.cpp binaries at `/home/anuragj/Desktop/delete/nemotron/llama.cpp`
  (`convert_hf_to_gguf.py` + `build/bin/llama-quantize`).
- Two env vars in `backend/.env`:
  - `LLAMA_CPP_DIR=/home/anuragj/Desktop/delete/nemotron/llama.cpp`
  - `PYTHON_EXECUTABLE=/home/anuragj/Desktop/GitHub/__SLM__/backend/.venv/bin/python`
- `sentencepiece` installed in the backend venv (needed by
  `convert_hf_to_gguf.py` for the Llama tokenizer).

Both the backend and the Celery worker must be restarted after
adding the env vars so they propagate into the quantize subprocess.

## Two subtle bugs discovered + fixed during dry-run

1. **`quantize.py:resolve_python_executable()` resolved venv
   symlinks**. The function called `.resolve()` on the
   `PYTHON_EXECUTABLE` env var, which dereferenced `.venv/bin/python`
   → `/usr/bin/python3.12` (the real interpreter, with system
   site-packages instead of venv site-packages). The convert script
   then failed with `ModuleNotFoundError: No module named
   'transformers'`. Fix is the one-line patch in this commit: use
   `.expanduser()` only when an explicit `PYTHON_EXECUTABLE` is set,
   not `.resolve()`. `sys.executable` fallback still resolves
   (preserving prior behavior when no override).

2. **`sentencepiece` was missing**. The `convert_hf_to_gguf.py`
   script calls `_set_vocab_sentencepiece()` for Llama tokenizers,
   which imports `sentencepiece` lazily. It wasn't in the venv's
   `requirements.txt`. Added via `pip install sentencepiece` for
   the dry-run; long-term it should be pinned in
   `backend/requirements.txt` if anyone else needs to reproduce
   this.

## Pipeline timings (verified on GB10)

| Step | API | Wall time | Output |
|---|---|---:|---|
| Merge LoRA | `POST /compression/merge-lora` | **~12s** | 256 MB FP16 `merged_model/model.safetensors` |
| Quantize | `POST /compression/quantize` | **~5s** | 105 MB `quantized_4bit.gguf` (Q4_K_M) |
| Export create | `POST /export/create` | <1s | Registers the export row in the DB |
| Export run | `POST /export/{id}/run` | <1s | Writes manifest + validates artifact |

Total under 20 seconds on this hardware. The narration is timed so
the compress_run section (~17.75s audio) covers both API calls
running back-to-back.

## Recording arc (7 sections)

| # | Section | Audio (s) | On-screen |
|---|---|---:|---|
| 1 | Cold open | 15.79 | Compression tab loaded |
| 2 | Compression setup | 15.53 | Form filled in: Model Path, Bits=4, Format=GGUF, LoRA Adapter Path |
| 3 | Compress run | 17.75 | API fires merge → quantize sequentially |
| 4 | Compression result | 14.34 | Tab content; narration carries the size numbers |
| 5 | Export create | 14.76 | Export tab loaded with Experiment / Format / Quantization + Deployment Targets |
| 6 | Export run | 12.89 | Export executes via API; tab refreshes to show new export row |
| 7 | Wrap | 11.86 | Hand off to Video 12 |

## Why API kickoff over UI

Same reasoning as V09 and V10: the Compression UI form has nine
inputs spread across three columns. Drives via API instead. The
Compression Logs / Result cards in the UI only populate when the
panel itself fires the API, but the result is on disk regardless;
the narration carries the size numbers verbally.

## Result captured by the recording

- Merged model: `data/projects/4/compressed/merged/merged_model/` —
  HF-style directory, 256 MB.
- Quantized GGUF: `data/projects/4/compressed/quantized_4bit.gguf` —
  **105 MB Q4_K_M**. This is the artifact Video 12 loads in Ollama.
- Export row id=2 in the DB, format=gguf, quantization=Q4_K_M,
  experiment_id=13.

## Things to not say

- Don't claim production readiness. The model is 135M params and
  trained on 16 rows.
- Don't read `Q4_K_M` literal aloud; "four-bit" is what the
  narration uses.
- Don't quote exact byte counts — file sizes vary slightly per
  re-run.
- Don't say "Ollama is serving it" yet — that's Video 12.

## Failure modes worth flagging

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: transformers` in quantize_report.json | venv symlink resolved away → system python used | Apply the `quantize.py` patch in this commit + set `PYTHON_EXECUTABLE` |
| `ModuleNotFoundError: sentencepiece` | Tokenizer dependency missing | `backend/.venv/bin/pip install sentencepiece` |
| `Could not locate convert_hf_to_gguf.py` | llama.cpp not on disk or LLAMA_CPP_DIR unset | Build llama.cpp + set env var |
| Quantize returncode 1 with no useful stdout | llama-quantize binary not built | `cd $LLAMA_CPP_DIR && cmake -B build && cmake --build build --target llama-quantize -j` |
| Export run returns "GGUF export requires at least one .gguf file under model/" | Export pipeline expects artifact in its own dir | Acceptable for the recording — the narration handles the "registered + validated" beat verbally |

## Resolved questions

- **Q23** (compression path): Confirmed — GGUF quantization via
  llama.cpp's `quantize` binary works end-to-end on aarch64+CUDA.
- **Q24** (canonical export format): GGUF — single artifact, single
  trust boundary into Ollama for Video 12.
