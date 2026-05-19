# Runtime Decisions

Decision date: 2026-05-19.

This file resolves the open runtime questions blocking Videos 08–12
of the series and documents the operational choices the recordings
will be made against. Once a video records against one of these
choices, that recording is the source-of-truth for the choice — this
file is upstream of the recordings.

## Resolution Table

| Open Q | Topic | Decision | Source |
|---|---|---|---|
| **Q16** | Training runtime | **Real Celery + external runtime** (default `TRAINING_BACKEND="external"` in `backend/app/config.py`). No simulated training in recordings. | Repo audit + operator choice 2026-05-19. |
| **Q17** | Smallest reliable training job per sample | **support-faq** for Video 09's training-run capture (16 train rows × 2 epochs ≈ 32 forward passes; finishes in seconds on GB10). PII and sentiment scaled-up runs deferred to later videos. | This file. |
| **Q18** | Redis/Celery required? | **Yes.** Real Celery means Redis + a Celery worker. Pre-flight commands documented below. | Codex's `04-recording-plan.md` runtime caveats; confirmed today. |
| **Q19** | Canonical Python env | The `backend/.venv` populated from `backend/requirements.txt`. No alternate env. | Repo audit; confirmed today. |
| **Q21** | Synthetic generation teacher | **Real endpoint** preferred; fall back to `ALLOW_SYNTHETIC_DEMO_FALLBACK=true` only if the endpoint isn't reachable at record time. The endpoint configured today is the local Ollama at `http://localhost:11434/v1` running `qwen2.5:7b-instruct-q4_K_M` (same model used for Q22 — Ollama serves multiple endpoints from one process). | Operator choice 2026-05-19. |
| **Q22** | LLM judge for eval | **Ollama, model `qwen2.5:7b-instruct-q4_K_M`**. Already running locally. Used for `evaluation/llm-judge` endpoints. | Operator choice 2026-05-19. |
| **Q23** | Compression path | **GGUF quantization via llama.cpp** (`backend/scripts/quantize.py` already supports this path). Reasons below. | Repo audit + operator choice "pick per your best knowledge". |
| **Q24** | Canonical export format for first recording | **GGUF**. Direct corollary of Q23 + Q25 — GGUF is what Ollama (Q25) consumes natively, and Q23's quantization output is GGUF. The recording pipeline lands at a quantized GGUF artifact that can be loaded directly by Ollama. | Derived from Q23 + Q25. |
| **Q25** | Local serve runtime | **Ollama**. Already running on `localhost:11434`. Used for both serve and judge (Q22). | Operator choice 2026-05-19. |

## Why GGUF is the right Q23/Q24 answer

- **One toolchain, one artifact**. Ollama's serve runtime (Q25)
  loads GGUF natively. Picking ONNX or HuggingFace as the export
  format would force a second runtime (ONNX Runtime or vLLM) just to
  prove the model works.
- **Already wired**. `backend/scripts/quantize.py` ships a real GGUF
  quantization path using llama.cpp's `quantize` binary; no stub
  involved. `backend/app/models/export.py` already lists `gguf` in
  the export-format enum.
- **Local-first**. Q19's canonical env is the project venv; GGUF
  quantization doesn't pull additional gigabytes of CUDA-specific
  wheels beyond what the trainer already needs.
- **Storyline alignment**. With local Ollama as the runtime + the
  judge, the demo loop becomes: train → quantize to GGUF → load in
  Ollama → smoke prompt → playground → LLM judge eval (also Ollama).
  Single tool, single trust boundary.

## Required local services before recording Videos 08–12

```bash
# 1. Redis for Celery's broker + result backend.
redis-server --daemonize yes
redis-cli ping     # → PONG

# 2. Celery worker (must be started from backend/ with .venv active).
cd backend
source .venv/bin/activate
celery -A app.worker.celery_app worker --loglevel=INFO --pool=threads --concurrency=2 &

# 3. Ollama with the chosen model pulled.
ollama serve &                                      # if not already running
ollama pull qwen2.5:7b-instruct-q4_K_M

# 4. Backend env additions in backend/.env:
cat >> backend/.env <<'EOF'
# Story-1.5/1.7 runtime defaults are fine; explicit overrides for recording:
TRAINING_BACKEND=external
TEACHER_MODEL_API_URL=http://localhost:11434/v1
TEACHER_MODEL_API_KEY=ollama
TEACHER_MODEL_NAME=qwen2.5:7b-instruct-q4_K_M
JUDGE_MODEL_API_URL=http://localhost:11434/v1
JUDGE_MODEL_API_KEY=ollama
JUDGE_MODEL_NAME=qwen2.5:7b-instruct-q4_K_M
EOF
```

> `TEACHER_MODEL_API_KEY=ollama` is a placeholder — Ollama doesn't
> validate the key, but the env variable being non-empty is what
> matters for the readiness banner to clear. Confirmed by reading
> `backend/app/config.py` defaults and the readiness check that fired
> in Codex's selector pass (`missing TEACHER_MODEL_API_KEY`).

## Video status updates after these decisions

`03-video-series-plan.md` modules previously gated by these
questions:

| Module | Old status | New status | Reason |
|---|---|---|---|
| 04 (Gold + Synthetic) | partial | **ready** | Q21 resolved with local Ollama endpoint. |
| 08 (Training Config) | partial | **ready** | Q16 resolved. No state change actually needed in the video since 08 stops at preflight, but it can now reference a known runtime. |
| **09 (Training Run)** | **blocked on Q16+Q17** | **ready** | Both decisions made. support-faq is the recording sample; Celery worker is the runtime. |
| 10 (Evaluation) | partial | **ready** | Q22 resolved (Ollama judge). |
| 11 (Compression / Export) | partial | **ready** | Q23 = GGUF quantization; Q24 = GGUF export. |
| 12 (Final Model Usage) | partial | **ready** | Q25 = Ollama serve. The full loop is now reachable. |

Modules 03, 05, 06, 07 (inspect-only sample walkthroughs) remain
unchanged at `partial` — they were already not blocked on these
decisions.

## The recording loop these decisions unlock

```
Train (Q16: Celery + external runtime, support-faq, 2 epochs)
   ↓
Eval against gold (Q22: Ollama qwen2.5:7b judge for LLM-judge mode)
   ↓
Compress (Q23: GGUF quantization via llama.cpp quantize binary)
   ↓
Export (Q24: GGUF format; export package includes manifest + serve.py)
   ↓
Serve (Q25: Ollama loads the GGUF; subprocess managed by serve_runtime_service)
   ↓
Smoke (curl from generated serve plan)
   ↓
Playground (UI call against the same Ollama endpoint)
```

Single trust boundary (local Ollama). Single artifact (GGUF). Single
worker (Celery → llama.cpp subprocess). This is the demo loop.

## Remaining open questions still affecting later videos

These weren't part of the 5 the operator resolved today; they're
either lower-priority or scope-creep for the demo series:

- **Q9** — eval handler dispatching for `support-faq`'s
  `instruction_sft` profile. Affects how Video 05's eval surface is
  narrated. Recommend resolving during the Video 09 actual training
  run — the eval handler picked at score time will answer this.
- **Q10** — does cleaning remove duplicates? Affects Video 03's
  cleaning narration. Recommend marking as "computes hashes; row
  removal unverified" and leaving narration accurate.
- **Q12** — classification-specific synthetic path. Affects Video 07.
  Recommend Video 07 narration explicitly say "the synthetic
  generator's classification path is not verified separately from
  the generic Q&A/conversation modes."
- **Q26–Q30** — final-model UI linkage details. Affect Video 12.
  Recommend resolving during the actual Video 12 recording — the
  playground model dropdown will answer them in real time.

## File deltas required to apply these decisions

Already in this commit:

- This file (`12-runtime-decisions-2026-05-19.md`) — new.
- `10-open-questions.md` — mark Q16/Q17/Q18/Q19/Q21/Q22/Q23/Q24/Q25
  resolved with link to this file.
- `03-video-series-plan.md` — flip status badges on modules 04,
  08–12.

To apply when recording actually starts:

- The four `backend/.env` lines under "Required local services"
  above. Already documented; not committed to the repo because
  `.env` is gitignored by design.

## Storyboard sequence after these decisions

If the operator can record one video per evening:

| Day | Video | Why this order |
|---|---|---|
| 1 | 02 quickstart | Smallest scope; tests recording pipeline. |
| 2 | 03 dataset lifecycle overview | Slide-heavy; low recording risk. |
| 3 | 04 gold + synthetic | First runtime-dependent video; tests Ollama wiring. |
| 4 | 05 support-faq walkthrough | First end-to-end sample tour. |
| 5 | 09 training run (support-faq) | First real training capture. |
| 6 | 10 eval (against day-5's experiment) | Captures the first F1 number from this series. |
| 7 | 11 compression + export | Lands on the first GGUF artifact. |
| 8 | 12 final model usage | Loop closes — Ollama serves the artifact. |
| 9+ | 06 PII pipeline, 07 sentiment pipeline, 13 BYO, 14 architecture | Can record in any order once the trained model exists. |
