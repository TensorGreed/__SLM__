# YouTube Playlist Metadata — BrewSLM Series

Channel: **@TensorGreed**
Author: Anurag Jain
Public, fully indexed.

---

## Playlist Title

**Build & Serve Small Language Models Locally — BrewSLM End-to-End**

**Why this title:**
- Front-loads the buyer intent ("build & serve") + the canonical search term ("small language models") + the unique value prop ("locally").
- Names the platform ("BrewSLM") so direct-brand searches land here.
- "End-to-End" tells viewers it's a complete series, not a single tutorial.
- 65 characters — under YouTube's 100-char cap, fully visible in playlist pages.

**Alternative if you want something punchier:**
> Fine-Tune & Serve Small Language Models on Your Own Hardware

---

## Playlist Description

```
Eleven short videos that take you from "what is a small language model"
to a fine-tuned model serving real predictions through Ollama — all on
local hardware, no cloud dependencies.

BrewSLM is an open workspace for the full SLM lifecycle: data
ingestion, cleaning, gold sets, synthetic generation, dataset prep,
tokenization, training (real Celery + LoRA), evaluation against gold,
GGUF compression via llama.cpp, and final serving through Ollama.

Each video is 1–3 minutes. No filler. Every claim is backed by a real
artifact on disk: a LoRA adapter, a 105 MB GGUF file, a predicted
response from the trained model.

Built for: ML engineers, AI/ML practitioners, and indie hackers who
want a local-first alternative to cloud fine-tuning platforms.

🔗 Source code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

What's covered:
• Foundations — what an SLM is, when it beats an LLM
• Three demo pipelines — Support FAQ, PII Detector, Sentiment Classifier
• BYO data — upload your own CSV and walk the pipeline
• Real training — Celery worker, LoRA adapter, 16 steps in 12 seconds
• Real evaluation — score against a 200-row hand-labelled gold set
• Compression — merge LoRA + GGUF quantize via llama.cpp
• Serving — Ollama loads the GGUF and answers a prompt

Stack: FastAPI · React · Celery · Redis · Ollama · PyTorch + LoRA · llama.cpp
Hardware: NVIDIA GB10 (any CUDA GPU works; CPU paths exist too)

#smallLanguageModel #LocalAI #FineTuning #LoRA #LLM #MachineLearning #OpenSourceAI
```

---

## Playlist Tags

Paste these comma-separated into YouTube's playlist tags field. 480 characters / 25 tags — comfortably under the limit.

```
small language model, SLM, fine tuning LLM, LoRA fine tuning, local LLM, private AI, on device AI, GGUF, llama.cpp, Ollama tutorial, fine tune small model, train LLM locally, open source AI platform, AI workspace, ML engineering, dataset pipeline, model compression, LLM evaluation, gold set evaluation, Hugging Face SmolLM, BrewSLM, TensorGreed, machine learning tutorial, AI tutorial, neural network fine tuning
```

---

## Playlist Thumbnail (cover image)

`docs-demo/youtube/thumbnails/playlist-cover.png` — 1280×720, same dark CRT aesthetic as individual video thumbnails. Use this as the playlist's representative thumbnail (YouTube auto-picks from videos, but you can override under playlist settings → image).

---

## Video Order (the canonical sequence)

| # | YouTube position | Local file | Length |
|---|---|---|---|
| 1 | 1 | `01-slm-101-narrated.mp4` | 1:55 |
| 2 | 2 | `02-brewslm-quickstart-narrated.mp4` | 2:10 |
| 3 | 3 | `03-support-faq-pipeline-narrated.mp4` | 2:45 |
| 4 | 4 | `04-pii-detector-pipeline-narrated.mp4` | 2:26 |
| 5 | 5 | `05-sentiment-classifier-pipeline-narrated.mp4` | 2:26 |
| 6 | 6 | `06-byo-custom-samples-narrated.mp4` | 1:24 |
| 7 | 7 | `09-training-run-narrated.mp4` | 1:26 |
| 8 | 8 | `10-evaluation-narrated.mp4` | 1:17 |
| 9 | 9 | `11-compression-export-narrated.mp4` | 1:43 |
| 10 | 10 | `12-final-model-usage-narrated.mp4` | 1:36 |
| 11 | 11 | `14-architecture-narrated.mp4` | 1:48 |

Total: ~21 minutes. Watch-through completion is realistic at this length — a viewer can consume the whole series in one sitting.

**Note on numbering:** The local file names skip 7, 8, and 13. That's intentional — those videos were planned but redundant with the videos before/after. YouTube viewers won't see the gaps; the playlist sequence is 1-11 visually.

---

## Privacy Review Before Publishing Public

The recorded videos sometimes show personal paths in URL bars and screen content. Before clicking "Public":

1. **File paths**: `/home/anuragj/Desktop/GitHub/__SLM__/` appears as the project directory in several frames (V02 onwards, in the workspace screen). This leaks your home directory + username.
   - **Impact**: Low. It's a Linux home dir, not credentials.
   - **Mitigation**: Acceptable. If you want, re-record with the repo at `~/brewslm/` or similar — but that's a heavy re-take.

2. **Git author**: `anugram@…` may appear if any frame shows DevTools or a terminal. Scan V09–V12 (which have the most terminal-adjacent UI).
   - **Impact**: Low — only if you don't want that handle public.

3. **Local IPs**: `localhost:8000`, `localhost:5173`, `localhost:11434` are visible. These are loopback addresses; safe.

4. **API key in narration**: V02's narration mentions "the local development token in the backend env file" without speaking the literal value. The on-screen password field is masked. Safe.

5. **Lab Journal achievements**: The XP/level chip is visible in many frames. This is a feature, not sensitive.

**Recommendation**: Publish as-is. The file-path exposure is the only real signal and it's at the level of "this person owns a Linux laptop" — not exploitable.
