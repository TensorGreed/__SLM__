# YouTube Per-Video Metadata — BrewSLM Series

Channel: **@TensorGreed** · Public · Audience: ML/AI engineers + indie hackers

Each entry includes:
- **Title** (≤60 chars where possible — YouTube truncates around 70 in mobile)
- **Description** (with GitHub link, channel link, timestamps, hashtags)
- **Tags** (comma-separated, ~480 chars max per video)
- **Thumbnail** filename (rendered at 1280×720)

YouTube imports the full description as-is. Copy each block into the "Description" field; the chapter timestamps render as clickable progress-bar markers.

---

## Video 1 — SLM 101

**Title**
```
Small Language Model 101: When to Use SLMs Over LLMs
```

**Description**
```
A short, opinionated intro to small language models. What "small"
actually means (it's a spectrum, not a number), why smaller models
beat big LLMs on narrow tasks, and the 11-stage lifecycle you'd
walk for any real project.

This is video 1 of an 11-part series that ends with a real
trained model serving predictions on local hardware. No cloud,
no API keys, no toy frameworks.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Full playlist: see channel
📡 More from @TensorGreed: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Cold open
00:11 What is a small language model?
00:31 Why smaller models matter
00:53 The SLM lifecycle
01:16 Where BrewSLM fits
01:37 Wrap & next up

#smallLanguageModel #LLM #FineTuning #LocalAI #MachineLearning
```

**Tags**
```
small language model, SLM, what is SLM, small vs large language model, fine tuning intro, LLM tutorial, machine learning lifecycle, local AI, on device AI, private AI, AI for indie hackers, BrewSLM, TensorGreed, ML engineering, neural network basics, AI tutorial, open source AI, model training, LoRA intro, Hugging Face
```

**Thumbnail**: `thumbnails/v01.png`

---

## Video 2 — BrewSLM Quickstart

**Title**
```
BrewSLM in 2 Minutes — Local Fine-Tuning Workspace Quickstart
```

**Description**
```
From a fresh repo clone to a fully-seeded demo project, in two
minutes. No training, no data prep — just login, click the demo
tile, and land on the Data tab with 20 raw rows + a 200-row gold
set already loaded.

The tile-driven seed flow is the fastest way to see what BrewSLM
actually does without committing your own dataset. Three official
demos ship in the repo: Support FAQ, PII Detector, Sentiment
Classifier.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:11 Login
00:20 Demo tiles
00:41 Seed Support FAQ
01:06 Cleaning tab
01:19 Gold Set tab
01:27 Dataset Prep tab
01:38 Training tab
01:44 Expand a row & wrap

#smallLanguageModel #BrewSLM #LocalLLM #FineTuning #AI
```

**Tags**
```
BrewSLM quickstart, local LLM workspace, fine tuning quickstart, small language model demo, AI workspace tutorial, SLM platform, train LLM on laptop, LoRA workspace, Hugging Face workspace, local AI demo, machine learning tutorial, AI for developers, open source LLM platform, TensorGreed, indie hacker AI, ML pipeline tour, no cloud AI
```

**Thumbnail**: `thumbnails/v02.png`

---

## Video 3 — Support FAQ Pipeline

**Title**
```
Fine-Tune an SLM on Support Tickets — Full Pipeline Walkthrough
```

**Description**
```
Walk a real fine-tuning pipeline end to end without launching any
training. Twenty customer-support tickets, a 200-row hand-labelled
gold set, dataset prep with a Q&A adapter, and a 16-train/2-val/2-
test split — already prepared by the demo seeder so you can see
what every stage actually surfaces.

Covers all 10 pipeline tabs (Data, Cleaning, Gold Set, Synthetic,
Dataset Prep, Tokenization, Training, Evaluation, Compression,
Export) plus the dedicated Training Config page. Real training
is Video 7 of this series.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:15 Data tab
00:34 Cleaning
00:55 Gold Set
01:12 Synthetic
01:29 Dataset Prep
01:48 Tokenization
02:04 Training Config
02:27 Evaluation & wrap

#smallLanguageModel #FineTuning #LoRA #BrewSLM #LLM
```

**Tags**
```
fine tune small language model, support FAQ SLM, customer support AI, instruction fine tuning, SFT tutorial, LoRA fine tuning, dataset pipeline, gold set evaluation, qa pair adapter, train LLM on tickets, support chatbot fine tuning, fine tuning workflow, BrewSLM pipeline, AI workspace, local LLM training, TensorGreed
```

**Thumbnail**: `thumbnails/v03.png`

---

## Video 4 — PII Detector Pipeline

**Title**
```
Build a PII Detector with a Small Language Model — End-to-End
```

**Description**
```
Train a span-level entity detector with a small language model.
The model takes a snippet of text and emits a structured JSON list
of every PII span it finds — email, phone, SSN, credit card,
ten entity types in total.

The headline beat: the disambiguation between two product features
that share the word "PII". Cleaning has a regex redaction step
that MASKS personal information before training. The PII Detector
model does the OPPOSITE — it FINDS personal information and emits
a structured output. Same word, very different feature.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:18 Data tab
00:32 Cleaning (PII disambiguation)
00:53 Gold Set
01:04 Synthetic (span mode)
01:25 Dataset Prep (span_set schema)
01:40 Tokenization
01:49 Training Config (LoRA for spans)
02:08 Evaluation & wrap

#PII #SpanExtraction #SmallLanguageModel #FineTuning #LLM
```

**Tags**
```
PII detection, PII detector AI, span level NER, span extraction LLM, structured extraction, PII span model, GDPR AI, LlamaFirewall, redaction model, entity extraction LLM, span set scoring, BrewSLM pipeline, fine tune LLM for NER, small language model PII, on device PII detection, TensorGreed
```

**Thumbnail**: `thumbnails/v04.png`

---

## Video 5 — Sentiment Classifier Pipeline

**Title**
```
Sentiment Classifier with a Small Language Model — Mobile CPU Target
```

**Description**
```
Train a three-way sentiment classifier (positive / neutral /
negative) sized for mobile CPU inference. Thirty source rows
balanced 10/10/10 across classes, a 200-row gold set with a slight
positive skew (70/65/65), and an ONNX-INT8 export path for on-
device deployment.

This is the simplest of the three sample task profiles — single
label per row, classification eval pack, per-class precision and
recall. The natural export target is ONNX with 8-bit quantization
for a small mobile footprint.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:14 Data tab (10/10/10 balance)
00:32 Gold Set (70/65/65 distribution)
00:50 Dataset Prep
01:06 Tokenization (mobile target)
01:20 Training Config (mobile_cpu)
01:36 Evaluation
01:49 Compression & Export (ONNX path)
02:07 Wrap

#SentimentAnalysis #SmallLanguageModel #ONNX #FineTuning #LLM
```

**Tags**
```
sentiment analysis LLM, sentiment classifier fine tuning, three way classification, mobile LLM, ONNX export LLM, on device sentiment, ONNX INT8 quantization, classification eval pack, mobile CPU inference, fine tune classifier, BrewSLM pipeline, small language model classification, edge AI, TensorGreed, ML pipeline
```

**Thumbnail**: `thumbnails/v05.png`

---

## Video 6 — BYO Custom Samples

**Title**
```
Bring Your Own Data: Train an SLM on Your Own CSV in 90 Seconds
```

**Description**
```
The three official demos are useful for learning the platform, but
the real point is to use your own dataset. This video shows the
shortest path from a CSV on your laptop to a project ready for
the rest of the pipeline.

Six-row coffee-shop FAQ as a stand-in for "your data." Create a
new project, upload the CSV, the file lands in ingestion. From
there the project goes through the same pipeline you saw on the
seeded samples (videos 3–5).

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:12 New project flow
00:23 Empty data tab
00:33 Upload CSV
00:52 Rows imported
01:11 Wrap

#BYOData #SmallLanguageModel #FineTuning #BrewSLM #LLM
```

**Tags**
```
bring your own data, BYO data LLM, train LLM on own CSV, custom dataset fine tuning, upload CSV to LLM, fine tune LLM with own data, small language model BYO, BrewSLM custom data, CSV to LLM, dataset import LLM, AI workspace BYO, indie AI training, TensorGreed
```

**Thumbnail**: `thumbnails/v06.png`

---

## Video 7 — Training Run (was V09 in the local file naming)

**Title**
```
Train a 135M LoRA in 12 Seconds — Real Celery, Real LoRA, Local GPU
```

**Description**
```
Launch a real training experiment. Tiny model (135M parameters),
two epochs, sixteen steps total. Celery worker on the local box,
PyTorch + transformers + PEFT under the hood. Twelve seconds wall
time on a GB10.

Not a simulated training loop. The output is a real LoRA adapter
on disk at data/projects/.../model/adapter_model.safetensors.
Final eval loss lands around 5 — high, because the dataset has
16 rows and the model has 135M params. The point is that the loop
fires end-to-end, not the quality of THIS model.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:17 Training Config recap
00:34 Kickoff (API)
00:46 Watching status: running
00:59 Completed with metrics
01:16 Wrap

#FineTuning #LoRA #Celery #SmallLanguageModel #LLMTraining
```

**Tags**
```
LoRA fine tuning, Celery worker LLM, train SmolLM, fine tune 135M model, PEFT tutorial, parameter efficient fine tuning, real training run, local GPU fine tuning, GB10 training, SFT loop, supervised fine tuning, LoRA adapter, BrewSLM training, fine tune in 12 seconds, fast LLM training, TensorGreed
```

**Thumbnail**: `thumbnails/v09.png`

---

## Video 8 — Evaluation (was V10)

**Title**
```
Evaluate a Fine-Tuned SLM Against a 200-Row Gold Set
```

**Description**
```
Score the trained model from the previous video against the
support-faq gold set. Twenty samples through the model, scored
against expected answers, aggregated. Real exact-match and
token-level F1 numbers. Real predictions in a side-by-side card.

Honest result: exact match lands at zero. F1 is in the low tens
of percent. The Auto-Gate fails on both required gates. That's
the right outcome for a 135M-param model trained on 16 rows —
the loop closing is what matters, not the score.

Resolves a real open question for the platform: the
instruction_sft task profile dispatches to the QA handler.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:11 Eval setup
00:29 Kickoff (POST /run-heldout)
00:38 Running eval on gold_dev
00:52 Results: Auto-Gate & predictions
01:08 Wrap

#LLMEvaluation #FineTuning #GoldSet #SmallLanguageModel #LLM
```

**Tags**
```
LLM evaluation, fine tuned model evaluation, gold set scoring, exact match scoring, F1 score LLM, eval against gold, SLM evaluation, BrewSLM evaluation, model scoring, evaluate fine tuned LLM, evaluation pipeline, sample predictions, eval handler, instruction fine tuning eval, TensorGreed
```

**Thumbnail**: `thumbnails/v10.png`

---

## Video 9 — Compression + Export (was V11)

**Title**
```
Compress an SLM to a 105 MB GGUF — Merge LoRA + llama.cpp Quantize
```

**Description**
```
Take a trained LoRA adapter and turn it into a quantized GGUF
artifact that Ollama can serve. Two steps. First merge the adapter
into the base model, producing a 256 MB half-precision checkpoint.
Then quantize via llama.cpp's quantize binary down to 4-bit Q4_K_M
— roughly 105 MB. Twelve seconds plus five seconds on a GB10.

Surfaces two real bugs that get fixed in this commit: a venv-
symlink-resolution issue in quantize.py that broke the
transformers import, and a missing sentencepiece dependency in
the backend venv. Both flagged in the recording plan for anyone
reproducing this.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:15 Compression form
00:31 Merge LoRA + quantize
00:49 GGUF on disk (105 MB)
01:03 Export tab
01:18 Export registered
01:31 Wrap

#GGUF #LoRAMerge #ModelCompression #llamacpp #SmallLanguageModel
```

**Tags**
```
GGUF quantization, llama.cpp quantize, merge LoRA into base model, model compression LLM, Q4_K_M quantization, 4 bit quantization, fine tuned model export, LoRA to GGUF, deploy fine tuned LLM, compress LLM 100MB, on device LLM compression, BrewSLM export, GGUF export, fine tuning compression, TensorGreed
```

**Thumbnail**: `thumbnails/v11.png`

---

## Video 10 — Final Model Usage (was V12)

**Title**
```
Serve a Fine-Tuned SLM with Ollama — Closing the Loop
```

**Description**
```
The compressed GGUF from the previous video gets loaded into
Ollama and answers a prompt through the OpenAI-compatible API.
End-to-end loop closes here: we trained the model, we evaluated
it, we compressed it, and now we serve it.

Real response from the trained model. Format is correct (numbered
support-ticket steps). Substance is generic — the 135M-param model
trained on 16 rows hasn't seen this company's actual password-
reset flow. Honest about what works and what doesn't.

Same loop scales to bigger samples, more rows, bigger base models.
That's the value proposition this 8-episode runtime arc proves.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:12 Ollama create from GGUF
00:29 Playground config
00:45 Send prompt
00:58 Model responds
01:14 Series wrap

#Ollama #FineTunedLLM #SmallLanguageModel #LLMServing #LocalLLM
```

**Tags**
```
Ollama tutorial, serve fine tuned LLM, Ollama GGUF, OpenAI compatible Ollama, serve LLM locally, deploy small language model, Ollama playground, fine tuned model inference, local inference LLM, BrewSLM playground, ollama create modelfile, end to end LLM, TensorGreed
```

**Thumbnail**: `thumbnails/v12.png`

---

## Video 11 — Architecture (was V14)

**Title**
```
BrewSLM Architecture: 5 Processes, 1 Machine, Local-First AI
```

**Description**
```
What makes the platform work under the hood. Five processes:
FastAPI backend, React frontend, Celery worker, Redis broker,
Ollama inference runtime. The data flow from raw rows to a
served model. Where each piece runs (GPU vs CPU vs disk). The
two trust boundaries the design draws.

This is the closer for the playlist. Watch this if you want to
understand WHY the rest of the series works the way it does —
or if you're considering using BrewSLM as a starting point for
your own local-first AI workspace.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Chapters:
00:00 Intro
00:06 The stack: 5 processes
00:30 Data flow: ingest → serve
00:51 Where things run
01:13 Trust boundaries
01:34 Wrap

#SystemDesign #AIArchitecture #LocalAI #BrewSLM #MachineLearning
```

**Tags**
```
AI architecture, LLM platform architecture, local AI stack, FastAPI Celery Redis Ollama, AI system design, machine learning platform, ML engineering stack, BrewSLM architecture, on prem AI, private AI infrastructure, AI workspace design, TensorGreed, software architecture LLM, AI engineering
```

**Thumbnail**: `thumbnails/v14.png`
