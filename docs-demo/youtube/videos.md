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
tile, and land on the project Overview with the sample's tickets and
a 200-row answer key already loaded.

The tile-driven seed flow is the fastest way to see what BrewSLM
actually does without committing your own dataset. Three official
demos ship in the repo: Support FAQ, PII Detector, Sentiment
Classifier.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Cold open
00:12 Login
00:24 Starter tiles
00:45 Seed Support FAQ & the Overview
01:09 Data · Sources
01:27 Data · Prepare
01:43 Data · Labels
01:55 Data · Split
02:06 Train
02:23 Wrap

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
Walk a real fine-tuning workspace end to end: the Support FAQ sample's
tickets and 200-row answer key, the five Data steps (Sources, Prepare,
Labels, Generate, Split), the Train form, the four Results workflows
(Verdict, Failures, Compare, Try it) and the four Ship stages (Package,
Publish, Deploy, Monitor). Real training is Video 7 of this series.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Overview
00:18 Data · Sources
00:32 Data · Prepare
00:43 Data · Labels
00:59 Data · Generate
01:16 Data · Split
01:33 Train
01:50 Train · More options
02:07 Results · Verdict
02:19 Results · Failures
02:29 Results · Compare
02:41 Results · Try it
02:50 Ship · Package
03:02 Ship · Publish
03:10 Ship · Deploy
03:22 Ship · Monitor
03:34 Wrap

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

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Intro
00:25 Data · Sources
00:44 Prepare (PII disambiguation)
01:08 Labels (answer key)
01:29 Generate (span mode)
01:50 Split (span set)
02:11 Train · Tokenization
02:25 Train · LoRA for spans
02:48 Results & wrap

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
balanced 10/10/10 across classes, a 200-row answer key with a slight
positive skew (70/65/65), and an ONNX-INT8 export path for on-
device deployment.

This is the simplest of the three sample task profiles — single
label per row, classification eval pack, per-class precision and
recall. The natural export target is ONNX with 8-bit quantization
for a small mobile footprint.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Intro
00:20 Data · Sources (10/10/10 balance)
00:39 Data · Labels (70/65/65 distribution)
00:59 Data · Split
01:14 Train · Tokenization (mobile target)
01:29 Train · LoRA (mobile_cpu)
01:47 Results
02:00 Ship · Compress & export (ONNX path)
02:22 Wrap

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
Bring Your Own Data: Train an SLM on Your Own CSV in Two Minutes
```

**Description**
```
The three sample projects are useful for learning the platform, but
the real point is to use your own dataset. This video shows the
shortest path from a CSV on your laptop to a project ready for the
rest of the pipeline.

A six-row coffee-shop FAQ stands in for "your data." Create a new
project from a one-line brief, open Data, upload the CSV, and the
rows land under Sources with their columns kept. From there the
project follows the same steps you saw on the samples (videos 3–5):
Prepare, Labels, Generate, Split, then Train, Results and Ship.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Intro
00:12 New project flow
00:30 Empty data tab
00:52 Upload CSV
01:16 Rows imported
01:39 Wrap

#BringYourOwnData #SmallLanguageModel #FineTuning #LLM #LocalAI
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
Train a 135M LoRA in 10 Seconds — Real Celery, Real LoRA, Local GPU
```

**Description**
```
Launch a real training run on the Support FAQ sample. Tiny base
model (SmolLM2, 135M parameters), 108 training rows, two epochs at
batch size two: 108 optimizer steps. A Celery worker on the local box
runs PyTorch + transformers + PEFT. About ten seconds of training on
an NVIDIA GB10.

Not a simulated loop: the output is a real LoRA adapter on disk, and
the run's dashboard shows the epoch counter, the loss curve and the
final losses (train 2.67, eval 3.10 in this take). Those are losses,
not accuracy, and still high: the point is that the loop runs end to
end. Video 8 scores the answers.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Train form
00:31 More options (hyperparameters, LoRA)
01:01 Kickoff (API)
01:21 Watching status: running
01:42 Run dashboard with metrics
02:08 Wrap

#LoRA #FineTuning #SmallLanguageModel #PyTorch #LocalAI
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
Evaluate a Fine-Tuned SLM Against Its Answer Key — Honest Numbers
```

**Description**
```
Score the run from the previous video against the Support FAQ
practice answer key (200 hand-written answers the model never trained
on). Twenty samples, greedy decoding, at most 64 new tokens each;
about ten seconds on a GB10.

Honest result: exact match 0, token F1 0.24. The pass/fail rules fail
on both required rules, and the scorecard says by how much. The
Failures tab puts each question next to the expected answer and what
the model actually said: that is where you start reading. A 135M model
after ten seconds of training is not a product; the loop closing is
the point.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Results · Verdict
00:19 Evaluate form (answer key)
00:46 Kickoff (API)
01:07 Scoring the practice answer key
01:20 Pass/fail rules, scorecard & predictions
01:47 Wrap

#ModelEvaluation #LLMEval #SmallLanguageModel #FineTuning #HonestMetrics
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
Compress an SLM to a 101 MB GGUF — Merge LoRA + llama.cpp Quantize
```

**Description**
```
Turn the trained LoRA adapter into a quantized GGUF that Ollama can
serve, from Ship → Package → Options → Compress first. Two steps:
merge the adapter into the base model (a ~260 MB half-precision
checkpoint), then quantize with llama.cpp to 4-bit: a 101 MB GGUF.
Each step takes about four seconds on a GB10.

Then the New export form registers the artifact against the run:
the export validates the file, writes a manifest with the model hash
and a deployment plan, and lands under Releases as completed (GGUF,
Q4_K_M, 100.57 MB). Export only ever packages that run's own
compressed file, never one made from another run.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Ship · Package
00:23 Options · Compress first
00:43 Merge LoRA + quantize
01:04 GGUF on disk (101 MB)
01:28 New export form
01:42 Export completed (Releases)
02:03 Wrap

#GGUF #Quantization #llamacpp #SmallLanguageModel #Ollama
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
The 101 MB GGUF from the previous video is registered with Ollama
and answers a prompt in Results → Try it, through the
OpenAI-compatible API. Trained, evaluated, compressed, served: the
loop closes here.

Asked how to reset a password (temperature 0, so the reply is the
same on every take), the model sends you to Settings → Security → a
Reset button, which is exactly where the training tickets point. Then
it drifts: the tickets say you get an emailed one-time link that
expires after 15 minutes, and the model never mentions the email,
invents a choice between "12 months" and "30 days", and points to a
Settings → Keyring page no ticket describes. Half right and
confidently worded, which is what the evaluation's 0.24 F1 was
saying. Same loop, bigger data and bigger base models, is how you get
past that.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18 on the six-destination workspace, awaiting upload.

Chapters:
00:00 Results · Try it
00:20 Ollama create from GGUF
00:41 Playground config (temperature 0)
01:08 Send prompt
01:22 Model responds
01:53 Series wrap

#Ollama #LocalLLM #SmallLanguageModel #FineTuning #GGUF
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
served model. Where each piece runs (GPU vs CPU vs disk), and the
one optional cloud piece: if you add a provider key, a project can
use a cloud model to label data, write an answer key or judge
answers, with every call metered against a monthly budget. The two
trust boundaries the design draws.

This is the closer for the playlist. Watch this if you want to
understand WHY the rest of the series works the way it does —
or if you're considering using BrewSLM as a starting point for
your own local-first AI workspace.

🔗 Code: https://github.com/TensorGreed/__SLM__
📺 Channel: https://www.youtube.com/@TensorGreed

Re-recorded 2026-10-18, awaiting upload.

Chapters:
00:00 Intro
00:12 The stack: 5 processes
00:42 Data flow: ingest → serve
01:06 Where things run (local by default)
01:43 Trust boundaries
02:07 Wrap

#SystemDesign #AIArchitecture #LocalAI #BrewSLM #MachineLearning
```

**Tags**
```
AI architecture, LLM platform architecture, local AI stack, FastAPI Celery Redis Ollama, AI system design, machine learning platform, ML engineering stack, BrewSLM architecture, on prem AI, private AI infrastructure, AI workspace design, TensorGreed, software architecture LLM, AI engineering
```

**Thumbnail**: `thumbnails/v14.png`

---

## Video 20 — Legal assistant: the gate that failed, and what worked (EP 13)

Local: `docs-demo/videos/20-legal-rag-vs-sft/final_captioned.mp4` (6:41).
Full kit (titles, description with chapters, tags, LinkedIn post):
`docs-demo/videos/20-legal-rag-vs-sft/PUBLISH.md`. Docs page:
`slm-docs/docs/demos/legal-documents-rag.md` (add the YouTube embed there
once published).

**Thumbnail**: `thumbnails/v20.png`

