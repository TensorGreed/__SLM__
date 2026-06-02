Here is the **updated compact learning path**, now including the **hands-on implementation track** with Python libraries plus a **UI-heavy BrewSLM path**.

---

# Fine-Tuning SLMs: Compact Learning Path

Audience:

* **Software developer** starting the SLM fine-tuning journey
* **ML engineer** with limited fine-tuning experience wanting to upskill
* Goal: learn fundamentals, terms, hands-on training, evaluation, deployment, and UI-driven fine-tuning workflows

---

## 0. Orientation: What Fine-Tuning Is For

* Understand fine-tuning as **behavior adaptation**, not massive knowledge injection.
* Learn when to use:

  * Prompting
  * RAG
  * Fine-tuning
  * RAG + fine-tuning
  * Distillation
  * Preference tuning
* Common SLM use cases:

  * Sentiment classification
  * Intent classification
  * PII detection
  * JSON extraction
  * Support FAQ bot
  * Tool/function calling
  * SQL generation
  * Domain-specific assistant
  * Small local/private AI workflows

---

## 1. Core LLM Basics

* Tokens and tokenization
* Vocabulary
* Context window
* Sequence length
* Embeddings
* Transformer basics
* Attention
* Causal language modeling
* Next-token prediction
* Base model vs instruct model
* Chat template
* Prompt template
* System/user/assistant messages
* Decoding settings:

  * temperature
  * top-p
  * top-k
  * max new tokens
  * repetition penalty
  * stop tokens
* Hallucination
* Quantization
* GPU memory
* Model size: 0.5B, 1B, 3B, 7B+

---

## 2. Fine-Tuning Concepts

* Supervised Fine-Tuning, or SFT
* Instruction tuning
* Chat fine-tuning
* Completion fine-tuning
* Full fine-tuning
* LoRA
* QLoRA
* PEFT
* Adapter weights
* Merging adapters
* Domain adaptation
* Continued pretraining
* DPO
* RLHF
* Reward models
* Distillation
* Model merging
* Catastrophic forgetting
* Overfitting

Recommended order:

```text
SFT → LoRA → QLoRA → Evaluation → Deployment → DPO / Preference tuning
```

---

## 3. Model Selection

Start with small instruct models:

* 0.5B
* 1B
* 1.5B
* 3B

Understand how to pick based on:

* Task complexity
* License
* Context length
* Instruct quality
* Tokenizer
* GPU/CPU availability
* Latency target
* Deployment target
* Community support

Avoid starting with 7B+ unless the learner understands GPU memory and quantization.

---

## 4. Dataset Fundamentals

Learn dataset formats:

* JSONL
* Alpaca format
* ShareGPT format
* Chat messages format
* Completion format
* Classification format
* Extraction format

Understand fields:

* instruction
* input
* output
* messages
* system
* user
* assistant
* label
* metadata

Dataset splits:

* train
* validation
* test
* golden test set

Core principle:

```text
Never evaluate on training data.
```

---

## 5. Data Curation

Teach how to create strong training data:

* Define the exact task.
* Define allowed outputs.
* Keep output style consistent.
* Add realistic production-like prompts.
* Include normal examples.
* Include edge cases.
* Include ambiguous cases.
* Include hard negatives.
* Include out-of-domain cases.
* Include refusal examples.
* Include formatting examples.
* Include failure examples from production later.

Dataset size guidance:

* 50 examples: toy learning
* 200 examples: first signal
* 500–1,000 examples: useful narrow fine-tune
* 2,000+ examples: stronger behavior
* 10,000+ examples: serious domain/task adaptation

---

## 6. Data Cleanup and Dedup

Important terms and tasks:

* Deduplication
* Near-deduplication
* Data leakage detection
* Label noise cleanup
* Class imbalance check
* Schema consistency
* Format normalization
* Synthetic data filtering
* PII removal
* Toxic data filtering
* Long-example trimming
* Contradiction detection
* Train/test overlap check
* Prompt-response mismatch cleanup

Practical cleanup checklist:

* Remove duplicate rows.
* Remove near-duplicate prompts.
* Fix wrong labels.
* Remove contradictory examples.
* Normalize labels.
* Normalize JSON structure.
* Remove broken Unicode or encoding issues.
* Remove examples with empty outputs.
* Remove examples where output does not answer input.
* Ensure all outputs follow the target style.

---

## 7. Python Hands-On Stack

Core libraries:

* `torch`
* `transformers`
* `datasets`
* `peft`
* `trl`
* `accelerate`
* `bitsandbytes`
* `scikit-learn`
* `pandas`
* `numpy`
* `evaluate`
* `pydantic`
* `jsonlines`
* `tqdm`

Useful later:

* `wandb`
* `mlflow`
* `tensorboard`
* `sentence-transformers`
* `faiss-cpu`
* `chromadb`
* `vllm`
* `llama-cpp-python`
* `fastapi`
* `uvicorn`
* `gradio`
* `streamlit`

Frameworks to explore:

* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* TRL
* Axolotl
* LLaMA-Factory
* Unsloth
* llama.cpp
* Ollama
* vLLM

---

## 8. Hands-On Track: Python Path

### Step 1: Run Base Model Inference

Learn:

* Load tokenizer
* Load model
* Use chat template
* Generate response
* Change decoding parameters
* Capture baseline outputs

Libraries:

```text
transformers
torch
accelerate
```

Goal:

```text
Run a small instruct model locally and understand its behavior before training.
```

---

### Step 2: Create Dataset

Learn:

* Create JSONL dataset
* Use instruction/input/output format
* Use chat messages format
* Split train/validation/test
* Validate schema

Libraries:

```text
pandas
datasets
json
jsonlines
pydantic
scikit-learn
```

Goal:

```text
Build a clean dataset for sentiment, intent, PII, JSON extraction, or FAQ answering.
```

---

### Step 3: Clean and Validate Data

Learn:

* Dedup exact matches
* Detect near-duplicates
* Validate labels
* Validate JSON outputs
* Remove invalid examples
* Detect train/test leakage

Libraries:

```text
pandas
scikit-learn
sentence-transformers
pydantic
```

Goal:

```text
Turn raw examples into training-ready examples.
```

---

### Step 4: Fine-Tune with LoRA

Learn:

* Configure LoRA
* Select target modules
* Set training hyperparameters
* Train adapter
* Save checkpoint
* Reload adapter

Libraries:

```text
transformers
peft
trl
accelerate
torch
```

Goal:

```text
Fine-tune a 0.5B–3B instruct model with LoRA.
```

---

### Step 5: Fine-Tune with QLoRA

Learn:

* 4-bit loading
* Quantized training
* Memory-efficient fine-tuning
* Gradient checkpointing
* Tradeoffs between memory and quality

Libraries:

```text
bitsandbytes
peft
trl
transformers
accelerate
```

Goal:

```text
Train larger SLMs on limited GPU memory.
```

---

### Step 6: Evaluate Model

Learn:

* Compare base vs fine-tuned model
* Run test set
* Compute metrics
* Validate output format
* Analyze failure cases

Libraries:

```text
scikit-learn
evaluate
pandas
pydantic
json
```

Metrics:

* Accuracy
* Precision
* Recall
* F1
* Confusion matrix
* Exact match
* Valid JSON rate
* Schema compliance
* Hallucination rate
* Task success rate

Goal:

```text
Prove whether fine-tuning actually improved the model.
```

---

### Step 7: Merge, Quantize, and Serve

Learn:

* Load base model + adapter
* Merge LoRA adapter
* Save merged model
* Quantize model
* Serve through API
* Stream responses

Libraries/tools:

```text
peft
transformers
llama.cpp
Ollama
vLLM
FastAPI
Docker
```

Goal:

```text
Deploy the fine-tuned SLM behind an API or local runtime.
```

---

### Step 8: Build Feedback Loop

Learn:

* Log prompts and outputs
* Capture failures
* Tag bad responses
* Add corrected examples to dataset
* Retrain
* Compare versions

Tools:

```text
MLflow
Weights & Biases
custom database
CSV/JSONL logs
```

Goal:

```text
Create a continuous improvement loop.
```

---

## 9. Training Hyperparameters

Must-know terms:

* Epochs
* Batch size
* Per-device batch size
* Gradient accumulation
* Effective batch size
* Learning rate
* Learning rate scheduler
* Warmup ratio
* Warmup steps
* Weight decay
* Max sequence length
* Packing
* Optimizer
* Mixed precision
* fp16
* bf16
* Gradient checkpointing
* Save steps
* Eval steps
* Logging steps
* Seed
* LoRA rank `r`
* LoRA alpha
* LoRA dropout
* Target modules

Starter values:

```text
epochs: 1–3
learning_rate: 1e-4 to 2e-4
LoRA rank: 8, 16, or 32
LoRA alpha: 16 or 32
LoRA dropout: 0.05
max_seq_length: 512–2048
batch size: as large as fits
gradient accumulation: use to increase effective batch size
```

---

## 10. Training Signals

Teach learners how to read:

* Training loss
* Validation loss
* Eval loss
* Perplexity
* GPU memory usage
* Samples/sec
* Tokens/sec
* Checkpoint quality
* Output quality over time

Interpretation:

* Train loss down + validation loss down: good
* Train loss down + validation loss up: overfitting
* No loss movement: wrong config, bad data, or bad learning rate
* Good loss + bad output: bad formatting or bad eval
* Repetitive output: overtraining or decoding issue
* Format drift: inconsistent dataset

---

## 11. Evaluation Terms

General:

* Baseline
* Golden set
* Regression test
* Human evaluation
* LLM-as-judge
* Task success rate
* Failure analysis

Classification:

* Accuracy
* Precision
* Recall
* F1
* Confusion matrix
* False positive
* False negative

Extraction:

* Exact match
* Field-level accuracy
* Valid JSON rate
* Schema compliance
* Missing field rate
* Extra field rate

Generative:

* Correctness
* Completeness
* Groundedness
* Hallucination rate
* Refusal accuracy
* Helpfulness
* Faithfulness

Performance:

* Latency
* Time to first token
* Tokens/sec
* Throughput
* Memory usage
* Cost per request

---

## 12. UI-Heavy BrewSLM Path

This path is for developers and ML engineers who want to understand the workflow through a **guided product UI** before or alongside raw Python.

### BrewSLM Learning Flow

* Create a project
* Choose task type
* Select or upload dataset
* Validate dataset
* Clean dataset
* Split train/validation/test
* Select base SLM
* Choose fine-tuning method
* Configure hyperparameters
* Start training job
* Monitor training
* View loss curves
* Compare model outputs
* Run evaluation
* Inspect failed examples
* Export adapter/model
* Deploy or test through inference UI

---

## 13. BrewSLM UI Modules to Learn Through

### Dataset UI

Learner should understand:

* Dataset upload
* JSONL validation
* Schema validation
* Label distribution
* Duplicate detection
* Train/test split
* Preview examples
* Bad example detection
* Synthetic data review
* Dataset versioning

Concepts learned:

```text
data quality
deduplication
schema consistency
class imbalance
data leakage
train/validation/test split
```

---

### Training UI

Learner should understand:

* Base model selection
* LoRA vs QLoRA
* Epochs
* Batch size
* Learning rate
* Max sequence length
* LoRA rank
* LoRA alpha
* LoRA dropout
* Checkpointing
* GPU memory estimate

Concepts learned:

```text
hyperparameters
adapter training
training loss
validation loss
overfitting
checkpointing
resource constraints
```

---

### Evaluation UI

Learner should understand:

* Base model vs fine-tuned model comparison
* Metric dashboard
* Classification metrics
* JSON validation
* Schema compliance
* Hallucination review
* Failure examples
* Confusion matrix
* Regression tests
* Golden set evaluation

Concepts learned:

```text
baseline
task success rate
precision
recall
F1
format compliance
failure analysis
regression testing
```

---

### Playground UI

Learner should understand:

* Prompt testing
* Chat template behavior
* Temperature
* Top-p
* Max tokens
* Output comparison
* Side-by-side base vs fine-tuned responses
* Edge case testing

Concepts learned:

```text
inference
decoding
prompt sensitivity
format drift
response quality
```

---

### Deployment UI

Learner should understand:

* Adapter export
* Merged model export
* Quantized export
* API endpoint
* Local deployment
* Ollama/llama.cpp export
* Versioning
* Monitoring
* Feedback collection

Concepts learned:

```text
serving
quantization
model versioning
latency
monitoring
continuous improvement
```

---

## 14. BrewSLM + Python Combined Track

The best learning path is not UI-only or code-only. It should combine both.

### Stage 1: Learn the workflow in BrewSLM UI

* Upload dataset
* Validate dataset
* Train with default LoRA settings
* Evaluate
* Compare outputs
* Inspect failures

Goal:

```text
Understand the full fine-tuning loop visually.
```

---

### Stage 2: Reproduce the same flow in Python

* Load the same dataset
* Train using `transformers`, `peft`, and `trl`
* Run the same evaluation
* Compare results with BrewSLM

Goal:

```text
Understand what the UI is doing under the hood.
```

---

### Stage 3: Return to BrewSLM for faster iteration

* Use UI for quick experiments
* Try different hyperparameters
* Compare model versions
* Review failure cases
* Export best model

Goal:

```text
Use the UI as an acceleration layer, not a black box.
```

---

## 15. Recommended Hands-On Projects

### Project 1: Sentiment Classifier

Learn:

* Labels
* Accuracy
* F1
* Confusion matrix
* Base vs fine-tuned comparison

Use:

```text
Python + BrewSLM UI
```

---

### Project 2: Intent Classifier

Learn:

* Class imbalance
* Routing behavior
* False positives
* False negatives

Use:

```text
JSONL dataset + LoRA fine-tune
```

---

### Project 3: JSON Extractor

Learn:

* Structured outputs
* Valid JSON rate
* Pydantic schema validation
* Format compliance

Use:

```text
Python validation + BrewSLM evaluation UI
```

---

### Project 4: PII Detector

Learn:

* Precision
* Recall
* False negatives
* Entity-level evaluation
* Safety-sensitive evaluation

Use:

```text
Python metrics + failure inspection UI
```

---

### Project 5: Support FAQ Bot

Learn:

* Hallucination
* Refusal behavior
* Grounded answers
* RAG vs fine-tuning

Use:

```text
BrewSLM playground + manual eval + golden set
```

---

### Project 6: Tool-Call Generator

Learn:

* Function calling
* JSON schema
* Tool routing
* Strict structured outputs

Use:

```text
Pydantic + fine-tuned SLM + API test harness
```

---

## 16. Software Developer Path

Focus order:

* Run small model inference locally
* Learn prompts and chat templates
* Learn JSONL
* Build small dataset
* Use BrewSLM UI to fine-tune first model
* Reproduce flow in Python
* Learn LoRA and QLoRA
* Build evaluation script
* Deploy with FastAPI/Ollama/llama.cpp
* Add logs and feedback loop
* Improve data from failures

Developer mindset:

```text
Fine-tuning is a software pipeline:
data → training → tests → deployment → monitoring → iteration
```

---

## 17. Limited-Knowledge ML Engineer Path

Focus order:

* Refresh transformers and tokenization
* Learn SFT, LoRA, QLoRA, PEFT
* Learn dataset curation deeply
* Learn hyperparameter effects
* Use BrewSLM UI to visualize training/eval loop
* Reproduce experiments in Python
* Learn failure analysis
* Learn evaluation design
* Learn quantization and deployment
* Learn DPO, preference tuning, and distillation later

ML engineer mindset:

```text
Fine-tuning is controlled behavior optimization with strong data and evaluation discipline.
```

---

## 18. Suggested 4-Level Curriculum

### Level 1: Foundations

* LLM basics
* Tokenization
* Prompting
* Chat templates
* Base vs instruct models
* Dataset formats
* Local inference

Output:

```text
Learner can run an SLM and prepare a small dataset.
```

---

### Level 2: First Fine-Tune

* SFT
* LoRA
* Hyperparameters
* Training loss
* Validation loss
* BrewSLM training UI
* Python training script

Output:

```text
Learner can fine-tune an SLM on a narrow task.
```

---

### Level 3: Evaluation and Deployment

* Base vs fine-tuned comparison
* F1, precision, recall
* JSON validation
* Hallucination checks
* Failure analysis
* Merging adapters
* Quantization
* API serving

Output:

```text
Learner can decide whether a fine-tuned model is production-worthy.
```

---

### Level 4: Advanced SLM Engineering

* QLoRA
* DPO
* Preference datasets
* Distillation
* RAG + fine-tuning
* Tool-use fine-tuning
* Multi-turn data
* Model monitoring
* Continuous retraining

Output:

```text
Learner can build real SLM workflows, not just run training scripts.
```

---

## 19. Final End-to-End Roadmap

```text
1. Learn LLM basics
2. Run a small instruct model
3. Learn dataset formats
4. Create and clean JSONL data
5. Validate and dedup dataset
6. Split train/validation/test
7. Fine-tune with BrewSLM UI
8. Reproduce with Python libraries
9. Understand hyperparameters
10. Track training and validation loss
11. Evaluate against base model
12. Inspect failures
13. Improve dataset
14. Retrain with LoRA/QLoRA
15. Merge or export adapter
16. Quantize if needed
17. Deploy behind API/local runtime
18. Monitor production failures
19. Feed failures back into dataset
20. Iterate continuously
```

---

## 20. Final Mental Model

The complete SLM fine-tuning journey is:

```text
Understand the model
→ define the task
→ curate the data
→ clean and dedup the data
→ train with LoRA/QLoRA
→ evaluate against baseline
→ inspect failures
→ improve dataset
→ deploy
→ monitor
→ retrain
```

For this audience, the best learning experience is:

```text
BrewSLM UI for visibility and guided workflow
+
Python libraries for deep understanding and reproducibility
```

That combination helps both personas:

* Developers understand the system end-to-end without drowning in ML theory.
* ML engineers deepen their fine-tuning, evaluation, and deployment skills with real implementation details.
