---
sidebar_position: 4
title: Picking a base model
---

# Picking a base model

A reference for selecting a base model in the **Base Model Registry**. The right family depends on the task more than the size — most beginner mistakes come from picking a causal LM for a classification job or vice versa.

## Mental model

| Family | Choose for | Avoid for |
|---|---|---|
| **Causal LM** | Chat, Q&A, instruction following, structured extraction with generated text. | Pure classification, deterministic rewrites. |
| **Seq2Seq** | Translation, summarisation, controlled rewriting, short Q&A. | Open-ended chat, multi-turn dialogue. |
| **Encoder / Classifier** | Label assignment, routing, moderation, intent detection. | Anything that needs a generated response. |

## Causal LM

The default. If you're not sure, start here.

| | |
|---|---|
| Typical task profiles | `instruction_sft`, `qa`, `structured_extraction`, `dpo`, `orpo` |
| Training modes supported | SFT, LoRA, DPO, ORPO |
| Example small models | `Qwen2.5-1.5B-Instruct`, `Phi-3.5-mini-instruct`, `Llama-3.2-1B-Instruct` |
| Strengths | Flexible output shape, strong instruction-following, broad tooling. |
| Costs | Inference cost scales with output length. Prompt template matters. |

## Seq2Seq

| | |
|---|---|
| Typical task profiles | `summarization`, `translation`, `seq2seq` |
| Training modes supported | SFT |
| Example small models | `flan-t5-small`, `mT5-small`, `bart-base` |
| Strengths | Deterministic-feeling transforms, good for narrow rewrite tasks. |
| Costs | Less flexible. Brittle on inputs outside the training distribution. |

## Encoder / Classifier

| | |
|---|---|
| Typical task profiles | `classification`, `regression` |
| Training modes supported | SFT (classification head) |
| Example small models | `distilbert-base-uncased`, `roberta-base`, `mdeberta-v3-base` |
| Strengths | Tiny, fast, easy to deploy on CPU. |
| Costs | Output is a label / score, not text. |

## Beginner recommendation flow

### UI

1. Project workspace → Training rail → **Base Model Registry**.
2. Filter by:
   - **Family** — pick from the table above.
   - **License** — `permissive` (MIT / Apache 2.0) vs `restricted` (Llama community, etc.). Beginner mode hides restricted models by default.
   - **Context length** — at least 2–4× your typical input length.
   - **Hardware fit** — `laptop` / `server` / `edge`.
3. Click **Validate for project** on a candidate. The validator runs a real (cheap) check:
   - Tokenizer loads.
   - Chat template (if any) is compatible with your task profile.
   - License rules are satisfied (e.g., commercial use allowed if your project is flagged commercial).
   - Adapter contract matches.
4. Read the **why recommended** / **why risky** rows. Each one is a `reason_code` from the model-compatibility taxonomy.
5. Click **Set as default** on the winner.

### CLI

```sh
# Browse the catalog
brewslm models list --family qwen --hardware-fit server --json

# Import a HuggingFace model id (refreshes the catalog row)
brewslm models import --hf-id "Qwen/Qwen2.5-1.5B-Instruct" --json

# Run the project-specific validator
brewslm models validate --project 1 --model 12 --json

# Pin as project default
brewslm models set-default --project 1 --model 12
```

### API

```sh
# Recommendation
curl "http://localhost:8000/api/projects/1/models/recommend?limit=5&hardware_fit=server"

# Validate
curl -X POST http://localhost:8000/api/projects/1/models/12/validate

# Set default
curl -X PUT http://localhost:8000/api/projects/1/models/default \
  -H "Content-Type: application/json" \
  -d '{"model_id": 12}'
```

## Choosing size

| Dataset size | Recommended starting model size |
|---|---|
| < 1k rows | 1B–2B parameters (Qwen 1.5B / Phi-3.5 mini). |
| 1k–10k rows | 1B–4B parameters. |
| 10k–100k rows | 3B–8B parameters. |
| > 100k rows | 7B+ — or stay small with longer training. |

**Going bigger doesn't help if your data quality is the bottleneck.** Run a small baseline first (1B, 1 epoch) before scaling up. The [autopilot's measured cost estimator](../reliability/measured-vs-estimated.md) recognises this and will refuse to escalate to a larger model when a smaller one's eval is still mediocre.

## See also

- [Adapter Studio examples](adapter-studio-examples.md) — how to bridge from raw data shape to a model's expected input.
- [Model compatibility matrix](../reference/model-compatibility-matrix.md) — exhaustive per-model support table.
- [Training](../workflows/training.md) — once you've picked a model, what to do with it.
