---
sidebar_position: 3
title: Model compatibility matrix
---

# Model compatibility matrix

How the Base Model Registry's compatibility engine evaluates whether a base model fits your project. The engine combines four orthogonal checks: **task family**, **modality**, **target profile**, and **integration (tokenizer / chat template / adapter)**.

## Legend

- ✅ Good default fit.
- ⚠️ Possible with caveats; check provenance.
- ⛔ Usually incompatible; the validator returns a blocker.

## Task family × architecture

| Task family | Causal LM | Seq2Seq | Classifier |
|---|---|---|---|
| `instruction_sft` | ✅ | ⚠️ | ⛔ |
| `qa` | ✅ | ✅ | ⛔ |
| `structured_extraction` | ✅ | ✅ | ⚠️ |
| `summarization` | ⚠️ | ✅ | ⛔ |
| `translation` | ⚠️ | ✅ | ⛔ |
| `classification` | ⚠️ | ⚠️ | ✅ |
| `routing` | ⚠️ | ⚠️ | ✅ |
| `preference` (DPO/ORPO) | ✅ | ⛔ | ⛔ |

Picking from this table is the first move. The [Model family guide](../getting-started/model-family-guide.md) has the longer explanation.

## Modality

| Model modality | Required runtime modality |
|---|---|
| `text` | `text` |
| `image` / `vision_language` | `vision_language` |
| `audio` / `audio_text` | `audio_text` |
| `multimodal` | `multimodal` |

When a multimodal model is paired with a text-only runtime, the validator returns blocker `RUNTIME_MODALITY_UNSUPPORTED`. Fix by either picking a different model, or by configuring a runtime that declares the matching modality (see the runtime plugin contract).

## Target profile

| Target | Typical weight budget | Best models | Avoid |
|---|---|---|---|
| `mobile_cpu` | ≤ 1.5 GB | Quantised 1B–2B causal LM, distilbert-class classifier | 7B+ models, anything FP16 |
| `browser_webgpu` | ≤ 800 MB | GGUF-Q4 of 1B–2B causal LM | Anything > 1B FP16 |
| `edge_gpu` | ≤ 4 GB | ONNX-INT8 / GGUF-Q8 of 1B–4B | Multi-GB FP16 |
| `vllm_server` | No hard cap | FP16 7B+, anything quantised | n/a |

If `weight_size_within_budget` returns `false` at plan time, the [Deployability score](../deployment/rollback-and-score.md) will block promote. Suggested fixes (in order of preference):

1. Quantise (GGUF-Q4 or ONNX-INT8).
2. Move to a higher-memory target.
3. Pick a smaller base model.

## Integration checks

| Check | Pass reason code | Fail reason code |
|---|---|---|
| Tokenizer loads cleanly | `TOKENIZER_OK` | `TOKENIZER_METADATA_MISSING`, `TOKENIZER_LOAD_FAILED` |
| Chat template present (when needed) | `CHAT_TEMPLATE_OK` | `CHAT_TEMPLATE_MISSING` |
| Adapter contract matches | `ADAPTER_COMPATIBLE` | `ADAPTER_TASK_MISMATCH` |
| License allows your project's use | `LICENSE_OK` | `LICENSE_RESTRICTED` |
| Context length ≥ project's typical input | `CONTEXT_OK` | `CONTEXT_TOO_SHORT` |

The validator emits **pass** reason codes too — useful for explainability. The UI's "Why recommended" / "Why risky" rows render exactly these.

## Compatibility scoring

Each model gets a 0–1 compatibility score per project. The score weights:

| Component | Weight |
|---|---|
| Task-family fit | 0.35 |
| Modality fit | 0.20 |
| Target profile fit | 0.20 |
| Integration checks | 0.15 |
| Historical reliability (rolling) | 0.10 |

A score ≥ 0.7 is "ready". 0.4–0.7 is "caution". < 0.4 is "block".

Provenance per component:

- Task / modality / target — usually `estimated` (read from metadata).
- Integration — `measured` when **Validate for project** has been run; `estimated` otherwise.
- Historical reliability — always `measured`.

## Unblock strategy

When a model comes back `block`:

1. **Target / runtime mismatch first** — easiest to fix (change target or runtime).
2. **Task-family mismatch** — usually means wrong model family; switch to one that fits.
3. **Integration warnings** — run **Validate for project** to convert them from `estimated` to `measured`; often resolves on its own.
4. **License** — talk to your legal / compliance owner before touching.

Re-run validate after each fix and watch the score + reason codes shift.

## Per-model compatibility lookup

### UI

Training rail → **Base Model Registry** → click a model row → **Compatibility** tab. Shows every check + pass/fail + provenance.

### CLI

```sh
brewslm models validate --project 1 --model 12 --json | jq '.checks'
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/models/12/validate
```

## See also

- [Model family guide](../getting-started/model-family-guide.md) — picking the right family.
- [Adapter examples](../getting-started/adapter-studio-examples.md) — adapter integration in detail.
- [Measured vs estimated](../reliability/measured-vs-estimated.md) — reading provenance.
