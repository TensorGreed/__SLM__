---
sidebar_position: 13
title: Model Compatibility Matrix
---

# Model Compatibility Matrix

This matrix summarizes how the compatibility engine evaluates model fit.

## Legend

- ✅ Good default fit
- ⚠️ Possible with caveats
- ⛔ Usually incompatible / blocked

## Task Family vs Architecture

| Task Family | Causal LM | Seq2Seq | Classifier |
| --- | --- | --- | --- |
| `instruction_sft` | ✅ | ⚠️ | ⛔ |
| `qa` | ✅ | ✅ | ⛔ |
| `structured_extraction` | ✅ | ✅ | ⚠️ |
| `summarization` | ⚠️ | ✅ | ⛔ |
| `classification` | ⚠️ | ⚠️ | ✅ |
| `routing` | ⚠️ | ⚠️ | ✅ |

## Runtime Modality Checks

| Model Modality | Required Runtime Modality |
| --- | --- |
| `text` | `text` |
| `image` | `vision_language` |
| `audio` | `audio_text` |
| `multimodal` | `multimodal` |

If runtime modality support is missing, compatibility returns blocker reason code:

- `RUNTIME_MODALITY_UNSUPPORTED`

## Common Reason Codes

Pass-oriented:

- `TASK_FAMILY_SUPPORTED`
- `INPUT_MODALITY_SUPPORTED`
- `TARGET_PROFILE_COMPATIBLE`
- `RUNTIME_MODALITY_SUPPORTED`

Risk/blocker-oriented:

- `TASK_FAMILY_UNSUPPORTED`
- `INPUT_MODALITY_UNSUPPORTED`
- `TARGET_PROFILE_INCOMPATIBLE`
- `ADAPTER_TASK_MISMATCH`
- `TOKENIZER_METADATA_MISSING`
- `CHAT_TEMPLATE_MISSING`

## Unblock Strategy

When blocked, prioritize in this order:

1. fix target/runtime mismatch,
2. fix task-family mismatch,
3. fix tokenizer/chat-template warnings,
4. re-run `validate` and confirm score + reason code improvement.
