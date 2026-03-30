---
sidebar_position: 4
title: Model Family Guide
---

# Model Family Guide

Use this guide when choosing a base model in the **Universal Base Model Registry**.

## Quick Mental Model

- **Causal LM**: best default for instruction-following assistants and open-ended generation.
- **Seq2Seq**: best for structured input-to-output transformations (summarization, translation, rewriting).
- **Encoder/Classifier**: best for labels/scores/categories, not long-form generation.

## Causal LM

Choose causal LM when you need:

- chat-style assistants,
- Q&A with free-form answers,
- extraction that still needs generated language.

Tradeoffs:

- Prompt/template formatting matters more.
- Inference cost can be higher for long outputs.

Typical tasks:

- `instruction_sft`
- `qa`
- `structured_extraction`

## Seq2Seq

Choose seq2seq when you need:

- controlled transformations,
- deterministic rewrite style,
- concise mapping from source text to target text.

Tradeoffs:

- Less flexible for open-ended chat behavior.
- Can require careful dataset pairing quality.

Typical tasks:

- `summarization`
- `translation`
- `qa` (especially short answer)

## Encoder / Classifier

Choose classifier families when you need:

- class labels,
- routing decisions,
- moderation / intent detection.

Tradeoffs:

- Not suitable for rich generated responses.
- You often need a separate generative model downstream.

Typical tasks:

- `classification`
- `routing`
- `moderation`

## Beginner Recommendation Flow

1. Start from your **Domain Blueprint task family**.
2. Open **Base Model Registry** page.
3. Filter by family/size/license/training mode.
4. Click **Validate For Project** and read:
   - Why recommended
   - Why risky
   - Unblock actions
5. Pick the model with the best risk-adjusted compatibility score.
