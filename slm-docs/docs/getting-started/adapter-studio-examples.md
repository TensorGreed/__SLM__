---
sidebar_position: 4
title: Adapter Studio Examples
---

# Adapter Studio Examples

Use these examples with Dataset Structure Explorer and Adapter Studio.

## Tabular QA (CSV/TSV)

Input columns:

- `question`
- `answer`
- optional `label`

Recommended adapter:

- `qa-pair` or `default-canonical` with field mapping:
  - `question -> question`
  - `answer -> answer`

## Extraction JSON (Nested)

Input record:

```json
{
  "doc": {"text": "Invoice #123 total is $98.50."},
  "entities": {"invoice_id": "123", "amount": "98.50"}
}
```

Recommended adapter:

- `structured-extraction`
- mapping hints:
  - `source_text -> doc.text`
  - `target_text -> entities`

## Chat SFT (JSONL transcripts)

Input record:

```json
{
  "messages": [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "Use account reset flow."}
  ]
}
```

Recommended adapter:

- `chat-messages`
- task profile: `chat_sft`

## Preference Data (Pairwise)

Input record:

```json
{
  "prompt": "Answer politely",
  "chosen": "Sure, I can help.",
  "rejected": "No."
}
```

Recommended adapter:

- `preference-pair`
- task profile: `preference`

## Practical Flow

1. Run `dataset profile` or Adapter Studio Profile first.
2. Run adapter inference.
3. Review mapping canvas and auto-fix suggestions.
4. Validate coverage until `status=pass` or acceptable warning level.
5. Save adapter version and export scaffold when needed.
