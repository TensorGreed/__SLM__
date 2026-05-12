---
sidebar_position: 5
title: Adapter examples
---

# Adapter examples

A data adapter is the translation layer between **your raw row shape** and **BrewSLM's canonical record**. This page shows the most common shapes with the recommended adapter, field mapping, and the UI / CLI / API entry points.

The Adapter Studio (Training rail; hidden in beginner mode) is where you tune adapters by hand. For most projects the built-in adapters work — `auto` detects the right one from a sample.

## Pattern 1 — Tabular Q&A (CSV / TSV)

### Input

```csv
question,answer
"How do I reset my password?","Use the account reset flow at settings → security."
"Where is my invoice?","Invoices are emailed to the address on file."
```

### Recommended adapter

| | |
|---|---|
| Adapter id | `qa-pair` or `default-canonical` |
| Field mapping | `question → question`, `answer → answer` |
| Task profile | `qa` or `instruction_sft` |
| Canonical output | `{"text": "Q: …\nA: …"}` |

### UI

Pipeline → **Dataset prep** → **Adapter preview**. Pick `qa-pair`. The preview pane shows 5 mapped rows. Click **Save adapter**.

### CLI

```sh
brewslm adapter preview --project 1 --source-type csv --source-ref ./faq.csv --adapter-id qa-pair --json
brewslm adapter validate --project 1 --source-type csv --source-ref ./faq.csv --adapter-id qa-pair --json
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/data-adapter/preview \
  -H "Content-Type: application/json" \
  -d '{"source_type": "csv", "source_ref": "./faq.csv", "adapter_id": "qa-pair"}'
```

## Pattern 2 — Structured extraction (nested JSON)

### Input

```json
{
  "doc": {"text": "Invoice #123 total is $98.50, due 2026-05-30."},
  "entities": {"invoice_id": "123", "amount": "98.50", "due_date": "2026-05-30"}
}
```

### Recommended adapter

| | |
|---|---|
| Adapter id | `structured-extraction` |
| Field mapping | `source_text → doc.text`, `target_text → entities` (serialised as JSON) |
| Task profile | `structured_extraction` |
| Canonical output | `{"source_text": "...", "target_text": "{\"invoice_id\": \"123\", ...}"}` |

### UI

Pipeline → **Dataset prep** → **Auto-detect adapter**. With nested data, `auto` usually proposes `structured-extraction`; accept or pick from the dropdown.

### CLI

```sh
brewslm adapter infer --project 1 --source-type jsonl --source-ref ./invoices.jsonl --json
brewslm adapter preview --project 1 --source-type jsonl --source-ref ./invoices.jsonl --adapter-id structured-extraction
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/data-adapter/infer \
  -H "Content-Type: application/json" \
  -d '{"source_type": "jsonl", "source_ref": "./invoices.jsonl"}'
```

## Pattern 3 — Chat SFT (JSONL transcripts)

### Input

```json
{
  "messages": [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "Use the account reset flow at settings → security."}
  ]
}
```

### Recommended adapter

| | |
|---|---|
| Adapter id | `chat-messages` |
| Field mapping | `messages → messages` (no transform) |
| Task profile | `chat_sft` |
| Canonical output | `{"messages": [...]}` (rendered through the base model's chat template at tokenisation) |

The chat template gets selected from the base model's metadata — `Qwen2.5-Instruct` and `Llama-3.2-Instruct` ship their own; the adapter respects whichever is registered for the active model.

### UI

Pipeline → **Dataset prep** → adapter dropdown → `chat-messages`. Preview shows the rendered chat template for sample row 1.

### CLI

```sh
brewslm adapter preview --project 1 --source-type jsonl --source-ref ./chat.jsonl --adapter-id chat-messages
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/data-adapter/preview \
  -H "Content-Type: application/json" \
  -d '{"source_type": "jsonl", "source_ref": "./chat.jsonl", "adapter_id": "chat-messages"}'
```

## Pattern 4 — Preference pairs (DPO / ORPO)

### Input

```json
{
  "prompt": "Answer politely.",
  "chosen": "Sure, happy to help.",
  "rejected": "No."
}
```

### Recommended adapter

| | |
|---|---|
| Adapter id | `preference-pair` |
| Field mapping | `prompt → prompt`, `chosen → chosen`, `rejected → rejected` |
| Task profile | `preference` |
| Training modes that consume it | DPO, ORPO |

### UI

Pipeline → **Dataset prep** → `preference-pair`. Then in **Training Configurations**, pick training mode `dpo` or `orpo` — the Resolved Defaults panel surfaces the matching trainer.

### CLI

```sh
brewslm adapter preview --project 1 --source-type jsonl --source-ref ./prefs.jsonl --adapter-id preference-pair
brewslm train start --project 1 --training-mode dpo --base-model 12
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/data-adapter/preview \
  -H "Content-Type: application/json" \
  -d '{"source_type": "jsonl", "source_ref": "./prefs.jsonl", "adapter_id": "preference-pair"}'
```

## Pattern 5 — Auto

If none of the patterns above quite match, set `--adapter-id auto`. The auto adapter samples 50 rows, scores every registered adapter against them, and picks the one that maps cleanest. Saves you the dropdown when the shape isn't obvious.

```sh
brewslm adapter preview --project 1 --source-type csv --source-ref ./weird.csv --adapter-id auto --json
```

## Custom adapters

When none of the built-ins fit:

1. Run `brewslm scaffold adapter --plugin-id my-adapter` to generate a Python module.
2. Edit `map_row` in the generated `.py` file.
3. Validate: `brewslm extensions validate --kind adapter --module path.to.my_adapter`.
4. Add to `.env`: `DATA_ADAPTER_PLUGIN_MODULES="path.to.my_adapter"`.
5. Reload: `brewslm extensions reload --kind adapter`.

See [Extensions → Scaffold](../extensions/scaffold.md) for the full flow.

## Practical sanity flow

1. **Profile** the file first — `brewslm dataset profile` shows columns, sample values, suspected schema.
2. **Infer** the adapter — `brewslm adapter infer` ranks adapter candidates.
3. **Preview** — confirm the first 5 rows look right.
4. **Validate** — runs the adapter against the full sample; flags coverage gaps.
5. **Save** — persist the adapter to the project so re-runs use the same mapping.

## See also

- [Pipeline overview](../workflows/pipeline-overview.md) — where the adapter sits.
- [Data ingestion](../workflows/data-ingestion.md) — the upstream stage.
- [Contracts](../extensions/contracts.md) — how the adapter contract is defined.
