---
sidebar_position: 3
title: PII / PCI Detector demo
---

# PII / PCI Detector demo

A pre-loaded BrewSLM project that trains a span-level PII / PCI detector
— given a text snippet, it emits a JSON object listing every sensitive
value found in the text along with its character offsets and entity
type. The output format is designed to drop into a redaction pipeline
or a [LlamaFirewall](https://github.com/meta-llama/PurpleLlama) scanner.

## What you get on click

Click **"Demo · PII / PCI Detector"** on the project list and you'll
land in a fully-seeded project:

- **61 training rows** of synthetic chat / log / form / DM-style
  snippets with entity offsets pre-labelled.
- **25 gold rows** for held-out evaluation.
- **`task_profile: structured_extraction`** so the evaluation pipeline
  routes through the `StructuredExtractionHandler` and scores at three
  layers: JSON validity, schema compliance, and field-level F1.
- **Pre-filled Autopilot brief** — open the Autopilot tab and click
  Preview Plan; the brief is already in the textarea.
- **Adapter preset**: `structured-extraction` data adapter, mapped to
  `text → entities_json`.
- **Target profile**: `vllm_server` with the `balanced` plan profile —
  optimised for recall over latency since false negatives leak PII.

## Entity types covered

The shipped bundle covers ten common types:

| Type             | Example value                                    |
| ---------------- | ------------------------------------------------ |
| `email`          | `jane.doe@example.com`                           |
| `phone`          | `555-0173` / `(555) 014-2381` / `+1-555-0188`    |
| `ssn`            | `000-12-3456` (US Social Security)               |
| `credit_card`    | `4242424242424242` (Visa/MC/Amex test PANs)      |
| `person_name`    | `Jane Doe`, `Priya Raman`                        |
| `street_address` | `742 Evergreen Terrace, Springfield, IL 62704`   |
| `date_of_birth`  | `1989-03-14`                                     |
| `ip_address`     | `192.0.2.55` (RFC 5737 reserved-doc range)       |
| `api_key`        | `sk_live_…` / `ghp_…` / `AKIA…` / JWTs           |
| `bank_account`   | IBAN-format account numbers                      |

All values in the bundle are **synthetic**: `555-` phone numbers,
`000-` SSNs (reserved range), test PANs, `@example.com` emails.
Re-running the bundle generator never produces real-world identifiers.

## Expected output format

For an input like:

```text
Hi support team, I'm Jane Doe and I can't log in.
Email me at jane.doe@example.com or call 555-0173.
```

The trained model is expected to emit:

```json
{
  "entities": [
    {"type": "person_name", "start": 21, "end": 29, "text": "Jane Doe"},
    {"type": "email", "start": 65, "end": 87, "text": "jane.doe@example.com"},
    {"type": "phone", "start": 96, "end": 104, "text": "555-0173"}
  ]
}
```

Empty case (no PII present): `{"entities": []}`. The bundle includes a
few clean rows so the model learns to emit `[]` rather than
hallucinate.

## How scoring works

The `StructuredExtractionHandler` (introduced in Phase 5.3.4) scores
three independent layers — each is a separate gate-able metric, so you
can tell apart "model emits garbage JSON" from "model emits clean JSON
with the wrong values":

- **`json_validity_rate`** — fraction of rows where the model's output
  parses as JSON (after stripping ` ```json ` code fences). A 30%
  malformed rate makes the model unshippable regardless of accuracy.
- **`schema_compliance_rate`** — fraction of rows where the parsed
  object has every required field (here: `entities`).
- **`exact_match`** — whole-blob equality of the predicted JSON vs the
  gold. With span-level offsets, this is the most demanding metric.
- **`f1`** — mean per-field F1 (here: F1 on the `entities` field's
  string-rendered list). Looser than exact_match; useful for tracking
  progress between checkpoints.

The Sample Predictions card on the Evaluation tab shows per-row
**"JSON: valid · X/Y fields"** badges and a **"Show field-by-field
comparison"** disclosure so you can eyeball failures without leaving
the UI.

## Expanding the dataset

The 61 + 25 rows are enough to demonstrate the full pipeline, but
production PII detectors need 10× more — both for diversity (more
real-world templates) and for recall (more entity-type coverage). Two
clean ways to scale:

### HuggingFace datasets

Recommended starting points (all CC-BY / synthetic-safe):

- [**ai4privacy/pii-masking-200k**](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
  — 200k+ rows, multi-language, includes 51 PII classes with
  character-level span annotations. The format is close to what
  BrewSLM expects; you'll need a small adapter to reshape
  `mbert_token_classes` into `{type, start, end, text}` triples.
- [**Isotonic/pii-masking-200k**](https://huggingface.co/datasets/Isotonic/pii-masking-200k)
  — mirror of the above with cleaner conversation framing.
- [**bigcode/the-stack-pii**](https://huggingface.co/datasets/bigcode/the-stack-pii)
  — code-specific PII (API keys, secrets in source). Useful if you're
  detecting secret leakage in commits.

To pull one into BrewSLM:

```sh
brewslm dataset import \
  --hf ai4privacy/pii-masking-200k \
  --project-slug pii-detector \
  --split train \
  --limit 5000 \
  --adapter-preset structured-extraction \
  --field-map source_text:text,privacy_mask:entities_json
```

The `--field-map` flag lets you rename the HF dataset's columns onto
the bundle's expected fields. The adapter preset wraps the result in
the `{"entities": [...]}` shape the handler scores against. See the
[Data ingestion docs](../workflows/data-ingestion.md) for a deeper
walkthrough of the importer.

### Kaggle datasets

- [**PII Detection Dataset (Kaggle Cup 2024)**](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)
  — student writing samples with PII; ~20k essays. Format is BIO-tagged
  tokens; you'll need a converter to span offsets before BrewSLM picks
  it up.
- [**Synthetic PII Detection**](https://www.kaggle.com/datasets/ekohrt/pii-data-detection-dataset)
  — pre-generated synthetic data with span annotations.

Pull via the Kaggle CLI then import:

```sh
kaggle competitions download pii-detection-removal-from-educational-data
unzip pii-detection-removal-from-educational-data.zip
python scripts/kaggle_pii_to_brewslm.py \
  --input train.json \
  --out backend/data/imports/kaggle-pii.jsonl
brewslm dataset import \
  --jsonl backend/data/imports/kaggle-pii.jsonl \
  --project-slug pii-detector \
  --adapter-preset structured-extraction
```

The `kaggle_pii_to_brewslm.py` converter is a small script — about
30 lines — that walks each essay's BIO-tagged tokens and emits
`{type, start, end, text}` records. There's a reference in
[Adapter Studio examples](../getting-started/adapter-studio-examples.md).

### Bundle generator (synthetic expansion)

If you want more synthetic data with full control over coverage, the
bundle ships its own generator:

```sh
cd backend/data/demo_samples/pii-detector
# Edit TRAINING_TEMPLATES / GOLD_TEMPLATES in _generate_bundle.py
python _generate_bundle.py
```

Each template is a list of literal-string and `E(type, value)` parts;
the script computes character offsets correctly on regeneration.
Re-seed the project after edits:

```sh
brewslm demo seed pii-detector --force
```

## Plugging into LlamaFirewall

Once you have a trained model exported (the demo defaults to a vLLM
target, but ONNX-INT8 also works), the model's output format is the
native shape LlamaFirewall's regex / classifier scanners consume.
Sketch:

```python
from llamafirewall import Firewall, ScannerConfig
from brewslm.runtime import LocalModel  # or your serving endpoint

model = LocalModel.load("exports/pii-detector/onnx-int8")

def brewslm_scanner(text: str) -> list[dict]:
    """Conforms to LlamaFirewall's scanner interface."""
    response = model.generate(text, max_new_tokens=512)
    parsed = json.loads(response)
    return [
        {
            "category": e["type"],
            "start": e["start"],
            "end": e["end"],
            "matched_text": e["text"],
            "score": 1.0,
        }
        for e in parsed.get("entities", [])
    ]

fw = Firewall(
    input_scanners=[ScannerConfig(name="brewslm_pii", scanner=brewslm_scanner)],
)
result = fw.scan_input("Hi, I'm Jane Doe at jane@example.com")
# result.detected → [{"category": "person_name", ...}, {"category": "email", ...}]
```

The contract on BrewSLM's side: emit valid JSON, every entity has
`{type, start, end, text}`, and `text == input[start:end]`. The
`StructuredExtractionHandler`'s gates already enforce the first two;
the third is checked by post-processing in your scanner wrapper (the
example above can add `if response[e["start"]:e["end"]] != e["text"]:
continue` to drop hallucinated offsets).

## End-to-end recipe

```sh
# 1. Seed the demo
brewslm demo seed pii-detector

# 2. Open Autopilot in the UI — the brief is pre-filled.
# Click Preview Plan, review, click Apply.

# 3. Wait for training. Default plan uses LoRA on a small open
#    model; finishes in ~10 minutes on a single A100, ~30 on CPU sim.

# 4. Run eval on the gold set
brewslm eval run --project pii-detector --dataset gold_dev --eval-type f1

# 5. Inspect Sample Predictions — every row should show "JSON: valid"
#    and ideally "1/1 fields" with the entities matching.
#    If you see "JSON: malformed" rates above 5%, raise max_new_tokens
#    on the training config and re-run.

# 6. Export
brewslm export onnx --project pii-detector --quantize int8

# 7. Drop the exported model into your LlamaFirewall scanner config.
```

## What's not in the demo (yet)

- **Entity-level span scoring** — the current handler scores whole-blob
  equality on the entities list. A future phase (5.3.X) will add a
  dedicated NER handler that computes precision/recall/F1 at the span
  level so partial-credit (one entity right, one missed) is visible.
- **Multi-language coverage** — the bundle is English-only. The HF
  datasets above ship multi-language data; the same handler works
  unchanged, just feed it different rows.
- **Production rate-limits** — if you plug into LlamaFirewall and the
  scanner becomes a hotspot, consider exporting to ONNX-INT8 and
  serving on CPU; latency is ~10ms / row for short snippets after
  quantisation.
