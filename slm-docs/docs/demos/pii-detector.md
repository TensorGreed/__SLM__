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
- **`task_profile: structured_extraction`** with
  **`scoring_mode: span_set`** — eval automatically produces
  per-class precision / recall / F1 per entity type (the
  compliance-grade signal) alongside JSON validity and schema
  compliance. No further config needed.
- **Pre-filled Autopilot brief** — open the Autopilot tab and click
  Preview Plan; the brief is already in the textarea.
- **Adapter preset**: `structured-extraction` data adapter, mapped to
  `text → entities_json`.
- **Target profile**: `vllm_server` with the `balanced` plan profile —
  optimised for recall over latency since false negatives leak PII.
- **Sample Predictions card** shows the entity-by-entity breakdown
  per row (matched / missed / hallucinated) so failure modes are
  visible without leaving the Evaluation tab.

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

The PII demo's `output_schema` declares
**`scoring_mode: "span_set"`**, which switches the
`StructuredExtractionHandler` (Phase 5.3.4 + 5.3.4b) into entity-level
NER scoring instead of the default whole-blob field comparison. This
is the load-bearing config — without it, you'd see a single coarse F1
that hides which entity type is weak. With it, you get the per-class
metrics compliance teams actually evaluate.

### Metrics emitted per eval run

**JSON-shape gates** (same in both scoring modes):

- **`json_validity_rate`** — fraction of rows where the model's
  output parses as JSON, after stripping ` ```json ` code fences and
  pulling the first balanced `{…}` block out of prose-and-JSON
  mixes. A 30% malformed rate makes the model unshippable regardless
  of entity accuracy.
- **`schema_compliance_rate`** — fraction of rows where the parsed
  object has every required field (here: `entities`).

**Entity-level matching** (`span_set` mode):

- **`precision`** — micro precision over all entities across all
  rows. *Of every entity the model claimed, how many were real?* Low
  precision = false positives = redaction destroys legitimate text.
- **`recall`** — micro recall. *Of every entity in the gold, how
  many did the model find?* Low recall = PII leaks the firewall
  failed to catch. **This is the metric compliance cares about most.**
- **`f1`** — micro F1 (harmonic mean of P and R).
- **`precision_macro` / `recall_macro` / `f1_macro`** — unweighted
  means across entity types. Treats every type equally regardless of
  support count, which is right for PII: SSN matters even if rare
  in your dataset.
- **`exact_match`** — row-level whole-set equality (every gold entity
  matched, none missed, none hallucinated). Strictest possible
  signal; useful as a ship/no-ship gate.

**Per-class breakdown** (the headline diagnostic):

```json
"per_class": {
  "email":          { "precision": 0.99, "recall": 0.98, "f1": 0.985, "support": 312, "tp": 305, "fp": 3,  "fn": 7  },
  "credit_card":    { "precision": 1.00, "recall": 0.997, "f1": 0.999, "support": 89,  "tp": 89,  "fp": 0,  "fn": 0  },
  "ssn":            { "precision": 0.91, "recall": 0.62, "f1": 0.737, "support": 47,  "tp": 29,  "fp": 3,  "fn": 18 },
  "person_name":    { "precision": 0.92, "recall": 0.95, "f1": 0.935, "support": 410, "tp": 388, "fp": 32, "fn": 22 }
}
```

This is what you stare at to improve the model. The fictional row
above says SSN recall is 62% — that's a leak. You'd:

1. Add more SSN training examples (use the [bundle generator](#bundle-generator-synthetic-expansion) or import from one of the [HF datasets](#huggingface-datasets) filtered for SSN-heavy rows).
2. Re-train.
3. Check `per_class.ssn.recall` next eval, not overall F1.

### What "strict matching" means

A predicted entity counts as a true positive **only if** there's a
gold entity with the **same `(type, start, end)`**. Off-by-one
boundary errors count as a miss + a hallucination. Type mismatches
(same span, wrong type) likewise. This is the right contract for
redaction: a "John" prediction for a gold "John Smith Jr." span
breaks redaction just as badly as missing the span entirely.

Duplicate entities use multiset semantics — if the same email
appears twice in the text and both are gold, the model has to find
both. One prediction can't free-pass two gold spans.

### Compliance-grade gating

To gate on per-class recall (e.g. "minimum 99% credit_card recall
before ship"), add a gate to your eval pack that keys on the
per-class metric path:

```yaml
gates:
  - id: min_credit_card_recall
    metric: per_class.credit_card.recall
    operator: gte
    threshold: 0.99
    required: true
  - id: min_ssn_recall
    metric: per_class.ssn.recall
    operator: gte
    threshold: 0.995
    required: true
  - id: min_email_recall
    metric: per_class.email.recall
    operator: gte
    threshold: 0.98
    required: true
```

These gates compose with the strict-mode autopilot — if any required
gate fails, deployment promotion is blocked.

### Sample Predictions card

The Evaluation tab's Sample Predictions card swaps into entity-level
mode automatically when `scoring_mode: span_set` is detected on the
preview rows:

- Inline counts per row: *"3 matched · 1 missed · 0 hallucinated · P 1.00 · R 0.75"*. Missed/hallucinated counts render in red when > 0.
- "Show entity-by-entity breakdown" disclosure: per-entity table
  with status badge (**✓ matched** in green, **✗ missed** /
  **✗ hallucinated** in red), entity type, text, and offset range.
  Every failure mode is visible without leaving the page.

Plus the JSON-shape badges still render:
**"JSON: valid"** / **"JSON: malformed"** badge for the parser
result, and a red *"missing: entities"* note if the model emits an
object that doesn't even have the required field.

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

#### Faster path: hf locator + `--auto`

The generic dataset-import pipeline can sniff the same HF dataset and
auto-pick the right mapper without spelling out `--field-map`:

```sh
# Inspect what the introspector sees on the HF dataset (streams the
# first 20 rows — no full download).
python -m app.cli.dataset_import introspect \
  --locator hf:ai4privacy/pii-masking-200k:train

# Land 5k rows in the project's synthetic dataset.
python -m app.cli.dataset_import run \
  --locator hf:ai4privacy/pii-masking-200k:train \
  --project <id> --auto --limit 5000
```

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) before running for gated
datasets. The introspector falls into `bio_to_spans` when the dataset
ships BIO-tagged tokens + labels, or `label_to_classification` when it
ships a flat `{text, label}` shape — see the
[generic dataset-import pipeline](../workflows/data-ingestion.md#generic-dataset-import-pipeline-sources-mappers)
docs for the full mapper catalog.

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
python backend/data/demo_samples/pii-detector/kaggle_pii_to_brewslm.py \
  --input train.json \
  --out backend/data/imports/kaggle-pii.jsonl
brewslm dataset import \
  --jsonl backend/data/imports/kaggle-pii.jsonl \
  --project-slug pii-detector \
  --adapter-preset structured-extraction
```

#### Faster path: `kaggle:` locator + `--auto`

The generic dataset-import pipeline can fetch + extract + introspect
the same competition in one command — no unzip, no bespoke converter:

```sh
# Auth once (or drop a kaggle.json into ~/.kaggle/).
export KAGGLE_USERNAME=<user>
export KAGGLE_KEY=<key>

# Inspect first — streams the first 20 rows for the sniffer.
python -m app.cli.dataset_import introspect \
  --locator 'kaggle:competition:pii-detection-removal-from-educational-data'

# Land 5k rows directly into the project's synthetic dataset.
python -m app.cli.dataset_import run \
  --locator 'kaggle:competition:pii-detection-removal-from-educational-data' \
  --project <id> --auto --limit 5000
```

The connector caches the download under `~/.cache/brewslm/kaggle/`,
so re-runs skip the network. Multi-file archives get disambiguated
via `?file=<path>` on the locator.

The converter ships alongside the demo bundle generator at
[backend/data/demo_samples/pii-detector/kaggle_pii_to_brewslm.py](https://github.com/anugram/__SLM__/blob/main/backend/data/demo_samples/pii-detector/kaggle_pii_to_brewslm.py).
It walks each essay's BIO-tagged tokens, merges B-X / I-X runs into
single spans, reconstructs character offsets (preferring `full_text`
alignment, falling back to token + `trailing_whitespace` reconstruction
when alignment drifts), and maps Kaggle's tag vocabulary
(`NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`,
`URL_PERSONAL`, `STREET_ADDRESS`) onto the demo's entity types. Pure
stdlib — no extra installs. Use `--limit N` to test against a small
sample before running the full ~22k-essay set.

#### Skip the converter with `--auto`

The same Kaggle file works directly via the generic dataset-import
pipeline — no domain-specific converter needed. The schema
introspector sniffs the BIO-tagged tokens + labels columns and proposes
a `bio_to_spans` mapping automatically:

```sh
# Inspect what the introspector sees before committing.
python -m app.cli.dataset_import introspect \
  --locator jsonl:./train.json

# Run with --auto: it picks `bio_to_spans`, populates field_map.
python -m app.cli.dataset_import run \
  --locator jsonl:./train.json \
  --project 1 \
  --auto
```

The introspector requires confidence ≥ 0.8 to proceed without
`--force`. Run `introspect` first to see the rationale; if the
proposal is correct but confidence is borderline (e.g. a tiny sample
or non-conventional column names), re-run with `--force`. To override
just the entity-type vocabulary, layer `--map-json` on top of `--auto`:

```sh
python -m app.cli.dataset_import run \
  --locator jsonl:./train.json \
  --project 1 \
  --auto \
  --map-json '{"entity_type_map": {"NAME_STUDENT": "person_name", "EMAIL": "email", "PHONE_NUM": "phone_number"}}'
```

See the [Schema introspection](../reference/glossary.md#schema-introspection)
glossary entry for the architecture.

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
import json

model = LocalModel.load("exports/pii-detector/onnx-int8")

# Per-class confidence thresholds — set these from your eval pack's
# per_class metrics. The numbers below are illustrative; pull yours
# from your latest eval run's metrics.per_class report.
PER_CLASS_THRESHOLD = {
    "credit_card": 1.00,   # at 99.7% recall, near-zero FP — accept all
    "ssn":         1.00,
    "email":       0.98,   # high precision; some FP risk — slight discount
    "person_name": 0.85,   # lower precision; common false positives
}

def brewslm_scanner(text: str) -> list[dict]:
    """Conforms to LlamaFirewall's scanner interface."""
    response = model.generate(text, max_new_tokens=512)
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return []  # malformed output — let downstream regex scanners catch the obvious ones
    out: list[dict] = []
    for e in parsed.get("entities", []):
        start, end = e.get("start"), e.get("end")
        # Hallucinated-offset guard: the trained model's strict
        # exact_match metric enforces this on eval data, but a
        # production text might still trip it. Drop entities whose
        # claimed text doesn't match the actual span.
        if text[start:end] != e.get("text"):
            continue
        out.append({
            "category": e["type"],
            "start": start,
            "end": end,
            "matched_text": e["text"],
            "score": PER_CLASS_THRESHOLD.get(e["type"], 0.9),
        })
    return out

fw = Firewall(
    input_scanners=[ScannerConfig(name="brewslm_pii", scanner=brewslm_scanner)],
)
result = fw.scan_input("Hi, I'm Jane Doe at jane@example.com")
# result.detected → [{"category": "person_name", ...}, {"category": "email", ...}]
```

The contract on BrewSLM's side: emit valid JSON, every entity has
`{type, start, end, text}`, and `text == input[start:end]`. The
`StructuredExtractionHandler`'s gates enforce the first two on eval
data (`json_validity_rate`, `schema_compliance_rate`); the third is
the offset-sanity check in the scanner wrapper above.

**Why per-class thresholds matter**: a class with 99%+ recall on
your eval should be trusted (low score discount); a class with 85%
precision should have its score discounted before LlamaFirewall
decides whether to redact. Read the per-class numbers directly off
your eval and update the threshold table — there's no substitute for
measuring on representative data.

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

# 5. Inspect the metrics on the Evaluation tab. The headline you
#    care about is `per_class.<type>.recall` — overall F1 hides
#    which type is weak.
#       - credit_card recall ≥ 0.99 → shippable
#       - ssn recall ≥ 0.99      → shippable
#       - email recall ≥ 0.98    → shippable
#       - person_name recall ≥ 0.92 → acceptable
#    If any class is below its target, look at Sample Predictions →
#    "Show entity-by-entity breakdown" to see what the model missed
#    or hallucinated for that class, then add training examples
#    (synthetic via the bundle generator, or import more from HF /
#    Kaggle for that specific class) and re-train.

# 6. Set per-class gates on your eval pack (see "Compliance-grade
#    gating" above) so the autopilot blocks ship if any class
#    regresses below threshold.

# 7. Export
brewslm export onnx --project pii-detector --quantize int8

# 8. Drop the exported model into your LlamaFirewall scanner config,
#    setting PER_CLASS_THRESHOLD from your final eval's per_class
#    precision numbers.
```

## What's not in the demo (yet)

Entity-level span scoring **is** in the demo (Phase 5.3.4b — see
[How scoring works](#how-scoring-works) above). Genuine remaining gaps:

- **Partial-credit / token-IoU boundary scoring** — the current
  matching is strict: same `(type, start, end)` or it's a miss.
  Off-by-one boundary errors get no partial credit. For tuning
  early checkpoints, a token-IoU partial-match metric would give
  a smoother optimization signal. Strict is the right contract for
  shipping; partial would be a useful diagnostic alongside.
- **Bipartite optimal matching for type-overlap mode** — strict
  matching is exact-key only. A "looser" mode (type matches +
  spans overlap, but boundaries differ) needs bipartite matching
  to avoid double-counting and isn't shipped.
- **Multi-language coverage** — the bundle is English-only. The HF
  datasets in [Expanding the dataset](#huggingface-datasets) ship
  multi-language data; the span_set handler works unchanged on
  non-English rows, just feed it different inputs.
- **Production rate-limits** — if you plug into LlamaFirewall and
  the scanner becomes a hotspot, export to ONNX-INT8 and serve on
  CPU; latency is ~10ms / row for short snippets after quantisation.
- **PII-specific eval pack** — the demo defaults to
  `evalpack.general.default`. A dedicated `evalpack.pii.default`
  with per-class recall gates pre-configured (see [Compliance-grade
  gating](#compliance-grade-gating)) would let users ship with a
  one-line pack reference instead of writing their own gates. Easy
  follow-up; not in this commit.
