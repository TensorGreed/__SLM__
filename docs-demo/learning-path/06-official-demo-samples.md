# Official Demo Samples

Three folders under `backend/data/demo_samples/` are the only official
templates: `support-faq`, `pii-detector`, `sentiment-classifier`.
Every claim below is backed by either the manifest, the source file,
or a selector-discovery pass on 2026-05-19.

| Sample | Source rows | Gold rows | Task profile | Target | Seeded split | Video |
|---|---:|---:|---|---|---:|---|
| `support-faq` | 20 | 200 | `instruction_sft` | `vllm_server` | 16 / 2 / 2 | Video 05 |
| `pii-detector` | 61 | 200 | `structured_extraction` | `vllm_server` | 45 / 8 / 8 | Video 06 |
| `sentiment-classifier` | 30 | 200 | `classification` | `mobile_cpu` | 22 / 4 / 4 | Video 07 |

> **Manifest text vs file truth**: each sample's `manifest.json`
> description currently mentions a smaller gold count (e.g. "6
> hand-labelled gold rows" for support-faq). The 200-row figure is
> measured from the file and is the source of truth; the manifest
> description text is stale. Open Q2 in
> `docs-demo/evidence/10-open-questions.md`.

## support-faq

Learning goal: show an instruction-style support assistant from ticket Q&A.

Dataset story: `tickets.csv` has `question,answer` rows. The manifest uses `instruction_sft` and maps input to `question`, output to `answer`.

Pipeline story: seeding creates raw documents, locked gold rows, and prepared splits. Later stages need runtime verification.

UI demo story: seed from the demo tile, inspect source Q&A, inspect gold rows, show dataset prep and training config.

Final model usage story: to verify. Likely playground or export/serve flow after a successful run.

Evidence needed: screenshots of source rows, gold rows, prepared split UI, and any successful training/eval/export path.

## pii-detector

Learning goal: show structured extraction for sensitive spans.

Dataset story: `pii_records.csv` has `text` and `entities_json`. Manifest defines `output_schema.scoring_mode=span_set` and entity types.

Pipeline story: seeding forwards schema/entity types into the prepared manifest and uses the structured extraction adapter.

UI demo story: inspect span JSON, explain offsets, show schema/entity types, then verify synthetic span generation and evaluation.

Final model usage story: to verify. A real ending could be JSON entity extraction for redaction/guardrails if runtime is working.

Evidence needed: prepared manifest screenshots/API output, span eval behavior, and proof of model output shape.

## sentiment-classifier

Learning goal: show a small three-way classifier.

Dataset story: `reviews.csv` has `text,label`. Manifest labels are positive, neutral, negative.

Pipeline story: seeding uses the classification adapter and forwards labels into the prepared manifest.

UI demo story: inspect review labels, run/inspect dataset prep, train/evaluate if runtime is available.

Final model usage story: to verify. The manifest says mobile CPU and ONNX-INT8 export, but this must be proven by export evidence.

Evidence needed: label handling, eval metrics, export format list, and successful export output.

