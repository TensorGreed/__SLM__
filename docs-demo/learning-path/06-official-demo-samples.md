# Official Demo Samples

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

