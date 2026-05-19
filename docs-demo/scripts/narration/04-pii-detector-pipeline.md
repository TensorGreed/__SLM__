# PII Detector Pipeline Narration Skeleton

Status: every pipeline step is to-verify until evidence is mapped and recorded.

Opening:
- This is the official `pii-detector` sample from `backend/data/demo_samples/pii-detector`.
- The manifest says the task profile is `structured_extraction`.
- The source CSV uses `text` and `entities_json`.
- The output schema uses span-set scoring.

To verify in UI:
1. Demo tile selection.
2. Raw text and entity JSON.
3. Gold rows and expected entity spans.
4. Prepared manifest schema/entity types.
5. Synthetic span generation if runtime is configured.
6. Evaluation scoring.
7. Final JSON output shape.

Narration caveat:
- Detection is verified by sample schema; removal or masking is not verified yet.

