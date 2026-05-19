# Video 07: Final Model Usage Outline

This video answers:
- After training, what does the user actually do with the model?
- How do they test it?
- How do they compare it?
- How do they export or deploy it?
- How do they prove it works?

## Candidate Flow

1. Start from a completed training run.
2. Inspect run metrics and manifest.
3. Run held-out evaluation.
4. Compare runs if more than one exists.
5. Create export.
6. Register/promote model if gates pass.
7. Generate local serve plan.
8. Start or dry-run serving.
9. Send a smoke request or use playground.
10. Show deployment telemetry only if sample telemetry is available.

## Current Evidence Status

Partial:
- Playground exists.
- Export panel exists.
- Registry APIs exist.
- Deployment assistant exists.

Unknown:
- Exact trained artifact path.
- Best local serve template.
- Whether playground directly uses trained/exported model.

