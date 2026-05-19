# Using A Trained Model

Status: unknown/partial.

Possible usage surfaces found:
- Playground: `/project/{id}/playground`.
- Export panel local serve plans and serve runs.
- Registry register/promote/deploy.
- Deployment assistant with telemetry, drift checks, and scores.

Evidence:
- `ProjectPlaygroundPage.tsx`
- `ChatPlaygroundPanel.tsx`
- `ExportPanel.tsx`
- `ProjectDeploymentsPage.tsx`
- `backend/app/api/export.py`
- `backend/app/api/registry.py`
- `backend/app/api/deployments.py`

Open questions (mirrors `10-open-questions.md` Q26–Q30):
- Which trained artifact is selectable in playground?
- Which serve template works locally?
- Which API endpoint should be used for final model smoke tests?
- Which deployment target is credential-free?

## Maps onto Video 12

This file is the conceptual companion to **Video 12 — Final Model
Usage** in `docs-demo/evidence/03-video-series-plan.md`. The recording
plan lives at
`docs-demo/videos/07-final-model-usage/recording-plan.md` (currently a
skeleton; will flesh out after Q25 — local serve runtime decision —
resolves).

## Verified surface inventory (2026-05-19 selector pass)

The selector pass confirmed these surfaces *render*; it did not
confirm they produce useful output without a real artifact:

- `/project/<id>/playground` renders with the
  `ChatPlaygroundPanel` mounted.
- `/project/<id>/pipeline/export` shows the export-format dropdown
  (`gguf`, `onnx`, `tensorrt`, `huggingface`, `docker`) and the
  registry side-panel.
- `/project/<id>/deployments` renders the deployment versions panel.

For real proof of a trained artifact reaching one of these surfaces,
see the end-to-end checklist in
`docs-demo/evidence/09-final-model-usage-plan.md`.

