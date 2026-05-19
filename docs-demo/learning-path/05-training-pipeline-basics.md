# Training Pipeline Basics

Status: partial.

## Verified Surfaces

- Training Config route: `/project/{id}/training-config`.
- Training runs appear in the pipeline training tab.
- Backend training APIs include experiment creation, effective config, preflight, start, cancel, reset, status, manifest, clone, compare, and task polling.

Evidence:
- `frontend/src/pages/ProjectTrainingConfigPage.tsx`
- `frontend/src/components/training/TrainingPanel.tsx`
- `backend/app/api/training.py`

## Runtime Reality

Training is runtime-dependent. `backend/app/config.py` defaults to external training and does not enable simulated training by default. Recorded demos must explicitly say whether training is real, simulated, or skipped.

## Demo Teaching Path

1. Explain the prepared dataset.
2. Pick task/profile/model settings.
3. Run preflight.
4. Start a run only after runtime prerequisites are known.
5. Monitor metrics/logs.
6. Preserve the run manifest.

