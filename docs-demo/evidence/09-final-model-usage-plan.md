# Final Model Usage Plan

## Verified Or Partial Usage Surfaces

| Usage surface | Evidence | Status | Notes |
|---|---|---|---|
| Prompt playground | `frontend/src/pages/ProjectPlaygroundPage.tsx`, `frontend/src/components/training/ChatPlaygroundPanel.tsx`, `backend/app/api/training.py` playground endpoints | partial | UI exists for runtime adapters, prompt presets, feedback logging. Need verify model selection after training/export. |
| Export package | `frontend/src/components/export/ExportPanel.tsx`, `backend/app/api/export.py`, `backend/app/services/export_service.py` | partial | Export creation and run endpoints exist. Need verify formats and local output. |
| Local serve plan/run | `ExportPanel.tsx`, `backend/app/api/export.py`, `backend/app/api/registry.py`, `backend/app/services/serve_service.py` | partial | Serve-plan and serve-runs endpoints exist. Need verify prerequisites and commands. |
| Registry | `ExportPanel.tsx`, `backend/app/api/registry.py` | partial | Register, list, promote, deploy, and serve-plan APIs exist. Need verify UI path and gates. |
| Deployment assistant | `frontend/src/pages/ProjectDeploymentsPage.tsx`, `backend/app/api/deployments.py` | partial | Deployment versions, telemetry, drift checks, score APIs exist. Telemetry is push-only. |
| API usage of deployed model | `backend/app/api/export.py`, `backend/app/api/deployments.py` | unknown | Need actual serve/deploy run before claiming. |

## Export Formats

Evidence is incomplete in this pass. `ExportPanel.tsx` and `backend/app/api/export.py` prove export APIs exist, but the exact supported formats and successful local outputs need deeper inspection and a real run.

## Limitations And Unknowns

- Which model artifact is produced by a training run in local demo mode.
- Whether the playground can directly call a newly trained model without export/serve setup.
- Which export format is best for each sample.
- Which deployment target works without external credentials.
- Whether registry promotion gates pass with demo metrics.

## Evidence Needed Next

- Inspect export schemas/models for format enum values.
- Run one short training path with explicit runtime mode.
- Create one export and inspect output manifest.
- Start one serve plan in dry-run first.
- Test playground against the resulting runtime.

