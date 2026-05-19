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

Open questions:
- Which trained artifact is selectable in playground?
- Which serve template works locally?
- Which API endpoint should be used for final model smoke tests?
- Which deployment target is credential-free?

