# Codex Demo Agent Instructions

Codex must inspect the repo before generating demo claims.

Official demo samples are exactly:
- `backend/data/demo_samples/pii-detector`
- `backend/data/demo_samples/sentiment-classifier`
- `backend/data/demo_samples/support-faq`

Rules:
- Do not invent other templates.
- Map each sample to real files, schemas, UI routes, APIs, and pipeline steps.
- Separate real, measured, simulated, estimated, conceptual, and unknown features.
- Cite repo evidence for every claim.
- Add `data-testid` attributes only when necessary and preferably after explicit approval.
- Avoid changing app behavior.
- Use Playwright for repeatable browser/UI recordings only after discovery is complete.
- Keep generated artifacts under `docs-demo/`.
- Create one working prototype before scaling to many videos.

