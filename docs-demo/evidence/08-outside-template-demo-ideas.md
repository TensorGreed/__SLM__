# Outside-Template Demo Ideas

These ideas are not official demo templates. The official templates remain only:
- `backend/data/demo_samples/pii-detector`
- `backend/data/demo_samples/sentiment-classifier`
- `backend/data/demo_samples/support-faq`

| Idea | Status | Evidence | Notes |
|---|---|---|---|
| Bring your own CSV dataset through import wizard | partially supported | `DatasetImportWizard.tsx`, `IngestionPanel.tsx`, `backend/app/api/ingestion.py` | UI/API exist; exact demo flow needs selector/API mapping. |
| Upload local files and inspect document samples | supported by current repo | `IngestionPanel.tsx`, `DocumentSampleAccordion.tsx`, `ingestion.py` | Needs real browser pass. |
| Custom cleanup rules | needs verification | `CleaningPanel.tsx`, `cleaning.py`, cleaning services | UI exposes cleaning config, but custom rule depth needs inspection. |
| Custom synthetic data generation | partially supported | `SyntheticPanel.tsx`, `synthetic.py`, `synthetic_service.py` | Teacher model or demo fallback env needed. |
| Custom span extraction data | partially supported | `SyntheticPanel.tsx`, `synthetic.py`, `kaggle_pii_to_brewslm.py` | PII converter is a helper script, not a template. |
| Custom evaluation pack/gates | partially supported | `EvalPanel.tsx`, `evaluation.py`, `evaluation_pack_service.py` | Pack generation endpoint exists; real UX needs verification. |
| Custom export/usage | partially supported | `ExportPanel.tsx`, `export.py`, `serve_service.py` | Export formats and runtime commands must be verified. |
| Custom final model API client | future/conceptual | export/deployment APIs exist | Need confirmed deployed/served model endpoint first. |

## Recording Rule

Any outside-template flow must start from repo-supported upload/import paths and must clearly say when a step is conceptual, simulated, or requires extra runtime configuration.

