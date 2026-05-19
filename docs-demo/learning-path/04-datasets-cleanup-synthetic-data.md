# Datasets, Cleanup, And Synthetic Data

## Dataset Setup

Evidence:
- `IngestionPanel.tsx`
- `DatasetImportWizard.tsx`
- `backend/app/api/ingestion.py`
- `demo_project_service.py`

Official demos are seeded from CSV files. Custom datasets can likely use upload/import paths, but custom recording flows need verification.

## Ingestion

Supported surfaces include upload, batch upload, remote inspect/import, queued remote import, document listing, document sample, and EDA.

## Cleanup

Evidence:
- `CleaningPanel.tsx`
- `backend/app/api/cleaning.py`

The UI starts async batch cleaning and polls task status. Exact cleaning transforms need deeper service inspection before narration.

## Deduplication

Status: partial.

Evidence:
- `GoldSetWorkbenchPanel.tsx` mentions dedup by row content when sampling rows into a gold set.
- Dataset semantic analysis may identify duplicate-like clusters, but demo-worthy dedup behavior needs verification.

## Normalization

Status: partial.

Evidence:
- Remote import has `normalize_for_training`.
- Demo seeder canonicalizes rows into `text`, `source_text`, and `target_text`.
- Dataset adapters preview and mapping acceptance exist.

## Gold Set

Official demos include `gold.jsonl`. The seeder creates a locked gold dev dataset and approved rows.

## Synthetic Data

Evidence:
- `SyntheticPanel.tsx`
- `backend/app/api/synthetic.py`
- `backend/app/services/synthetic_service.py`

Supported endpoints include Q&A pairs, conversations, spans, and async span tasks. Runtime depends on teacher model or demo fallback settings.

## Dataset Prep

Evidence:
- `DatasetPrepPanel.tsx`
- `backend/app/api/dataset.py`

The UI includes preview, profile, semantic analysis, adapter catalog/preview, and split.

## Quality Checks

Evidence:
- Adapter preview validates mapping/contract.
- Evaluation packs and gates exist later in the pipeline.

## What Slides Should Show

- Raw rows -> cleaned chunks -> gold rows -> synthetic examples -> prepared splits.
- Mark seeded demo behavior separately from user-triggered behavior.
- Show evidence references on speaker notes.

## What Browser Demo Should Show

- One official sample seeded from a tile.
- Raw data inspection.
- Gold set inspection.
- Dataset prep and split evidence.
- Synthetic generation only when runtime is configured.

## Evidence Needed

- Exact selectors and stable UI states.
- Cleaning output examples.
- Synthetic output examples.
- Dataset split preview for each sample.

