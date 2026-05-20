# Video 06 — BYO Custom Samples · Recording Plan

Status: **shipped 2026-05-20**. Demonstrates the "bring your own
data" path: create a fresh non-demo project, upload a tiny custom
CSV, and confirm the file is staged in the project's ingestion
state. Matches the recording pipeline used for V02–V05.

## Goal

Show that you don't need to use a seeded sample to learn the
platform. A six-row coffee-shop FAQ CSV stands in for "your data";
the spec creates a new project, uploads the CSV via the canonical
API, and processes the document.

## Final length

**1:24** (audio 84s; muxed close to that). One of the shortest
videos in the series — BYO is intentionally a fast surface tour,
not a full pipeline walk.

## Sample data

`docs-demo/byo-sample/byo-coffee-shop-faq.csv` — six rows, two
columns (`question`, `answer`). Deliberately distinct from the
support-faq seeded sample so the viewer sees the platform
treating it as new data, not just re-importing a known shape.

## Spec design

1. **Pre-roll**: log in, capture JWT for API-driven upload (same
   pattern as V09 / V11).
2. **Cold open** — project list view.
3. **New project** — click **+ New Project**, switch to **Advanced
   Mode** (the default is the Beginner 3-step brief wizard which
   doesn't fit this short demo), fill Name + Description, click
   Create.
4. **Empty Data tab** — direct navigation to
   `/project/<id>/pipeline/data`. Fresh projects render the
   `GettingStartedWizard` because `progress_percent === 0` — the
   recording shows that welcome card with the "Ingest Data" step
   highlighted.
5. **Upload CSV** — `POST /ingestion/upload-batch` via the
   Playwright `request` fixture with a multipart payload. The
   spec then fetches the new document id and calls
   `POST /ingestion/documents/{docId}/process` to kick off
   extraction.
6. **Rows imported** — page reload to surface the result.
7. **Wrap**.

## Why API-driven upload

Same reasoning as V09 / V11: the UI's file picker is fragile to
drive programmatically via `setInputFiles` (the `IngestionPanel`'s
async upload + auto-refresh dance has a 10-minute hang failure
mode if anything stalls). The `/ingestion/upload-batch` endpoint
is the canonical path the UI itself calls, so the spec just hits
it directly.

## Known UX gap surfaced by this recording

On a fresh project, the `GettingStartedWizard` overlay only
dismisses when `pipelineStatus.progress_percent > 0`. The upload
+ process API calls succeed (the document is on disk and the
ingestion record exists), but pipeline status doesn't auto-
increment off zero from those calls alone — it tracks
`pipeline_stage` advancement, not document count.

Consequence: the recording's "rows imported" section still shows
the welcome wizard, even though the data is actually in the DB.
Narration acknowledges this loosely ("the rows are in") but
visually the viewer sees the same wizard panel as the empty-data
section.

If we want this video to land "perfectly," the fix is either:
1. Add a doc-count-aware branch to `showWizard` in
   `ProjectPipelinePage.tsx:68` (so any uploaded doc dismisses the
   wizard), OR
2. Have the spec click the "Ingest Data" wizard card explicitly,
   which calls `setWizardDismissed(true)` and reveals the
   underlying IngestionPanel.

Option 2 is the smaller change; deferred for now in favor of
shipping the video.

## Verification artifacts

- New project created (id varies per run; latest in this take
  was `BYO Coffee Shop FAQ 50288`, project id=8).
- 1 document at status=pending → processed → 6 logical rows.
- API JSON for the upload: `{"uploaded": 1, "errors": []}`.

## Things to not say

- Don't claim "uploading triggers automatic CSV parsing" — the
  process step is separate (the spec calls it explicitly).
- Don't promise the wizard goes away after upload — it doesn't,
  for the reason above.
- Don't read literal API endpoints aloud. Narration speaks of
  "upload zone" and "ingestion" in plain English.
