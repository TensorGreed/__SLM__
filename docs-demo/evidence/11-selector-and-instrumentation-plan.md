# Selector And Instrumentation Plan

Discovery date: 2026-05-19.

Companion to `11-selector-route-evidence.md` (the *as-observed*
record of which selectors work today). This file is the
*proposed-changes* record: what stable selectors would make
recordings less brittle, where to add them, what the risk is, and
what app behavior **must not change** to ship them.

**No product code has been modified.** This file is purely a
proposal pending approval. If approved, the changes are markup-only
(adding `data-testid`); no behavior delta.

## Status legend

- **already-stable** — current selector is sufficient; no action.
- **proposed-add** — recommended `data-testid` addition; safe markup
  change.
- **proposed-rename** — current `data-testid` exists but could be
  renamed for clarity; medium-risk because Playwright specs depend on
  the old name.
- **deferred** — proposal exists but is not blocking any near-term
  recording.

## Recording-blocking UI actions, current selector, and verdict

| UI action | Current selector observed | Status | Risk if left as-is | Proposal |
|---|---|---|---|---|
| Login username input | `getByPlaceholder("Enter your username")` | already-stable | Placeholder text could be renamed in a future translation pass. | None now; revisit if i18n ships. |
| Login password input | `getByPlaceholder("API Key or Password")` | already-stable | Same as above. | None now. |
| Login submit | `getByRole("button", { name: /^Sign in$/ })` | already-stable | Button label is short and unlikely to change. | None now. |
| Demo tile container | `.demo-project-tiles` | already-stable | CSS class only; not a styling-driven name. | None now. |
| Demo tile button (one of three) | `.demo-project-tile` filtered by visible text `Support FAQ` / `PII / PCI Detector` / `Sentiment classifier` | proposed-add | Selector currently couples to the manifest's display name (with the centered-dot punctuation). A docs-only rename of the sample name would break recordings silently. | Add `data-testid="demo-tile-<slug>"` where `<slug>` is `support-faq`, `pii-detector`, `sentiment-classifier`. |
| Raw document expander | `[data-testid^="expand-doc-<id>"]` | already-stable | Concrete ids (`expand-doc-20`, `expand-doc-61`, `expand-doc-91`) observed; the prefix selector is robust. | None. |
| Pipeline tab button (any of 10) | `button.tab[title="Data"]` / `title="Cleaning"` / etc. | already-stable | Title attribute could be removed by a future a11y refactor. | Add `data-testid="pipeline-tab-<key>"` where `<key>` is one of `data` / `cleaning` / `goldset` / `synthetic` / `dataprep` / `tokenization` / `training` / `eval` / `compression` / `export`. **Strongly recommended** — Videos 03–07 each cycle through every tab. |
| Essentials vs Advanced training-config toggle | `getByRole("button", { name: /Advanced/ })` (inferred from `ProjectTrainingConfigPage.tsx:36-54`) | proposed-add | Button label "Essentials"/"Advanced" is short and could collide with another control. | Add `data-testid="training-config-mode-essentials"` and `-advanced`. |
| Training Config Base Model input | placeholder-driven (text input within the Config tab) | proposed-add | Default value defaults to hardcoded `microsoft/phi-2` (Story 1.7 known UX bug; line 734 of `TrainingPanel.tsx`); selectors that rely on the *value* will break when this default changes. | Add `data-testid="training-config-base-model"` to the input. |
| Synthetic generation: Source-text textarea | inferred from `SyntheticPanel.tsx` | proposed-add | Form fields have no stable selectors today. | Add `data-testid="synthetic-source-text"`, `data-testid="synthetic-num-rows"`, `data-testid="synthetic-entity-types"`, `data-testid="synthetic-generate-btn"`. |
| Eval result schema-mismatch banner | `[data-testid="eval-schema-mismatch-banner"]` | already-stable | Added by Story 1.5 commit `92cf7a5`. | None. |
| Recommender data-shape blocked banner | `[data-testid="recommender-data-shape-banner"]` | already-stable | Added by Story 1.5 commit `92cf7a5`. | None. |
| Experiment row Reset button | `[data-testid="experiment-reset-<id>"]` | already-stable | Added by Story 1.7 commit `65a439a`. | None. |
| Experiment row Delete button | `[data-testid="experiment-delete-<id>"]` | already-stable | Added by Story 1.7 commit `65a439a`. | None. |
| Bulk-archive-failed banner button | `[data-testid="bulk-archive-failed-button"]` | already-stable | Added by Story 1.7 commit `65a439a`. | None. |
| Annotation Promote-to-synthetic button | `[data-testid="annotate-promote-synthetic"]` | already-stable | Added by Story 1.6 commit `8c5d109`. | None — orthogonal to demo recordings. |
| Annotation Promote-to-gold button | `[data-testid="annotate-promote-gold"]` | already-stable | Same. | None. |
| Span labeler & classification labeler controls | `[data-testid^="classification-label-"]`, `[data-testid^="span-mark-"]`, `[data-testid="span-submit"]`, etc. | already-stable | Added by Story 1.2/1.3. | None — orthogonal to demo recordings. |

## Proposed `data-testid` additions (consolidated)

These eight markup-only additions would make recordings 03–08
visibly less brittle. None of them changes app behavior. None of
them changes visible UI text or layout. Risk is low; review is markup-
only.

| # | Component file | Element | Proposed `data-testid` value | Why |
|---|---|---|---|---|
| 1 | `frontend/src/components/dashboard/DemoProjectTiles.tsx` | each `.demo-project-tile` | `demo-tile-<slug>` (one of `support-faq`, `pii-detector`, `sentiment-classifier`) | Decouples recordings from the manifest display name + centered-dot punctuation. |
| 2 | `frontend/src/components/layout/ProjectSidebar.tsx` (or wherever `PIPELINE_TABS` renders) | each `button.tab` | `pipeline-tab-<key>` (matches `TabKey` in `frontend/src/types/index.ts:610`) | Videos 03–07 each cycle through 10 tabs. Stable id eliminates `title` coupling. |
| 3 | `frontend/src/pages/ProjectTrainingConfigPage.tsx` | Essentials/Advanced mode buttons | `training-config-mode-essentials` and `training-config-mode-advanced` | Video 03+ flips this toggle. |
| 4 | `frontend/src/components/training/TrainingPanel.tsx` | Base Model input | `training-config-base-model` | Video 03+ types here; default-value brittleness (`microsoft/phi-2`) is a known bug. |
| 5 | `frontend/src/components/data/SyntheticPanel.tsx` | Source-text textarea | `synthetic-source-text` | Video 04 + sample videos. |
| 6 | `frontend/src/components/data/SyntheticPanel.tsx` | Rows-to-generate number input | `synthetic-num-rows` | Same. |
| 7 | `frontend/src/components/data/SyntheticPanel.tsx` | Entity-types input | `synthetic-entity-types` | Same. |
| 8 | `frontend/src/components/data/SyntheticPanel.tsx` | Generate button | `synthetic-generate-btn` | Same. |

## Risk assessment

| Item | Risk level | Reasoning |
|---|---|---|
| Add 8 `data-testid` props | **low** | Markup-only attribute. Renders with empty effect on visual layout. No state, no behavior. |
| Renumber existing `data-testid` props | **n/a** | Not proposed. |
| Remove existing selectors | **n/a** | Not proposed. |
| Any backend / API change | **n/a** | None proposed. |

## Are app behavior changes required?

**No.** Every proposal in this file is a `data-testid` markup
addition. No router change, no state-shape change, no API contract
change, no styling change.

## Sequencing recommendation

1. **Record Video 02 first** with the *current* selectors. They work
   — selector pass on 2026-05-19 proves it. The screenshots from
   Codex's pass back up the selectors and confirm visual layout.
2. **Then** open a small PR adding the 8 `data-testid` additions
   above. Each is a one-line markup change with no test impact.
3. **Then** record Videos 03–07 with the new ids as the
   first-choice selector and the title-based selectors as the
   fallback.

This ordering avoids the dependency of Video 02 on a markup PR
landing first.

## What this file does not propose

- Adding `data-testid` to every leaf input in the app — that's
  scope-creep.
- Renaming any existing `data-testid`.
- Refactoring `PIPELINE_TABS` or the `DemoProjectTiles` component
  beyond adding the attribute.
- Adding any test-only fixture / seed / route. Recording remains
  evidence-driven against the real app.

## Cross-references

- `docs-demo/evidence/10-open-questions.md` Q32: "Which routes need
  stable `data-testid` attributes before reliable recording?" — this
  file is the answer.
- `docs-demo/evidence/11-selector-route-evidence.md` "Selector
  caution" section: flagged the demo-tile selector as the
  highest-value addition; this file picks that up plus seven more
  identified during the per-sample recording-plan write-up.
