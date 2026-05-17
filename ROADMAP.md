# BrewSLM Roadmap — Any-Domain Phase

The six epics below close the gaps between BrewSLM's current state and its
real-world promise: **"brew any SLM, for any domain, from zero."** The
existing pipeline handles ingest → train → eval → export when the user
already has labeled task-shaped data on HuggingFace or Kaggle. These
epics handle the harder case: the user has a domain and a problem,
nothing else.

Companion to [ROADMAP22APR.md](ROADMAP22APR.md), which captures the
pre-May-2026 sprint plan. This file is the forward backlog from
2026-05-14 onward.

---

## Status board

| Epic | Story | Status | Impact |
|---|---|---|---|
| 1. Annotation & active learning | 1.1 Annotation foundation (schema + service + API) | SHIPPED · 45e17a2 | 🔥 unblocks "any domain" |
| 1. Annotation & active learning | 1.2 Text-classification + span annotation UI | SHIPPED · a52a164 | 🔥 |
| 1. Annotation & active learning | 1.3 Preference-pair annotation UI | SHIPPED · fd34cc6 | high |
| 1. Annotation & active learning | 1.4 Active-learning ranker + IAA | NOT STARTED · needs baseline model | high |
| 1. Annotation & active learning | 1.5 Training-eval contract gates (any task) | SHIPPED · 92cf7a5 | 🔥 |
| 1. Annotation & active learning | 1.6 Promote labeled rows → training dataset | SHIPPED · 8c5d109 | 🔥 closes annotation loop |
| 1. Annotation & active learning | 1.7 Experiment lifecycle hygiene + checkpoint-resume compat gate | SHIPPED · 65a439a | 🔥 incident-driven |
| 2. Knowledge distillation | 2.1 Teacher logit capture | NOT STARTED | 🔥 differentiator |
| 2. Knowledge distillation | 2.2 KD training recipe | NOT STARTED | 🔥 |
| 2. Knowledge distillation | 2.3 Student-vs-teacher eval | NOT STARTED | high |
| 3. Closed feedback loop | 3.1 Feedback queue + ingestion | NOT STARTED | high |
| 3. Closed feedback loop | 3.2 Reviewer UI | NOT STARTED | high |
| 3. Closed feedback loop | 3.3 One-click retrain | NOT STARTED | high |
| 4. Cross-experiment slice analysis | 4.1 Slice definitions | NOT STARTED | medium |
| 4. Cross-experiment slice analysis | 4.2 Slice eval at score time | NOT STARTED | medium |
| 4. Cross-experiment slice analysis | 4.3 Slice comparison UI | NOT STARTED | medium |
| 5. Model card auto-gen | 5.1 Model card service | NOT STARTED | medium |
| 5. Model card auto-gen | 5.2 Card export to MD/HTML/PDF | NOT STARTED | medium |
| 6. Hyperparameter search | 6.1 Sweep API + grid backend | NOT STARTED | medium |
| 6. Hyperparameter search | 6.2 Pareto comparison UI | NOT STARTED | medium |

When a story moves: edit the **Status** column above and the **Status**
field on the story itself. Use `IN PROGRESS`, `SHIPPED`, `BLOCKED`,
or `DEFERRED`.

---

## How to use this file

The same story works for either agent — pick the prompt block that
matches the agent you're handing it to.

### With Claude Code (this CLI / VS Code extension)

1. Open this file.
2. Find the story.
3. Send the **Claude prompt** (one line). I'll re-read this file +
   navigate the repo + ask clarifying questions if needed.

### With ChatGPT Codex (one-shot brief)

1. Find the story.
2. Expand the `<details>` "Codex brief" block.
3. Paste the full brief into Codex. It's self-contained — file paths,
   acceptance criteria, test plan, quality gates, commit message
   template all included.
4. If the brief feels too thin for Codex, ask Claude Code:
   `"expand Story X.Y's Codex brief to a full self-contained spec"` —
   I'll write the longer version inline.

### General rules

- Each story is a vertical slice: backend + tests + (UI when relevant) +
  docs in one commit. No partial deliveries.
- Stories within an epic are ordered by dependency. Don't pick 1.2 before 1.1.
- Cross-epic dependencies are called out in each story's prereqs.

---

# Epic 1 — Annotation & active learning

**Why this is first**: every other epic in this roadmap assumes you
have labels. Today's pipeline gets labels from HuggingFace / Kaggle (a
small fraction of real-world datasets) or from synthetic generation
(useful but needs ground-truth seed labels to validate quality). The
middle path — **"I have raw documents in my domain, help me label
500 of them efficiently"** — has no in-tool surface. Closing this
unlocks ~80% of the domains that aren't already covered by public
benchmarks.

Scope contract for the epic:
- Backend stores per-project label jobs + per-row labels + reviewer
  attribution.
- UI lives on the existing Data tab; one new pipeline stage between
  *Cleaning* and *Synthetic*.
- Plugin-extensible label types (text-classification, span,
  preference-pair) — same registry/dispatcher pattern as mappers.
- Active learning is a service that ranks unlabeled rows; not a
  separate model.

---

### Story 1.1 — Annotation foundation: schema + service + API

**Status**: SHIPPED · 45e17a2
**As a** project owner
**I want** to define a label job (task shape, label set, instructions, target N rows)
**so that** the pipeline knows what we're labeling and the audit log captures it.

**Acceptance criteria:**
- New `label_jobs` table: `id`, `project_id`, `name`, `label_type`
  (`classification` | `span` | `preference_pair`), `label_schema` (JSON:
  allowed labels, span types, etc.), `instructions` (markdown),
  `status`, `target_rows`, `created_at`, `updated_at`.
- New `label_rows` table: `id`, `job_id`, `source_row_id`, `raw_payload`
  (JSON), `assigned_to` (user_id|null), `label_payload` (JSON|null),
  `labeled_at` (datetime|null), `reviewer_notes` (text).
- Service: `app/services/annotation_service.py` with `create_job`,
  `seed_rows_from_dataset(job_id, dataset_id, n)`, `assign_next(job_id, user_id)`,
  `submit_label(row_id, label_payload)`, `job_stats(job_id)`.
- API endpoints under `/api/projects/{id}/label-jobs/` for full CRUD +
  `next-row` (returns one assigned row) + `submit` (saves a label).
- Audit: each label-submit emits a RunEvent with new
  `reason_code="annotation_label_submitted"`. Job creation emits
  `annotation_job_created`. Both added to `app.models.reason_codes`.

**Files likely touched:**
- `backend/app/models/label_job.py` (new), `backend/app/models/label_row.py` (new)
- `backend/alembic/versions/20260515_0031_label_jobs_label_rows.py` (new)
- `backend/app/services/annotation_service.py` (new)
- `backend/app/api/annotation.py` (new)
- `backend/app/main.py` (register router)
- `backend/app/models/reason_codes.py` (two new codes)

**Tests:**
- `backend/tests/test_phase108_annotation_foundation.py`:
  - Create job + seed N rows from a dataset → row count matches.
  - Assign next-row returns an unlabeled row + marks it assigned.
  - Submit label persists + emits RunEvent.
  - Stats endpoint reports `total / labeled / assigned / unlabeled`.
  - Concurrent `assign_next` doesn't hand the same row to two users.

**Claude prompt:**
> Read ROADMAP.md Story 1.1 and ship it end-to-end (model + migration + service + API + tests + audit hook).

<details>
<summary>Codex brief — Story 1.1</summary>

You're working in the BrewSLM monorepo. Add the foundation for an
in-product annotation flow. Two new tables, one service, one router,
one alembic migration, full test coverage.

**Reference patterns to mirror:**
- `backend/app/models/dataset_import_config.py` for the model file
  shape (Mapped types, default factories, `__table_args__`).
- `backend/alembic/versions/20260514_0030_project_gamification.py`
  for the migration skeleton.
- `backend/app/services/dataset_import/configs.py` for service
  helpers (create / get / list / delete + idempotency).
- `backend/app/api/dataset_import.py` for the router pattern
  (`APIRouter(prefix=...)`, pydantic request models, 4xx translation).
- `backend/app/services/dataset_import/service.py:_emit_import_audit_event`
  for the best-effort RunEvent emission hook.

**Schema** — two tables:

```sql
label_jobs (
  id PK, project_id FK→projects.id (indexed),
  name VARCHAR(120) NOT NULL,
  label_type VARCHAR(32) NOT NULL,  -- 'classification' | 'span' | 'preference_pair'
  label_schema JSON NOT NULL,        -- {allowed_labels: [...], span_types: [...]}
  instructions TEXT,
  status VARCHAR(32) NOT NULL DEFAULT 'active',  -- 'active' | 'paused' | 'completed'
  target_rows INTEGER,
  created_at, updated_at
)

label_rows (
  id PK, job_id FK→label_jobs.id (indexed),
  source_row_id VARCHAR(128),
  raw_payload JSON NOT NULL,
  assigned_to INTEGER FK→users.id (nullable),
  assigned_at DateTime (nullable),
  label_payload JSON (nullable),
  labeled_at DateTime (nullable),
  reviewer_notes TEXT
)
```

**Endpoints** (mount under `/api/projects/{project_id}/label-jobs`):
- `POST   /` create
- `GET    /` list
- `GET    /{job_id}` detail + stats
- `DELETE /{job_id}`
- `POST   /{job_id}/seed-from-dataset` body: `{dataset_id, n}`
- `POST   /{job_id}/next-row` body: `{user_id}` → returns one
  unlabeled row, marks it assigned to that user (transactional!)
- `POST   /{job_id}/rows/{row_id}/submit` body: `{label_payload, reviewer_notes?}`

**Reason codes** to add to `backend/app/models/reason_codes.py`:
- `ANNOTATION_JOB_CREATED = "annotation_job_created"`
- `ANNOTATION_LABEL_SUBMITTED = "annotation_label_submitted"`
Both added to the `STAGE_INGESTION` set in `REASON_CODES_BY_STAGE`.

**Tests** in `backend/tests/test_phase108_annotation_foundation.py`:
1. Create job; seed 50 rows from a fixture dataset; assert row count.
2. Call next-row twice in two sessions — assert different row ids.
3. Submit label → row's `labeled_at` populated + RunEvent emitted with
   reason_code `annotation_label_submitted`.
4. Job stats reflect `total / labeled / assigned / unlabeled`.
5. List + delete CRUD operations.

**Quality gates:**
- `python -m pytest tests/test_phase108_annotation_foundation.py -v`
- `python -m pytest tests/test_phase101_dataset_import_foundation.py -q`
  (regression smoke for the RunEvent emit path).

**Commit message template:**
```
Annotation foundation: schema + service + API for in-product labeling

[two-table schema + service layer + endpoints + audit hook]
[Per-row submit emits annotation_label_submitted RunEvent.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
</details>

---

### Story 1.2 — Text-classification + span annotation UI

**Status**: SHIPPED · a52a164
**Depends on**: Story 1.1
**As a** human labeler
**I want** a fast, keyboard-driven UI for text-classification and span (NER) tasks
**so that** labeling a row takes seconds, not a minute.

**Acceptance criteria:**
- New page `/project/:id/annotate/:job_id` under the workspace layout.
- Keyboard shortcuts: `1`–`9` to apply single-label classification;
  `j`/`k` next/prev row; `a`/`b`/`c` for span types; `esc` cancel.
- For span tasks: drag-to-select character range, click span to delete,
  inline color-coded types matching the label_schema's `span_types`.
- Progress bar showing `labeled / target`; click skip = unassign +
  put back in queue.
- Auto-saves on submit; next row loads instantly.
- Loading + error states are first-class (use the existing toast).

**Files likely touched:**
- `frontend/src/pages/ProjectAnnotatePage.tsx` (new)
- `frontend/src/components/annotation/ClassificationLabeler.tsx` (new)
- `frontend/src/components/annotation/SpanLabeler.tsx` (new)
- `frontend/src/components/annotation/AnnotationProgress.tsx` (new)
- `frontend/src/api/annotation.ts` (new — typed wrappers)
- `frontend/src/App.tsx` (route)
- `frontend/src/pages/ProjectSidebar.tsx` (nav entry under Data)

**Tests:**
- `ClassificationLabeler.test.tsx`: keyboard shortcut applies the right label, calls submit.
- `SpanLabeler.test.tsx`: drag-to-select creates a span; click-to-delete removes it.
- `AnnotationProgress.test.tsx`: renders the right counts; updates after a submit.

**Claude prompt:**
> Read ROADMAP.md Story 1.2 and ship it. Depends on 1.1 — assume that's already merged.

<details>
<summary>Codex brief — Story 1.2</summary>

Frontend-only story. The backend (Story 1.1) is already done; the
API is at `/api/projects/{id}/label-jobs/...`. Build a keyboard-driven
labeling page.

**Reference UI patterns:**
- `frontend/src/components/data/DatasetImportWizard.tsx` for modal /
  step navigation patterns.
- `frontend/src/components/data/DocumentSampleAccordion.tsx` for the
  loading / error / refresh pattern over an API.
- `frontend/src/stores/toastStore.ts` for the toast helper.

**Component structure:**
```
ProjectAnnotatePage
├── AnnotationProgress       (header: "12 / 50 labeled" + progress bar)
└── Labeler                  (branches on job.label_type)
    ├── ClassificationLabeler (1–9 shortcuts, button strip)
    └── SpanLabeler           (drag-to-select, span color chips)
```

**API client** (`frontend/src/api/annotation.ts`):
```ts
fetchJob(projectId, jobId) → LabelJob
fetchNextRow(projectId, jobId) → LabelRow
submitLabel(projectId, jobId, rowId, payload) → void
skipRow(projectId, jobId, rowId) → void
```

**Keyboard contract:**
- Classification: `1`–`9` apply the indexed label; if labelSchema has
  10+, additional via mouse only.
- Span: highlight text → press `a`/`b`/`c`/etc for the indexed span
  type; the resulting span gets that type and color.
- Universal: `j` next row (saves + advances), `k` previous row,
  `esc` skip + unassign.

**Tests** with Vitest + RTL:
- ClassificationLabeler: mount with 3 labels, press "2", assert
  submit called with label `labels[1]`.
- SpanLabeler: simulate text-range selection, press "a", assert
  the in-component span list has one entry of type `span_types[0]`.
- AnnotationProgress: render with `total=50, labeled=12`, assert
  the text + the progress bar width.

**Quality gates:**
- `npx tsc --noEmit` clean.
- `npx vitest run src/components/annotation/` all pass.
- `npx vite build` clean (catches JSX mismatch issues that tsc
  doesn't — see commit `1690a49` for prior example).

**Constraints:**
- No emojis in code.
- Inline styles match house style elsewhere (see DatasetImportWizard).
- Add `data-testid` attrs liberally — the test plan above relies on
  them.
</details>

---

### Story 1.3 — Preference-pair annotation UI

**Status**: SHIPPED · fd34cc6
**Depends on**: Story 1.1
**As a** human labeler
**I want** to rank two model completions side-by-side
**so that** I can generate DPO/ORPO training data efficiently.

**Acceptance criteria:**
- Same `/project/:id/annotate/:job_id` page; renders `PreferencePairLabeler` when `job.label_type === 'preference_pair'`.
- Side-by-side panel: prompt at top, two completions A / B below.
- Keyboard: `←` prefer A, `→` prefer B, `=` tie / skip, `r` mark both bad.
- Optional comment field per submission.
- Submit emits a `label_payload` shaped `{chosen: "A"|"B", tie: bool, both_bad: bool, comment?: string}` per Story 1.1's contract.

**Files likely touched:**
- `frontend/src/components/annotation/PreferencePairLabeler.tsx` (new)
- `frontend/src/components/annotation/PreferencePairLabeler.test.tsx` (new)

**Claude prompt:**
> Read ROADMAP.md Story 1.3 and ship it. Depends on 1.1 + 1.2.

<details>
<summary>Codex brief — Story 1.3</summary>

One new component slotted into the `ProjectAnnotatePage` from Story 1.2.
The page already branches on `job.label_type`; you're adding the
`'preference_pair'` branch.

The job's `label_schema` for preference pair carries no allowed_labels
(it's always {A, B, tie, both_bad}). The raw_payload per row is shaped:
```json
{
  "prompt": "...",
  "completion_a": "...",
  "completion_b": "...",
  "metadata": {...optional...}
}
```

Submitted label_payload:
```json
{
  "chosen": "A" | "B",
  "tie": false,
  "both_bad": false,
  "comment": "..."  // optional
}
```

UI: 3-row grid — prompt across top, two completion panes below
(prompt-styled card each, monospace for any code content), and an
action bar at the bottom with the four buttons + a comment textarea.

Keyboard handlers: `useEffect` with `keydown` listener on `document`;
clean up on unmount. Same shortcuts contract as 1.2 (`←` `→` `=` `r`).

Tests:
- Mount with a sample row; press `←` → submit called with
  `{chosen: "A", tie: false, both_bad: false}`.
- Press `=` → `tie: true`, `chosen: null`.
- Type a comment, press `→` → submit called with the comment in
  payload.
</details>

---

### Story 1.4 — Active-learning ranker + inter-annotator agreement

**Status**: NOT STARTED
**Depends on**: Story 1.1, 1.2
**As a** project owner
**I want** unlabeled rows ranked by model uncertainty
**so that** my labelers label the 100 most informative rows, not a random 100.

**Acceptance criteria:**
- New service `app/services/annotation/active_learning.py` with
  `rank_unlabeled(job_id, *, n)` that returns up to N row ids ordered
  by uncertainty (highest first). Implementations to ship:
  - `entropy` for classification (over the project's current best
    model's softmax outputs).
  - `margin` for span (min margin between top-2 token tags).
  - `disagreement` for preference pair (when an ensemble of teacher
    models was used).
- `POST /api/projects/{id}/label-jobs/{job_id}/rank` returns the
  ordered list. The next-row endpoint optionally consumes this order
  via `?strategy=active`.
- IAA: when a row has labels from ≥2 reviewers, compute Cohen's κ
  for classification, span F1 for span. Surface in job stats.

**Files likely touched:**
- `backend/app/services/annotation/__init__.py` (new)
- `backend/app/services/annotation/active_learning.py` (new)
- `backend/app/services/annotation/iaa.py` (new)
- `backend/app/services/annotation_service.py` (extend stats endpoint)
- `backend/app/api/annotation.py` (rank endpoint + strategy query param)

**Tests:**
- `test_phase108_active_learning.py`: fixture with a fake model that
  emits known probabilities, assert ranking puts the highest-entropy
  rows first.
- `test_phase108_iaa.py`: two reviewers, three classification labels,
  expected κ; same for span F1.

**Claude prompt:**
> Read ROADMAP.md Story 1.4 and ship it. Depends on 1.1, 1.2.

<details>
<summary>Codex brief — Story 1.4</summary>

Two small services + one new endpoint. No UI changes.

**Active learning ranker** (`backend/app/services/annotation/active_learning.py`):

Strategy dispatch keyed by `job.label_type`:
- `classification` → entropy: `-sum(p_i * log(p_i))` over softmax outputs.
  Source the probabilities by running the project's current best model
  via `app/services/training_service.py:predict_probas` (helper to add
  if it doesn't exist; should be ~30 lines wrapping the existing
  inference path).
- `span` → margin: for each token, `top1_prob - top2_prob`. Row's
  uncertainty = mean of (1 - margin) over tokens.
- `preference_pair` → disagreement: when teacher ensemble exists,
  count how often A/B picks disagree across teachers. Without
  ensemble, fall back to random.

Returns `list[(row_id, score)]` sorted by score DESC.

**IAA** (`backend/app/services/annotation/iaa.py`):
- `compute_iaa(job_id) → dict`:
  - For classification: pairwise Cohen's κ for each reviewer pair
    that has overlapping rows. Report mean + per-pair.
  - For span: pairwise span-F1 (exact match on `(type, start, end)`)
    — reuse the existing `bio_to_spans` test utilities if applicable.
  - For preference_pair: agreement = fraction of rows where two
    reviewers picked the same chosen.

API change to `backend/app/api/annotation.py`:
- New: `GET /api/projects/{id}/label-jobs/{job_id}/rank?n=50` →
  `{strategy, row_ids: [...]}`
- Modify: `POST .../next-row` accepts `?strategy=active|random`
  (default `random`).
- The job stats endpoint grows an `iaa` field when ≥2 reviewers.

Tests as called out in the acceptance criteria.

Quality gates: backend pytest only for this story; no UI changes.
</details>

---

### Story 1.5 — Training-eval contract gates (any task)

**Status**: SHIPPED · 92cf7a5
**As a** project owner running any task profile (classification, span,
QA, RAG, preference-pair, summarization — anything)
**I want** the platform to detect and surface training/eval contract
drift in three places where it currently fails silently
**so that** a training run can never "complete" with eval F1 ≈ 0
because the model was trained on a different objective than the eval
measures.

**The failure class** (domain-agnostic — the same pattern bites NER
projects, classification projects, QA projects, JSON-extraction
projects, anything that has a non-trivial output schema):
- The trainer's data has shape X.
- The eval handler scores shape Y.
- X ≠ Y, no gate catches it before launch, training optimizes the
  wrong objective for hours, eval drops F1 ≈ 0 with a tiny "missing
  field" footnote buried under the headline FAIL chip.

Commit `222bc5d` plugged one of three holes (a pre-training data-shape
gate). This story plugs the other two and adds a status-state
reconciliation pass for stuck runs.

**Acceptance criteria** (three independent gates, ship together):

1. **Model-recommender data-shape gate.** When the project has a
   prepared `train.jsonl` and the existing data-shape gate would
   refuse the run, the recommendation API returns a "data shape
   blocks model choice" banner *instead of* a model list. Today the
   recommender scores only language/device/VRAM
   ([model_selection_service.py:649](backend/app/services/model_selection_service.py#L649))
   so it cheerfully recommends a model for data that no model could
   succeed on. Reuses the existing `verify_training_data_has_targets`
   helper — no task-specific logic.
2. **Eval-time schema-mismatch banner.** When an eval pass completes
   and ≥80% of predictions have a top-level JSON key set disjoint
   from the gold answers' key set, the result API surfaces a
   top-level `schema_mismatch` warning above the headline metric.
   Names the expected/observed keys + suggests checking the adapter
   contract's `target_fields`. Generic — works whether the gold
   answers have `entities`, `summary`, `intent`, `messages`,
   `chosen/rejected`, anything.
3. **Run-status reconciliation.** When `experiments.status = RUNNING`
   but the experiment's `training_report.json` has a `finished_at`
   field (subprocess exited cleanly, DB write-back didn't land), the
   read path returns the correct terminal status and a startup
   reaper updates the row. Same logic applies to any training_mode.

**Files likely touched:**
- Backend:
  - `backend/app/services/model_selection_service.py` (data-shape
    probe before scoring; return shape with `blocked_by_data_shape`)
  - `backend/app/api/models.py` (recommendation endpoint forwards
    the banner)
  - `backend/app/services/evaluation_service.py` (key-set comparison
    at eval finalize; pluggable across every task handler)
  - `backend/app/services/training_service.py` (reconciliation
    helper + reaper)
  - `backend/app/main.py` (call reaper from `lifespan` startup)
- Frontend:
  - `frontend/src/components/training/ModelRecommendations*.tsx`
    (renders banner when blocked)
  - `frontend/src/components/evaluation/EvalResultPanel.tsx`
    (banner above metrics)

**Tests:**
- `backend/tests/test_phase109_recommender_data_gate.py` — prepared
  `train.jsonl` with no target → recommendation API returns blocked
  banner; well-shaped data → model list as before. Fixtures
  parameterized across task profiles (classification, span, qa,
  preference_pair) to prove the gate is task-agnostic.
- `backend/tests/test_phase109_eval_schema_mismatch_banner.py` —
  parameterized cases: gold shape `{entities: [...]}` vs prediction
  shape `{value, label}`; gold `{summary: "..."}` vs `{text: "..."}`;
  gold `{chosen: ..., rejected: ...}` vs free-form string. Each fires
  the banner with the right key sets.
- `backend/tests/test_phase109_status_reconciliation.py` — insert a
  RUNNING experiment, write a `training_report.json` with
  `finished_at`, run the reaper, assert state transition.

**Claude prompt:**
> Read ROADMAP.md Story 1.5 and ship it — three task-agnostic gates in one vertical slice. Reuse training_data_gate; don't add task-specific logic.

<details>
<summary>Codex brief — Story 1.5</summary>

Three independent contract-drift gates, all task-agnostic. Reuse the
existing `training_data_gate` helper rather than introducing
task-specific checks. None of the three gates should mention any
specific dataset, label vocabulary, or domain.

**Gate 1 — Model-recommender data-shape gate.**

`backend/app/services/model_selection_service.py:recommend_training_base_models`
currently scores models on language / device / VRAM only. Add a
pre-scoring check:

```python
from app.services.training_data_gate import verify_training_data_has_targets

prepared = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"
if prepared.exists():
    gate = verify_training_data_has_targets(prepared, training_mode="sft")
    if not gate["ok"]:
        return {
            "blocked_by_data_shape": True,
            "data_shape_message": gate["message"],
            "recommendations": [],
        }
```

The API at `backend/app/api/models.py` forwards the
`blocked_by_data_shape` field. The frontend component renders the
banner instead of the model list when the flag is true.

**Gate 2 — Eval-time schema-mismatch banner.**

In `backend/app/services/evaluation_service.py`, after metrics are
computed, sample the first N predictions + gold answers. For each
pair, JSON-parse if possible and extract top-level keys. If ≥80% of
prediction-key-sets are disjoint from the corresponding
gold-key-set, populate `result.metrics["schema_mismatch"]`:

```python
{
    "ratio": 0.97,
    "sample_size": 100,
    "expected_top_keys": [...],         # from gold
    "observed_top_keys_top3": [
        {"keys": [...], "count": 67},
        ...
    ],
    "hint": "Predictions don't share top-level keys with gold answers; check the adapter contract's target_fields + that the training data carries the eval schema."
}
```

The eval result panel renders this above the headline metric chip
when present. No task-specific code paths — the check is purely
key-set comparison.

**Gate 3 — Run-status reconciliation.**

Two pieces:
1. Read-side helper in `training_service.get_training_status` that
   loads `training_report.json` from `exp.output_dir` and, if it has
   a `finished_at` field but `exp.status == RUNNING`, returns the
   correct terminal status and flips the DB row.
2. Startup reaper in `app.main.lifespan` that scans
   `experiments WHERE status = 'RUNNING' AND started_at < NOW() - 1h`
   for any whose `training_report.json` is on disk with a
   `finished_at`, and bumps them to the right terminal state.

**Quality gates:**
- `python -m pytest tests/test_phase109_*.py -v`
- `python -m pytest tests/test_training_data_gate.py -q`
  (regression for the gate this story builds on)
- Frontend: `npx tsc --noEmit && npx vitest run && npx vite build`

**Commit message template:**
```
Training-eval contract gates: recommender · eval schema · status

[Three task-agnostic gates in one commit — recommender refuses
when prepared train.jsonl has no target field; eval result surfaces
schema-mismatch banner above the headline metric when predictions
and gold disagree on top-level keys; RUNNING-but-finished
experiments reconcile to their correct terminal status on read and
via a startup reaper.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
</details>

---

### Story 1.6 — Promote labeled rows → training dataset

**Status**: SHIPPED · 8c5d109
**As a** project owner who labeled 200 rows via the in-product annotation UI
**I want** those labels to actually reach my training data
**so that** the annotation feature isn't a dead-end — Stories 1.1–1.3
ship a fully-functional UI, but submitted labels currently sit in the
``label_rows`` table forever and never flow into ``train.jsonl``.

**Audit confirming the gap** (verified 2026-05-15):
- `submit_label` in [annotation_service.py](backend/app/services/annotation_service.py)
  persists labels into ``label_rows``.
- No code path reads from ``label_rows`` into the synthetic dataset
  or the prepared train file.
- `combine_datasets` only pulls from ``CLEANED`` / ``SYNTHETIC`` /
  ``GOLD_DEV`` — ``label_rows`` is not a dataset_type and isn't
  considered.
- Net: a project that uses annotation as its primary data path has
  a beautiful UI and zero training signal. The PII demo project
  worked around it by using synthetic generation + a pre-bundled
  gold set, not by labeling. Without 1.6, Epic 1 is vertical-slice
  incomplete.

**Acceptance criteria:**
- Service: `app/services/annotation/promotion.py` with
  `promote_labeled_rows(db, *, project_id, job_id, target_dataset_type)`
  that writes labeled rows to the project's SYNTHETIC dataset (or
  GOLD_DEV when caller passes that). Rows get
  ``source: "annotation_job"``, ``annotation_job_id``,
  ``original_row_id``, ``reviewer_user_id`` provenance fields. Each
  label_type's `label_payload` maps to the canonical
  ``{question, answer}`` shape:
  - classification → `{question: text, answer: json.dumps({label})}`
  - span → `{question: text, answer: json.dumps({entities})}`
  - preference_pair → `{prompt, chosen, rejected}` rows for the
    alignment training path.
- API: `POST /api/projects/{id}/label-jobs/{job_id}/promote` body
  `{target_dataset_type: "synthetic" | "gold_dev"}`. Returns
  promoted_count + skipped_count + the resulting dataset's
  record_count.
- Audit: each promotion emits a RunEvent with reason
  `annotation_rows_promoted` under STAGE_INGESTION.
- UI: "Promote N labeled rows → Synthetic" button on the Annotation
  page once `stats.labeled > 0`. Confirms before promoting; toasts
  on success / failure.
- Idempotency: rows already promoted (tracked via ``promoted_at``
  column on `label_rows`) aren't re-promoted on subsequent calls;
  the response reports them as `skipped_already_promoted`.

**Files likely touched:**
- Backend:
  - `backend/alembic/versions/<next>_label_row_promoted_at.py` (new — add column)
  - `backend/app/models/label_job.py` (add ``promoted_at`` field)
  - `backend/app/models/reason_codes.py` (new code:
    `ANNOTATION_ROWS_PROMOTED`)
  - `backend/app/services/annotation/__init__.py` (new package
    boundary — make room for 1.4's `active_learning.py` later)
  - `backend/app/services/annotation/promotion.py` (new)
  - `backend/app/api/annotation.py` (new endpoint)
- Frontend:
  - `frontend/src/pages/ProjectAnnotatePage.tsx` (Promote button +
    confirm dialog)
  - `frontend/src/api/annotation.ts` (new wrapper)

**Tests:**
- `backend/tests/test_annotation_promotion.py` parameterized across
  the three label_types:
  - Classification job → labeled rows land in SYNTHETIC with
    ``{question, answer: "{\"label\": ...}"}`` shape.
  - Span job → labeled rows land in SYNTHETIC with
    ``{question, answer: "{\"entities\": [...]}"}`` shape.
  - Preference pair → labeled rows land in the alignment dataset
    file with ``{prompt, chosen, rejected}`` shape.
- Idempotency: calling promote twice doesn't duplicate rows; the
  second call reports `skipped_already_promoted == N`.
- Cross-project rejection: a job from project A cannot promote into
  project B's dataset (returns 404 / 400).

**Claude prompt:**
> Read ROADMAP.md Story 1.6 and ship it. Closes the annotation loop end-to-end so Stories 1.1–1.3 finally reach the trainer.

<details>
<summary>Codex brief — Story 1.6</summary>

End-to-end story: schema migration + service + API + UI + tests.
No model changes to ``LabelJob`` itself; one new nullable column on
``label_rows``.

**Schema migration:**

```python
op.add_column(
    "label_rows",
    sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
)
op.add_column(
    "label_rows",
    sa.Column("promoted_to_dataset_id", sa.Integer(),
              sa.ForeignKey("datasets.id"), nullable=True),
)
```

The `promoted_at` field is the idempotency guard; `promoted_to_dataset_id`
captures the target so the operator can trace which run consumed
which labels.

**Service signature:**

```python
async def promote_labeled_rows(
    db: AsyncSession,
    *,
    project_id: int,
    job_id: int,
    target_dataset_type: DatasetType = DatasetType.SYNTHETIC,
) -> dict:
    """Materialize every labeled, unpromoted row in ``job_id`` into
    the project's target dataset JSONL. Idempotent — rows already
    carrying ``promoted_at`` are skipped. Returns counts."""
```

Per-label_type rendering — all three end as JSONL lines in the
target dataset's file:

```python
def _render_for_classification(row: LabelRow) -> dict:
    label = row.label_payload.get("label")
    text = (row.raw_payload.get("text") or row.raw_payload.get("question") or "")
    return {
        "question": text,
        "answer": json.dumps({"label": label}),
        "source": "annotation_job",
        "annotation_job_id": row.job_id,
        "original_row_id": row.id,
        "reviewer_user_id": row.assigned_to,
    }

def _render_for_span(row: LabelRow) -> dict:
    spans = row.label_payload.get("spans") or []
    text = (row.raw_payload.get("text") or "")
    return {
        "question": text,
        "answer": json.dumps({"entities": spans}),
        # ...same provenance fields...
    }

def _render_for_preference_pair(row: LabelRow) -> dict:
    rp = row.raw_payload
    lp = row.label_payload
    return {
        "prompt": rp.get("prompt") or "",
        "chosen": rp["completion_a"] if lp.get("chosen") == "A" else rp["completion_b"],
        "rejected": rp["completion_b"] if lp.get("chosen") == "A" else rp["completion_a"],
        # ...same provenance fields...
    }
```

The preference-pair path writes to `projects/{id}/alignment/preferences.jsonl`
instead of the synthetic file, since the alignment trainer expects
that location.

**API:**

```python
@router.post("/{job_id}/promote")
async def promote_labels(
    project_id: int,
    job_id: int,
    body: PromoteRequest,  # {target_dataset_type: str}
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await promote_labeled_rows(
        db, project_id=project_id, job_id=job_id,
        target_dataset_type=DatasetType(body.target_dataset_type),
    )
    await db.commit()
    return result
```

**UI:**

On `ProjectAnnotatePage` in the per-job detail view, render a primary
button when `stats.labeled > 0`:
- "Promote 142 labeled rows → Synthetic dataset" (count from stats)
- Click → confirm dialog → POST to the new endpoint → toast result
- Disable when `stats.labeled === stats.promoted` (need to add a
  `promoted` field to `job_stats`).

**Audit:** new reason code `ANNOTATION_ROWS_PROMOTED` under
`STAGE_INGESTION`; emit per promotion with the payload
`{job_id, promoted_count, skipped_count, target_dataset_type}`.

**Quality gates:**
- `python -m pytest tests/test_annotation_promotion.py tests/test_phase108_annotation_foundation.py -v`
- Frontend: `npx tsc --noEmit && npx vitest run src/pages/ProjectAnnotatePage* && npx vite build`

**Commit message template:**
```
Annotation: promote labeled rows → training dataset

Closes the Epic 1 annotation loop. submit_label rows now flow into
the project's synthetic / alignment dataset via a new
promote_labeled_rows service, with provenance fields preserved and
idempotency guarded by a new label_rows.promoted_at column. UI adds
a "Promote N labeled rows" CTA on the job detail page.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
</details>

---

### Story 1.7 — Experiment lifecycle hygiene + checkpoint-resume compat gate

**Status**: SHIPPED · 65a439a
**As a** project owner whose training keeps failing because the trainer
auto-resumes from a stale checkpoint left over in the output directory
**I want** the platform to either (a) refuse the incompatible resume
with an actionable message or (b) give me one-click recovery from any
FAILED experiment without hand-crafted SQL
**so that** I never burn another 23 seconds (or 10 hours) on a config
mismatch that the platform could have caught up front.

**Context — the incident series this closes** (2026-05-15..17):
Experiments 9 → 10 → 11 in project 3 all failed within seconds with
identical torch shape stack traces (`size mismatch for ...lora_A...
copying a param with shape torch.Size([16, 1536]) from checkpoint,
the shape in current model is torch.Size([8, 1536])`). Root cause:
`_resolve_resume_checkpoint` defaulted to "auto", which scanned
`output_dir` for any `checkpoint-*` dir and tried to resume from it
even when the LoRA rank or base model differed from the current run.
Recovery required three rounds of hand-crafted SQL + `mv` commands
under operator supervision.

**Acceptance criteria** (four pieces, one vertical slice):

1. **Checkpoint-resume compatibility gate** in `backend/scripts/train.py`.
   Before resuming, read the checkpoint's `adapter_config.json` and
   compare `lora_r`, `base_model_name_or_path`, `target_modules`
   against the current config. On mismatch, raise
   `CheckpointCompatibilityError` with the specific diff. Also
   change the default: stop auto-picking the latest checkpoint when
   the caller didn't explicitly set `resume_from_checkpoint`.
2. **Recovery service** `app/services/experiment_recovery_service.py`
   with three functions:
   - `reset_experiment(db, project_id, exp_id)` — FAILED → PENDING,
     archive output dir, drop checkpoint rows. Idempotent. Refuses
     RUNNING.
   - `delete_experiment(...)` — hard delete DB row + output dir.
   - `bulk_archive_failed(...)` — sweep every FAILED row in a project.
3. **API endpoints**:
   - `POST /api/projects/{id}/training/experiments/{exp_id}/reset`
   - `DELETE /api/projects/{id}/training/experiments/{exp_id}`
   - `POST /api/projects/{id}/training/experiments/bulk-archive-failed`
4. **UI surfaces them** in `TrainingPanel.tsx`:
   - 🔄 Reset button on every FAILED experiment row.
   - 🗑 Delete button on every non-RUNNING row (with type-the-name
     confirm).
   - "Archive all failed" header banner when ≥2 FAILED rows exist.
5. **CLI** in `brewslm.py`:
   - `brewslm experiment reset --project <id> --exp <id>`
   - `brewslm experiment delete --project <id> --exp <id>`
   - `brewslm experiment archive-failed --project <id>`

**Bonus shipped in the same commit**: eval-time POST in
`EvalPanel.tsx` now sends an explicit 30-min axios timeout. Default
Vite-proxy ~10-min cut was killing 200-row structured evals mid-run
on local-GPU Qwen-1.5B (same bug pattern as the cleaning + synthetic
fixes from earlier sprints).

**Files touched:**
- Backend:
  - `backend/scripts/train.py` (compat gate + helpers)
  - `backend/app/services/experiment_recovery_service.py` (new)
  - `backend/app/api/training.py` (3 endpoints)
  - `backend/scripts/brewslm.py` (3 CLI subcommands)
- Frontend:
  - `frontend/src/components/training/TrainingPanel.tsx` (Reset /
    Delete / bulk-archive banner)
  - `frontend/src/components/evaluation/EvalPanel.tsx` (eval timeout)
- Tests:
  - `backend/tests/test_phase110_experiment_recovery.py` (19 tests:
    7 for the compat gate, 12 for service + API)

**Lesson recorded for future stories**: this is the third
incident-driven contract gate (Story 1.5 = data-shape +
schema-mismatch + status reconciliation; Story 1.7 = checkpoint
compat + experiment recovery). The pattern is consistent: long-
running ML operations need cheap upfront gates that refuse
inconsistent state before the GPU spins up, plus operator-facing
recovery actions when something does slip through. Worth keeping
that lens for Epic 2+ (knowledge distillation will have its own
contract surface — teacher logits format, KD vs hard-label data
shape mismatch, etc.).

---

# Epic 2 — Knowledge distillation

**Why this matters**: BrewSLM is explicitly an SLM platform. The
biggest unique value an SLM platform can deliver is "your small model
that punches above its weight class because it learned from a strong
teacher." The teacher concept is already wired (synthetic generation,
LLM-assisted mapping); KD just connects the dots: capture the teacher's
soft labels and train the student to match them, not just the hard
labels.

Scope contract for the epic:
- Add KD as a *training mode*, not a separate pipeline.
- Reuse the existing teacher-model config (`TEACHER_MODEL_API_URL`).
- Eval surfaces a student-vs-teacher gap so the user sees the win.

---

### Story 2.1 — Teacher logit capture

**Status**: NOT STARTED
**As a** project owner
**I want** to capture the teacher model's logit / token-distribution outputs alongside hard labels
**so that** my student model can train against soft targets.

**Acceptance criteria:**
- New service: `app/services/distillation/teacher_capture.py` with
  `capture_teacher_outputs(project_id, dataset_id, *, top_k=10)`.
- For each row in the input dataset, call the teacher model with
  `logprobs=true` + `top_logprobs=k` (OpenAI-compat); store the top-k
  token distributions inline next to the row.
- Output rows land in the project's synthetic dataset with a
  `teacher_logits` field per token plus the original hard label.
- Emits a RunEvent with new reason_code `distillation_teacher_capture`.

**Files likely touched:**
- `backend/app/services/distillation/__init__.py` (new)
- `backend/app/services/distillation/teacher_capture.py` (new)
- `backend/app/api/distillation.py` (new — `POST /capture`)
- `backend/app/models/reason_codes.py` (one new code)

**Tests:**
- `test_phase109_teacher_capture.py`: mock teacher returns known
  logprobs; assert captured rows carry the expected structure.

**Claude prompt:**
> Read ROADMAP.md Story 2.1 and ship it.

<details>
<summary>Codex brief — Story 2.1</summary>

The teacher-call infrastructure already exists at
`backend/app/services/synthetic_service.py:call_teacher_model`.
You're adding a sibling helper that asks for logprobs and persists
them.

**Service signature:**
```python
async def capture_teacher_outputs(
    db: AsyncSession,
    project_id: int,
    dataset_id: int,
    *,
    top_k: int = 10,
    teacher_model_name: str | None = None,
    limit: int | None = None,
) -> CaptureResult:
    """For each row in dataset_id, call the teacher with
    logprobs=true + top_logprobs=top_k, persist captured logits
    inline. Returns counts + written_path.
    """
```

**Per-row output shape** (one JSONL line written to the project's
synthetic dataset):
```json
{
  "id": 1,
  "text": "...",
  "label": "positive",                       // hard label
  "teacher_logits": [
    {"token": "It", "top_k": [["pos", -0.12], ["neg", -2.1], ...]},
    ...
  ],
  "source": "teacher_capture",
  "captured_at": "2026-05-15T...",
  "status": "accepted"
}
```

**API**: `POST /api/projects/{id}/distillation/capture` body
`{dataset_id, top_k?: int, limit?: int}` → 202 + task_id (this is a
long-running call; use the same background-task pattern as
`cleaning_service`'s `start_clean_batch_task`).

**Status polling**: `GET /api/projects/{id}/distillation/tasks/{task_id}` —
identical contract to cleaning's task endpoint.

**Reason code**: add `DISTILLATION_TEACHER_CAPTURE = "distillation_teacher_capture"`
to `backend/app/models/reason_codes.py` under `STAGE_INGESTION`.

**Tests**: mock `call_teacher_model` to return a fixed logprobs
structure; verify the captured row carries the right shape and that
the audit RunEvent fires.

**Quality gates**: backend pytest only.
</details>

---

### Story 2.2 — KD training recipe (loss + temperature + mode)

**Status**: NOT STARTED
**Depends on**: Story 2.1
**As a** project owner
**I want** a `training_mode: "distillation"` option on the training config
**so that** the trainer optimizes a KD loss (cross-entropy on hard labels + KL on soft teacher distribution).

**Acceptance criteria:**
- New training recipe: `kd_classification`, `kd_qa`, `kd_span_extraction`
  (one per task profile that has KD-friendly outputs).
- Loss: `α * hard_label_loss + (1 - α) * T² * KL(student / T, teacher / T)`
  with default `α=0.5, T=2.0`.
- Training config UI shows a "Distillation" toggle that pulls the
  teacher_logits column from the prepared manifest.
- Per-step loss components logged separately (hard / soft / total).

**Files likely touched:**
- `backend/app/services/training_recipe_service.py` (extend)
- `backend/app/services/training_service.py` (loss wiring)
- `backend/app/services/training_runtime_service.py` (KD step in the trainer)
- `frontend/src/components/training/TrainingConfigForm.tsx` (toggle)

**Tests:**
- `test_phase109_kd_loss.py`: synthetic logits + known teacher
  distribution → verify the combined loss math.

**Claude prompt:**
> Read ROADMAP.md Story 2.2 and ship it. Depends on 2.1.

<details>
<summary>Codex brief — Story 2.2</summary>

KD lives inside the existing trainer; you're adding a loss variant
+ a config toggle.

**Loss in PyTorch terms:**
```python
import torch.nn.functional as F

# hard loss — standard cross-entropy on the gold label
hard = F.cross_entropy(student_logits, target_labels)

# soft loss — KL divergence between softened distributions
T = config.kd_temperature
student_soft = F.log_softmax(student_logits / T, dim=-1)
teacher_soft = F.softmax(teacher_logits / T, dim=-1)
soft = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)

total = config.kd_alpha * hard + (1 - config.kd_alpha) * soft
```

**Where to wire it**:
- `training_runtime_service.py`'s training step. Find the loss
  computation; when `config.training_mode == "distillation"`, branch
  to KD loss. The teacher_logits come from the prepared manifest
  (added by Story 2.1's capture).

**Recipe additions** in `training_recipe_service.py`:
- `kd_classification`: base = classification, defaults
  `{kd_alpha: 0.5, kd_temperature: 2.0}`.
- `kd_qa`: same.
- `kd_span_extraction`: same.

**UI**: one new section in the training config form — when teacher
logits exist on the manifest, show a "Distillation" toggle + two
sliders (α, T). When teacher logits don't exist, show a banner
linking to Story 2.1's capture flow.

**Tests**: unit-test the loss math against a synthetic case (known
inputs → expected output) — no full training run required.
</details>

---

### Story 2.3 — Student-vs-teacher eval comparison

**Status**: NOT STARTED
**Depends on**: Story 2.2
**As a** project owner
**I want** the eval report to compare student model performance against teacher model performance on the same eval set
**so that** I can see how much quality the distilled student preserved.

**Acceptance criteria:**
- Eval pack carries an optional `teacher_baseline_run_id`.
- When set, the eval report shows side-by-side metrics: student vs
  teacher per-class / per-slice.
- "Quality retained" headline metric: `student_score / teacher_score`.

**Claude prompt:**
> Read ROADMAP.md Story 2.3 and ship it. Depends on 2.2.

<details>
<summary>Codex brief — Story 2.3</summary>

Minimal extension to the eval pack + evaluation service.

**Eval pack schema**: add an optional `teacher_baseline_run_id` field.
When set, the evaluation service additionally runs the same eval set
against that baseline (typically a previous teacher-model eval run)
and emits a `comparison` block in the result.

**Result shape extension**:
```json
{
  "metrics": {...student metrics...},
  "comparison": {
    "teacher_run_id": "exp-42",
    "teacher_metrics": {...},
    "quality_retained": 0.93,     // student F1 / teacher F1
    "by_slice": [
      {"slice": "short_prompts", "student": 0.88, "teacher": 0.91, "retained": 0.97},
      ...
    ]
  }
}
```

**UI**: extend the EvalResultPanel to show the comparison block when
present.

**Tests**: fixture eval pack with `teacher_baseline_run_id` → assert
the comparison block populates.
</details>

---

# Epic 3 — Closed feedback loop

**Why this matters**: every other piece exists in isolation —
production telemetry (`served_model_telemetry_service`), drift checks,
gold-set versioning, the eval remediation service. What's missing is
the wiring: a thumbs-down in production → a labeled row → a next
training set → a one-click retrain. Three small services close it.

---

### Story 3.1 — Feedback queue + thumbs-down ingestion

**Status**: NOT STARTED
**As a** product engineer using a BrewSLM-served model
**I want** to send a thumbs-down (with the prompt + response + optional reason) to BrewSLM
**so that** the row lands in a review queue for retraining.

**Acceptance criteria:**
- New table `feedback_queue` with `id`, `project_id`, `model_run_id`,
  `prompt`, `response`, `signal` (`negative` | `positive`), `reason`
  (text), `metadata` (JSON), `status` (`new` | `reviewed` |
  `incorporated`), `submitted_by`, `submitted_at`.
- API: `POST /api/projects/{id}/feedback` (open endpoint — token
  optional for production-side calls).
- Audit: each submission emits RunEvent with reason
  `feedback_submitted`.

**Claude prompt:**
> Read ROADMAP.md Story 3.1 and ship it.

<details>
<summary>Codex brief — Story 3.1</summary>

Single table + service + endpoint. Mirror the
`dataset_import_configs` pattern (model + alembic migration +
service + router).

**Schema:**
```sql
feedback_queue (
  id PK,
  project_id FK→projects.id (indexed),
  model_run_id VARCHAR(128),
  prompt TEXT,
  response TEXT,
  signal VARCHAR(16) NOT NULL,  -- 'negative' | 'positive'
  reason TEXT,                  -- user-supplied
  metadata JSON,
  status VARCHAR(16) NOT NULL DEFAULT 'new',
  submitted_by VARCHAR(128),
  submitted_at DateTime
)
```

**Endpoint**: `POST /api/projects/{id}/feedback` body
`{prompt, response, signal, reason?, metadata?, model_run_id?, submitted_by?}`
→ 201 + feedback id.

**Reason code**: `FEEDBACK_SUBMITTED = "feedback_submitted"` added to
`STAGE_INGESTION` reason codes.

**Tests**: insert + list + filter by status.
</details>

---

### Story 3.2 — Reviewer UI for the feedback queue

**Status**: NOT STARTED
**Depends on**: Story 3.1
**As a** reviewer
**I want** to triage feedback rows in a queue UI
**so that** I can label them, dismiss noise, and mark the row for incorporation into the next training set.

**Acceptance criteria:**
- New page `/project/:id/feedback` listing the queue.
- Each row expandable: shows prompt, response, signal, reason, metadata.
- Reviewer actions: "Correct response →" (opens an inline labeler;
  saves the ideal response as the label), "Skip / not actionable",
  "Mark as incorporated."
- Linkage: an "incorporated" row gets a `label_payload` referencing
  the user's corrected response.

**Claude prompt:**
> Read ROADMAP.md Story 3.2 and ship it. Depends on 3.1.

<details>
<summary>Codex brief — Story 3.2</summary>

Frontend-only. Patterns from `SavedMappingsPanel.tsx` apply.

Page route: `/project/:id/feedback` → `ProjectFeedbackQueuePage.tsx`.

Tabs for filter status: `New | Reviewed | Incorporated`. Default New.

Each row card: prompt + response side-by-side (use the same dual-pane
treatment as the wizard), plus the signal badge, reason, and an
action bar.

Inline labeler for "Correct response": a textarea pre-filled with the
original response — reviewer edits it; on save, sets row's
`label_payload = {ideal_response: <text>}` and bumps `status` to
`reviewed`.

API: extend Story 3.1's router with a `PATCH /api/projects/{id}/feedback/{row_id}`
endpoint that accepts `{status?, label_payload?}`.
</details>

---

### Story 3.3 — One-click retrain from feedback set

**Status**: NOT STARTED
**Depends on**: Story 3.2
**As a** project owner
**I want** to merge the `incorporated` feedback rows into a training set and trigger a new training run with one click
**so that** the model continuously improves from production telemetry.

**Acceptance criteria:**
- New CTA on the Training Config page: "Retrain with N feedback rows."
- Clicking triggers: merge feedback rows into the project's synthetic
  dataset (each row gets `source="feedback_loop"`), then kicks off
  training with the existing autopilot defaults.
- Feedback rows used in a run get their `status` flipped to
  `incorporated` + `run_id` recorded.

**Claude prompt:**
> Read ROADMAP.md Story 3.3 and ship it. Depends on 3.2.

<details>
<summary>Codex brief — Story 3.3</summary>

Small wiring story. Two parts:

1. **Service helper** in `backend/app/services/feedback_loop_service.py`:
   `merge_feedback_into_synthetic(project_id) → row_count`. Pulls all
   `status='reviewed'` rows with non-null `label_payload`, writes them
   to the synthetic JSONL with `source="feedback_loop"` and an
   `original_feedback_id` link, marks the rows `incorporated`.

2. **CTA on the Training Config page**. When `count(feedback_queue
   WHERE status='reviewed') > 0`, render a card: "N reviewed feedback
   rows ready. Retrain now." Button calls
   `POST /api/projects/{id}/training/run` with the autopilot defaults
   after first calling the merge endpoint.

Audit: emit RunEvent with reason `feedback_loop_retrain` carrying
the row count.
</details>

---

# Epic 4 — Cross-experiment slice analysis

**Why this matters**: failure clusters today are per-eval. The real
analytical question is **across** experiments: "did exp-47 fix the
short-prompts regression that exp-42 had?" Without this, every fine-tune
iteration is partially blind.

---

### Story 4.1 — Slice definitions per project

**Status**: NOT STARTED
**As a** project owner
**I want** to define reusable slices via predicates (e.g. `token_count < 20`, `language == "es"`, `label == "negative"`)
**so that** I can evaluate every experiment against the same slices.

**Acceptance criteria:**
- New table `eval_slices` with `id, project_id, name, predicate` (JSON Logic or similar).
- API: full CRUD.
- Built-in slices auto-seeded per project: `short_prompts` (token
  count < 20), `long_prompts` (token count > 200), `each label`
  (one slice per declared label).

**Claude prompt:**
> Read ROADMAP.md Story 4.1 and ship it.

<details>
<summary>Codex brief — Story 4.1</summary>

Single table + CRUD service + API. Mirror the dataset_import_configs
pattern.

Predicate format: use JSON Logic
([https://jsonlogic.com/](https://jsonlogic.com/)) since it's compact
and easy to evaluate in both Python (`json-logic-py` package) and JS
(`json-logic-js`) — keeps frontend and backend consistent.

Example predicates:
```json
{"<": [{"var": "token_count"}, 20]}                       // short_prompts
{"==": [{"var": "language"}, "es"]}                       // spanish
{"and": [{">=": [{"var": "token_count"}, 100]}, {"==": [{"var": "label"}, "negative"]}]}
```

Auto-seed: when a project is created, insert `short_prompts`,
`long_prompts`, plus one slice per allowed_label if the project's
declared task_profile is classification.
</details>

---

### Story 4.2 — Slice evaluation at score time

**Status**: NOT STARTED
**Depends on**: Story 4.1
**As an** eval engineer
**I want** the eval handler dispatcher to compute per-slice metrics alongside the headline metrics
**so that** every experiment's eval result carries slice breakdowns.

**Acceptance criteria:**
- `EvalResult.metrics` grows a `slices` field: `{slice_name: {metric_name: value, ...}}`.
- Each task handler's `score()` is unchanged externally; the dispatcher
  applies slice predicates to the prediction list, calls `score()` on
  each filtered list, attaches results.
- Audit RunEvent's payload grows the slice metric summary.

**Claude prompt:**
> Read ROADMAP.md Story 4.2 and ship it. Depends on 4.1.

<details>
<summary>Codex brief — Story 4.2</summary>

Modify `app/services/evaluation_service.py` (the score entry-point)
to run slice-eval as a post-step.

```python
# After the main metrics:
slices = await fetch_active_slices(db, project_id)
slice_metrics = {}
for slice in slices:
    filtered = [p for p in predictions if eval_predicate(slice.predicate, p)]
    if not filtered:
        continue
    slice_metrics[slice.name] = handler.score(filtered, ctx)
result.metrics["slices"] = slice_metrics
```

The `eval_predicate` helper uses `json-logic-py`. Add to
`backend/requirements-base.txt`.

Tests: fixture predictions with a known mix; assert the slices field
populates with the right per-slice metric.
</details>

---

### Story 4.3 — Slice comparison UI across experiments

**Status**: NOT STARTED
**Depends on**: Story 4.2
**As an** ML engineer
**I want** to pick 2+ experiments and see a slice-by-slice comparison table
**so that** I can tell whether a new run fixed a slice regression.

**Acceptance criteria:**
- New page `/project/:id/eval/compare` with multi-select for
  experiments + a slice-vs-experiment matrix.
- Cells render the slice metric for that experiment; deltas vs the
  first selected experiment are color-coded.

**Claude prompt:**
> Read ROADMAP.md Story 4.3 and ship it. Depends on 4.2.

<details>
<summary>Codex brief — Story 4.3</summary>

Frontend-only. A page with an experiment-picker on top, then a table
where rows are slices and columns are experiments. Each cell shows
the primary metric for that handler (F1 / EM / faithfulness / etc).

Background color logic: green if better than the leftmost experiment
by ≥0.01, red if worse by ≥0.01, neutral otherwise.

API: `GET /api/projects/{id}/eval/compare?exp_ids=42,47,51` →
`{experiments: [...], slices: [...], cells: {...}}`.
</details>

---

# Epic 5 — Auto-generated model cards & data sheets

**Why this matters**: every production-promotion review asks for a
model card. The pieces exist (manifest, eval, run events); the
consolidated artifact doesn't. One service away.

---

### Story 5.1 — Model card service

**Status**: NOT STARTED
**As a** governance reviewer
**I want** a generated model card for each export
**so that** I can review provenance, intended use, limitations, and metrics in one document.

**Acceptance criteria:**
- New service `app/services/model_card_service.py:generate_model_card(export_id)` returning a structured dict (Mitchell et al model-card sections).
- API: `GET /api/projects/{id}/exports/{export_id}/model-card`.

**Claude prompt:**
> Read ROADMAP.md Story 5.1 and ship it.

<details>
<summary>Codex brief — Story 5.1</summary>

Pure read-side service that joins existing tables.

Card sections (return one dict per):
1. **Model details**: name, version, base model, training recipe, KD
   flag, owner.
2. **Intended use**: from the project's `task_profile` + the domain
   pack description.
3. **Factors**: from the project's eval slices.
4. **Metrics**: latest eval pass-rate + per-slice metrics + per-class
   breakdown.
5. **Evaluation data**: gold set version + size + provenance.
6. **Training data**: synthetic dataset source breakdown
   (dataset_import audit log grouped by mapper).
7. **Quantitative analyses**: from `failure_cluster_service`.
8. **Ethical considerations**: from any safety eval handler results.
9. **Caveats and recommendations**: free-form, pulled from the
   project's `model_card_notes` field (add this column to projects;
   nullable; editable in the UI).

Endpoint returns the dict; serialization (Story 5.2) is downstream.
</details>

---

### Story 5.2 — Card export to Markdown / HTML / PDF

**Status**: NOT STARTED
**Depends on**: Story 5.1

**Acceptance criteria:**
- `GET /api/.../model-card?format=md|html|pdf` returns the formatted
  artifact.
- UI: "Download model card" button on the Deployments + Exports pages.

**Claude prompt:**
> Read ROADMAP.md Story 5.2 and ship it. Depends on 5.1.

<details>
<summary>Codex brief — Story 5.2</summary>

Add a renderer with three backends:
- `md`: jinja2 template → markdown string.
- `html`: same template + a CSS file → HTML response.
- `pdf`: `weasyprint` (already a common Python PDF lib; or
  `reportlab`).

Cache the rendered artifact on the export row (`model_card_cache`
JSON column or a file path); regenerate on demand.

Tests: render md/html, assert key sections are present.
</details>

---

# Epic 6 — Hyperparameter search

**Why this matters**: autopilot picks one config; the next 10× of
quality comes from sweeping. Without it, every project plateaus at
the autopilot default.

---

### Story 6.1 — Sweep API: search space → N runs (grid backend first)

**Status**: NOT STARTED
**As a** training engineer
**I want** to define a search space (LR ∈ {1e-5, 5e-5}, LoRA rank ∈ {8, 16}) and fire N runs
**so that** the platform produces a Pareto frontier I can compare.

**Acceptance criteria:**
- `POST /api/projects/{id}/training/sweep` body `{search_space, N, base_recipe}`.
- Backend: grid expansion (Cartesian product), N runs queued as separate Experiments with a shared `sweep_id`.
- Each run's manifest carries the resolved hyperparameter values.

**Claude prompt:**
> Read ROADMAP.md Story 6.1 and ship it.

<details>
<summary>Codex brief — Story 6.1</summary>

Small service + endpoint; reuses the existing training-run path.

`backend/app/services/sweep_service.py`:
- `expand_search_space(spec) → list[dict]` (Cartesian product over
  enum values; clamp to `N` if grid is larger).
- `start_sweep(project_id, recipe, configs) → sweep_id` — kicks off
  one training run per config, all tagged with the same `sweep_id`.

Schema change: add `sweep_id` column to `experiments` table
(nullable; alembic migration).

API: `POST .../training/sweep` accepts search-space schema:
```json
{
  "search_space": {
    "learning_rate": [1e-5, 5e-5, 1e-4],
    "lora_rank": [8, 16, 32]
  },
  "N": 9,
  "base_recipe": "kd_classification"
}
```

For now, grid only. Bayesian / Hyperband is a follow-up story.
</details>

---

### Story 6.2 — Pareto comparison UI for sweeps

**Status**: NOT STARTED
**Depends on**: Story 6.1, ideally Story 4.2 (slice eval) for full power

**Acceptance criteria:**
- New page `/project/:id/sweeps/:sweep_id` showing a table of runs:
  hyperparameters × metric × cost.
- Highlight the Pareto frontier (runs that are non-dominated on
  metric-vs-cost).

**Claude prompt:**
> Read ROADMAP.md Story 6.2 and ship it. Depends on 6.1.

<details>
<summary>Codex brief — Story 6.2</summary>

Frontend-only. The data is already at
`GET /api/projects/{id}/sweeps/{sweep_id}` (add this endpoint
trivially if 6.1 didn't include it — it just lists experiments
filtered by `sweep_id`).

Render a table with columns: each hyperparameter from the resolved
config + primary metric + cost (sum of GPU-seconds × $ rate).

Pareto: for each row, mark it "frontier" if no other row has both
higher metric AND lower cost. Highlight frontier rows.

Allow click on a row → link to that experiment's detail page.
</details>

---

# Lesser gaps — backlog (not yet broken into stories)

The following gaps are real but lower-impact for the "any-domain"
thesis. Promote to stories as they become priorities.

- **Distributed multi-GPU training** — DeepSpeed / FSDP wiring in
  `training_runtime_service`. Single-GPU LoRA covers most SLM
  use-cases; this becomes interesting at the larger end of the SLM
  range (10B+).
- **On-device export** — CoreML / TFLite targets beyond GGUF.
- **A/B routing / canary at serve time** — online traffic split, not
  offline drift.
- **Domain template marketplace** — depth + curation of starter packs
  for healthcare / legal / retail / code / agriculture etc.
- **Notebook export** — auto-generated Jupyter notebook that
  replays an experiment.
- **Reproducibility lockfile** — env + seeds + data hash captured at
  run time.

---

# Adding new epics

When a new epic emerges:

1. Add an `# Epic N — Title` heading, a paragraph framing it, then
   2–4 stories.
2. Add rows for the stories to the **Status board** at the top.
3. Each story carries: status, As/I want/So that, acceptance
   criteria, files likely touched, tests, Claude prompt, and a
   collapsible Codex brief.
4. Keep status board sorted by epic + dependency order, not by
   priority — the priority is conveyed by the Impact column.

When a story ships:

1. Change Status to `SHIPPED` in both the status board and the story
   body.
2. Add the commit SHA next to the status (e.g. `SHIPPED · 5333132`).
3. Cross out (`~~`) or remove the Claude/Codex prompt blocks — they're
   no longer queueable.
