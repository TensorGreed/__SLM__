# Generic Dataset Import Plan

A staged rollout for a general-purpose data import pipeline that lets a
newbie go from "here's a dataset URL" to "trained model" without
hand-writing a converter. Generalizes the
[kaggle_pii_to_brewslm.py](backend/data/demo_samples/pii-detector/kaggle_pii_to_brewslm.py)
one-off into the *category of which it's an instance*: pluggable
sources × pluggable mappers × pluggable target shapes.

Each phase is independently shippable; the architecture is purely
additive, so any phase you skip just leaves that capability for later.

## Why this exists

Today, a newbie who wants to train a PII / classification / QA model
on real data has to:

1. Find a dataset (HF / Kaggle / their own CSV).
2. Figure out the format on their own.
3. Write a one-off Python script to transform it into BrewSLM's
   canonical shape — different code for every source × task
   combination.
4. Run that script. Hope offsets / column maps / encodings are right.

That last point is where most newbies bounce. The kaggle converter
we shipped on Phase 4.x was 175 lines of careful BIO-to-spans logic;
nobody arriving at BrewSLM cold should have to write that themselves.

The fix is a three-layer pipeline that **introspects the source,
proposes a mapping, lets the user confirm, then runs the transform**
— the same shape any modern ETL tool uses, but constrained to the
specific shapes BrewSLM's task handlers consume.

## Architectural shape

Three pluggable layers, each with a small contract. The same registry
+ dispatcher pattern we used for eval task handlers (every new entrant
is one file, no edits to existing entrants).

```
┌─ Source loader ──────┐   ┌─ Schema introspector ──┐   ┌─ Target mapper ──────┐
│ (gets raw rows from  │ → │ (sniffs columns,       │ → │ (transforms rows     │
│  external system)    │   │  proposes mapping)     │   │  into canonical      │
│                      │   │                        │   │  shape per           │
│  - jsonl             │   │  - column-type sniffer │   │  task_profile)       │
│  - csv               │   │  - shape detector      │   │                      │
│  - parquet           │   │  - mapping proposer    │   │  - bio_to_spans      │
│  - hf                │   │                        │   │  - label_to_class    │
│  - kaggle            │   │                        │   │  - kv_to_structured  │
│  - …                 │   │                        │   │  - qa_pair_passthrough│
└──────────────────────┘   └────────────────────────┘   └──────────────────────┘
                                       ↓
                              proposed mapping JSON
                                       ↓
                          user confirms via UI or CLI flag
                                       ↓
                          rows land in project's synthetic dataset
```

### Source loader contract

```python
class DatasetSource(Protocol):
    source_id: str  # "jsonl" | "csv" | "hf" | "kaggle" | "parquet" | …

    def load(self, locator: str, *, limit: int | None = None) -> Iterable[dict]:
        """Stream raw row dicts from the external system.

        `locator` is the source-specific reference: a path for local
        files, "ai4privacy/pii-masking-200k" for HF, the competition
        slug for Kaggle, etc.
        """

    def describe(self, locator: str) -> dict:
        """Return metadata for the introspector — column names, row
        count, sample rows, license, schema if the source provides one.
        """
```

Connectors register themselves into a global `_SOURCE_LOADERS` dict.
New connector = one file; the import pipeline auto-picks it via the
`locator`'s prefix.

### Schema introspector

Takes the first N rows from a `DatasetSource.describe()` call and
produces a `ProposedMapping`:

```python
@dataclass
class ProposedMapping:
    target_task_profile: str       # "classification" / "span_set" / "qa" / …
    confidence: float              # 0-1
    field_map: dict[str, FieldMap] # source column → target field
    mapper_id: str                 # which target mapper to use
    rationale: str                 # why the introspector chose this
    warnings: list[str]            # things the user should know
```

Detection rules are conservative and **never silently auto-pick**.
They emit confidence scores + rationale; the user sees a preview
before any data lands. This is the load-bearing architectural rule —
silent mappings poison training data, and the cost of a bad mapping
is catastrophic (you train against garbage and ship it).

### Target mapper contract

```python
class TargetMapper(Protocol):
    mapper_id: str  # "bio_to_spans" / "label_to_classification" / …

    def transform(
        self, rows: Iterable[dict], field_map: dict, *, project_ctx: ProjectContext
    ) -> Iterable[dict]:
        """Yield canonical rows for the project's task_profile."""

    def declared_target(self) -> str:
        """Return the task_profile this mapper feeds (e.g. 'classification',
        'structured_extraction')."""
```

Mappers are reusable across sources. The BIO→spans mapper that handles
the Kaggle PII format also handles HF's `ai4privacy/pii-masking-200k`
and any other BIO-tagged dataset.

## Cross-cutting decisions (locked in once at Phase A)

- **No silent auto-mapping.** The introspector proposes; the user
  confirms. `--auto` flag on CLI accepts the highest-confidence
  proposal explicitly. Confidence < 0.8 → CLI requires `--force` (UI
  shows a warning banner).
- **Connectors register themselves.** Adding a new source = one file,
  no edits to the import service or CLI. Same pattern as task
  handlers.
- **Mappers are domain-agnostic.** A `bio_to_spans` mapper handles
  PII, healthcare, legal, financial NER — the entity-type mapping is
  config, not code. Don't write `pii_bio_to_spans` or
  `medical_bio_to_spans`.
- **Output goes to the project's synthetic dataset.** Same path
  `save_synthetic_batch` already writes to. Imported rows are first-
  class members of the dataset alongside teacher-LLM-generated rows.
- **Dry-run is first-class.** `--dry-run` on CLI; UI's "Preview"
  button. Shows the proposed mapping + the first 5 transformed rows
  without writing anywhere.
- **Mapping configs are JSON.** Auditable, version-control-able,
  reproducible. You can save a mapping config and re-run it on a
  refreshed source.
- **Per-row validation gates land in the dataset, not in transit.**
  Bad rows are written with a `status: rejected` flag and a reason,
  not silently dropped. Same pattern as the synthetic-save batch.

## Codebase layout

```
backend/app/services/dataset_import/
  __init__.py
  sources/                # Source loaders (one file per connector)
    jsonl.py
    csv.py
    hf.py
    kaggle.py
    parquet.py
  mappers/                # Target mappers (one file per transform)
    bio_to_spans.py
    label_to_classification.py
    kv_to_structured.py
    qa_pair_passthrough.py
    preference_pair.py
  introspector.py         # Schema sniffing + mapping proposal
  registry.py             # Dispatcher + register_source / register_mapper
  service.py              # Orchestrates source → introspector → mapper → save

backend/app/api/
  dataset_import.py       # POST /preview, POST /run, GET /sources, GET /mappers

backend/cli/commands/
  dataset_import.py       # `brewslm dataset import …`

frontend/src/pages/
  ProjectDatasetImportPage.tsx     # UI wizard

frontend/src/components/dataset_import/
  SourcePicker.tsx
  MappingPreview.tsx
  ConfirmRun.tsx
```

---

## Phase A — Foundation (shipped)

**Goal**: end-to-end import pipeline working with two sources (jsonl,
csv) and two mappers (bio_to_spans, label_to_classification). No
introspection yet — user supplies the mapping config explicitly. Ships
as a CLI command + a backend service; no UI yet.

### What landed
- `backend/app/services/dataset_import/` package with
  `protocols.py` (DatasetSource + TargetMapper Protocols + the
  RawRow / TransformedRow / RejectedRow / ImportContext /
  ImportResult dataclasses), `registry.py` (source + mapper
  dispatchers with locator-prefix parsing), and `service.py`
  (orchestrator: `preview_import` + `run_import`).
- Two source loaders:
  - `sources/jsonl.py` — streams `dict` rows from a JSONL file;
    unparseable lines become sentinel rows with `__parse_error__`
    so mappers reject them with a stable reason code rather than
    silently dropping them.
  - `sources/csv.py` — wraps `csv.DictReader`; first row is the
    header.
- Two target mappers:
  - `mappers/bio_to_spans.py` — generalizes the kaggle PII
    converter. Full-text alignment with token+whitespace fallback,
    B-X / I-X run merging, configurable entity-type mapping.
    Declared target: `structured_extraction`.
  - `mappers/label_to_classification.py` — type-coercing
    passthrough (bool/int labels → canonical strings, whitespace
    collapsed). Optional `allowed_labels` filter. Declared
    target: `classification`.
- API endpoints (registered in `backend/app/main.py`):
  - `GET /api/dataset-import/sources` — list registered sources
  - `GET /api/dataset-import/mappers` — list registered mappers
  - `POST /api/projects/{id}/dataset-import/preview` — dry-run
  - `POST /api/projects/{id}/dataset-import/run` — persist
- CLI module at `backend/app/cli/dataset_import.py`, runnable as
  `python -m app.cli.dataset_import <subcommand>`. Subcommands:
  `sources`, `mappers`, `preview`, `run`. Flags: `--locator`,
  `--mapper`, `--map K=V` (repeatable), `--map-json '<json>'` for
  nested values like `entity_type_map`, `--limit`, `--drop REASON`
  (repeatable bulk-drop), `--project`, `--json`, `--sample-cap`.
  Future `brewslm` binary can wrap this module unchanged.
- Bulk-drop UX contract honored end-to-end:
  `rejection_counts` always reflects the full tally;
  `--drop REASON` filters specific reason codes out of the
  surfaced `rejected_sample` but keeps counts intact.
- 42 tests (32 service + 10 CLI) covering registry dispatch +
  unknown-id KeyErrors, source loaders (streaming, parse-error
  sentinels, limit, describe, missing-file), bio_to_spans
  (full-text + token-fallback offset reconstruction, B/I merge,
  rejection codes), label_to_classification (coercion,
  whitespace collapse, allowed_labels filter, field remapping),
  preview_import (happy path, rejection grouping, drop-reasons
  filter, unknown id errors), and CLI (catalog commands, JSON
  output, field-map pairs + JSON form, drop filter, error
  messages).

### Out of scope (lands in later phases)
- Schema introspection / auto-mapping (Phase B).
- More target mappers (Phase C).
- HuggingFace + Kaggle connectors (Phases D, E).
- UI wizard (Phase F).

---

## Phase B — Schema introspector + dry-run mapping preview (shipped)

**Goal**: when the user doesn't supply `--map` flags, the introspector
samples rows, classifies columns by content, detects a likely task
shape, and proposes a mapping. The user reviews + confirms.

### What landed
- `backend/app/services/dataset_import/introspector.py`:
  - Per-column content sniffer: `text_like`, `categorical`,
    `bio_tag_list`, `entity_list_json`, `chat_messages`, `tokens_list`,
    `numeric`, `boolean`, `path_like`, `unknown`. Multi-word + length
    signals separate text from labels; categorical detection requires
    a small repeated label set OR a tiny all-unique short-token sample
    (weak categorical 0.6).
  - Shape detector returning ranked `ShapeHypothesis` per registered
    mapper. Two hypotheses today (`bio_to_spans`, `label_to_classification`);
    Phase C's mapper expansion adds more detection rules.
  - One-shot `propose_mapping(sample_rows)` returns the top
    hypothesis as a `ProposedMapping`.
  - `CONFIDENCE_HIGH = 0.8` is the gate above which `--auto` runs
    without `--force`.
- Service: `introspect_locator(locator, sample_size=20)` wires the
  sniffer to the source connector's `describe()` and serializes
  column signatures + ranked hypotheses + top proposal for API/CLI.
- API: `POST /api/dataset-import/introspect` on the catalog router
  (no project required — it's a pre-project orientation step).
- CLI:
  - New `introspect` subcommand: human-readable column rundown +
    ranked hypotheses + safety-gated proposal.
  - `--auto` on `preview` / `run`: pulls mapper + field_map from the
    introspector. Refuses below the confidence threshold unless
    `--force`. Explicit `--map` / `--map-json` / `--mapper` layer on
    top of the auto suggestion (so you can override entity-type maps
    or specific keys without dropping the auto-detection).
  - `--force` overrides the confidence gate.
  - `--mapper` is now optional (still required when `--auto` isn't
    set; argparse moved the requirement check into the resolver so
    the error message names the escape hatch).
- Tests: `backend/tests/test_phase102_dataset_import_introspector.py`
  — 15 tests covering the column sniffer (BIO/chat/tokens/categorical
  splits), shape detector (NER + classification), service-level
  introspect, and the CLI gate (`--auto` high-confidence pass-through,
  low-confidence block, `--force` override, explicit `--map` overrides
  on top of `--auto`).
- Docs: PII demo gets an [`--auto` walkthrough](slm-docs/docs/demos/pii-detector.md)
  showing how the same Kaggle file works *without* the bespoke
  converter; glossary entries for [Schema introspection](slm-docs/docs/reference/glossary.md#schema-introspection),
  [Confidence threshold](slm-docs/docs/reference/glossary.md#confidence-threshold-dataset-import),
  [Source connector](slm-docs/docs/reference/glossary.md#source-connector-dataset-import),
  [Target mapper](slm-docs/docs/reference/glossary.md#target-mapper-dataset-import).

### Architectural rule (locked in)
Introspection **never silently auto-picks**. It emits a
`ProposedMapping` with confidence + rationale; the CLI (`--auto`) and
the future UI wizard (Phase F) confirm with the user. Below the
threshold the CLI exits with a message naming the override path.

### Out of scope (now Phase C+ work)
- LLM-assisted mapping suggestion → Phase H.
- Mapping configs persisted to project state → Phase G.
- UI introspection wizard → Phase F.
- Detection rules for `preference_pair` / `rag_passthrough` / etc. →
  Phase C (add a mapper, add its detection rule).

---

## Phase C — Mapper catalog expansion (shipped)

**Goal**: cover the common dataset shapes BrewSLM cares about. Each
mapper is a thin wrapper around a well-understood transform.

### What landed
- Six new mappers, one file each, registered at import time:
  - `text_only` (`language_modeling`) — single-column LM passthrough,
    optional `min_chars` gate.
  - `qa_pair_passthrough` (`qa`) — `{question, answer}` with fallback
    column precedence (prompt / instruction / input,
    response / completion / output / target_text) mirroring
    `evaluation_service._extract_prompt_and_reference`.
  - `chat_messages_passthrough` (`chat_sft`) — list of
    `{role, content}` dicts; alt `value` / `text` keys; rejects
    no-assistant-reply turns by default; emits `prompt` (rendered
    history) + `reference` (final assistant content) + the cleaned
    `messages` list for chat-template inference.
  - `preference_pair` (`dpo`) — `{prompt, chosen, rejected}` with
    DPO/ORPO-friendly fallback names; rejects degenerate
    `identical_pair` rows.
  - `rag_passthrough` (`rag_qa`) — `{question, context, answer}` with
    legacy `prompt` / `reference` aliases for non-RAG handlers
    reading the same row.
  - `kv_to_structured` (`structured_extraction`) — flat key-value
    extractions → `{"entities":[{"field","value"}]}`; supports list-
    or dict-form field config; `skip_empty_fields` toggle.
- Each mapper's `declared_target()` is registered in the eval task
  handler dispatcher — no fall-through to `GenericHandler`. A test
  pins the mapper → handler-profile mapping so it can't drift.
- Introspector detection rules added (`backend/app/services/dataset_import/introspector.py`):
  - chat_messages column → `chat_messages_passthrough` (0.95).
  - `{prompt, chosen, rejected}` triple → `preference_pair` (~0.92).
  - `{question, context, answer}` triple → `rag_passthrough` (~0.85+);
    `context` column name is the strong signal so RAG outranks
    classification even when the answer is short enough to be sniffed
    as categorical.
  - `{question, answer}` *without* a context column →
    `qa_pair_passthrough` (~0.92).
  - Single text column + no labels / structured columns → `text_only`
    (~0.85+). Stays below the stronger hypotheses when those are
    present.
  - `kv_to_structured` is deliberately NOT detection-eligible —
    it needs an explicit `fields` config the introspector can't
    invent. The UI wizard (Phase F) will surface it as a "build
    my own" option.
- Tests: `backend/tests/test_phase103_dataset_import_mapper_catalog.py`
  — 31 tests covering each mapper's happy path + rejection codes,
  the new detection rules, and an end-to-end `--auto` CLI
  preview on a preference dataset.
- Docs: mapper catalog refreshed in
  [glossary "Target mapper" entry](slm-docs/docs/reference/glossary.md#target-mapper-dataset-import);
  data-ingestion workflow callout enumerates the new mapper ids;
  this Phase C section marked shipped.

### Out of scope (later phases)
- Custom mappers via plugin system → Phase H.
- Multi-stage mappers (chain two transforms) — not planned; the
  pipeline contract treats each row as a single source → mapper hop.
- Detection rules for `kv_to_structured` — would require schema
  hints the introspector can't infer; deferred to Phase F's UI
  wizard.

---

## Phase D — HuggingFace source connector

**Goal**: import any HF dataset directly. The HF `datasets` library is
the de facto standard; once this connector lands, hundreds of public
datasets become click-import-able.

### User stories
- *As a researcher exploring PII detection*, I want
  `brewslm dataset import --source hf:ai4privacy/pii-masking-200k
  --project pii-detector --auto` to fetch + introspect + map +
  import that dataset's first 5k rows in one command.
- *As a developer iterating on a model*, I want
  `--source hf:dataset-id:train` to pick a specific split,
  `--source hf:dataset-id:train:revision-sha` to pin a revision.
- *As an offline operator*, I want HF datasets to cache locally so
  re-runs don't re-download.

### Work
- New connector `dataset_import/sources/hf.py` that wraps
  `datasets.load_dataset`.
- Locator format: `hf:<dataset_id>[:<split>[:<revision>]]`.
- Auth: reads HF_TOKEN from settings / project secrets when present
  (some datasets are gated).
- Streaming mode for >1GB datasets (uses HF's `streaming=True`).
- Adds `datasets` to `requirements-base.txt` (already a dep!).

### Out of scope
- Pushing datasets back to HF (one-way for now).
- HF Spaces integration.

---

## Phase E — Kaggle source connector

**Goal**: import Kaggle competition + dataset data via the Kaggle CLI.

### User stories
- *As a developer doing the Kaggle PII competition*, I want
  `--source kaggle:competition:pii-detection-removal-from-educational-data`
  to download + extract + introspect + map + import — same one-line
  flow as HuggingFace.
- *As a developer with a Kaggle dataset (not competition)*, I want
  `--source kaggle:dataset:user/dataset-name` to work the same way.

### Work
- New connector `dataset_import/sources/kaggle.py`. Shells out to
  `kaggle competitions download` / `kaggle datasets download`.
- Locator format: `kaggle:competition:<slug>` /
  `kaggle:dataset:<user/slug>`.
- Auth: reads `KAGGLE_USERNAME` + `KAGGLE_KEY` from settings /
  project secrets. Surfaces clear error when missing.
- Auto-extracts the downloaded zip; finds the train.json / .csv
  inside.

### Out of scope
- Kaggle submission (one-way).
- Kaggle kernels.

---

## Phase F — UI wizard

**Goal**: 3-step wizard that does everything the CLI does, plus a
visual mapping editor for power users.

### User stories
- *As a UI-first newbie*, I want a "Import Dataset" button on the
  Data tab that opens a 3-step wizard: Source → Preview & Mapping →
  Confirm & Run. No terminal required.
- *As an ML engineer*, I want the Preview step to show side-by-side
  source columns ↔ proposed target fields, with editable dropdowns
  for each mapping I want to override.
- *As anyone*, I want the Confirm step to show "X rows will be
  written, Y will be rejected (low confidence), here are 5 sample
  outputs" before I click Run.

### Work
- `frontend/src/pages/ProjectDatasetImportPage.tsx` with the
  3-step flow.
- `SourcePicker` component with source type dropdown +
  locator input + auth fields (when needed).
- `MappingPreview` component with editable per-field overrides + a
  table of the first 5 transformed rows.
- `ConfirmRun` component with stats + sample + final go button.
- Wired to the existing `POST /dataset-import/preview` + `/run`
  endpoints from Phase A.

### Out of scope
- Mapping config save/load to project state (Phase G).
- Visual BIO-tag annotator (would be a separate tool).

---

## Phase G — Persistent mapping configs + audit log

**Goal**: imports become reproducible. Save a mapping config to the
project; re-run later against a refreshed source with one click.

### User stories
- *As an ML engineer running periodic dataset refreshes*, I want to
  save my mapping config the first time, then re-run weekly against
  the refreshed source without re-doing the field mapping.
- *As a compliance reviewer*, I want an audit log showing every
  import: source, locator, mapping config, row counts, who ran it,
  when.

### Work
- New `dataset_import_configs` table.
- "Save mapping" button on the Confirm step.
- "Re-run from saved" entry point on the Import page.
- Audit log in the project's RunEvent stream
  (`stage: dataset_import`).

### Out of scope
- Scheduled / automatic refresh (cron-like). That's a separate
  workflow feature.

---

## Phase H — Plugin mappers + LLM-assisted suggestion (optional)

**Goal**: power-user extensibility.

### User stories
- *As a researcher with an unusual dataset shape*, I want to write a
  custom mapper in Python and register it as a plugin so my
  team can re-use it.
- *As a newbie facing an unfamiliar dataset*, I want BrewSLM to ask
  an LLM ("describe what these rows mean") and get a more confident
  mapping suggestion than pure column-content sniffing.

### Work
- Plugin contract for mappers: subclass `TargetMapper`, register
  via a manifest entry. Same plugin pattern BrewSLM already uses
  for adapters and runtimes.
- Optional LLM-assist mode: introspector sends sample rows + column
  names to a teacher model and asks for a mapping suggestion. The
  LLM's output is a *proposal*, never a silent action — same
  user-confirms rule.

### Out of scope
- Marketplace / sharing of mappers across organizations.

---

## Rollout order

| Order | Phase | User pain unblocked | Effort |
|------:|-------|---------------------|--------|
| 1 | A — Foundation | dev-internal scaffolding | medium |
| 2 | B — Introspector + dry-run | newbies finally see proposals instead of writing code | medium |
| 3 | C — Mapper catalog | covers 80% of common dataset shapes | medium |
| 4 | D — HuggingFace connector | hundreds of public datasets become click-import | medium |
| 5 | F — UI wizard | UI-first newbies don't need the CLI | medium |
| 6 | E — Kaggle connector | competition / Kaggle dataset users | small |
| 7 | G — Persistent configs + audit | reproducibility for production teams | medium |
| 8 | H — Plugins + LLM-assist | power users + edge-case shapes | medium |

Phases A → C → D give newbies the highest leverage path: from "find a
dataset on HF" to "trained model" in two commands without writing
code. Phase F closes the UI-first user's loop. The rest is hardening.

---

## Open questions

1. **Mapping config schema** — JSON Schema vs Python dataclasses vs
   YAML. Lean JSON Schema for portability (configs round-trip
   through HTTP + storage cleanly).
2. **Streaming for huge datasets** — Phase D mentions HF streaming
   mode. Same applies to Kaggle. How big before we force streaming?
   Probably anything over 1GB.
3. **Source-side data licenses** — Kaggle datasets have license
   restrictions; HF datasets too. The import flow should surface
   the license on the Preview step so users know what they're
   ingesting (compliance teams care). Lands in Phase B (intro
   shows the license string from `describe()`).
4. **Rejected rows** — write-with-flag (cross-cutting decision) is
   the current intent. But for very large imports (>100k rows),
   the rejected log can balloon. Cap the rejected log at a
   configurable size with a "showed first N, dropped M more"
   summary?
5. **Concurrency** — large imports take minutes. Should this go
   through the existing Celery worker queue (the eval / synthesis
   timeout fix we just shipped is a band-aid; for >10min imports
   the job pattern is right)? Probably yes for Phase D onward;
   add to the cross-cutting decisions at that point.
6. **Mapping versioning** — once configs are persisted (Phase G),
   should re-runs require a config version match? Or auto-migrate
   to the latest config schema? Defer until Phase G.

---

## What this is NOT

- **A general-purpose ETL platform.** This is a focused on-ramp for
  BrewSLM's specific task-handler shapes. Don't grow it into Airflow.
- **A data labeling tool.** The kaggle converter assumed pre-labeled
  data; this pipeline does too. Labeling unlabeled data is a separate
  problem (Active Learning Studio in a future phase, or external
  tools like Argilla / Label Studio).
- **A data cleaning pipeline.** We already have one (the Data
  Cleaning tab). The import pipeline takes rows already-cleaned by
  the source and routes them; quality checks live in cleaning.
