# Data: intake, synthetic rows, splits, task shape, health, seed documents

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Synthetic data** — two paths: (a) modern `synth_playbook_service`
  with recipe-aware playbooks + per-row `review_status="pending"`;
  (b) legacy `synthetic_service` with task framework + bridge to Jobs.
  Both auto-save to `data/projects/{id}/synthetic/synthetic.jsonl`.
  Synthetic rows land `review_status="pending"`;
  `synth_review_queue_service` lists them grouped by `synth_source` and
  flips status via `bulk_update_review_queue` (by row-id) or
  `bulk_update_by_source` (every pending row in a source group by key —
  `POST /synthetic/review-queue/bulk-update-by-source`). The **Data Studio
  review-queue panel** (`DataStudioReviewQueuePanel`) is actionable: its
  synthetic-pending `by_source` groups carry the raw `synth_source` and
  render inline "Accept all / Reject all" (Epic E). Reject is soft (rows
  stay on disk, grouped by `reject_reason`, bulk-purgeable). Training-bound reads
  (`dataset_service._load_records_from_file`) drop BOTH "pending" and
  "rejected" rows (rejected rows used to train — a rejected eval-only answer
  key reached project 31's split on 2026-10-14); the review queue reads all
  (`include_pending_synth=True`) and gives rows missing an integer id the next
  free one (`_ensure_row_ids`).
- **Versioned prepared splits** (Epic E) — `split_dataset` writes the active
  `prepared/{train,val,test}.jsonl` + `manifest.json` (the single source the
  trainer/export/coverage read) AND snapshots each run to
  `prepared/versions/{v}/` (`snapshot_prepared_version`). `restore_prepared_version`
  copies a snapshot back to active; `activate_prepared_version` (→ `POST
  /data-studio/dataset-versions/{version}/activate`) restores + records
  `runtime_config["active_prepared_version"]`. "Make active" and "Retrain from
  this version" on `DataStudioDatasetVersionsPanel` both call activate (retrain
  then opens Training, which reads the now-active version). The active version
  defaults to the latest prepared run when none is explicitly activated.
- **Generic intake (Wave 2a)** — `app/utils/tabular_io.py` is the ONE reader
  for row files (CSV/TSV/JSON(L)/XLSX/Parquet; encoding: utf-8 → charset_normalizer
  only when it finds a non-Latin script → cp1252 → latin-1; delimiter sniffed).
  `file_parsers` raise `DocumentParseError` (never return error strings — they
  used to become ACCEPTED training text); scanned PDFs error out; HTML/DOCX
  tables supported. Cleaning a row file is row-preserving
  (`cleaning_service._clean_structured_document`: per-field PII redaction,
  dup/empty drop, columns kept) — only documents are chunked. **Scanned
  PDFs are OCR'd** (`file_parsers.parse_pdf`: pages with no text layer →
  `ocr_pdf_pages` = `pdftoppm -r 300` + `tesseract … stdout`, subprocess,
  no Python dep) when `ocr_available()` (both binaries on PATH and
  `BREWSLM_OCR` not off; `BREWSLM_OCR_LANG` / `_DPI`); otherwise the
  DocumentParseError says "apt install tesseract-ocr poppler-utils". CI has
  no tesseract — the OCR test skips there. `process_document` runs `parse_file` in
  `asyncio.to_thread` — OCR inline froze the whole backend for minutes. Import wizard
  takes browser uploads: `POST /projects/{id}/dataset-import/upload` stages
  under `projects/{id}/imports/` → `file:` locator (`sources/file.py`).
  **Page furniture** (`services/page_furniture.py`, pure): `parse_pdf` joins
  pages with `file_parsers.PAGE_SEPARATOR` (`"\n\f\n"`; cleaning turns it into
  the old blank line). `clean_document(remove_page_furniture=)` (None = the
  doc's last `metadata_.page_furniture.enabled`, else
  `DEFAULT_REMOVE_PAGE_FURNITURE` True; API `remove_page_furniture`, Prepare
  checkbox "Remove repeated page headers and footers") runs
  `remove_page_furniture(text)` before boilerplate/chunking: only the first /
  last `EDGE_LINES` (4) non-empty lines of a page; rules `repeated`
  (digit-normalised form on ≥ `FURNITURE_PAGE_SHARE` 0.5 of pages, min 3 — a
  fused head/tail is cut off a body line when it has a number and ≥ 12 chars,
  or is a whole line elsewhere), `page_number`, `running_head` (words repeat
  on ≥ 3 pages, numbers change, ≥ 3 letters — a table row isn't one). Result
  `furniture_lines_removed` + `page_furniture` {pages, segments trimmed,
  by_rule, examples}. < 3 pages / no `\f` / row files untouched. The seeder
  cleans with it OFF (paged / scanned layouts keep furniture on purpose).
  IRS Pub 17: 173/1,103 passages with furniture → 1/1,063.
- **Task shape (Wave 2b)** — `task_shape_service` is the single detector +
  vocabulary: `TASK_SHAPES` (canonical profile → adapter/recipe/mapper),
  `detect_task_shape(rows, intent=)` = introspector column rules (evidence) ×
  adapter map-rate (validator) + intent prior (+0.08, never an override);
  row-fit-only guesses are capped below the 0.8 confirm threshold.
  `confirm_task_shape` writes `dataset_adapter_preset` (what split/train/eval
  read) + recipe snapshot; `apply_recipe_to_project` now syncs the preset too.
  Split manifest `task_profile` = majority resolved `_task_profile` of the rows
  (`task_profile_requested` kept). UI: `TaskShapeConfirmCard` in the import
  wizard + DatasetPrepPanel. Other detectors (newbie-autopilot keywords, brief
  keywords, recipe header sniffer) still exist — fold them in as priors next.
- **`json-extraction` recipe (typed JSON records)** — `recipe_service`
  `json-extraction` = text in, one JSON object with a fixed field set out:
  task_profile `structured_extraction`, adapter `structured-extraction`,
  `scoring_mode="field_match"`, columns `text` → `structured_output`.
  `span-extraction` stays for entity lists (`{entities: [...]}`, `span_set`).
  `task_shape_service.structured_recipe_for_rows` picks between them: gold
  payload is a JSON object WITHOUT an `entities` list -> json-extraction
  (candidate `recipe_id`; `confirm_task_shape(recipe_id=)` / the confirm
  request's optional `recipe_id`; absent, it re-samples the project rows).
  **Gold column**: use `structured_output` (the adapter `json.dumps` the
  payload, keeping types and nulls). **Do NOT use `kv_to_structured` for typed
  records** — it stringifies values and drops nulls, so the gold loses exactly
  what the typed scorer measures. **Declared output schema**
  (`output_schema_service`, JSON-Schema subset): `{"required": [...],
  "properties": {"f": {"type": "string"|[...], "enum": [...]}}}`; types
  string/integer/number/boolean/null (object/array type-checked strictly);
  enum members may include `null`. `PUT /api/projects/{id}/recipe/output-schema`
  (400 + message when malformed or the recipe isn't json-extraction; `null`
  clears) stores it on `Project.selected_recipe["output_schema"]` (re-applying
  the same recipe keeps it); `dataset_service._recipe_output_schema` merges
  recipe `scoring_mode` + the project's `properties`/`required` into the
  prepared manifest's `output_schema`, and `combine_datasets` pins the
  training prompt's field list to `sorted(properties)` so train and eval name
  the same fields. An explicit `required` (including `[]`) is kept in the manifest and honoured by the scorer; only an undeclared one means "all fields". The schema takes effect only the next time the dataset is prepared, and the training prompt pins the field list, so retrain afterwards. (UI hint: "Changes apply the next time the dataset is prepared; retrain afterwards so the training prompt lists the same fields.") No properties declared = field list derived from the
  references, as before. UI: `OutputSchemaEditor` on the task-type picker page
  (shown when json-extraction is applied). Tests:
  `tests/test_json_extraction_scoring.py`, `OutputSchemaEditor.test.tsx`.
- **Data defaults (Wave 2d)** — split sources default to
  `dataset_service.DEFAULT_TRAINING_SOURCE_TYPES` = CLEANED + SYNTHETIC (Wave 3b:
  GOLD_DEV is no longer a default source — the gold set is eval-only, and the
  train↔gold leakage check flagged the defaults' own gold rows). `split_dataset` defaults `dedup_rows=True`
  (reported, same matcher as leakage) and `auto_stratify=True` (stratify by a
  categorical `label` when no stratify/disjoint given → manifest
  `stratify_auto`). The split API only inherits the active version's config
  for an EXPLICIT `dedup_rows` request (`dedup_resplit`). When synthetic rows
  exist, `resolve_training_dataset_types` keeps CLEANED as `labelled_only`:
  `combine_datasets` drops just unlabelled passages (rows without
  `row_index`), counts in `include_types_resolution.cleaned_filter`.
  `train.py` classifier head uses `_class_weights` (≥3× imbalance, mean 1,
  cap 10) via a weighted-CE Trainer (`class_weighting` auto|none).
  Autopilot: `task_shape_service.project_data_task_profile` (confirmed, else
  confident detection) beats intent keywords (`task_profile_source`); the
  brief's `TASK_KEYWORDS` feed the same intent prior.
- **Split keeping the test rows** (`split_dataset(keep_test=True)`, API
  `keep_test`, split-form checkbox "Keep the current N test examples",
  shown when the active manifest has a test split): `_keep_test_rows` keeps
  `prepared/test.jsonl` verbatim, re-divides everything else train / val by
  their ratio, and drops any row whose `_entry_text_key` (text/input/prompt…
  whitespace-normalised, lower-cased) matches a test row — so adding data
  never moves the answer key into training. Manifest `keep_test` report.
  Tests: `tests/test_split_keep_test.py`.
- **Autofix engine** — `data_health_autofix_service` exposes 9 safe
  one-click transforms via `POST /data-health/autofix/{preview|apply}`.
  Original four (drop_failed_docs, dedupe_duplicate_docs, redact_pii,
  canonicalise_labels) plus Coach-stage-2 phase 4's five "rewrite
  cleaned text" fixes: `near_duplicate_dedup` (aggressively-normalised
  500-char-prefix hash), `normalize_whitespace`, `strip_html`,
  `length_cap` (truncates to `max_seq_length × CHARS_PER_TOKEN_APPROX`
  — closes the gap the phase-3 truncation signal flags),
  `normalize_schema` (renames gold-row variants `class→label`,
  `text→input`, `target→label`, `prompt→input`). Each is idempotent —
  second run finds nothing to change. Signals in `data_health_service`
  emit the matching `autofix_kind` (sampled-read for the cleaned-text
  signals to keep the poll endpoint fast). See `ROADMAP.md` Epic E0
  phase 4.
- **Train↔gold leakage** — `data_health_service.scan_train_gold_leakage`
  is the honesty guard on the eval gate: it scans GOLD_DEV/GOLD_TEST
  rows against the training corpus (TRAIN ∪ CLEANED ∪ SYNTHETIC) and
  flags any that are identical (exact-normalised) or near-identical
  (token-set Jaccard ≥ `LEAKAGE_FUZZY_THRESHOLD` 0.9, via an inverted
  index for bounded compute) — a leaked gold set makes the pass-rate
  measure memorisation, not skill. Renders as the `leakage` group's
  `leakage.gold_train_overlap` signal (any overlap warns, ≥10% of gold
  rows blocks; GOLD_TEST contamination called out separately). **No
  `autofix_kind`** — deleting a gold/train row is a judgement call, not
  a safe one-click delete; the Coach gold-set nudge
  (`gold_set:train-leakage`) routes to `data-studio-splits` to re-split.
  Self-hides until both a gold set and a training corpus exist.
  **Phase 7** adds `scan_prepared_split_leakage` (train↔val / train↔test
  / val↔test over the *prepared* TRAIN/VAL/TEST splits — NOT against
  CLEANED/SYNTHETIC, which the splits derive from) → `leakage.split_overlap`
  signal, and a generic leaked-row **drill-down** in `DataHealthReportPanel`
  (any signal whose `context.examples` is populated renders an expandable
  table of split / match-kind / leaked-row / matched-source-row). The
  shared matcher core is `_build_leakage_index` + `_match_row_against_index`.
  **Phase 8** adds the training-stage Coach nudge `training:split-leakage`
  (mirrors `gold_set:train-leakage`) for prepared-split leakage.
- **Seed documents** (`services/seed_documents_service.py`, Job kind
  `seed_documents`, `POST /projects/{id}/flows/seed-documents`): a synth
  backend (`pick_backend`, local first) writes `count` invented documents over
  `genres` (default `DEFAULT_GENRES`) containing `kinds`; `ingest_file(...,
  source="seed_documents")` → `process_document` → `clean_document` (one
  `seed-documents-<stamp>.txt`). Raises when nothing usable was written.
  **Prompt gate** (`services/prompt_contamination_service.py`, 2026-10-18):
  the seed-documents route (`kinds` / `genres` / `domain_hint`) and both
  span-synthesis routes (`entity_types` / `focus_types` + the project's
  definitions and custom-pattern descriptions, `span_prompt_fields`) call
  `refuse_if_contaminated` → 400 `PROMPT_CARRIES_BENCHMARK_VALUES`
  (`http_detail`: `metadata.offending` [{value, type, benchmarks, fields}])
  when a prompt field holds a benchmark gold value as a whole word
  (`benchmark_gold_index` over every BENCHMARK dataset's rows.jsonl, via
  `benchmark_contamination_service.gold_values`); request
  `ignore_benchmark_values: true` overrides (seed Job params
  `prompt_check.overridden`). Dry run `POST /synthetic/prompt-check`
  {fields, include_conventions}; UI buttons "Check against benchmarks"
  (`SeedDocumentsCard`, + "Write anyway") and "Check prompt against
  benchmarks" (`SyntheticPanel` span mode, + the "Generate anyway" checkbox);
  `api/promptCheck.ts` (`checkPrompt`, `promptCheckText`,
  `offendingValuesFromError`). Tests `tests/test_prompt_contamination.py`.
  `SeedDocumentsCard` is the first card on the Data tab (`IngestionPanel`);
  bell deep-links `#seed-documents`. **`layout="paged"`** (card checkbox
  "Multi-page, with headers and footers"; `words_per_page`, default 110):
  `paginate_document` splits each document at sentence ends and inserts
  invented header / footer lines by code (`_HEADERS` / `_FOOTERS`: org
  letterheads, CONFIDENTIAL, "Page i of n", "- i -", print dates…). The
  bracketed "[i/n]" footer is deliberately never generated — it is the
  held-out style for measuring. Cleaning flattens the line breaks, so a
  footer reads inline in the passage. **`layout="scanned"`** (P1f): `render_scanned_pdf`
  typesets the paged documents on A4 pages (a footer line ends a page),
  degrades them (`_degrade_page`: skew, blur, speckle, contrast, JPEG) and
  ingests an image-only PDF that `parse_pdf` OCRs — raises when OCR isn't
  available. Project 31 run #67 (+125 OCR'd passages) = run #65 on the real
  scan (0.72 vs 0.71). Tests: `tests/test_seed_documents.py`
  (read the ingested doc back IN THE SAME LOOP — a later TestClient read can
  lose the write on the StaticPool).
