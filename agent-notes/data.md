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
  their ratio, and drops any row whose `_entry_text_key` (the prepared row's
  `text` / `input` / `prompt` / `question` / `instruction` / `source_text` string fields joined,
  whitespace-normalised, lower-cased; unchanged by the predefined import: prepared rows carry
  no extra key field and the random path is byte-identical to before it) matches a test row — so adding data
  never moves the answer key into training. Manifest `keep_test` report.
  Tests: `tests/test_split_keep_test.py`.
- **Predefined-split import** (`POST /dataset/predefined-split`, multipart `train` /
  optional `val` / `test`, `on_leak` fail|drop_from_train, `field_mapping`; Data tab split
  form button "Use my own train / val / test files" → `PredefinedSplitCard`):
  `predefined_split_service.import_predefined_split`. Shares the random split's row
  pipeline — `dataset_service._resolve_adapter_for_rows` (adapter + candidates / field
  list resolved ONCE over all three files' rows, so splits get one prompt shape) +
  `_normalize_rows_for_training(rejected=[...])`, then the shared tail
  `_write_prepared_splits` (files + Dataset + DatasetVersion rows, tagged
  `split_source`/`source_filename`) and `_finalize_prepared_manifest` (manifest +
  snapshot). Exact membership: no ratio, shuffle or dedup. Adapter-rejected rows → 422
  (`metadata.rejected.<split>.by_reason`, 20 sample rows). Reasons: `unparseable_line` (JSONL
  line / JSON-array item that isn't a JSON object), `malformed_row` (CSV/TSV wrong column
  count), `review_status_pending|rejected` (the random split's review gate, reported
  instead of dropped), `adapter_unmappable_or_empty`, `normalizer_hook_dropped`. Row numbers
  are PHYSICAL FILE LINES (strict reader `_parse_upload`: blank lines skipped but counted; CSV
  header = line 1; JSON array / XLSX / Parquet = 1-based record index).
  Re-importing WITHOUT a test file while an ACCEPTED test-role doc exists is 409
  `PREDEFINED_TEST_WOULD_BE_DROPPED` unless the form field `drop_existing_test=true` is sent (then the old
  test doc is REJECTED as before and the manifest carries `dropped_existing_test_document_ids`); the card
  shows a "Drop my current test file (N rows)" checkbox while its own test file is active.
  Leak/duplicate key = `predefined_split_service._input_side_key(source row, mapped entry,
  adapter_id=…)` (used ONLY by the import; prepared rows are not stamped and the random split /
  `keep_test` never see it): the normalised text the MODEL IS GIVEN, read from the adapter-mapped
  entry (the canonical row written to prepared/*.jsonl) before the chat template renders the
  answer. Never a label / answer / target column, an id-like column (`id`, `idx`, `uid`,
  `guid`, `*_id`, `_*` ...), or the adapter's shared prompt boilerplate (candidate list, field
  list). A surviving row whose mapped input column is empty is REJECTED in the 422 (`no_input_field`).
  There is NO heuristic fallback ("first three content columns", label-like columns): for
  default-canonical / plugin adapters where no input column is recognised (by name, `field_mapping`
  or adapter config fields) in ANY row, the import is 422 `ROWS_REJECTED` with
  `metadata.reason=input_column_unrecognised` + `metadata.columns` (the file's columns) and asks for
  a `field_mapping` (question/text = input column, answer = output column); the card shows the
  columns, two selects and a resubmit button. Input fields per adapter:

  | Adapter | Input fields in the key |
  |---|---|
  | `qa-pair` | `question` |
  | `rag-grounded` | `question` + `context` (the same question over a different passage is NOT a leak; deliberately unlike the question-only `question_leakage_filter`, which guards generated rows against a gold set) |
  | `seq2seq-pair` | the source text (`question`, i.e. before the Summarize/Translate wrap) |
  | `structured-extraction` | `text` (the document; not `structured_output`, not the field-list prompt) |
  | `classification-label` | `text` (not the label, not the candidate-list prompt). The adapter reads `text, content, input, question, prompt, instruction, sentence, comment, review` (in that order), else a pair joined as `Premise: ..\nHypothesis: ..` / `Sentence 1: ..\nSentence 2: ..` (premise+hypothesis, sentence1+sentence2) so the prompt's `Text:` block shows both sides. A row whose text would equal its label (no input column found: the canonical fallback sets text = answer) is rejected (`classification_input_is_label`), never trained. Same for `structured-extraction` when no document column is recognised and the canonical fallback would put the gold JSON in the prompt (`structured_input_contains_gold`). A random split reports such rows in the manifest `include_types_resolution.input_rejected` and errors when nothing is left. |
  | `tool-call-json` | `question` (the user prompt; not tool name / args / result) |
  | `preference-pair` | `prompt` (not chosen / rejected) |
  | `vision-language-pair` / `audio-transcript` | asset path + `question` (not caption / transcript) |
  | chat rows (`messages` / `conversations`, any adapter) | every non-assistant turn, in order |
  | `default-canonical` and plugin adapters | the SOURCE columns the adapter reads as input, resolved with `record_normalization._get_nested_value` (dotted `field_mapping` paths work; only the exact answer path is excluded). Question side: `field_mapping` `question` + config `question_fields`/`prompt_fields` + aliases instruction, input, prompt, query, question, ask. Context (always added): config `context_fields` + context, passage, document, article, chunk. Text side, used ONLY when the row has no question-side column: `field_mapping` `text` + config `source_fields`/`text_fields` + text, content, body, message, sentence, sentence1/2, premise, hypothesis, text_a/b, text1/2, tokens, words (a token list is joined). So `{prompt, text:"p 3 -> resp"}` keys on `prompt` alone. Provenance columns (`record_normalization.NON_CONTENT_KEYS`: `source`, `id`, `split`, `timestamp`, `metadata` ...) and target-named columns never enter via aliases. Span / NER rows key on `tokens`, never `ner_tags`; SST-2 `{sentence,label,idx}` on `sentence`; NLI on premise + hypothesis. |

  → 409 (`metadata.pairs`); `drop_from_train` removes
  train rows only; val↔test always 409. Within-split repeats kept, counted in
  `duplicate_report`. Validation happens before anything is written. Files are registered
  as ACCEPTED raw documents (`source=predefined_split:<role>`; a re-import marks the
  previous ones REJECTED), stamped on rows via `_source_document*`, so a later random
  split (`include_types:["raw"]`) re-pools the train/val files only. With `on_leak=drop_from_train`
  the registered TRAIN document holds exactly the KEPT rows (the upload's rows minus the dropped
  ones, in the upload's format; XLSX/Parquet stored as JSONL), so a later keep-test / plain
  re-split never pools the dropped overlap back in; the original filename + sha256 and the
  dropped file lines stay in the document metadata (`original_filename`, `original_sha256`,
  `dropped_row_lines`) and in `source_files.train` (`registered_rows`, `registered_sha256`,
  `dropped_row_lines`; keys only present when rows were dropped). **One predicate keeps
  the test file out of every training source** (`split_role_service.is_test_role_document` /
  `test_role_document_ids` / `drop_test_role_rows`; no-op for projects that never imported
  splits): `combine_datasets` skips a `split_role="test"` RAW doc and drops CLEANED/SYNTHETIC
  rows whose `source_document_id` is a test-role doc of ANY status (superseded imports
  too); `clean_document` refuses test-role docs (clean-batch lists the refusal in `errors`;
  chosen over silently skipping so the user sees why), so no cleaned.jsonl row / chunk file /
  cleaned.txt exists for them, which also keeps them out of the passage loaders, RAFT /
  teacher-answer prompts, documents→Q&A flow and span relabels; `GET /cleaning/chunks` and
  `synthetic_service._load_project_cleaned_chunks` (synthetic + gold seed pickers) skip them.
  Read-only inference (task-shape detection, adapter-studio preview, EDA, quality scan, doc
  preview) still samples them: they feed no training row. `keep_test` keeps the prepared test
  rows. A re-split WITHOUT `keep_test` is refused 409
  (`PREDEFINED_TEST_SPLIT_WOULD_BE_REPLACED`, whatever `include_types`) when an ACCEPTED
  test-role doc exists or the active manifest is predefined with a non-empty test whose doc
  is still accepted; exits: `keep_test`, or `discard_predefined_test: true` (test doc(s)
  marked REJECTED + logged, normal random split, old test rows NOT pooled; both flags = 400;
  split form checkbox "Replace my own test file with a random test split", shown while the
  manifest says `predefined_test_active`). Every split manifest (random, keep-test or
  predefined) carries `predefined_test_active: true` while the project has an ACCEPTED
  test-role doc; the key is ADDED only when true, so manifests of projects that never imported
  are byte-identical to before. The UI drives the replace checkbox + the "re-draws validation"
  note from it (fallback for older manifests: `split_source=="predefined"` with test rows; the
  409 error code also shows it). The guard's manifest-doc lookup is scoped to the project.
  `POST /ingestion/documents/{id}/process` refuses (409 `PREDEFINED_SPLIT_DOCUMENT_REJECTED`) a
  REJECTED `predefined_split:*` document instead of re-accepting it. Rejecting or deleting the test doc also clears the guard.
  The import saves its resolved adapter wiring (incl. scanned label/field
  list) as the project adapter preference so per-doc re-normalisation builds identical
  prompts. On failure after `ingest_file`, the registered docs/files are removed and the
  superseded import is restored. Per-file upload cap = `dataset_import.MAX_UPLOAD_BYTES`
  (413). `chat_template` form field / profile split default resolve like `/split`; adapter
  wiring rule matches `/split` (any of adapter_id/adapter_config/field_mapping/task_profile
  sent → request is the whole wiring, default-canonical if no adapter_id; the saved
  preference gets `origin: "predefined_split"`). Manifest carries
  the `/split` enrichment (`_enrich_split_manifest`). Intentional differences: observed ratios
  in `resolved_split_config`, `dedup_requested:false`, `keep_test:null`, no `seed` shuffle.
  Rows are byte-identical to a random split of the same files. Manifest
  `split_source: "predefined"`, `source_files`, `leak_report`, `duplicate_report`;
  `ratios` are the observed proportions. Not a Job (synchronous). Tests:
  `tests/test_predefined_split_import.py`, `PredefinedSplitCard.test.tsx`.
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
