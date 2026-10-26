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
  | `qa-pair` | `question` (adapter reads question, prompt, query, instruction, input; answer response/output/target/completion; SST-2 / NLI / `sentence` rows are NOT qa-pair input) |
  | `rag-grounded` | `question` + `context` (the same question over a different passage is NOT a leak; deliberately unlike the question-only `question_leakage_filter`, which guards generated rows against a gold set) |
  | `seq2seq-pair` | the source text (`question`, i.e. before the Summarize/Translate wrap). Adapter source aliases: source, input, question, prompt, instruction, text, content, article, document, body, passage, dialogue, transcript |
  | `structured-extraction` | `text` (the document; not `structured_output`, not the field-list prompt). Adapter document aliases: text, content, input, question, prompt, document, passage (or `field_mapping` text); a file with none of them (a `narrative` column) is `input_column_unrecognised`, never trained with the gold JSON in the prompt |
  | `classification-label` | `text` (not the label, not the candidate-list prompt). The adapter reads `text, content, input, question, prompt, instruction, sentence, comment, review` (in that order), else a pair joined as `Premise: ..\nHypothesis: ..` / `Sentence 1: ..\nSentence 2: ..` (premise+hypothesis, sentence1+sentence2) so the prompt's `Text:` block shows both sides. SST-2 (`sentence`) and NLI (`premise`+`hypothesis`) work under THIS adapter only; under default-canonical they are keyed by the default text-side aliases below. A row whose text would equal its label (no input column found: the canonical fallback sets text = answer) is rejected (`classification_input_is_label`), never trained. Same for `structured-extraction` when no document column is recognised and the canonical fallback would put the gold JSON in the prompt (`structured_input_contains_gold`). A random split reports such rows in the manifest `include_types_resolution.input_rejected` (the split card shows the count + reasons) and errors when nothing is left. These two rejections fire ONLY when no input column was mapped: an explicit `field_mapping` `text` (else `question`; dotted paths) is read DIRECTLY as the input (`data_adapter_service._mapped_input_text`, ahead of every alias / canonical fallback) for `classification-label` and `structured-extraction`, so `{utterance:"help", intent:"help"}` with a mapping is kept and a saved `{question: X}` preference trains `X`, never `Question: X\nAnswer: <label>`. The mapper sends `text` for these two adapters, `question` for the rest. |
  | `tool-call-json` | `question` (the user prompt; not tool name / args / result) |
  | `preference-pair` | `prompt` (not chosen / rejected) |
  | `vision-language-pair` / `audio-transcript` | asset path + `question` (not caption / transcript) |
  | chat rows (`messages` / `conversations`, any adapter) | every non-assistant turn, in order |
  | `default-canonical` and plugin adapters | the SOURCE columns the adapter reads as input, resolved with `record_normalization._get_nested_value` (dotted `field_mapping` paths work; only the exact answer path is excluded). Question side: `field_mapping` `question` + config `question_fields`/`prompt_fields` + aliases instruction, input, prompt, query, question, ask. Context (always added): config `context_fields` + context, passage, document, article, chunk. A blank cell is skipped (a blank `question` falls through to the text side). If ONLY a context alias matched and the row has another unrecognised content column (an `utterance`), the row is `input_column_unrecognised` (`{document, summary}` still keys on `document`). Text side, used ONLY when the row has no question-side column: `field_mapping` `text` + config `source_fields`/`text_fields` + text, content, body, message, sentence, sentence1/2, premise, hypothesis, text_a/b, text1/2, tokens, words (a token list is joined). So `{prompt, text:"p 3 -> resp"}` keys on `prompt` alone. Provenance columns (`record_normalization.NON_CONTENT_KEYS`: `source`, `id`, `split`, `timestamp`, `metadata` ...) and target-named columns never enter via aliases. Span / NER rows key on `tokens`, never `ner_tags`; SST-2 `{sentence,label,idx}` on `sentence`; NLI on premise + hypothesis. |

  Every canonical-key adapter additionally checks (`_input_is_real`) that the input is the text of ONE real column and
  not the row's answer/label (an input equal to the answer is still real when a non-target, non-provenance, non-answer
  column holds the same text: a GEC pair left unchanged, `{text:"help", label:"help"}`; `highlights`, `abstract`,
  `summary`, `target_text` are target names). Flagged rows are ALWAYS rejected per row as `input_is_answer` (listed by file line in
  the ROWS_REJECTED 422, bulk-droppable in the card); the whole-file `input_column_unrecognised` 422 is only for "no row has a real input column". 409 and 422 errors carry `metadata.overlap_input_columns`
  (the card prints it under the error). Otherwise (the canonical fallback fabricated it from the answer or concatenated columns)
  the import is 422 `input_column_unrecognised` + `metadata.columns` (also when the adapter rejected EVERY row for
  that reason, or none of the file's columns is a recognised input). `no_input_field` 422s carry `metadata.columns`
  too, so the card offers the mapper. The manifest's `overlap_input_columns` names the columns the overlap check
  keyed on (the result card prints "Overlap checked on: ...").
  A `field_mapping` sent ALONE (the card's mapper) is merged onto the project's adapter preference (adapter, config,
  task profile kept; the card also sends the split form's adapter wiring with any mapping). Sending `adapter_id` /
  `adapter_config` / `task_profile` still makes the request the whole wiring.

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

- **Follow-up candidate (seq2seq input == target):** on the random split, the
  seq2seq-pair adapter maps `{"input": "", "output": o}` / `{"target": t}`-only /
  `{"summary": s}`-only rows to question == answer (pre-existing on main). The
  predefined-split import now reports such rows as `input_is_answer`; a matching
  `seq2seq_input_is_target` rejection on the random split is a candidate follow-up.

## Document sources (website / git repo) + code chunker

Pull documents from a site or a repo instead of uploading them (phase G1).
No migration: provenance lives in `RawDocument.metadata_` (+ `source`,
`license_info`); a provenance ledger is G2.

- **Readers** (`services/document_sources/`): `web_crawl.py` (polite BFS, stdlib
  + httpx; robots per RFC 9309 via `robots.py` — product-token group, longest match, Allow wins
  ties, `*` / `$`, percent-encoding normalised; unreachable / 5xx → host rejected
  `robots_unreachable`, 4xx → allow; robots.txt capped 500 KB, sitemap 10 MB;
  a `threading.Event` cancels the crawl; allowed-prefix limit, sitemap + one level of
  index, ETag / Last-Modified cache under `DATA_DIR/web_cache`, in-prefix
  redirects only) and `git_repo.py` (`git` CLI shallow fetch of one ref into a
  temp dir, commit pinned, licence → SPDX by keyword, file classes `doc` /
  `code`). Rejections carry stable reason codes — web: `http_error` `robots_disallowed` `robots_unreachable`
  `outside_prefix` `not_html` `too_large` `excluded_by_pattern` `duplicate_content`
  `fetch_error`; repo: `excluded_by_glob` `too_large` `binary` `unsupported_type`;
  importer: `parse_error`. Over-limit pages/files are only *counted*
  (`limit_reached`). Off-site links are simply not followed (not rejected).
- **Git safety**: URL starting `-` refused; only `https://` `http://` `ssh://`
  `git@host:path`; `file://` / local paths only with
  `settings.DOCUMENT_SOURCES_ALLOW_LOCAL_GIT` (default off; tests patch it on).
  `GIT_TERMINAL_PROMPT=0`, `GIT_ALLOW_PROTOCOL` pinned, symlinks never followed.
  A URL with embedded credentials (`https://user:token@host/…`) or an ssh host starting `-` is
  refused (`DOCUMENT_SOURCE_INVALID`); ssh / scp targets on localhost / 127.0.0.0/8 / `::1` count
  as local (same setting). `exclude_globs` are *added* to `DEFAULT_EXCLUDE_GLOBS`
  (`**/.git/**`, `**/node_modules/**`, `**/vendor/**`, `**/dist/**`, `**/build/**`, `*.min.js`,
  `*.lock`, `package-lock.json` — any depth); `default_excludes=false` drops the defaults. A glob
  with no `/` matches the file name at any depth, one with `/` is anchored at the root. The
  wider fallback fetch only runs when git says the ref wasn't found / not our ref (shallow, or
  `--filter=blob:none` for a commit id). A cancelled Job sets a `threading.Event` the clone /
  crawl polls (`git` is killed). Dataset-import `introspect` / `preview` / `run` run the source
  in `asyncio.to_thread`; the `web` source caps preview / introspect crawls (`PREVIEW_MAX_PAGES`).
- **Importer** (`importer.py`): `ingest_file` → `process_document` per page / file
  (an upload's path). Web docs: `source="web_crawl"`, metadata `source_url,
  final_url, retrieved_at, etag, title, strip_navigation, import_job_id`. Repo docs:
  `source="git_repo"`, `license_info=<spdx>`, metadata `source_url, ref, commit, path,
  language, content_kind, licence_file, retrieved_at, import_job_id`; filename =
  path with `/` → `__` (an extension-less file gets its language's extension).
  Nothing imported raises `ValueError` with the rejection counts (Job fails / inline 422).
- **API** (`api/document_sources.py`): `POST /api/projects/{id}/document-sources/web-crawl`
  and `.../git-repo`, `?async_job=true` (default) → 202 Job kind
  `document_source_web_crawl` / `document_source_git_repo`; `false` → 201 summary.
  `DocumentResponse` now also lifts `license_info`, `source_url`, `path`, `commit`,
  `content_kind`, `language` for the provenance label. Dataset-import sources
  `web:<seed url>` and `git:<url>[@ref]` yield rows for the `text_only` mapper.
- **HTML**: `parse_html(path, strip_navigation=True)` (set from
  `metadata_["strip_navigation"]` by `process_document`) drops nav / header / footer /
  aside, navigation roles and breadcrumb / nav / menu / sidebar / toc / footer class or id
  tokens (container tags only: div ul ol section span table form), and keeps only
  `<main>` / `role="main"` (else `<article>`) text when present. `html` / `body` / `main` /
  `article` / `role="main"` are never dropped; the page is parsed into a small element tree
  (implied `</p>` / `</li>` / cell closes, stray end tags ignored) so an unclosed tag can't
  swallow the rest. With no `<main>`, if stripping leaves < 20% of the visible text it retries
  by tag / role only, then keeps everything. Uploads keep everything.
- **Code files**: `utils/file_parsers.parse_code` keeps text verbatim; `PARSERS` now
  accepts code extensions (a `.json` upload stays a row file; a repo `.json` is code via
  `metadata_["content_kind"] == "code"`). `ingest_file` also treats an existing
  `<stem>.extracted.txt` as a name clash (`a.py` / `a.java` must not share sidecars).
- **Code cleaning** (`cleaning_service._clean_code_document`, runs before the tabular
  branch): text byte-for-byte, **no redaction** (PII / secret counts recorded,
  `metadata_["redaction"] == "skipped_for_code"`), chunks from
  `code_chunker.chunk_code(text, language, max_chars=chunk_size)`. cleaned.jsonl /
  `.chunks.jsonl` rows add `path, language, start_line, end_line, symbol, kind`.
  `chunk_overlap` is ignored for code. `auto_rag_service` passes those keys through,
  re-chunks code with `chunk_code` (never sentences) and labels a passage
  `[n] path:start-end (symbol)` + newline + code.
- **Code chunker** (`services/code_chunker.py`, pure): heuristic scanners (no
  tree-sitter) over comment/string-masked text. Java / C#: one chunk per method
  (`Class.method`), class header + fields = `class` chunk, nested classes qualified,
  anonymous classes stay in their method. Python by indentation (decorators,
  docstring belong to the unit); go / js / ts / c / cpp by brace balance; sh functions;
  tf / hcl blocks (`resource.t.n`); yaml top-level keys; json top-level keys; xml
  depth-1 children; sql statements; ini sections; properties groups; Dockerfile stages.
  Python lines that start inside an open triple-quoted string or bracket are body of the
  current unit, never dedent boundaries; JS / TS regex literals (`/[{]/g`) are masked; GNU-style
  C (`int` / `name(args)` / `{`) is recognised; HCL heredoc bodies are masked; one line longer than
  `max_chars` is cut into `max_chars` slices that all cite that line.
  Invariant: with `context_lines=0` every source line is in exactly one chunk in order;
  blank lines after a unit and a lone closing brace after the last member stay with the
  previous chunk, so `end_line` can include them. Oversize units split at line
  boundaries (blank preferred), later pieces get symbol `#2`, `#3`. Unknown language →
  blank-line blocks packed to `max_chars`. Tests: `tests/test_code_chunker.py`.
- **BM25 identifiers**: `_tokenize` splits `getUserName` / `HTTPServer` / `user_id` into
  parts and keeps the whole lowercased identifier; prose tokenizes as before.
  `BM25_TOKENIZER_VERSION = 3` is stamped in every index; a missing / different
  version is stale (`document_index_status`, `qa_index_is_current`) and rebuilt, `retrieve()` raises
  `AutoRagUnavailable` on one, and `scripts/auto_rag_ab.py` fails loudly on an `index_dir_override`
  built by another version. An all-caps plural (`IDs`) is one token and a digit→capital step
  (`5G`) is not a boundary.
- Tests: `tests/test_web_crawl_source.py` (MockTransport site), `tests/test_git_repo_source.py`
  (real `git init` fixture repo), `tests/test_bm25_tokenizer.py`.
- **Not done / follow-ups**: re-importing the same site or repo creates duplicate
  documents (no cross-import dedupe yet — G2 ledger); no SSRF guard for the crawler
  (single-user local tool); `.json` code documents are still named `*.json`, so any
  consumer that decides "tabular" by extension alone sees them as row files.
  Row loaders that go through documents (`task_shape_service.sample_project_rows`,
  `dataset_service` combine + profile sampling, `sample_document_rows`) skip / line-sample
  them via `code_chunker.is_code_document` (the one definition; cleaning reuses it).
