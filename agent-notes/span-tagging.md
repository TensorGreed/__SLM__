# Span tagging (token classification, labelling, answer key, thresholds)

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Token classification** (`task_type="token_classification"`,
  `scripts/train_token_classification.py`, dispatched from
  `train.py::_run_training_attempt` right after `_normalize_task_type`):
  encoder + `AutoModelForTokenClassification` on BIO labels from char spans
  (`row_to_example` validates `text[start:end]`, drops overlaps;
  `spans_to_token_labels` / `token_labels_to_spans`; `span_prf` strict
  (type,start,end) = the span_set match; `predict_spans` shared with eval).
  Saves `model/` (+ `label_space.json`), `metrics.jsonl`, returns the
  training_report dict (`task_metrics.span_set`, `entity_types`). Eval:
  `_resolve_token_classifier_artifacts` (config.json `*ForTokenClassification`
  under `model_ref` or `model_ref/model`) → `_run_token_classification_inference`
  tags the pair's `structured_input` and emits `{"entities": [...]}` JSON for
  the span_set scorer; dispatched before the classifier-head check.
  **Packaging (Epic P5)**: HF/Docker export of a span run →
  `span_model_package_service.package_token_classifier` writes
  `model/README.md` (pure `build_span_model_card`: evidence-only — label
  space, row sources + **Labelled by** (`row_labeller` tally) only when the
  train file's row count still matches, an **External benchmarks** table
  (`benchmarks_for_run` rows with a result for the run),
  per-type P/R/F1 from the latest span EvalResults, answer key first, usage
  with `stride` = ¼ window + the leading-space trim, limitations) +
  `model/thresholds.json` (all 0 = argmax); manifest `span_package`.
  `predict_spans(with_scores=, thresholds=, min_score=, recall_bias=)` →
  `decode_scored_spans` (score = mean P(B-t)+P(I-t); bias subtracted from P(O)
  before argmax); all defaults → the old argmax path. `POST
  /projects/{id}/redact` {text, experiment_id?, mode mask|tag|spans,
  min_score?, recall_bias?} (`redaction_service`, one cached model via
  `load_tagger` — tests patch it). `RedactionTryBox` in the span
  EvalSummaryCard. Tests: `tests/test_span_packaging.py`. No ONNX
  (optimum/onnx not installed).
  **"Evaluate this run"** works for these runs: the lift job
  (`run_post_training_lift_eval`, auto after training too) routes
  `is_token_classification` runs to `run_span_evaluation` — no base model
  (an untrained head means nothing), scores on `test` and, when a GOLD_TEST
  dataset exists (`project_has_answer_key`), on `gold_test`; result
  `{kind: "span_eval", test, answer_key}` (`span_block`: P/R/F1, rows,
  `weakest_types`). `eval_summary_service.span_eval_summary` →
  `kind="span_eval"`, `verdict="evaluated"`, answer key first (+
  `written_by` from gold rows' `source`), failures from the answer key.
  `EvalSummaryCard`: "Span F1 x against the answer key", both score lines,
  weakest types, "Re-evaluate". Bell: "span F1 … on the answer key · … on
  the test examples". Tests: `tests/test_span_eval_summary.py`. Data gate
  accepts `entities` targets for this task type; UI option in
  `TrainingModelEssentialsSection`. Tests: `tests/test_train_token_classification.py`.
  **Long texts (Epic P1)**: `encode_windows` (fast tokenizer overflow,
  `window_stride` = config `window_stride` or ¼ window, < ½ window) is used
  by training (one example per window, labels from each window's offsets;
  B- only on the token holding a span's first char, so a window starting
  mid-span continues it), validation (`decode_window_predictions` merges the
  eval windows back into texts), `predict_spans` and therefore eval
  (`merge_window_scores`: a token's scores come from the window where it is
  most central). Report `windowing` {max_length, stride, train_windows,
  train_long_rows, eval_windows}. The saved tokenizer gets
  `model_max_length = max_seq_length` (DeBERTa-v3 ships ~1e30, so the HF
  pipeline never windowed and fed whole documents to a 512-trained
  encoder); `export_service.cap_token_classifier_window` writes it at export
  for older runs (manifest `token_window`, release-notes line). The HF
  pipeline needs `stride=128` to window. Measured on project 31's test
  texts as 4 multi-page docs: cut-at-512 0.59 → windowed 0.82 (per-text
  0.84); page-footer lines drop precision to 0.65 (data gap). The test
  file builds a tiny word-level tokenizer + random BERT offline — and swaps
  `sys.modules` entries with `_swap_module`, never `mock.patch.dict` (it
  drops numpy/tokenizers imported inside, which can't be re-imported).
  Docs: `workflows/sensitive-data-tagging.md` (the from-zero walkthrough).
- **Labelling conventions** (`services/span_type_definitions_service.py`):
  `runtime_config["span_type_definitions"]` = `{type: definition}`
  (`normalise_definitions`, `project_definitions`, `load_definitions(db, pid)`);
  `definitions_clause(types, defs)` is the prompt block, sent by BOTH
  `generate_span_extraction_rows` (training labels) and the relabeller
  (`build_relabel_prompt(texts, types, conventions)`); `suggest_definitions`
  offers starting wording for common type names (never applied silently).
  `GET/PUT /synthetic/type-definitions` (GET: definitions, the project's
  types, suggestions for undefined ones). `SpanTypeDefinitionsEditor` in the
  Synthetic tab's span mode. The relabeller's core is `label_texts(llm,
  texts, types, conventions, …)`; `relabel_training_rows` relabels
  `synthetic.jsonl` in place (+ regex pass; failed rows keep old labels) —
  `POST /synthetic/relabel-answer-key {target: "training"}`, button
  "Relabel training rows with <pin>". Summaries carry `conventions_types`.
  Tests: `tests/test_span_type_definitions.py`.
- **Custom infotypes** (`services/span_custom_patterns_service.py`):
  `runtime_config["span_custom_patterns"]` = `[{type, pattern, description,
  flags}]` (flags i/m/s). Validated on write (`validate_pattern`): type
  normalised like the definitions, compiles, ≤ 300 chars, no empty match,
  nested unbounded repeat (`(a+)+`) refused, then a timed scan of ~2k-char
  adversarial strings in a killable child process (`_speed_check`, 3 s — `re`
  holds the GIL, a thread can't time it out). The regex pass
  (`_extract_entities_via_regex` / `_merge_regex_entities` /
  `augment_saved_rows_with_regex(custom_patterns=)`) runs built-ins + custom
  under the vocabulary filter (custom adds, never overrides); loaded via
  `load_patterns(db, pid)` in span generation, `relabel_training_rows`,
  `_label_and_append` and regex-augment — not the second-model answer key.
  `GET/PUT /synthetic/type-patterns` (PUT all-or-nothing, 422
  `INVALID_TYPE_PATTERN`), `POST /synthetic/type-patterns/test` (count + first
  10 matches over saved rows + passages). UI: `SpanAddTypeForm` (name +
  definition + pattern with Test). Tests: `tests/test_span_custom_patterns.py`.
- **Focus types ≠ vocabulary** (span synthesis): `generate_span_extraction_rows(
  entity_types=<vocabulary>, focus_types=<must contain>)`; the task / API carry
  `focus_types`; the prompt says "MUST contain several of: … Still label EVERY
  other entity of the allowed types"; the regex merge always runs over the
  full vocabulary. `augment_saved_rows_with_regex(project_id, entity_types)` →
  `POST /synthetic/spans/regex-augment` (idempotent; `added_by_type`) repairs
  rows generated under a restricted list (project 31: +438 entities; run #57
  vs the Haiku key 0.74 → 0.79, phone 0.61 → 0.96, email 0.71 → 1.00). The
  Refine button sends `all_types` as the list and the weak ones as focus.
- **Refine weak types** (`services/span_refine_service.py`): `latest_span_eval`
  (newest EvalResult of the project with `metrics.scoring_mode == span_set`),
  `weak_types_from_metrics(metrics, threshold)` (pure; weakest first),
  `weak_entity_types(db, pid, threshold)` → `{eval_result_id, experiment_id,
  types, all_types, suggested_rows}` (`ROWS_PER_WEAK_TYPE` 25 within 60–300);
  `GET /synthetic/weak-types?threshold=0.8`. `SyntheticPanel`'s "Refine weak
  types" button (span mode) fills the entity types, sets all-passages + the
  suggested rows and runs the normal generate path; the async span body now
  sends `backend: "ollama"` for the local provider. Seed prompt asks for
  varied surface forms (separators, locale id formats, names in greetings).
  Tests: `tests/test_span_refine.py`, `SyntheticPanel.test.tsx`.
- **Label passages as training rows** (`span_relabel_service.label_passages_as_training_rows`,
  `POST /synthetic/relabel-answer-key {target: "passages", document_id?}`,
  Synthetic-tab button "Label document passages with <pin>"): the cleaned
  passages (`load_document_passages`: cleaned.jsonl rows without
  `row_index`, optionally one document) are labelled AS THEY ARE under the
  conventions (+ regex pass) and appended to `synthetic.jsonl`
  (`synth_source="passage_labels:doc=<id>"`, pending). The span generator
  rewrites passages into snippets and loses layout; this keeps letterheads /
  page numbers in training text. Passages already present (same text) are
  skipped; raises when none are new. Summary `split="passages"`.
- **Label whole documents** (`label_documents_as_training_rows`,
  `target: "documents"`, `max_chars` 4000; button "Label whole documents"):
  `load_document_texts` (raw/<name>.cleaned.txt per source doc, line breaks
  kept) → `pack_long_rows` (whole blank-line blocks up to max_chars; an
  oversized block cut by `chunk_text`) → rows `synth_source=
  "document_labels:doc=<id>"`, `unit: "document"`, pending, with ids.
  **All labelling goes through `label_texts_any`**: texts ≤
  `LABEL_PIECE_CHARS` (1000) as they are, longer ones via `_label_in_pieces`
  (`piece_offsets` at line / sentence ends, edge whitespace trimmed — the
  span validator strips, so an untrimmed piece came back one char early —
  spans shifted back; a text is labelled only if every piece was; a
  misaligned span drops the row). A whole document labelled at once got
  ~4.1 entities / 1,000 chars vs ~6.8 in pieces. `relabel_training_rows(
  source=)` / API `source` relabels one synth_source group. Every labeller
  warns when texts come back with no labels. `_label_and_append` is the
  shared append core (passages + documents).
- **OCR-noise augmentation** (token classification; `TrainingConfig.
  ocr_noise_augment_rate` 0–0.3 + `ocr_noise_augment_copies` 1–5; Power
  Tools "OCR-noise copies", shown for token_classification; form hook
  `ocrNoiseRate` / `ocrNoiseCopies`, sent only when > 0):
  `train_token_classification.augment_with_ocr_noise` adds copies with
  `ocr_noise` (length-preserving `TRAIN_CONFUSIONS` swaps, per-copy rate in
  [rate/4, rate], seeded) — spans keep offsets, `text` follows; val/test
  untouched; auto-epochs counts the copies; report `augmentation`
  {ocr_noise_rate, copies, rows_added}. `h g u Z 9 2 @ -` are deliberately
  NOT in TRAIN_CONFUSIONS (held-out noise for measuring). Project 31 run
  #65: 10% noise 0.69 → 0.79, 20% 0.51 → 0.75, clean 0.85 → 0.83.
- **Scanned-text mode** (P1h): `runtime_config["span_scan_tolerant"]`
  (`span_type_definitions_service.SCAN_TOLERANT_KEY`, `project_scan_tolerant`,
  `load_scan_tolerant`); `definitions_clause(types, defs, scan_tolerant=)`
  appends `SCAN_TOLERANT_CLAUSE` (a value broken by stray spaces / marks is one
  span, no marks or label word at its edges; a prefix that is part of the id —
  ACCT-…, EMP-…, PT-… — stays; the clause text is part of `conventions_version`) — passed by span synthesis, the
  relabeller (all three paths) and the gold-LLM prompt. Decoding:
  `train_token_classification.ocr_tidy_spans` (trim `OCR_EDGE_MARKS`; join
  same-type neighbours whose gap ≤ `OCR_MAX_GAP` 3 chars of whitespace /
  stray marks, never a comma / letter / digit) via `predict_spans(
  ocr_tolerant=)`; `run_heldout_evaluation` marks the pairs (`ocr_tolerant`)
  when the flag is on and `_run_token_classification_inference` reads it; the
  redact service reads the flag too. `PUT /synthetic/type-definitions` now
  takes `definitions` optional (None = unchanged, {} clears) +
  `scan_tolerant`; GET returns it. UI: checkbox in `SpanTypeDefinitionsEditor`
  (saves immediately). Project 31: decoding +0.012–0.014 F1 on the scan key,
  0 on clean; retraining under the clause didn't help (run #70 ≈ #69).
- **Conventions version** (P1k): `span_type_definitions_service.conventions_version(
  defs, scan_tolerant)` = 12-char hash (None with no conventions);
  `CONVENTIONS_VERSION_KEY` stamped on rows by `relabel_training_rows`,
  `_label_and_append` (+ `labelled_by`) and span synthesis (kept by
  `save_synthetic_span_batch` and the legacy auto-save). `conventions_status(rows,
  current)` → {span_rows, current_rows, stale_rows, untracked_rows, is_stale,
  labelled_by (`row_labeller`), relabel_pin (`relabel_pin`: cloud pin or
  `ollama`)}; the review queue (source grouping only) puts it on each group as
  `conventions` + top-level `conventions_version`. `relabel_training_rows`
  without entity_types now uses the project vocabulary (defined types ∪ every
  saved row's types). `POST /synthetic/review-queue/mark-conventions-current
  {source}` (`mark_conventions_current`) stamps a group without relabelling
  (`conventions_marked_current_at`). UI: `StaleConventionsStrip` in
  `SynthReviewQueue` ("Relabel under current conventions with <pin>" /
  "Mark as current") + "older conventions" badge on accepted groups.
  P1l: queue top-level `labeller_pins` (every model that labelled span rows,
  as pins) → the strip's "Relabel with" picker; `set_group_retired(source,
  retired)` / `POST /synthetic/review-queue/retire-group {source, retired}`
  (accepted/legacy rows → rejected, `reject_reason="retired"`, `retired_at`;
  restore only touches rows rejected AS retired) → "Retire group" on accepted
  cards + `RetiredGroupsList` with Restore. `relabel_training_rows(source=)`
  matches groups with the queue's `_resolve_source_label` (rows without a
  synth_source are `legacy:<source>`).
  `relabel_split_in_place(split)` / request `target: "split"` relabels a
  prepared split's rows in place (a `keep_test` split never follows relabelled
  source rows) + every `prepared/versions/*/<split>.jsonl` byte-identical to it;
  Synthetic-tab button "Relabel test examples with <pin>".
  `relabel_training_rows(only_stale=True)` / request `only_stale` sends only
  rows not stamped with the current conventions version (raises when none);
  the strip always sends it ("Relabel N rows under current conventions").
- **Span thresholds (P1t)** (`services/span_threshold_service.py`): `tune_from_scored`
  (pure; per type the `GRID` threshold maximising strict F1, kept at 0 unless
  it gains ≥ `MIN_GAIN` 0.005), `tune_run_thresholds(db, pid, eid, split="val")`
  (refuses test / benchmarks) writes the run's `model/thresholds.json`
  (`per_type` + a `tuned` report); `POST /projects/{id}/span-thresholds/tune`;
  "Tune thresholds" on the span Eval card. `thresholds_for_decoding(model)` (non-zero
  per-type only) + `recall_bias_for_decoding(model)` are applied by
  `_run_token_classification_inference` — so held-out eval and benchmark scores use
  the same decoding as /redact and the export (runtime `thresholds_applied` /
  `recall_bias_applied`). **Recall floors (P1u)**: `runtime_config["span_recall_floors"]`
  = `{type: 0–1, "_default": 0–1}` (`FLOORS_KEY`, `normalise_floors(raw, known_types)`,
  `floor_for`, `known_span_types` = data types ∪ definitions ∪ compliance types);
  `GET/PUT /projects/{id}/span-recall-floors` (422 out of range / unknown type; GET
  carries `suggestions` from `compliance_bundles_service.suggested_recall_floors` —
  each built-in bundle has a `recall_floor`, PCI/HIPAA 0.95, GDPR/CCPA 0.9 — never
  applied server-side). `tune_from_scored(floors=, before_pred=)`: a floored type takes
  the best-precision threshold whose recall ≥ floor (ties → higher recall, then higher
  threshold; 0 unless +`MIN_GAIN` precision), `floor_status` met / not_reachable /
  no_support; no floor → F1 behaviour unchanged. `suggest_recall_bias` re-decodes with
  `BIAS_GRID` (0.05–0.5) → smallest bias meeting every unreachable floor + per-type /
  micro precision cost; `tune_scored_split(gold, decode, types, floors=,
  apply_recall_bias=)` (pure over a `decode(bias)` callable) re-tunes under the bias
  only when asked. The split is scored ONCE: `train_token_classification.score_tokens`
  (model call → merged per-token logits) + `decode_token_scores` (model-free;
  `predict_spans` = the two). Tune request `{experiment_id, floors?, apply_recall_bias}`
  (floors None → project setting). thresholds.json adds `recall_floors`, `recall_bias`
  (tuning owns it — 0 unless applied), `tuned.report` (recall/precision before → after,
  floor, floor_status), `tuned.recall_bias_suggestion`; the model card
  (`recall_floor_lines`) adds a "Recall floors" table. UI `SpanThresholdTuner`
  (floors editor + "Use compliance suggestions" + save + tune + result table; api
  `api/spanThresholds.ts`) opens from the card's "Tune thresholds". Tests
  `tests/test_span_thresholds.py` (patch `_load_model`), `SpanThresholdTuner.test.tsx`.
- **Labelling cost**: `span_relabel_service.relabel_prompt_parts` → (stable, varying);
  a backend with `supports_prompt_cache` (CloudLlmBackend) gets `cache_prefix=stable`
  → `call_anthropic_chat(cached_system=)` (cached system block; usage
  `cache_write_tokens` / `cache_read_tokens` on `CloudLlmResponse`); other
  providers get it prepended. `CloudLlmBackend.usage` counts per instance (one per
  job); `usage_block(*llms)` + `llm_pricing.estimate_cost` (list prices by family,
  `LLM_PRICES_JSON` override, cache write 1.25× / read 0.1×, `usd_uncached`) →
  `RelabelSummary.usage` / `cost_usd` / `tokens` on every relabel path; bell line.
  Targeted: `span_type_definitions_service.HISTORY_KEY` snapshots each conventions
  version (`remember_conventions`, on every type-definitions PUT before + after, and
  by relabels); `changed_types(old, new)` (None = not type-local → all rows);
  `row_affected(row, changed, pattern_hit)`; `relabel_training_rows(only_affected=,
  dry_run=)` relabels only touched stale rows, carries the rest forward
  (`conventions_carried_forward_at`), reports `targeting`. Tiered:
  `review_disagreements` — rows where the bulk labels ≠ the latest span run's
  predictions (`latest_span_run_predictions`) go to `review_override`; request
  `review_backend` / `review_experiment_id`; strip "Review disagreements with".
  Tests `tests/test_labelling_cost.py`. Measured on project 31 (114 rows):
  cache −15% (output tokens dominate; Haiku's prefix is below its minimum
  cacheable length); tiered cost MORE (113/114 rows disagreed — any span
  difference counts); targeted relabel after a one-type change: 154 of 1,927
  rows (−92%) — the real saving.
  Tiered threshold: `row_agreement` (row span F1) < `review_min_agreement`
  (`DEFAULT_REVIEW_MIN_AGREEMENT` 0.8; request field; 1.0 = any difference) →
  review; `review.agreement_distribution`. Project 31: @0.8 104/114 rows ($0.53),
  @0.5 39 rows ($0.27, agreement 0.73) vs Sonnet-only $0.43 — Haiku too far from
  the conventions there. Batch mode verified live (16 rows, 3/3, ~3 min).
- **Batch relabels** (Anthropic Message Batches, half price): request `use_batch`
  (400 `BATCH_NEEDS_ANTHROPIC` unless `cloud:anthropic:*`) sets
  `CloudLlmBackend.use_batch`; `label_texts` (via `uses_message_batch`) sends every
  call as ONE batch through `complete_batch` (`cloud_llm_service.create_anthropic_batch`
  / `wait_for_anthropic_batch` (poll `ANTHROPIC_BATCH_POLL_SECONDS`, cancels the
  batch on Job cancel) / `fetch_anthropic_batch_results`; params from
  `anthropic_message_params`; a temperature-rejected batch is resubmitted once
  without it); `label_texts_any` puts short texts + long-text pieces in the same
  batch. Errored / expired requests go back through the sync path (unparseable
  results split, as before). `batch_usage` → `usage_block` entry `"(batch)"` priced
  `estimate_cost(batch=True)` (`llm_pricing.BATCH_DISCOUNT` 0.5); summary `batch` /
  `via_batch` → bell "via batch". Checkbox "Batch (half price, slower)" in the
  Synthetic span tools + StaleConventionsStrip. Tests `tests/test_relabel_batch.py`.
- **Compliance bundles** (`services/compliance_bundles_service.py`,
  `api/compliance.py`): a bundle = regime tag + entity types with one-line
  definitions, stored as DATA (`BUILTIN_BUNDLES`: pci_dss / gdpr / hipaa /
  ccpa). `apply_bundles(cfg, ids, remove_ids)` (pure) adds types to
  `runtime_config["compliance"] = {bundles, custom, type_regimes}`, fills
  `span_type_definitions` ONLY for undefined types; removal drops the tag but
  keeps definitions; `set_type_regimes` replaces the map (tags outside the
  built-ins = custom). `GET /projects/{id}/compliance/bundles`, `POST
  .../apply`, `PUT .../type-regimes`. `regime_scores(per_class, type_regimes)`
  = micro P/R/F1 per regime (weakest recall first; unseen types →
  `unscored_types`) → `span_block(result, type_regimes)["by_regime"]` in
  `span_eval_summary` and the `run_span_evaluation` Job result. UI:
  `ComplianceBundlePicker` (Synthetic span mode; fires
  `brewslm:span-definitions-changed` so `SpanTypeDefinitionsEditor` reloads)
  + `SpanRegimeTable` on the span `EvalSummaryCard`. Tests:
  `tests/test_compliance_bundles.py`, `ComplianceBundlePicker.test.tsx`.
- **LLM answer key** (`gold_llm_service`, `POST /gold/generate-via-llm`):
  count drift is a 200 with `warnings`, never a 400 (the call is already
  paid); only no usable row → 400 `LLM_RESPONSE_UNPARSEABLE`. Span rows:
  wrong offsets are re-located from the entity text (`_locate_entity`), a row
  is dropped only when an entity isn't in it, and the prompt carries
  `span_type_definitions` as vocabulary + conventions. Frontend
  `parseApiErrorDetail` reads the envelope's top-level `error_code` /
  `message` (it used to show every 400 as UPSTREAM_ERROR) — tests should mock
  the real envelope. (Project 31's three 400s on 2026-10-14 were
  LLM_RESPONSE_COUNT_MISMATCH.)
- **Answer-key review (P6 Trust)** (`services/gold_review_service.py`,
  `GET/PUT /projects/{id}/gold/review`): a person checks the key row by row.
  Verdicts live ON the gold row (`review` = {verdict accepted / corrected /
  rejected, reviewed_at, note, original}); `corrected` replaces the answer
  fields in place (span rows: `entities`, validated against the text, no
  overlaps; other shapes: answer / label / summary) and keeps the model's
  original so `verdict: null` restores it; `is_rejected` rows are dropped by
  `evaluation_service._load_eval_rows` (never scored) and don't count in the
  GOLD dataset's `record_count`. Queue order (`order_queue`): unreviewed
  first, then lowest `row_agreement` between the key and the latest span
  run's spans (`model_predictions` → `latest_span_run_predictions(as_served=True)`
  = `span_threshold_service.score_split_as_served`: the run's thresholds +
  recall bias, the eval's decoding — plain decoding showed spans the served
  model drops; cached per run in `gold/review_predictions.json`, stamped
  `decoding: "served"`, older caches rebuilt); rows are identified by `text_key` (sha1 of
  the text) — `_row_scores_for_pairing` now stores `inputs` (same hash) and
  per-row `span_tp/fp/fn`, so `split_scores` re-totals a stored span eval
  over checked vs unchecked rows (`stale` when a correction / rejection is
  newer than the eval). Surfaces: `answer_key.review` on the span eval
  summary (`EvalSummaryCard` line via `answerKeyReviewText`), the model
  card's "Human review: …" line (`_answer_key_review_block`), and
  **Reviews survive a relabel**: `relabel_split_as_answer_key` runs
  `gold_review_service.carry_reviews(old_rows, new_rows)` (pure, pairs by
  text) before writing — corrected → the reviewer's entities win and
  `review.original` becomes the new labels; accepted → carried only when the
  span set is unchanged, else kept as `previous_review` on the row; rejected
  → carried; `RelabelSummary.reviews` {carried, dropped, …} + a warning line;
  record_count skips rejected rows. Tests in `test_gold_review` +
  `test_span_relabel`.
  `AnswerKeyReviewPanel` (Answer Key tab, test set): passage with key spans
  (orange = model missed, dashed = model-only), Accept / Fix (inline span
  editor, select text to add, "Use the model's spans") / Reject / Clear.
  Tests: `tests/test_gold_review.py`, `AnswerKeyReviewPanel.test.tsx`.
- **Answer-key sizing (P6)** (`services/span_eval_sizing_service.py`, pure):
  per type, `spans_needed(f1, gap)` = ceil(8·p·(1−p)/gap²) reference spans for
  a `DECISION_GAP` 0.05 F1 difference between two runs on the same key to
  clear noise (paired rule, same as the documents flow's `target_test_rows`;
  p = F1 clamped to [0.5, 0.95]); `detectable_gap(f1, support)` =
  2·sqrt(2p(1−p)/n) is the gap the support it has can show; `size_types(
  per_class)` → {gap, types (largest shortfall first: support, needed,
  shortfall, detectable_gap, enough), short, enough, shortfall}; `sizing_line`.
  Attached by `span_block` as `sizing` (Eval card `SpanSizingTable`: one line +
  per-type table + link to the Synthetic tab), stated on the model card under
  each evaluation, and read by `weak_entity_types` → `under_sampled`,
  `sizing_gap`, `types_to_refine` (weak ∪ under-sampled, weak first; the
  Refine button focuses these; `suggested_rows` counts them). New rows only
  reach the key through a re-split WITHOUT keep_test + a re-key. Honest
  numbers: at F1 0.8 a type needs 512 spans; project 31's key (≤ 37 per type)
  can show ~0.10–0.95, never 0.05. Tests: `tests/test_span_eval_sizing.py`.
- **Answer key from a second model** (`services/span_relabel_service.py`,
  Job `span_relabel`, `POST /synthetic/relabel-answer-key` {split, backend,
  entity_types}): `relabel_split_as_answer_key` prompts the backend (synth
  name or cloud pin via `_resolve_cloud_backend_override` → `backend_override`)
  with `TEXTS_PER_CALL` 6 texts, parses `{rows:[{index, entities}]}`,
  validates through `_validate_span_rows` (an entity with NO offsets is now
  located from its text — `start,end=-1`), writes `gold/gold_test.jsonl`
  (`source: "answer_key:<backend>"`) + the GOLD_TEST Dataset (file_path,
  record_count, metadata source=span_relabel) that `_resolve_dataset_alias
  ("gold_test")` reads, and returns `agreement` = `span_prf(new, old)`.
  Synthetic-tab button "Write answer key with <pin>" (pin = `ollama` or
  `cloud:<provider>:<model>` from the panel's provider/model); bell outcome
  "answer key: N rows · M entities · agreement … F1". The async span task's
  shadow Job title names the entity types (refine runs read as such in the
  bell); the shadow Job AUTO-SAVES rows — never call /save-spans after it.
  Tests: `tests/test_span_relabel.py`.
