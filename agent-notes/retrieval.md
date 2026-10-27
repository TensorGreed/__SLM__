# Retrieval / Auto-RAG / reroute-to-RAG

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Document retrieval (Wave 2c-2)** — `auto_rag_service` also indexes cleaned
  *document passages* (cleaned.jsonl rows without `row_index`; corpus key
  `documents`) under `auto_rag/documents/bm25_index.json`, built lazily by
  `ensure_document_index` and rebuilt when cleaned.jsonl is newer.
  `build_preamble_from_query(corpus="auto")` = Q&A index if the recipe has one,
  else passages with a cite-[n]/say-you-don't-know preamble; result carries
  `corpus`. `GET /auto-rag/documents`; playground toggle "Answer from your
  documents" (default on when available) + Sources chip with `source_doc ·
  passage n`. `rag_sandbox_service` (RAG-compare) now uses the passage index /
  extracted text (it read PDF bytes before).
- **Auto-RAG** — `auto_rag_service` builds BM25 indexes at training
  completion; `playground_chat` prepends top-K retrievals. Comparison
  panel + UI-triggered `auto_rag_ab --project` Job. The Q&A corpus is the
  project's **training rows** (`dataset_service.load_training_corpus_rows`:
  `prepared/train.jsonl`, else CLEANED + accepted SYNTHETIC) — never GOLD_* or
  val/test (it used to index the answer key, so retrieval could return the
  scored question with its answer). Indexes carry `corpus_source`
  (`QA_CORPUS_SOURCE`); an unstamped one is rebuilt (or dropped when there are
  no training rows) by `build_preamble_from_query`, the preview endpoint and
  `run_project_comparison`. `api/curriculum` preview ranks the same rows.
  The harness (`scripts/auto_rag_ab.py`) prompts with the **tokenizer's chat
  template** (`_build_inference_prompt`; Llama-3 headers only as the
  no-template fallback), decodes only the new tokens, and loads either
  base + LoRA adapter or a full fine-tuned `model/` dir. Generation is capped
  at 1.5× the longest training answer (`_generation_cap`, 32–128 tokens).
  `--project N --base-only` scores the **base model** with/without retrieval
  (the RAG-first question) → `auto_rag/comparison_base.json`; it never
  overwrites the fine-tuned `comparison.json`. `--decoding model` decodes
  greedily with the model's own `generation_config` instead of neutral settings
  → `comparison*_model_decoding.json` (the passage-reader check's second base
  bar; refuses `--sweep-retrieval`). `GET /auto-rag/comparison`
  returns the fine-tuned comparison at the top level (`summary` null if not
  run) plus `base` (or null); 404 only when neither exists. `POST
  /auto-rag/comparison/run?model=base` runs the base one (same Job kind, same
  one-in-flight guard). `AutoRagComparisonPanel` shows the two as side-by-side
  cards with run id / base model provenance + a "highest of the four" line.
  `--corpus documents` (CLI) / `?corpus=documents` (run endpoint) retrieves
  the project's document passages (`ensure_document_index`, cite preamble)
  instead of Q&A pairs → `comparison[_base]_documents.json`
  (`comparison_file_name`); the GET returns `base_documents` /
  `fine_tuned_documents`, the panel renders a third card ("Base model +
  document passages"). Both arms are judged by `answer_judge_service` when
  it applies (`_judge_comparison_rows`): `summary.judge` (label + per-arm
  score/counts), per-row `without_rag.judge` / `with_rag.judge`, and the
  GET's `judge_evidence` (`_lift_evidence(score_key="judge")`).
  **Passages vs fine-tune verdict** — `services/passage_rag_verdict_service.py`:
  `passages_vs_finetune(db, pid)` compares the judged base+documents
  comparison (val rows) with the latest trained run's judged lift (test
  rows; a seed leader = mean of its judged children; an unjudged latest run
  falls back to the newest judged one with `finetune_is_latest=False`).
  `passages_win` needs the same judge, retrieval evidence `better`, and a
  higher passages score. Read by: the GET (`passages_vs_finetune`), the
  Coach eval nudge `eval:passages-beat-finetune` (action kind
  `reroute_to_rag` → frontend `rerouteToRagAsync`), and
  `resolve_playground_corpus` (explicit `runtime_config.auto_rag_corpus` >
  passages-win → `documents` > pre-training gate ready/promising
  (`read_passages_gate`) → `documents` (`pretraining_gate`) > `auto`), used by `_apply_playground_auto_rag`
  (request `auto_rag_corpus` overrides; audit block `corpus_reason`).
  `clone_project_for_rag` stamps `auto_rag_corpus="documents"` on the
  sibling when the source resolves to passages.
  Harness `--split val|test` (`run_project_comparison(split=)`; documents
  corpus defaults to **test** so the verdict's `same_rows` is True; payload
  + GET carry `split`). `POST /comparison/run?experiment_id=N` (and the CLI's
  `--experiment-id`) scores that COMPLETED run instead of the latest — 400
  `EXPERIMENT_ID_WITH_BASE_MODEL` / `EXPERIMENT_NOT_COMPLETED`, 404 for another
  project's run; Job params + title + result carry it, the bell says "run #N ·
  … · K test|val rows". Same output file, so it replaces the cached comparison
  (GET then reports `stale` vs the latest run). The comparison Job start lives in
  `services/auto_rag_comparison_job_service.start_auto_rag_comparison_job`
  (raises `AutoRagComparisonInFlight`; the endpoint maps it to 409) so
  `rag_project_service.start_sibling_passages_check` can spawn it from the
  reroute clone (async + sync paths; result `passages_check`) for a sibling
  with `auto_rag_corpus=documents`. `eval_summary_service.passages_summary`
  (`kind="rag_passages"`, headline `judge_correct` without → with passages,
  judge-row evidence, failures = rows retrieval got wrong) is what
  `build_eval_summary` returns for a project with no trained run but a
  judged documents comparison; `EvalSummaryCard` labels it "base + your
  passages" and offers "Re-run passages check".
  **Retrieval quality** — `services/retrieval_reranker.py` (cross-encoder
  via plain transformers, `DEFAULT_RERANKER` MiniLM, `rerank(query, hits,
  top_k, model_name)` over `candidate_pool(k)` BM25 hits, best-effort →
  BM25 order on failure). `auto_rag_service.retrieval_settings(project)`
  reads `runtime_config["auto_rag_retrieval"] = {k, reranker}`;
  `retrieve_ranked` applies it; `build_preamble_from_query(k=None)` uses
  it (playground `auto_rag_k` None → project setting; audit block carries
  `k` + `reranker`). Harness `--k / --reranker / --sweep-retrieval`
  (`RETRIEVAL_SWEEP`: top-3, top-5, ± reranker; shared without-RAG arm;
  `pick_best_retrieval(arms, current_retrieval=)` — the incumbent is the
  project's `runtime_config.auto_rag_retrieval` (read from sqlite in
  `run_project_comparison`, inserted into the configs if missing) else the
  first arm; a candidate replaces it only when `paired_difference_evidence`
  on the per-row judge (or F1) scores says `better`, then highest mean,
  ties → earlier; arms without records fall back to highest mean. Each
  sweep arm carries `current` + `vs_current`; `sweep_decision` →
  `summary.retrieval_decision` {current, chosen, switched, reason}, also
  on the Job result, the flow gate and the rag_passages summary;
  `summary.retrieval` + `summary.retrieval_sweep`,
  `auto_rag/retrieval_sweep.json`). `PUT /auto-rag/retrieval` {k, reranker} pins
  (`runtime_config.auto_rag_retrieval.source="pinned"`; `apply_retrieval_choice`
  returns `kept_pinned` and never overwrites a pin), `DELETE` unpins; the
  comparison GET carries `retrieval_setting` {k, reranker, source, pinned};
  the panel's documents card renders the sweep table with Pin / Unpin
  (`onPin` / `onUnpin` props on `ModelComparisonCard`).
  Playground: the audit block carries `retrieval_source` (`request` when
  `auto_rag_k`/`auto_rag_reranker` were sent, else the setting's source);
  request `auto_rag_reranker` (`none`/`default`/model id) overrides the
  reranker for one call (`build_preamble_from_query(reranker_override=)`).
  `ChatPlaygroundPanel`: retrieval chip per reply (`retrievalChipText`),
  "Compare retrievals" toggle → `fetchRetrievalComparison` re-asks with the
  chosen alternative (`otherRetrieval`: the `compareAgainst` sweep config
  `"k|default"` or flip-reranker; sends `auto_rag_k` + `auto_rag_reranker`,
  unsaved) and `RetrievalComparisonBlock` shows A/B with "Pin this
  retrieval" → `comparisonDecision` POSTs `/auto-rag/retrieval/decisions`
  {question, candidates, pinned} then PUT /auto-rag/retrieval.
  **Retrieval setting = four fields** (`auto_rag_service.normalize_retrieval`
  → `{k, reranker, pool, chunk_size}`; `same_retrieval`, `retrieval_label`,
  `RETRIEVAL_FIELDS`): `pool` = the reranker's BM25 candidate count
  (`candidate_pool(k, pool)`, clamped [k, `MAX_POOL` 100]; dropped without a
  reranker), `chunk_size` = a re-chunked passage variant — `ensure_document_index(pid,
  chunk_size)` cuts `raw/*.cleaned.txt` with `cleaning_service.chunk_text`
  (overlap size/10) into `auto_rag/documents/chunk-<n>/` (`load_rechunked_passages`;
  stale when a cleaned.txt is newer; the cleaning's own index at `.../documents`
  is never touched); `current_chunk_size_estimate` = longest cleaned passage.
  `build_preamble_from_query(retrieval_override={reranker|pool|chunk_size})`
  (was `reranker_override`); chat request `auto_rag_pool` / `auto_rag_chunk_size`;
  pin PUT accepts `pool` / `chunk_size`; `retrieval_key` appends `|pool|chunk`
  only when set (old two-field keys still match). **Staged sweep**
  (`sweep_stages(current, documents=, own_chunk_size=)` in the harness):
  `k_reranker` (= `RETRIEVAL_SWEEP`: top-3/5 × {none, `default`, `strong` =
  `retrieval_reranker.STRONG_RERANKER` BAAI/bge-reranker-v2-m3, via
  `RERANKER_ALIASES`} + incumbent first) → `pool`
  (`RETRIEVAL_SWEEP_POOLS` 20/50, only when the running best reranks) →
  `chunk_size` (documents; `RETRIEVAL_SWEEP_CHUNK_SIZES` 600/900/1200 minus the
  one within `_OWN_CHUNK_TOLERANCE` 15% of the cleaning's own). Each stage
  re-picks with `pick_best_retrieval` over ALL arms vs the project incumbent;
  arms carry `stage`. CLI `--pool` / `--chunk-size`. The Job drainer shows the
  arm being scored ("scoring row 12/46 · retrieval pool 1/2 (stage 2/3): …")
  instead of "loading model". Frontend: `components/evaluation/retrievalLabel.ts`
  (`RetrievalSetting`, `retrievalLabel`, `retrievalKey`, `retrievalPinBody`) is
  the one label/key/pin-body helper — don't re-derive labels in components.
  **Pin decisions** — `services/retrieval_decisions_service.py`: append-only
  `auto_rag/pin_decisions.jsonl`, `preference_tally` ({offered, chosen,
  share} per `retrieval_key` "k|reranker"), `preferred_over` (≥
  `MIN_CHOICES_FOR_PREFERENCE` 2 chosen, higher share). The harness reads
  `project_preference` and `pick_best_retrieval(user_preference=)` uses it
  ONLY to break a within-noise tie against the incumbent (`user_preferred`,
  arm `user_choices`; decision reason names the head-to-head count). The
  comparison GET carries `user_preference` {decisions, tally}; the sweep
  table has a "users chose" column. `PinDecisionLog` under the sweep table
  lazy-loads `GET /auto-rag/retrieval/decisions` (newest first, winner
  starred; toggle says "that broke this tie" when `retrieval_decision`
  switched on users' choices). Job
  `sweep_retrieval=True` → `apply_retrieval_choice` writes the winner to
  the project; the sibling auto-check sweeps. Summary card line
  "Retrieval served: …". Tests patch `retrieval_reranker.score_pairs`.
  A rag_first project's playground with `provider="experiment"` serves the
  base model in-process (`_resolve_playground_run` returns the base model
  id; `local_chat_service` loads a plain checkpoint) — the frontend
  defaults a rag_first project to that provider. Docs demo video 20
  (`docs-demo/videos/20-legal-rag-vs-sft/`, gitignored like all videos)
  reuses video 13's builder via importlib + a retrying per-sentence TTS
  (`synthesize_checked`); page `slm-docs/docs/demos/legal-documents-rag.md`.
  The GET also returns `latest_experiment_id` + `stale` (fine-tuned cache is
  for an older run → warning on the card). The panel reads its project's
  `auto_rag_comparison` Jobs from `useJobsStore((s) => s.jobs)`: in flight →
  progress on that card + all Run buttons disabled; a newly succeeded Job →
  silent re-fetch (no loading flash). Tests mock the store as a callable with
  `getState`.
  Each comparison carries `evidence` (`services/paired_comparison_stats.py`,
  pure: rows better / worse / same + paired-t 95% interval on the mean per-row
  change → `better` / `worse` / `within_noise` / `too_few_rows`), computed at
  read time from the cached rows. The card shows it under the numbers; the
  lift is only coloured when the verdict is better/worse, and a fine-tuned
  verdict adds a one-training-run caveat (the interval covers row sampling,
  not seed-to-seed variance). Reuse the helper for any paired before/after
  metric — don't show a bare lift on a few rows.
  The **lift check** uses it too: `run_heldout_evaluation` stores
  `details["row_scores"]` (`keys` = hash of prompt+reference, `correct`, and
  `f1` when the handler records per-row F1) for every row;
  `eval_summary_service` pairs base vs fine-tuned rows by key and returns
  `evidence` (headline `f1` / `exact_match` / `accuracy` only);
  `EvalSummaryCard` softens its title when the lift is within noise. A cached
  baseline result without `row_scores` is not reused. Shared frontend wording
  lives in `components/evaluation/liftEvidence.ts`. The same verdict gates two
  more surfaces: the `post_training_lift_eval` Job result carries a compact
  `evidence` (bell line says "within noise" / "too few rows to tell" + row
  counts instead of better/worse), and the Coach export stage emits
  `export:within-noise` (warning) instead of `export:worse-than-base` /
  `export:newer-better-run` when the evidence is `within_noise` or
  `too_few_rows`. The summary's own `verdict` field is still the bare delta.
  Evidence is attached per lift row in `sft_lift_summary_service`
  (`_attach_row_evidence`; `metric_lifts[i].evidence`) — `eval_summary`, the
  lift Job and `SftLiftPanel` all read it from there. The
  `auto_rag_comparison` Job result carries a compact `evidence` too, and the
  bell qualifies its lift with `shortEvidenceText`.
  **LLM-judge correctness (long answers)** — `services/answer_judge_service.py`:
  for generative profiles (`JUDGE_PROFILES`: qa / rag_qa / instruction_sft /
  summarization) with median reference ≥ `MIN_REFERENCE_WORDS` (8),
  `run_heldout_evaluation` (eval_type exact_match / f1) also judges every row
  correct / partial / wrong (1 / 0.5 / 0) via `_safe_judge_answers` →
  `metrics["judge_correct"]` + `metrics["judge"]` snapshot (judge label,
  counts, calls / cached / tokens; `{"skipped": "not_long_answer_task" |
  "no_judge_available"}` otherwise) and `row_scores["judge_correct"]` (None
  for unjudged rows — `_paired_row_scores` drops those rows, not the
  comparison). Judged rows decide `_prediction_failed` (a correct paraphrase
  is not a failure). `judge_correct` heads `_PREFERRED_HEADLINE_METRICS`.
  Resolution `resolve_answer_judge`: `EVAL_JUDGE_BACKEND` env (`none` /
  `ollama[:model]` / `anthropic|openai|deepseek[:model]`) → project
  `runtime_config["eval_judge"]` → first available local synth backend →
  probe-judge cloud config; never cloud over a reachable local model.
  The project pin needs the project: a caller with a session but no project
  gets it loaded (`db.get`); the auto-RAG harness (no async session) passes
  `SimpleNamespace(runtime_config=…)` from its SQLite project row
  (`_judge_comparison_rows(..., runtime_config)`). Before this, both paths
  skipped a pinned judge for the automatic pick (gemma4 judged a sweep pinned
  to gpt-oss, 2026-10-27).
  File cache `data/projects/<id>/eval_judge_cache.json` keyed by judge label +
  question + reference + prediction. The lift runner resolves ONE judge for
  both evals (`_resolve_lift_judge`, passed as `answer_judge=`) and
  `_reusable_baseline_result(judge_label=)` re-runs a cached base result
  judged by another judge (`_baseline_judge_matches`). `POST
  /evaluation/summary/lift-check` (`start_post_training_lift_job(manual=True)`
  — bypasses the auto opt-outs) = "Evaluate this run" / "Re-run lift check" on
  `EvalSummaryCard`; summary payload `judge` block + per-failure
  `row_judge_verdict` / `row_judge_reason`. Tests inject `answer_judge=` /
  patch `_resolve_lift_judge` — a dev box with Ollama would otherwise judge
  for real (`tests/test_answer_judge.py`).
  **Retrieval judge (retrieval vs reading)** — `answer_judge_service`:
  `build_retrieval_judge_prompt` / `parse_retrieval_verdict` (`found` /
  `missed` + 1-based `passage`), `ResolvedJudge.retrieval_judge` (same
  model, second prompt; None on test fakes), `judge_retrieval(rows, judge,
  label=, cache=)` annotates `row_retrieval_verdict/passage/reason` and
  returns found / missed / `found_rate` / `wrong_missed` / `wrong_misread` /
  `correct_missed`; cache key `"<label>:retrieval"` + passages joined
  (`AnswerJudgeCache.get_retrieval`, `set(passage=)`); no call for a row with
  no passages. Harness records carry `retrieved_passages` (documents corpus,
  not written to rows); `_judge_comparison_rows` runs it on the with-RAG arm
  → row `with_rag.retrieval_judge`, `summary.judge.with_rag.retrieval`, arm
  `found_rate` (sweep table "passage found" column), summary card
  `judge.retrieval` + failure `row_retrieval_*`. Frontend helper
  `components/evaluation/retrievalSplit.ts` (`retrievalSplitText`,
  `retrievalRowText`). **A non-sweep comparison scores the SERVED retrieval**
  (`run_project_comparison(rag_k=None, …)` → `current_retrieval`; it used to
  score plain top-3); CLI `--k` default None.
  **Multi-seed lift check**: a seed-group *leader* (`seed_group_id` set,
  `seed_value` None) is lift-checked as a group — `_run_seed_group_lift_eval`
  scores every COMPLETED child against one cached baseline and returns
  `seeds` + `seed_evidence` (`paired_comparison_stats.seed_spread_evidence`:
  paired-t across seeds, `kind="seeds"`, verdicts better / worse /
  within_noise / too_few_seeds, `all_better`); it used to be evaluated as a
  single run via the borrowed first-child output_dir (or skipped as
  `no_model_weights`). `build_eval_summary` rolls a leader up the same way
  (`_seed_group_summary`: mean ± std headline, `seeds`, `seed_evidence`,
  failures + row evidence from the median seed →
  `representative_experiment_id`); `_latest_trained_experiment_id` skips
  seed children. Seed evidence outranks row evidence in the Coach export
  stage and the Eval card title. `POST /evaluation/summary/check-seeds`
  clones a run's config with `num_seeds` and starts the group ("Check across
  3 seeds" on `EvalSummaryCard`).
  **Ship the best seed** (`services/seed_selection_service.py`, `api/seed_groups.py`):
  `pick_seed` (pure) picks a group's child by a **validation** metric from each
  child's `training_report.json` (+ its `model/thresholds.json`) only
  (`validation_values(report, loss, thresholds)`: token_classification →
  `span_f1_tuned` = `thresholds.json` `tuned.micro_f1_after` when the tune ran
  on val (the served decoding; the auto metric is the first one EVERY child
  has, so an untuned seed drops the group to `span_f1`), then
  `task_metrics.span_set.f1`, classification → `final_eval_macro_f1` else
  `final_eval_accuracy`, else lowest `final_eval_loss`; ties → lower run id; a
  child without the metric is listed, never picked) — never test / answer key /
  benchmarks. `GET /projects/{id}/seed-groups/{leader}/pick` (`?metric=` override;
  `recorded` = last selection), `POST .../ship` {export_format=huggingface, metric,
  run_smoke_tests, evaluate_if_missing} → `evaluate_before_ship` (a span child
  with no EvalResult is lift-checked inline first — the card reads its
  evidence from EvalResults and a seed child has none until then; False →
  409 `SeedNotEvaluated` with the lift-check hint; response
  `evaluated_before_ship`) → `record_selection` on the leader's
  `config["seed_selection"]` + `create_export`/`run_export` of the child. `run_export` reads
  `selection_for_experiment` → manifest `seed_selection`, RELEASE_NOTES line, span
  card "Seed selection" line (`build_span_model_card(seed_selection=)`). Summary
  flag `seed_group_leader` → `SeedShipTable` on `EvalSummaryCard`. Tests:
  `tests/test_seed_selection.py`, `SeedShipTable.test.tsx`.
- **Reroute-to-RAG** — `rag_project_service.clone_project_for_rag()`
  creates a sibling project with `runtime_config.rag_first=True`. The
  playground uses base model + retrieval (no LoRA).
