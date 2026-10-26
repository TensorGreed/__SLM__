# Export + Hugging Face Hub publish

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Export (Wave 1b)** — GGUF/ONNX/TensorRT export packages only
  `compressed/exp-<run>/` (compression_service files outputs there when the
  source path is under `experiments/<run>/` OR `compressed/exp-<run>/` — the
  merge-lora → quantize chain; until 2026-10-18 quantizing the merged model
  wrote the GGUF to the project-wide `compressed/` root and every GGUF export
  of a merged run failed with "No Q4_K_M GGUF artifact"); no raw-weights fallback under a
  compressed label. HF/Docker export of a LoRA run merges the adapter
  (`adapter_merge_service.merge_adapter`: fp32 merge, saved in the base
  checkpoint's dtype read from `AutoConfig` — `from_pretrained(dtype=fp32)`
  overwrites `config.dtype`). `scripts/quantize.py` auto-detects an
  adapter-only `--model`.
- **Shipped decoding = evaluated decoding (2026-10-26)** — HF/Docker exports
  rewrite `model/generation_config.json` with
  `decoding_settings.neutral_generation_config`: sets `do_sample` false,
  `num_beams` 1, `repetition_penalty` 1.0, `no_repeat_ngram_size` 0,
  `temperature` 1.0; removes top-k / top-p / min-p / top-h / typical / epsilon
  / eta (writing `top_k: 0` under greedy fails transformers' strict
  `save_pretrained` validation); keeps every other key (token ids, forced /
  suppressed tokens, Whisper language maps, length limits). What it changed is
  in `manifest.decoding` and a `Decoding:` line in RELEASE_NOTES.md; the source
  run's file is untouched. Compressed formats (GGUF/ONNX/TensorRT) are not
  rewritten. The reader card snippet passes `repetition_penalty=1.0`.
- **Passage-reader model card** (`services/reader_model_package_service.py`; span taggers keep
  `span_model_package_service`). HF/Docker export of a run with `config.passage_prompts` writes
  `model/README.md` from evidence the RUN owns: `auto_rag/passage_reader/<run>.json` (the four arms,
  the paired evidence, the retrieval judge's found / missed split), `passage_prompts/report.json`
  (trained retrieval, targets, counts), the run's `adapter_config.json` + `training_report.json`, and
  question-generator / source-document tallies from `passage_prompts/train.jsonl` only while its
  sha256 matches `reproducibility.train_file_sha256`. Never the project-wide
  `comparison*_documents.json` (overwritten by any later comparison, on any split). The claim
  follows `evidence.verdict` (CI), not the raw delta; no lift check → the card says so; a trained ≠
  evaluated retrieval and `comparable: false` are called out. The prompt example renders
  `auto_rag_service._DOCUMENT_PREAMBLE_TEMPLATE`, so it can't drift from serving. Manifest key
  `reader_package` {model_card, evaluated, split, judge, verdict} + a RELEASE_NOTES line. Teacher
  targets: the report's `filter` {judge, keep, judged} + `revise` {critic, judge, rounds, kept} and the
  train counts `revised_kept` / `filtered_revised` / `filtered_revised_abstained` give the Filter and
  Critique-and-revise lines, plus a Limitations line when the filter judge IS the eval judge. A run
  trained before the report recorded them (e.g. #162) falls back to `teacher_provenance` over the cache,
  only while its train file's sha256 matches the report AND every trained answer matches its cache
  record (no dropped-answer breakdown then — that needs the training-time counts). **Across seeds**:
  export passes `seed_siblings(db, project, run)` — COMPLETED passage runs in the project with the
  same base model, the same `train_file_sha256` + `val_file_sha256` in their training reports and the
  same config bar `seed` / `training_runtime_id` / `seed_selection` / `_`-keys (a formal seed group
  or hand clones like p38's #160–162). Table: seed, run, test score, paired diff vs bar (CI),
  validation loss, this run bold; only siblings whose lift check matches this run's split, judge,
  rows, retrieval and bar score (the rest are named with the reason); the across-seed line is
  `paired_comparison_stats.seed_spread_evidence` on those scores (p38: +0.091, CI +0.028 to
  +0.155). The table never says how the shipped seed was picked — only `seed_selection` does. The card's caution line is
  generic; a Hub card still needs the user's education-only warning + blog links added by hand.
  UI: each row of ExportPanel's Hub-publish card shows `modelCardLine(manifest)`
  (`components/export/exportModelCard.ts`): span card + its evaluations, reader card + split · judge ·
  verdict, or a warning that a card-less export publishes a blank Hub page
  (`data-testid="export-publish-card-line-<id>"`).
- **Publish to the Hugging Face Hub** (`services/hub_publish_service.py`, `POST
  /projects/{id}/export/{export_id}/publish-hub` {repo_id, private, token?, license,
  commit_message}): a COMPLETED huggingface-format export's `model/` becomes the
  repo root (`upload_folder`), `manifest.json` + `RELEASE_NOTES.md` go under
  `brewslm/`; `card_with_hub_metadata` (pure) adds `license` / `language` to the
  card's front matter only when absent; token = request > `HF_TOKEN` > the
  `huggingface-cli login` cache, never stored; the record lands on the export
  manifest as `hub_publish` (disk + DB; the list route returns it). `api=` is
  injectable (tests use a fake HfApi, never the network — `tests/test_hub_publish.py`).
  **No local paths travel** (`services/portable_paths.py`, pure: `portable_text` / `portable_paths`
  — run-dir paths → relative, `.` for the dir itself, home → `~`): `run_export` stores the manifest
  through it with `run_dir` = the run folder's NAME (relative to `export.output_path`); the deploy
  re-validation and `last_deploy_execution` too; the publish uploads `brewslm/*` as scrubbed bytes
  (older exports hold absolute paths — the 2026-10-17 tagger manifest named the home dir, fixed on
  the Hub 2026-10-20). `_resolve_export_run_dir_from_export` (serve_service delegates) reads both
  spellings — never `Path(manifest["run_dir"])` directly. API responses (serve / deploy plans) keep
  absolute paths: they are for the local user.
  UI: the expanded Export-History row of a HF export (`ExportPanel`,
  `data-testid="export-hub-publish-<id>"`). Project 31's tagger: repo
  `TensorGreed/brewslm-sensitive-data-tagger` (see the workflow doc for which export).
- **Data provenance in the manifest** (phase G2): every format's `manifest.json` gains `provenance`
  = `{counts_from, sources[{id, kind, title, locator, ref, licence, licence_verified, usage,
  retrieved_at, train, val, test, excluded}], licences{<licence or "unknown">: train+val rows},
  warnings[{source_id,title,flag,licence,message}]}` (`provenance_service.export_provenance`).
  `counts_from` = `training_manifest` (the run's captured `data_provenance`, what it trained on) else
  `current_prepared_manifest` (experiments record no prepared version) else null. Warnings
  (`unknown` / `non_commercial` licences of sources that fed the run) never block an export.
  `RELEASE_NOTES.md` gets a "Data sources & licences" section (`release_notes_section`).
