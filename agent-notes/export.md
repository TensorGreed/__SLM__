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
- **Export converts the run itself (2026-11-19, P5 gap 10)** — a GGUF / ONNX export with no matching file in
  `compressed/exp-<run>/` calls `_auto_convert_for_export` -> `compression_service.convert_run_for_export`: the
  `QUANTIZE_EXTERNAL_CMD` command run in-process (no Celery, like the HF merge), same output folder +
  `quantize_report.json`, model path `<output_dir>/model` (quantize.py merges an adapter-only run). Labels map
  explicitly in `quantization_plan` (GGUF `4-bit` / `q4` / `int4` / `Q4_K_M` -> 4 = `Q4_K_M`, per
  `GGUF_TYPE_BY_BITS`; none -> 16 = F16; ONNX none / `8-bit` / `int8` -> int8). Any other label (`Q4_0`, `Q4_K_S`,
  `7-bit`) raises `QuantizationLabelError` and the export fails with the supported list -- never converted to a
  nearby type. Only when `can_convert_inline()` (external backend + command); stub / TensorRT keep the "Open
  Compression" error, a failed conversion adds "Converting it here failed: ..." to it. Atomic + locked: the
  command writes into `compressed/exp-<run>/.convert-<id>/`, files move into place with `os.replace` (sidecars
  first, the model file last) only on exit 0, and the temp folder is removed on failure / timeout / cancel;
  `_collect_compressed_files` skips `.convert-*` folders, `*.partial` / `*.tmp`, the lock file and `*.ok`
  markers, and any `quantized_<bits>bit.<fmt>` without `is_complete_conversion` (its `<file>.ok` marker records
  the file's size; legacy files: `quantize_report.json` exit 0 for that path, written after it). The queued
  Compression -> Quantize job is atomic too: `quantize_model` renders the command into a temp folder and passes
  `finalize` to the Celery `run_quantization_job`, which runs it under `conversion_file_lock` (flock on
  `<folder>/.convert.lock`) and publishes with `finish_conversion` (os.replace, then the marker). Exports take the
  same lock (`run_conversion_lock`: async, non-blocking flock tries, nothing kept in memory), so the API's inline
  conversion and the worker never race; the second export re-checks inside the lock and reuses only the file at
  its own bit width (`quantized_<bits>bit.<fmt>`, F16 included; `auto_conversion.reused`). `manifest.auto_conversion` {bits, label, output_format, duration_seconds, reused} +
  a `Converted:` release-notes line; ExportPanel's status line says so. `manifest.quantization_label` /
  the `Quantization:` line name the packaged file's type (`packaged_quantization_label`: `Q4_K_M`, `int8
  (dynamic)` for ONNX even with no label). quantize.py's LoRA merge loads with an explicit dtype (bf16 on a
  bf16-capable GPU, else fp32).
- **Shipped decoding = evaluated decoding (2026-10-26)** — HF/Docker exports
  rewrite `model/generation_config.json` with
  `decoding_settings.neutral_generation_config`: sets `do_sample` false,
  `num_beams` 1, `repetition_penalty` 1.0, `no_repeat_ngram_size` 0,
  `temperature` 1.0; removes top-k / top-p / min-p / top-h / typical / epsilon
  / eta (writing `top_k: 0` under greedy fails transformers' strict
  `save_pretrained` validation); keeps every other key (token ids, forced /
  suppressed tokens, Whisper language maps, length limits). What it changed is
  in `manifest.decoding` and a `Decoding:` line in RELEASE_NOTES.md; the source
  run's file is untouched. ONNX/TensorRT are not rewritten. GGUF (2026-11-19, P5 gap 11):
  `write_gguf_decoding` writes `Modelfile` (Ollama: `FROM "./model/<gguf>"` (JSON-quoted) + `PARAMETER` lines)
  next to `model/`. The params come from `gguf_decoding_params`: the run's newest built-in eval's recorded
  decoding (`metrics.inference.decoding`; do_sample false -> temperature 0, `repetition_penalty` ->
  `repeat_penalty`, top_k / top_p / min_p when sampling, `details.inference.max_new_tokens` -> `num_predict`)
  over `decoding_settings.GGUF_EXPORT_PARAMS` (temperature 0 + `LLAMA_CPP_NEUTRAL`) for anything unrecorded.
  `manifest.decoding` = {modelfile, written, source: "eval #<id>" | "default", replaced: {}}; the GGUF `serve.py`
  uses the same values (path via `json.dumps`, `max_tokens` default = `num_predict` else 256). The packaged
  GGUF is `pick_gguf` (the `quantized_<bits>bit.gguf` the export's label asks for, else the newest). The
  `exporter.gguf` deployment report's launch example names that file (`_packaged_launch_example`, Modelfile
  first). Predefined-split manifests now carry `provenance` (rows per imported file), so export provenance
  lists real train / val / test counts.
  The reader card snippet passes `repetition_penalty=1.0`.
  Chat template (2026-11-19, P5b gap 6): `gguf_chat_template.modelfile_template(gguf_template_dirs(exp),
  use_tokenizer_chat_template=)` reads `chat_template.jinja` / `tokenizer_config.json` `chat_template` from the run's
  `output_dir/model` then `output_dir`; a TEMPLATE only when `training_chat_wrap(exp)` (the run's
  `training_report.json` `runtime_environment.chat_template_rewrap`) is "tokenizer_chat_template" (raw_prompt /
  preset:<name> / unrecorded -> note); a `"""` in the text -> note (a Modelfile value cannot carry it); converts only ChatML / Llama 3 (no "Cutting Knowledge" preamble) /
  Gemma (no system folding) after a sandboxed Jinja render check (a request's own system message must render as a
  system turn); the ChatML default system = the rendered system turn of a chat without one (`SYSTEM`). The Go
  templates render `.System` once and skip system turns in `.Messages`. Otherwise `{written: false, reason, note}`
  -> a `# No TEMPLATE written: ...` Modelfile comment + "Chat template:" release-notes line. `write_gguf_decoding(...,
  template=, num_ctx=)`; `num_ctx` = `gguf_num_ctx(max_seq_length, num_predict)`: max_seq_length + num_predict (else 256), rounded up to
  256, at least 2048; only when `config.max_seq_length` is set; `manifest.decoding.template`. serve.py
  params unchanged. Tests: `test_export_scoping_and_merge.py` (`GgufChatTemplateTests` + the Modelfile test).
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
- **Serving gateway + MCP server (G11)**: `/v1` (`api/gateway.py` `v1_router`, mounted OUTSIDE the `/api` auth
  deps; `GatewayError` handler gives the OpenAI envelope `{error:{message,type,code}}`). Auth is always on
  (`gateway_api_keys`: sha256 hash, `bsk_` + 12-char prefix, `project_ids`, per-minute limit via an in-memory
  sliding window; Bearer or `X-API-Key`, never `?token=`). Keys: `/api/gateway/keys` (EDITOR on every listed
  project when auth is on). Model ids (`services/gateway_service.py`): computed slug (no column; oldest project
  keeps it, a later collision gets `-<id>`), `p<id>`, `<slug>` = `runtime_config.gateway.default_run_id` else
  latest servable run (`servable_run_service`, shared with the playground), `<slug>:run-N`, `<slug>:base`.
  Runtime: `runtime_config["gateway"]` `{upstream: in_process|openai_compatible, base_url, upstream_kind:
  vllm|ollama|openai, upstream_model_map, default_run_id, tools}`; in-process uses `local_chat_service.
  generate_reply` (temperature 0), upstream names default to the canonical `<slug>:run-N` / `<slug>:base`.
  Retrieval = ONE `auto_rag_service.retrieve_document_passages` call (to_thread; `AutoRagUnavailable` -> `applied:false, reason:no_index`) + `document_preamble` as ONE system message
  (client system appended), default ON for passage-trained runs, never a 500 without an index; `citations[n]`
  match the `[n]` numbering, `source_url` joined from `RawDocument`. Guided JSON from `runtime_config.
  guided_json.schema`: `response_format` json_schema (strict) for EVERY upstream kind (Ollama `/v1` ignores `format`, new vLLM dropped
  `guided_json`) and the reply is validated -> `{mode:enforced, via:response_format, valid, errors}`; client's own
  wins (`mode:client`); in-process `validated_only`; 502 never echoes the upstream body. `upstream_kind` is a hint only.
  Project delete -> `gateway_service.purge_project` (ids removed from keys, emptied keys revoked, logs deleted);
  keys list/revoke: creator, admin or owner of every scoped project; Bearer + differing `X-API-Key` = 401. `/v1`
  errors (404/405/422/500) use the OpenAI envelope (`main.py` handlers branch on `/v1`). MCP HTTP: Origin/Host/
  Content-Type checks, `--allow-remote` for non-loopback. Telemetry: `gateway_request_logs` via `served_model_telemetry_service.
  record_gateway_request` (timing/status/tokens only, NEVER prompt/reply text). In-platform callers use `run_internal` / `internal_key` (key id 0, never rate-limited, `execute_chat(record_telemetry=False)` so no log row; pipeline eval). Named tools
  (`gateway_tools_service`): safe `{field}` substitution, `@recipe` = `build_code_findings_prompt` (labels from
  the prepared manifest, else the taxonomy), name clash -> `<name>__<slug>` (lookup by exact exposed name, never suffix parsing; reserved: list_models/run_model/retrieve). Multi-LoRA:
  `serve_service.build_multi_lora_serve_plan` (shared base, rank rounding) + `/serve/multi-lora-*`.
  `app/mcp_server.py` is stdlib + httpx only and must import nothing from `app.models/database/services`
  (asserted in `tests/test_mcp_server.py`). Docs: `slm-docs/docs/deployment/gateway-and-mcp.md`.
