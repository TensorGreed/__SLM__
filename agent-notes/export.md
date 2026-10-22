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
