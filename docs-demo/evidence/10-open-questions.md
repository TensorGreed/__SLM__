# Open Questions

Discovery date: 2026-05-19.

These questions should be answered before creating real recording scripts,
slides, or narration that make product-specific claims.

## Official Sample Data

1. Why does `backend/data/demo_samples/pii-detector/manifest.json` describe 60
   snippets while `pii_records.csv` currently contains 61 data rows?
2. Why do sample manifest descriptions mention smaller gold sets while each
   current `gold.jsonl` contains 200 rows?
3. Are `_generate_bundle.py` and `kaggle_pii_to_brewslm.py` intended only as
   PII sample helper scripts, or should they appear in a creator/developer
   appendix?
4. Should the demo mention exact label/entity counts, or keep those as internal
   evidence only?

## Seeded Project Behavior

5. Does a seeded demo project starting at `PipelineStage.TRAINING` affect how
   earlier tabs are visually presented? Selector passes showed all pipeline
   tabs unlocked with current stage `training` and progress `60%`; narration
   still needs to explain that the seed preloads data/gold/prepared splits.
6. Does the frontend expose prepared `manifest.json`, split counts, adapter
   selection, and sample rows clearly enough for a browser recording? Raw
   samples and gold counts are visible; prepared split/adapter/schema evidence
   was verified through API, not as clearly visible UI text.
7. Are legacy gold set UI and gold workbench UI both expected to be shown, or
   should one be treated as the canonical demo path?
8. Does the locked seeded gold set prevent editing in ways that need narration?

## Pipeline Semantics

9. Which evaluation task handler is used for the `support-faq` sample with
   `task_profile=instruction_sft` and seeded `qa-pair` adapter behavior?
10. Does cleaning actually remove duplicate rows anywhere, or only compute
    hashes and expose duplicate/redundancy signals?
11. Should `slm-docs/docs/workflows/data-ingestion.md` be treated as stale where
    it describes cleaning-stage deduplication?
12. Is there a verified classification-specific synthetic data generation path
    for `sentiment-classifier`, or only generic Q&A/conversation/import paths?
13. Which normalization steps should be narrated as user-visible actions versus
    backend adapter behavior?
14. Does dataset import intentionally persist accepted rows to the synthetic
    dataset for all mappers, and should that be called "import" or "synthetic"
    in the demo narration?
15. What is the exact UI path for showing PII cleaning redaction separately from
    the PII detector model sample?

## Runtime Choices

> **Resolved 2026-05-19 — see `12-runtime-decisions-2026-05-19.md`
> for the full rationale and operational setup.**

16. ✅ Training runtime: **real Celery + external runtime**. No
    simulated training in recordings.
17. ✅ Smallest reliable training job: **support-faq, 2 epochs**
    (16 train rows × 2 ≈ 32 forward passes). PII + sentiment runs
    scaled up later.
18. ✅ Redis/Celery required: **yes**. Pre-flight commands in
    `12-runtime-decisions-2026-05-19.md`.
19. ✅ Canonical Python env: `backend/.venv` from
    `backend/requirements.txt`. No alternate env.
20. Still open — which tokenizer/model presets are guaranteed to
    load without gated access or a large download? Not blocking any
    current video; revisit when Video 09's training run actually
    happens.
21. ✅ Synthetic generation teacher: **local Ollama
    `qwen2.5:7b-instruct-q4_K_M` at `http://localhost:11434/v1`**.
    Real endpoint, not fallback.
22. ✅ LLM judge: **local Ollama `qwen2.5:7b-instruct-q4_K_M`**
    (same Ollama process; same model).
23. ✅ Compression path: **GGUF quantization via llama.cpp**.
24. ✅ Canonical export format: **GGUF**. Direct corollary of
    Q23 + Q25 — single artifact format, single trust boundary.
25. ✅ Local serve runtime: **Ollama**.

## Final Model Usage

26. Does a trained experiment artifact appear in playground model options
    without export or registry registration?
27. Does an exported model appear in playground options through registry or
    artifact discovery?
28. What is the most reliable proof that the final model works: playground,
    generated curl smoke test, local serve run status, registry readiness, or a
    combination?
29. Which registry promotion gates are required for staging or production in the
    demo environment?
30. Can deployment telemetry/drift be demonstrated with real data, or should it
    stay out of the first recording series?

## UI Recording Readiness

31. Should Playwright login through visible `/login`, or set authentication
    state after documenting the login flow? Visible login worked in selector
    passes and is evidence-backed; pre-auth storage remains an optimization.
32. Which routes need stable `data-testid` attributes before reliable
    recording? Demo tiles and pipeline tabs have usable text/title selectors,
    but stable `data-testid`s would reduce brittleness if approved.
33. Are any tabs lazy-loaded or hidden behind gate checks that need a seeded
    project status change? Selector passes showed all standard pipeline tabs
    unlocked for seeded official demos.
34. Which browser widths should be recorded for the main videos?
35. Should the first prototype recording use only the project list and one
    seeded project, leaving runtime-heavy steps for later?

## Documentation And Config Consistency

36. README worker command and any docs worker commands should be reconciled if
    they differ.
37. Are `slm-docs` pages current enough to cite, or should code be treated as
    the sole authority for recording claims?
38. Should environment variables for simulated/stub/demo fallback modes be put
    into a separate "recording profile" file?
39. Should demo recordings run against a disposable SQLite database to avoid
    reusing existing user data?
40. Should seeded demo projects be cleaned up between takes, or should the
    idempotent seeder behavior be part of the narration?

## Next Evidence Tasks

1. Decide whether to add stable `data-testid` values to demo tiles and pipeline
   tabs, or proceed with text/title selectors.
2. Run one safe API preflight for dataset prep and training configuration on a
   disposable official sample project.
3. Decide the runtime path for one tiny real or explicitly simulated training
   demo.
4. Verify whether tokenization can run with a small non-gated tokenizer on the
   recording machine.
5. Verify whether synthetic generation should use a real teacher endpoint or a
   clearly labeled demo fallback.
6. Only after those decisions, create real Playwright recording specs.

## Validation Notes (2026-05-19 repo audit)

The repo-audit agent spot-checked the following claims from this file
and the surrounding evidence pack. No additional questions were
added; existing items remain valid.

- Q1 (pii-detector manifest says 60, CSV has 61): still open; manifest
  prose at `backend/data/demo_samples/pii-detector/manifest.json` is
  unchanged.
- Q2 (manifests say smaller gold sets than the current 200 rows):
  confirmed stale for all three samples; see headline note at
  `support-faq/manifest.json` ("6 hand-labelled gold rows") and the
  others. Worth a small docs-only manifest update separately.
- Q5–Q8 (seeded-project visual semantics): partially answered by the
  selector-discovery pass — all pipeline tabs render unlocked, but
  prepared manifest/split count visibility in the UI is still an open
  recording-readiness question.
- Q31–Q35 (recording readiness): largely answered by the screenshot
  set under `docs-demo/screenshots/selector-pass-*.png`. Remaining
  hard question is Q32 (data-testid additions); see
  `11-selector-and-instrumentation-plan.md` for the explicit proposal.

## Answered By Selector Passes

- All three official demos can be seeded through real dashboard tiles.
- All three seeded projects render data, cleaning, gold set, synthetic, dataset
  prep, tokenization, training, evaluation, compression, export, and training
  config routes.
- All three seeded projects expose raw row expansion through
  `[data-testid^="expand-doc-"]`.
- Gold set UI shows 200 entries for all three official demos.
- Prepared split/adapter/schema/label details are reliably available through
  `/api/projects/{id}/prepared-manifest`.
