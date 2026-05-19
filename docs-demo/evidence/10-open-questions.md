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
   earlier tabs are visually presented?
6. Does the frontend expose prepared `manifest.json`, split counts, adapter
   selection, and sample rows clearly enough for a browser recording?
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

16. Which training runtime should recorded demos use: real external training,
    explicitly enabled simulated training, or a hybrid with clear labels?
17. What is the smallest reliable real training job for each official sample on
    the recording machine?
18. Is Redis/Celery required for the chosen training and compression demo path?
19. Which Python environment and dependency set should be treated as canonical
    for training, evaluation, compression, export, and serving?
20. Which tokenizer/model presets are guaranteed to load without gated access or
    a large download?
21. Which teacher model endpoint should be used for synthetic generation, or
    should `ALLOW_SYNTHETIC_DEMO_FALLBACK=true` be used and labeled simulated?
22. Which judge model endpoint should be used for LLM judge evaluation, or
    should demos avoid LLM judge until local setup is confirmed?
23. Which compression path is practical locally: GGUF, ONNX, benchmark only, or
    explicitly marked stub?
24. Which export format should be canonical for the first real final-model demo?
25. Which local serving runtime is installed or installable for recording:
    Ollama, llama.cpp, vLLM, TGI, ONNX Runtime, Docker, or none?

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
    state after documenting the login flow?
32. Which routes need stable `data-testid` attributes before reliable
    recording?
33. Are any tabs lazy-loaded or hidden behind gate checks that need a seeded
    project status change?
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

1. Start the app against a disposable local database and seed one official
   sample through the UI.
2. Capture screenshots of the project list, data tab, gold tab, dataset prep
   tab, and training config tab.
3. Verify exact selectors/routes without writing full recording scripts.
4. Run one safe API preflight for dataset prep and training configuration.
5. Decide the runtime path for one tiny real or explicitly simulated training
   demo.
6. Only after that, create real Playwright recording specs.
