# Open Questions

1. Why does `pii-detector/manifest.json` describe 60 snippets while `pii_records.csv` currently counts 61 data rows?
2. Why do the manifest descriptions mention smaller gold sets for all three samples while each current `gold.jsonl` file counts 200 rows?
3. Does a seeded demo project being set to `PipelineStage.TRAINING` affect how earlier tabs should be narrated?
4. Which runtime mode should recorded demos use: real external training, simulated training, or a deliberately marked hybrid?
5. Which env vars are required for synthetic generation in the intended demo environment?
6. Which compression/export path is practical on the recording machine without GPU or external tools?
7. Does the frontend expose prepared split files and prepared manifest clearly enough for a visual demo?
8. Should Playwright login through the UI or inject a JWT/local storage token for setup speed?
9. Which routes need `data-testid` attributes after selector discovery?
10. Which final model usage surface should be the canonical ending: playground, local serve, registry deploy, or API smoke request?
11. Are the docs in `slm-docs` fully current with the code defaults for auth and database paths?
