**Great. Here is a concrete Phase 7/8 sprint backlog (2-week sprints).**

**Phase 7: Reliability + Newbie Zero-Knowledge (Sprints 1-4)**

1. **Sprint 1: Strict Reliability Mode**
- API task: Add strict execution toggle (`STRICT_EXECUTION_MODE`) that blocks all simulated fallbacks for training/compression/cloud jobs.
- API task: Standardize error envelope (`error_code`, `stage`, `actionable_fix`, `docs_url`) across ingestion/training/export.
- API task: Add `GET /api/projects/{id}/runtime/readiness` to validate GPU/dependencies/paths/secrets before run.
- UI task: Add “Readiness” panel in Training and Import pages with blocker/warning badges and direct fix tips.
- CLI task: Add `brewslm doctor --project <id>`.
- Acceptance criteria: With strict mode on, no stage silently falls back to simulation.
- Acceptance criteria: 100% failed jobs return structured error envelope.
- Acceptance criteria: `doctor` output matches API readiness status in CI integration test.

2. **Sprint 2: Universal Connector Hardening**
- API task: Add remote source “inspect” endpoint (`/ingestion/import-remote/inspect`) returning available splits/configs/features before import.
- API task: Harden HF import path for script-based datasets (snapshot/parquet fallback, better exception mapping).
- API task: Add connector retry policy with backoff + resumable queued imports.
- UI task: Upgrade import UI to two-step flow (Inspect -> Select split/config -> Import).
- UI task: Show connector-specific remediation (“set HF token”, “dataset requires script fallback”, etc.).
- Acceptance criteria: >=90% success across curated 50-dataset connector test suite.
- Acceptance criteria: duplicate raw dataset rows not created on retry/idempotent re-submit.
- Acceptance criteria: import failures always include at least one actionable next step.

3. **Sprint 3: Autopilot v2 (True Zero-Knowledge)**
- API task: Add `POST /training/autopilot/plan-v2` returning one simple plan + 2 alternatives with speed/quality/cost labels.
- API task: Add run-cost/time estimator endpoint for selected plan + hardware profile.
- API task: Add guardrail reason codes for “why blocked” and “how to unblock in one click”.
- UI task: Replace jargon-heavy controls in wizard path with 3 plain choices: `Fastest`, `Balanced`, `Best Quality`.
- UI task: Add one-click “Fix and Continue” actions for common blockers.
- Acceptance criteria: new user can launch first run in <=6 clicks from wizard.
- Acceptance criteria: >=80% first-run completion in internal novice usability test.
- Acceptance criteria: blocker states are explainable in plain language (no raw stack traces).

4. **Sprint 4: Target-First Project Setup Foundation**
- API task: Add target registry endpoint (`/targets/catalog`) with training/inference constraints.
- API task: Add compatibility endpoint (`/targets/compatibility`) for model/runtime/export combinations.
- API task: Add memory/latency estimator per target profile (mobile CPU, edge GPU, vLLM server, etc.).
- UI task: Make “Target deployment” a first-class step in wizard before training config.
- UI task: Show “fit/not-fit” checks for selected base model against target.
- Acceptance criteria: every new project stores a target profile.
- Acceptance criteria: impossible model-target combos are blocked before training starts.
- Acceptance criteria: recommendation API uses target profile as mandatory input.

---

**Phase 8: Optimization + Feedback Loop + Production Confidence (Sprints 5-8)**

5. **Sprint 5: Inference Optimization Autopilot**
- API task: Add `POST /export/optimize` to search quantization + runtime template combinations for selected target.
- API task: Add lightweight benchmark harness for latency/memory/quality deltas on sampled eval prompts.
- UI task: Add “Optimize for Target” button with tradeoff cards and recommended artifact.
- CLI task: Add `brewslm optimize --project <id> --target <target>`.
- Acceptance criteria: optimizer returns ranked candidates with measurable metrics.
- Acceptance criteria: chosen artifact includes reproducible benchmark report in export manifest.
- Acceptance criteria: default recommendation improves either latency or memory without violating minimum quality gate.

6. **Sprint 6: Playground Feedback -> Alignment Retrain Loop**
- API task: Auto-materialize thumbs up/down + edited responses into preference pairs.
- API task: Add `POST /training/alignment/retrain-from-feedback`.
- API task: Add dataset provenance tags linking feedback items to retrain runs.
- UI task: Add “Use feedback in next run” with preview of candidate pairs and quality filter.
- UI task: Show before/after comparison panel for last two runs.
- Acceptance criteria: feedback appears in alignment workspace within 60 seconds.
- Acceptance criteria: retrain-from-feedback creates valid DPO/ORPO-ready dataset and run config.
- Acceptance criteria: full audit trail from playground item -> alignment row -> experiment id.

7. **Sprint 7: Domain Eval Packs + Ship Gate**
- API task: Add domain eval pack registry and versioned pack assignment.
- API task: Add mandatory gate policy config per project (`must_pass`, `min_score`, `blocked_if_missing`).
- API task: Add deployment gate check endpoint consumed by export/deploy actions.
- UI task: Add single scorecard page with “Ship / No-Ship” decision and exact failed gates.
- UI task: Add domain-specific starter eval templates (legal, support, healthcare, finance, general).
- Acceptance criteria: export/deploy is blocked when required gates fail.
- Acceptance criteria: every experiment has domain-aware scorecard with missing-metric diagnostics.
- Acceptance criteria: gate decisions are deterministic and reproducible from stored artifacts.

8. **Sprint 8: Cloud Burst GA + Adoption Packaging**
- API task: finalize provider plugin contract for submit/status/cancel/logs/sync across providers.
- API task: add budget caps, spend alerts, and auto-cancel guardrails.
- API task: add resumable artifact sync with checksums and retry audit.
- UI task: add cloud run timeline with cost burn, status transitions, and controlled cancel/resume.
- DX task: publish “newbie quickstart pack” (3 templates + one-click sample datasets + CLI guide).
- Acceptance criteria: managed cloud burst success rate >=95% on integration suite.
- Acceptance criteria: median time-to-first-working-model <=45 minutes for a clean install path.
- Acceptance criteria: top 3 support issues covered by guided remediation docs and in-app hints.

---

**Cross-sprint non-negotiables**
1. Add tests for every new endpoint and one end-to-end path per sprint.
2. No silent fallback in strict mode.
3. Every blocker must include actionable remediation.
4. Telemetry must track conversion: ingest success, first run success, first deploy success.

If you want, I can now break Sprint 1 into a **ticket-ready board** (`backend`, `frontend`, `tests`, `docs`) with estimated points and file-level implementation targets.