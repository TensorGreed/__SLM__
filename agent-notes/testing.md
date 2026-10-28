# Testing — detail

Split out of CLAUDE.md (which keeps the commands and the hard rules).

## Backend

- **CI test scope.** `ci.yml`'s `backend-tests` job runs the `test_phase*`
  modules under `python -m unittest` (one shared process) **plus** a second
  step that runs every **non-`test_phase`** file in **its own `pytest`
  process** (`pip install pytest` then a per-file loop). Per-file isolation is
  mandatory: several non-phase files (`test_jobs_service`, `test_demo_project_reset`,
  `test_classification_training_launch`, the `test_probe_*` files) override
  `DATABASE_URL` at import, so two in one pytest process collide on the single
  StaticPool engine. The non-phase files are pytest-developed (they rely on
  `conftest.py`'s per-PID `/tmp` DB pin) and several **fail under bare
  `unittest`** (no conftest) — keep them on the pytest loop, don't fold them
  into the unittest job. The loop retries each file up to 3×. The original reason —
  async-job tests (`test_jobs_service`'s heldout / quickstart / llm-judge
  endpoints, `test_distillation_teacher_capture::test_run_event_emitted`)
  failing ~1-in-8 with "Project N not found" / a missing RunEvent — is fixed
  at the root: on the shared StaticPool connection a read-only session's
  ROLLBACK (`Session.close()` / pool reset-on-checkin) discarded another
  session's flushed-but-uncommitted writes; `app/database.py`'s rollback
  guard skips the DBAPI rollback for checkouts that executed no DML (only on
  the SQLite StaticPool engine; regression test
  `tests/test_shared_connection_rollback_guard.py`). The retry stays as a
  safety net for load-dependent timeouts (e.g. the 1.2 s base-model
  introspection in `training_preflight_service._infer_model_capability`); a
  genuinely broken file still fails all 3 attempts. Tests that start Jobs
  must still wait for them in tearDown — `from _job_wait import
  wait_for_jobs_idle` (raises naming the in-flight Jobs) — because a Job
  alive at TestClient shutdown hangs the process.

- **SQLAlchemy is pinned `<2.1`** (`requirements-base.txt`). 2.1 gives the aiosqlite adapter a
  per-connection `asyncio.Lock` (`_execute_mutex`); StaticPool shares one connection across every
  loop (uvicorn, `asyncio.run` in executor threads, Celery per-task loops, tests seeding beside a
  TestClient), and the lock binds to the first loop that contends on it — afterwards contention on any
  other loop raises "bound to a different event loop" or hangs. CI picked up 2.1.6 unpinned on
  2026-10-28 (`test_provenance_api` stuck Jobs / 600 s timeouts) while the dev venv had 2.0.48. Moving
  to 2.1 needs a per-loop mutex on the shared connection (prototype: branch
  `worktree-agent-abb9c491cf02cd691`, commit 937e531) and a full test pass on 2.1.
- **Real-runtime gate** (`tests/test_real_training_path.py`, CI job
  `real-training`): drives a LIVE stack over HTTP (backend on
  `TRAINING_BACKEND=external` + Redis + a Celery worker) — project → support-faq
  sample → experiment → start → Celery → `python scripts/train.py` → COMPLETED →
  watcher Job → automatic lift check → `/evaluation/summary` with row evidence.
  Skips unless `BREWSLM_REAL_TRAINING_BASE_URL` is set, so the per-file loop
  ignores it. CPU-only on CI (SmolLM2-135M, `num_epochs=1`,
  `AUTO_LIFT_EVAL_MAX_SAMPLES=8`, `AUTO_LIFT_EVAL_MAX_NEW_TOKENS=48`); the HF
  model is cached by `actions/cache`. Path-integrity, not quality: the lift may
  read "within noise". **Documents-project gate** (same file,
  `DocumentsProjectGateTests`): synthetic statute → upload → clean → the
  documents → Q&A flow with `backend="template"` → the pre-training passages
  gate (real base-model inference on CPU, judged by the template judge) →
  stop + Coach/summary assertions, "train anyway" → COMPLETED → judged lift
  → `passages_vs_finetune` readable. The job env sets
  `BREWSLM_TEMPLATE_SYNTH=1 EVAL_JUDGE_BACKEND=template`, so the support-faq
  test's headline is `judge_correct` there (it accepts f1 or judge_correct).
  **Template backend** (`synth_backends/template.py`): deterministic stand-in
  for a generation / judge model — pairs lifted from passage sentences,
  verdicts by token overlap; registered LAST and available only with
  `BREWSLM_TEMPLATE_SYNTH=1` (never auto-picked on a dev box); refuses any
  other prompt. The synthetic statute lives in both the pytest file and
  `tests/golden-path/golden-path-documents.spec.ts` — keep them in sync. Locally: boot a scratch backend + worker with the same
  env and run the file with the URL set — the dev worker must NOT be running
  on the same broker DB, or it consumes the task with the wrong DATA_DIR.
  **CPU dtype**: transformers ≥ 5 loads a checkpoint in its *saved* dtype
  (bf16 for most small instruct models). On a CPU without native bf16 (AVX2
  x86 — half of GitHub's runners) bf16 is emulated: a training step took
  ~230 s instead of ~5 s. `train.py`, the eval loaders
  (`evaluation_service._inference_dtype`) and `local_chat_service` load
  **fp32 on CPU**, bf16/fp16 on CUDA. Any new `from_pretrained` must pass a
  dtype — never rely on the default.
- **Real-training tests** (`test_training_correctness`, `test_continued_pretraining`,
  `test_export_scoping_and_merge`, `test_playground_trained_run`) fine-tune /
  load SmolLM2-135M. They run only with a GPU or `BREWSLM_REAL_TRAINING=1`
  (and a cached model); `BREWSLM_SKIP_REAL_TRAINING=1` forces a skip. CI has
  neither, so they skip there.
- **Never open a `TestClient(app)` outside pytest** (an ad-hoc `python -`
  probe, a script). Without `conftest.py`'s DB pin it boots the app against
  the DEV `data/slm_platform.db`, and the lifespan's startup sweep marks every
  QUEUED/RUNNING job there FAILED (`lost_during_restart`) — on 2026-10-13 it
  swept a documents→Q&A flow that was still running in the real backend
  (the runner's own `_finish_job` later overwrote FAILED with SUCCEEDED, so
  the result survived, but any waiter keyed on the status moved on early).
  Probe the running backend over HTTP instead.
- **Tests that spawn background Jobs must let them finish** (cancel runs they
  start, wait on `/api/jobs/active?include_recently_completed=false` in
  `tearDown`) and must never poll the engine via `asyncio.run` from the test
  thread — on the shared StaticPool connection both wipe the next test's
  writes ("Project N not found", "Could not refresh instance", "Lock … bound
  to a different event loop"). And never let a Job run a real model eval
  (patch `run_heldout_evaluation`): on CI it downloads + runs on CPU and the
  process hangs at exit. Also stub synth backends (patch **both**
  `synth_backends.BACKEND_REGISTRY` and `synth_playbook_service.BACKEND_REGISTRY`
  — the service binds its own copy) or the runner probes/uses a real Ollama and
  outlives the test; and make fake runners hold ~0.5s before failing so a
  follow-up request doesn't race the FAILED write (see `test_jobs_service`).
  CI caps each non-phase file at 600s
  (`::error::<file> timed out`) and the job at 90 min.


## Frontend


- **EvalPanel tests** (`EvalPanel.test.tsx` + the `.alignment` / `.extraction` /
  `.qa` / `.rag` / `.spanset` handler files, 24 tests) are green — the old
  "23 stale after a8bf40c" note is obsolete (verified 2026-10-14: all pass,
  full suite 148 files / 1202 tests). The per-row badges (`Preferred chosen`,
  `JSON: valid`, …) live in the predictions card under **"Advanced
  evaluation"**, which is open only when the active project is NOT
  `beginner_mode`; the handler tests therefore pin
  `useProjectStore.setState({ activeProject: null })` in `beforeEach`. Set a
  beginner project and click `data-testid="eval-advanced-toggle"` to reach the
  badges from the beginner path (the alignment file's beginner test). The
  earlier CoachStrip failures have since been fixed (27/27 as of 2026-06-02).

### Real-browser E2E (Playwright)

Playwright + chromium are installed at the repo root (NOT under
`frontend/`). For ad-hoc E2E loops use the entry from the root
`node_modules/`:

```js
import { chromium } from '/home/anuragj/Desktop/GitHub/__SLM__/node_modules/playwright/index.mjs';
```

Auth pattern: `POST /api/auth/local/login` with
`{username:"admin", password:"sk-mock-admin-key"}`, then seed the
token into `localStorage.slm_token` via `ctx.addInitScript` before
`page.goto` — that's the key the frontend reads at
`App.tsx:27`. Headless mode works on the dev box.

**Golden-path regression gate (Epic G phase G1).**
`tests/golden-path/golden-path.spec.ts` + `playwright.golden-path.config.ts`
(fast, no video/slowMo, distinct from the narrated `tests/demo-recordings`
suite). It locks the newbie path: login → launch the support-faq sample →
beginner-mode pipeline renders → **train on the simulate runtime → assert
`completed`**. Run: `npm run test:e2e:golden` (root). It needs an isolated
stack — backend with `AUTH_ENABLED=true API_KEY=sk-mock-admin-key
AUTH_BOOTSTRAP_API_KEY=sk-mock-admin-key ALLOW_SIMULATED_TRAINING=true
TRAINING_BACKEND=simulate` + a scratch DB, and the frontend pointed at it
via the new **`VITE_API_PROXY_TARGET`** env (vite proxy target is now
env-configurable so a second backend on a non-default port works). CI runs
it as the `e2e-golden-path` job. Training step + completion poll go through
the API (`page.request` with the login token) for robustness; the UI drives
login + sample launch + pipeline render. **Path-integrity gate, not a
quality gate** (simulate runtime). The eval→export tail is deferred
(G1-tail — eval on a simulated checkpoint needs real inference).
**G2 — documents project** (`golden-path-documents.spec.ts`, same config):
upload + clean via `page.request`, the Training tab's flow card, one click,
poll the `documents_qa_flow` Job, assert the card shows the pre-training
gate's verdict (reroute/train-anyway when `retrieval_ready`, "training went
ahead" otherwise). The E2E stack also boots with `BREWSLM_TEMPLATE_SYNTH=1
EVAL_JUDGE_BACKEND=template HF_HUB_OFFLINE=1` and caches SmolLM2 (the gate
runs real CPU inference). ~4 min locally.
