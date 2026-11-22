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

- **Uploads and the shared client.** Every `api.post(url, formData)` goes through
  `src/api/client.ts`, whose default `Content-Type: application/json` made axios serialise a
  `FormData` body to JSON (`{"train":{}}`), dropping the files (predefined-split import, eval-set
  upload, answer-key file import all 422'd in a real browser, 2026-11-19). The request
  interceptor now clears `Content-Type` for a `FormData` body so the browser sets the multipart
  boundary. Unit tests of upload helpers mock `api`, so they cannot see this:
  `src/api/client.multipart.test.ts` drives the REAL client through axios's xhr adapter with a
  recording `XMLHttpRequest` and asserts every upload helper sends `FormData` with no forced
  `Content-Type`. A new upload helper belongs in that test.

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

## Main-path counter (UI simplification pass, phase 0 / 0b)

Two specs use `tests/golden-path/pathCounter.ts` (`createPathCounter(page)`: UI inputs go through
`counter.click / fill / select / check / setFiles`) and the constants in `tests/golden-path/pathBudget.ts`:

| Spec | Budget | What it measures |
|---|---|---|
| `main-path.spec.ts` | `MAIN_PATH_BUDGET` (**enforced**) | The real path: New Project -> Data (own train / val / test files) -> Train -> Results (Score this run, then read the Verdict card: score of record + provenance + rules line) -> Ship (HuggingFace export), all through the UI |
| `golden-path.spec.ts` | `SAMPLE_LAUNCH_BUDGET` (enforced) | Only the sample-project launch (training goes via the API). A guard for that launch, NOT a measure of the main path |
| (hand-measured) | `REFERENCE_PATH` 23 / 6 (not enforced) | The 2026-11-19 UI-only reference run; the plan's target is 12 / 5 |

**Counting rule.** An action is one click / fill / select / check / file pick on the first successful path. Not counted:
waits and polling, retries, opening an "Advanced" / "More options" disclosure (`{ uncounted: 'advanced' }`), setup such as
login (`'setup'`), `page.request` API calls, and **navigation** (`'navigation'`): `page.goto` and the sidebar / top-bar
links between destinations (Overview, Data, Train, Results, Ship, Settings, project list). Switching a step, tab or stage
inside a page (Data steps, Results tabs, Ship stages) is a counted click. A screen is a distinct view after
`counter.start()`: path with numeric ids and `annotate/<job>` folded to `:id`, plus the `step` / `tab` / `stage` value, where
the page's default (`data` step `data`, `results` tab `verdict`, `ship` stage `package`) folds to the bare path; `/login`
and legacy-redirect bounce URLs are not screens.

**Ratchet.** A phase may only LOWER `MAIN_PATH_BUDGET`, in the same commit as the UI change that earns it. Raising it needs a
stated reason in the commit message (a step added on purpose); never raise it to make a run pass.

**Phase 6 (Create, Coach, wording, Overview): 11 actions, 5 screens, 4.0 min** (fresh stack; training 125 s, 28
steps, final train loss 3.228; HF export completed). The name is left empty (the spec asserts the derived name in the
dialog and in the POST body) and the new project lands on Data › Sources, so the screens are Projects, Data, Train,
Results, Ship: the plan's ≤ 12 / 5 (HF) is met. GGUF is the default format, so the same path with GGUF is 10 actions
(not run by the spec: it skips the Format pick). Budget lowered to 11 / 5.

**Phase 5 (short Ship form): 12 actions, 6 screens, 3.8 min** (fresh stack; training 129 s, 28 steps, final train
loss 3.228; HF export completed). Ship is now Format + Run Export: the spec asserts the preselected run
(`export-run-select` = this run, option "#id · …") and the prefilled quantization None instead of clicking them.
Budget lowered to 12 / 6. The plan's ≤ 5 screens is not met: the six are Projects, the project, Data, Train,
Results, Ship (Phase 6 territory).

**Phase 4 (Train Common / Advanced, scoring settings at Verdict): unchanged at 14 / 6, 3.8 min** (fresh stack;
training 117 s, 28 steps, final train loss 3.228 — the same as phase 3). Train is still Train + Launch.

**Measured (2026-11-20, phase 3, fresh stack + fresh DATA_DIR): 14 actions, 6 screens, 4.6 min** (training 157 s, 28
steps, final train loss 3.228; eval ~1.5 min; export < 1 s). Data is now one pick of `train.jsonl` / `val.jsonl` /
`test.jsonl` on `pdsplit-files` (roles asserted from the names) + Import on Data › Sources: the real multipart
`POST /dataset/predefined-split` (its response must be OK, the result card visible, and its leak line must read "No
train, validation or test rows are exact repeats of each other", the plan's unseen G5 sentence). Data › Split is no
longer a screen on the path.

**Measured before (2026-11-20, phase 2 review fixes, fresh stack + fresh DATA_DIR): 18 actions, 7 screens, 11.3 min**
(training 274 s real CPU, 28 steps, final train loss 3.228, adapter written at completion; eval job ~6 min for 7 rows;
HuggingFace export ~4 s, a 269 MB merged `model.safetensors`). The first phase 2 figure (2.5 min) was INVALID: its
stack reused a DATA_DIR, so run 1 completed at once on an old adapter. Earlier phases (phase 1: 21 / 7, 4.3 min;
phase 0b: 22 / 8, 8.4 min) predate the freshness checks. Phase 2's short "Score this run" form is open + Run (2
actions, was 5): the 7-row test split is prefilled "All 7" and max new tokens 128 (the suggestion from its longest
answer, 70 characters, raised to the 128 floor; the spec logs the form's provenance lines and the stored result's
`samples 7, max_new_tokens 128`), temperature stays 0; no base result exists (auto-lift off), so nothing is matched. Phase 1 reads the verdict
on Results › Verdict (the spec waits for `verdict-score-value-*`, checks the provenance names the run and logs the
rules line) instead of clicking the Failures tab: one action and one screen fewer. Differences from the 23 / 6 reference:
- Actions: the reference had a "Start run #N" click after a page reload; the spec trains without a reload, so Train, Launch
  is 2 actions (the plan's gap #12); and since phase 1 there is no Failures click.
- Screens: the spec also lands on the project Overview after Create (the reference run's screen list did not count
  it). Phase 1 dropped Results › Failures from the path; phase 3 dropped Data › Split (the import is on Data › Sources).
- The base model is typed into the create dialog's main-step field (`create-project-base-model`, phase 7b; it was under
  Advanced options) as uncounted `'setup'`: a user who keeps the suggested model leaves it empty, so the spec's tiny
  CPU model is test setup, not a product action. The count is unchanged (11 actions / 5 screens).
- Score this run: open, Run (the reference filled Max Samples / Max New Tokens / Temperature: 3 counted fills, gone since phase 2).

**Running `main-path.spec.ts`** (`npm run test:e2e:main-path`; not part of CI, which ignores it: it needs real training).
Use `tests/golden-path/main-path-stack.sh start` (prints `STACK_DIR` and `E2E_BASE_URL`; `BACKEND_PY=<venv python>` in
a worktree without `backend/.venv`) and `... stop <STACK_DIR>` (kills each process group by its recorded pid, flushes
the stack's Redis dbs, says whether the ports are free; logs stay in `STACK_DIR`). **Every run needs a brand-new, empty
`DATA_DIR`** (the script makes one per start with `mktemp`): the fresh scratch DB numbers its first project / run 1
again, and a reused `DATA_DIR` hands run 1 the old `projects/1/experiments/1` directory — its old adapter marked the
new run COMPLETED at once, so the eval and export read the old weights while new training was still writing (the
first phase 2 measurement, 2026-11-20, was invalid for this reason). The spec now fails on that: the run must start
after the test began, be seen `running` before `completed`, take >= 30 s with steps + a loss, and (when its output dir
is on this machine) every weight file under it must be newer than the test; the eval result must be the one the job
the form queued wrote (`/api/jobs/{id}` → `result.eval_result_id`, the only result on the run, stored after training
ended, `samples` = the posted `max_samples`); the export must be the one the form created (`/export/create` response
id, `experiment_id` = the run), created and written after training ended.
Manual recipe (what the script does): backend on its own port with a scratch `DATABASE_URL` / a NEW `DATA_DIR`,
`AUTH_ENABLED=true API_KEY=sk-mock-admin-key AUTH_BOOTSTRAP_API_KEY=sk-mock-admin-key ALLOW_SQLITE_AUTOCREATE=true
DB_REQUIRE_ALEMBIC_HEAD=false TRAINING_BACKEND=external HF_HUB_OFFLINE=1 AUTO_LIFT_EVAL_ENABLED=false`, and a **Celery
worker** (`celery -A app.worker.celery_app worker --pool=threads --concurrency=2`) on a broker of its own
(`CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` / `REDIS_URL` on unused Redis db numbers). The in-process runtime
(`TRAINING_BACKEND=simulate`) produces no weights, so real training needs Redis + the worker; set `CUDA_VISIBLE_DEVICES=""`
on a shared-GPU box to keep it on CPU. Vite: `VITE_API_PROXY_TARGET=http://localhost:<backend port> vite --port <port>
--strictPort`; run with `E2E_BASE_URL` set to it. Needs SmolLM2-135M-Instruct in the HF cache. The fixtures are the 26 / 7 / 7
Q&A rows under `tests/golden-path/fixtures/` (disjoint questions). Training ~4.5 min, eval ~6 min (7 rows at 128 new tokens), export seconds on a 20-core CPU.
Stop the stack by the pids / ports of that stack only.
