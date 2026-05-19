# Task: Convert synthetic data generation to chunked + background-task pattern

You are working in the BrewSLM monorepo. The Data tab's **Synthetic Data**
step currently freezes and surfaces a "Network error" when the user
asks for a large batch (e.g. 50 Q&A pairs against a local Ollama). Two
problems compound:

1. Each `/synthetic/generate*` endpoint makes a **single** teacher-model
   HTTP call to produce all N rows at once. The request stays open the
   entire generation; the Vite dev proxy's 10-minute ceiling severs it
   long before slow local models finish.
2. Asking the model to emit 50 structured items in one JSON output is
   load-bearing on the model's ability to stay coherent. Local models
   routinely truncate mid-array, drift into hallucinations, or burn
   their budget inside `<think>` reasoning blocks before any JSON
   starts. One bad generation dumps the whole batch.

The fix has three parts, applied consistently across all three
synthetic generators (Q&A pairs, multi-turn conversations, span
extraction):

- **Chunk** the teacher calls into batches of ~5–10 rows each, executed
  serially, and accumulate the partial results.
- **Background-task pattern**: the API endpoint returns a `task_id`
  within milliseconds; the frontend polls a status endpoint for
  progress + final results. Mirrors the existing cleaning fix.
- **Raise the per-request cap** to ~200 across all three generator
  schemas, since each underlying teacher call is now small.

The reference implementation already shipped is the cleaning fix
(commit `dc88e04`). Read it first; the new code should follow the
exact same patterns (in-memory task registry, `asyncio.create_task`
with its own DB session via `async_session_factory`, 202 on start, GET
for polling, frontend swaps the synchronous POST for start + 1.5s poll
loop). **Do not introduce Celery or any new infra.**

## Files you will touch

### Backend (Python, FastAPI + SQLAlchemy async)

- `backend/app/services/synthetic_service.py` — the three async
  generator functions live here. **Existing entry points** (do not
  remove or rename):
  - `generate_qa_pairs(db, project_id, source_text, num_pairs, api_url, api_key, model_name)` — line ~652
  - `generate_conversation_dialogues(db, project_id, source_text, num_dialogues, min_turns, max_turns, api_url, api_key, model_name)` — line ~769
  - `generate_span_extraction_rows(db, project_id, source_text, num_rows, entity_types, api_url, api_key, model_name)` — line ~1258
  - These currently make **one** `call_teacher_model(...)` per
    invocation. Refactor each so that internally they loop over chunks
    of ~5–10 rows (configurable via a module-level constant
    `SYNTHETIC_CHUNK_SIZE = 8`), aggregate the per-chunk results, and
    return the union. The existing teacher-prompt + JSON-parsing
    helpers (`_parse_teacher_pairs`, `_parse_teacher_conversations`,
    `_validate_span_rows`, etc.) stay as-is — you're just calling them
    per chunk instead of once.
  - **Per-chunk failure handling**: if one chunk's
    `call_teacher_model` raises or `_parse_teacher_*` returns empty,
    record the failure in a per-task `chunk_errors: list[dict]`
    structure but continue with the remaining chunks. Goal: one bad
    chunk costs you 5–10 rows, not the whole batch.

- `backend/app/services/synthetic_service.py` — append a
  background-task plumbing block at the end of the file. **Follow the
  pattern from `cleaning_service.py` exactly**:
  - `@dataclass class SyntheticGenerationTask` with fields:
    `task_id: str`, `project_id: int`, `mode: Literal["qa", "conversation", "span"]`,
    `status: str` (pending/running/completed/failed), `requested_count: int`,
    `completed_chunks: int`, `total_chunks: int`, `produced_count: int`,
    `results: list[dict]`, `chunk_errors: list[dict]`, `error: str | None`,
    `started_at: datetime`, `updated_at: datetime`,
    `finished_at: datetime | None`. Add a `to_dict()` method that
    serializes ISO-format timestamps.
  - Module-level `_SYNTHETIC_TASKS: dict[str, SyntheticGenerationTask]`
    with a `threading.Lock` (mirror cleaning_service exactly) +
    `_MAX_TRACKED_TASKS: int = 64` cap + `_trim_finished_tasks()`
    helper.
  - `async def _run_synthetic_task(task: SyntheticGenerationTask,
    *, kwargs_for_chunk: dict)` — opens its own session via
    `async_session_factory()`, computes `total_chunks = ceil(requested_count / SYNTHETIC_CHUNK_SIZE)`,
    loops over chunks calling the underlying generator with
    `num_pairs` / `num_dialogues` / `num_rows` set to the chunk size,
    appends results, catches per-chunk exceptions into
    `chunk_errors`, updates `completed_chunks` + `produced_count` +
    `updated_at` on every chunk so the frontend sees live progress.
    Sets `status` to `"completed"` (even if some chunks failed) or
    `"failed"` (only if a fatal non-chunk error occurred).
  - `start_synthetic_task(*, project_id, mode, ...kwargs) -> SyntheticGenerationTask`
    that builds the task, registers it, kicks off
    `asyncio.create_task(_run_synthetic_task(...))`, and returns the
    record so the API can immediately serialize + return it. **Task ID
    format**: `synth-{uuid4().hex[:12]}`.
  - `get_synthetic_task_status(task_id) -> dict | None` reading via
    the lock.

- `backend/app/api/synthetic.py` — add three new endpoints alongside
  the existing ones (keep the old ones in place; they call the
  refactored chunked generators directly, which is fine for small N):
  - `POST /projects/{project_id}/synthetic/generate-async` → 202,
    accepts the same body as `/generate` but with `num_pairs` cap
    raised to **200** (`Field(5, ge=1, le=200)`). Body field name
    stays `num_pairs`.
  - `POST /projects/{project_id}/synthetic/generate-conversations-async`
    → 202, mirrors `/generate-conversations` body schema with
    `num_dialogues: Field(3, ge=1, le=200)`.
  - `POST /projects/{project_id}/synthetic/generate-spans-async` →
    202, mirrors `/generate-spans` body schema with
    `num_rows: Field(5, ge=1, le=200)`.
  - `GET  /projects/{project_id}/synthetic/tasks/{task_id}` returns
    the task's `to_dict()`. **404 if the id is unknown**, **404 if
    the task belongs to a different project** (don't leak cross-project
    state — match the cleaning endpoint's check).
  - **Also raise the cap on the existing sync endpoints** to 200 (so
    `/generate`, `/generate-conversations`, `/generate-spans` get
    bumped to `le=200`); the sync versions stay useful for small
    counts where users don't want the polling overhead.

### Frontend (TypeScript + React + axios)

- `frontend/src/components/data/SyntheticPanel.tsx` — find
  `handleGenerate` (around line 274). Replace the synchronous POST for
  each `generationMode` branch with the async start + 1.5s poll
  pattern. The shape is **already implemented** for cleaning at
  `frontend/src/components/data/CleaningPanel.tsx` — copy that loop
  structure (the `while (true) { poll; if terminal break; }` block,
  the live status string updates, the terminal-state result
  materialization). Drop the `LONG_REQUEST_TIMEOUT_MS` axios override
  on the start call (the new endpoint returns in <1s). Show progress
  like `Generating 12/40 pairs (chunk 3/8)…` using the task's
  `produced_count` / `requested_count` and `completed_chunks` /
  `total_chunks`. Surface any non-empty `chunk_errors[]` array under
  the result list as a small "N partial-batch errors" warning row
  (don't block the save flow — partial results are still useful).
- TypeScript types: add `SyntheticTaskStatus` interface mirroring the
  backend dataclass; share it across the three modes since the shape
  is identical.
- **Raise the numeric input caps** on the three sliders/inputs from
  50 → 200 (the JSX inputs near the top of the component).

## API contract (new endpoints)

```http
POST /api/projects/{project_id}/synthetic/generate-async
Content-Type: application/json
{
  "source_text": "<at least 10 chars>",
  "num_pairs": 100,
  "api_url": "",
  "api_key": "",
  "model_name": "llama3"
}
→ 202
{
  "task_id": "synth-3f9e1a2c0bd4",
  "project_id": 17,
  "mode": "qa",
  "status": "pending",
  "requested_count": 100,
  "completed_chunks": 0,
  "total_chunks": 13,
  "produced_count": 0,
  "results": [],
  "chunk_errors": [],
  "error": null,
  "started_at": "2026-05-14T16:00:00+00:00",
  "updated_at": "2026-05-14T16:00:00+00:00",
  "finished_at": null
}

GET /api/projects/{project_id}/synthetic/tasks/{task_id}
→ 200 (same shape; "status" eventually transitions to "completed" or "failed")
→ 404 if unknown or wrong project
```

Same shape applies to `generate-conversations-async` and
`generate-spans-async`, except `results` contains conversation dicts
and span-row dicts respectively, and the request body uses
`num_dialogues` / `num_rows`.

## Tests (these are part of the deliverable, not optional)

### Backend

Create `backend/tests/test_synthetic_async_task.py` modeled after
`backend/tests/test_cleaning_async_task.py`. Pin:

1. `POST /generate-async` returns 202 with a task_id and an initial
   status payload that carries the right `mode`, `requested_count`, and
   `total_chunks`.
2. Task transitions to terminal status (`completed` or `failed`)
   within a reasonable timeout (use the existing `_wait_terminal`
   helper pattern, ~10s with 50ms polling).
3. With the teacher unconfigured (no `TEACHER_MODEL_API_URL`) and
   `ALLOW_SYNTHETIC_DEMO_FALLBACK=true`, all three modes complete
   end-to-end using the demo-mode fallbacks and produce non-empty
   `results`.
4. With the teacher mocked via `unittest.mock.patch` on
   `app.services.synthetic_service.call_teacher_model`, a chunk that
   raises `RuntimeError` lands one entry in `chunk_errors`, the task
   still reaches `completed`, and the other chunks' results are
   present in `results`. The mock should be set up to raise on the
   2nd chunk only and return valid JSON on chunks 1 / 3+.
5. `GET /tasks/{unknown_id}` returns 404.
6. `GET /tasks/{task_id_from_other_project}` returns 404.

You should additionally extend the **synthetic unit tests** (find any
existing `test_*synthetic*.py` in `backend/tests/` first) so the
existing chunk-free assertions still pass after your refactor — that
is, `generate_qa_pairs` still returns a list of N items when called
directly, regardless of how it now chunks internally.

### Frontend

Create `frontend/src/components/data/SyntheticPanel.test.tsx` (or
extend an existing one if present) using the patterns from
`frontend/src/components/data/DatasetImportWizard.test.tsx` (mock
`api/client` via `vi.hoisted`). Pin at minimum:

1. Clicking Generate in `qa` mode hits `POST .../synthetic/generate-async`
   with the right body, then begins polling `GET .../synthetic/tasks/{task_id}`.
2. While the task is `running`, the status line shows `Generating
   <produced>/<requested>` progress text.
3. When the task terminates as `completed`, the generated rows render
   in the results table.
4. `chunk_errors[]` non-empty surfaces a visible warning element.
5. Numeric input cap is 200 across the three modes.

## Quality gates

Before declaring done, all of the following must be green:

```bash
# Backend
cd backend
python -m pytest tests/test_synthetic_async_task.py -v
python -m pytest tests/test_cleaning_async_task.py tests/test_document_sample_endpoint.py tests/test_phase101_dataset_import_foundation.py -q
# Run any other test_*synthetic* files you find.

# Frontend
cd ../frontend
npx tsc --noEmit              # must be silent
npx vitest run                # must end with all tests passing
npx vite build                # production build must succeed
```

A previous regression-class bug to watch for: when adding new JSX
inside a `<table>`, **always run `npx vite build`** in addition to
`tsc --noEmit`. `tsc` doesn't catch mismatched JSX closing tags
(e.g. `<Fragment>...</React.Fragment>`); Vite/Babel does. The build
must be clean.

## Constraints

- **Do not** add Celery, Redis, or any external task queue. Use
  `asyncio.create_task` + an in-memory dict guarded by
  `threading.Lock`, identical to `cleaning_service.py`.
- **Do not** rename or remove the existing sync endpoints
  (`/generate`, `/generate-conversations`, `/generate-spans`) or the
  existing service functions. Other callers (CLI, tests, scripts) may
  still hit them with small N. Just raise the cap to 200.
- **Do not** change the request body shapes. The async endpoints
  accept the same JSON the sync ones do, plus the response is the
  task envelope instead of the inline result.
- **Do not** disable the confidence-threshold filtering in
  `save_synthetic_batch` / `save_synthetic_conversation_batch` /
  `save_synthetic_span_batch` — those are downstream from your work
  and stay untouched.
- **No emojis in code or comments** unless the existing file already
  uses them (some do — match the local style of the file you're
  editing).
- **`SYNTHETIC_CHUNK_SIZE` lives as a module-level constant** in
  `synthetic_service.py` (not an env var, not a setting). 8 is the
  default; the test suite assumes this value when computing
  `total_chunks`.
- **Per-chunk DB commits**: after each chunk, call `await db.flush()`
  but **not** `await db.commit()` — the task owns the session and
  commits once at the end of `_run_synthetic_task`. If `_run_synthetic_task`
  raises a fatal (non-chunk) error before committing, roll back. This
  matches the cleaning task's behavior.

## Commit hygiene

When done, produce a single commit on `main` with this message
shape (Co-Authored-By line included verbatim):

```
Synthetic generation: chunked teacher calls + background-task pattern

Three generators (Q&A pairs, multi-turn conversations, span
extraction) were each making a single teacher-model HTTP call to
produce all N rows at once. With a slow local Ollama and N=50, that
single call exceeded the Vite proxy's 10-minute timeout, surfaced as
"Network error" on the Synthetic tab. Two follow-on quality issues:
LLMs drift when emitting long structured JSON, and one bad
generation dumped the entire batch.

Fix:
- Each generator now loops internally over chunks of
  SYNTHETIC_CHUNK_SIZE=8 rows per teacher call and aggregates.
  Per-chunk failures land in chunk_errors[] but the rest of the
  batch still runs.
- New background-task endpoints alongside the existing sync ones:
    POST /generate-async, /generate-conversations-async,
         /generate-spans-async  → 202 + task envelope.
    GET  /synthetic/tasks/{task_id}  → status + partial results.
  SyntheticGenerationTask in synthetic_service.py mirrors the
  CleaningTask pattern from cleaning_service.py: in-memory dict
  capped at 64 entries, asyncio.create_task drives the work
  against its own DB session via async_session_factory.
- SyntheticPanel.tsx replaces the synchronous POST with the async
  start + 1.5s poll loop. Status string updates live ("Generating
  12/40 pairs (chunk 3/8)…") so the user sees progress. Numeric
  input caps raised 50 → 200 across all three modes.

Tests: test_synthetic_async_task.py pins 202 start, terminal
status, chunk-error isolation, 404 paths; SyntheticPanel test
covers async start, live progress, and chunk-error surface. Full
frontend regression + vite production build green.
```

## How to verify the user-facing fix

After you ship, the manual smoke-test the user will run is:
1. Open the Data tab → Synthetic Data step.
2. Paste a multi-paragraph source text.
3. Set "Number of pairs" to 100.
4. Click Generate.

Expected behavior:
- Within ~1 second, the status flips to "Generating 0/100 (chunk 0/13)…".
- The counters tick forward as chunks complete.
- After all chunks finish, the results table renders with up to 100
  rows. If any chunks failed (e.g. teacher timeouts), a small warning
  shows N chunk errors but the save flow still works on the
  successful ones.
- No "Network error" appears regardless of how long the total
  generation takes, because the HTTP request returned in step 1 and
  the rest is polling.
