import fs from 'node:fs';
import path from 'node:path';
import { test, expect, type Page, type Response } from '@playwright/test';

import { MAIN_PATH_BUDGET, REFERENCE_PATH } from './pathBudget';
import { createPathCounter } from './pathCounter';

/**
 * Main-path gate (UI simplification pass, phase 0b; phase 1 reads the verdict on Verdict).
 *
 * Drives the whole product path through the UI and counts it with
 * `pathCounter.ts` (rule in its header): New Project (name derived from the brief; lands on Data, phase 6) -> Data (own train / val /
 * test files) -> Train -> Results (Score this run: Run both — phase 7e: a trained run with no evals opens the form by
 * default; phase 7c: a first eval with no base-model score on the set defaults to Both, so one click queues the
 * fine-tuned eval then the base eval; then read the Verdict card:
 * score of record, rows, rules line; no Failures click) -> Ship (HuggingFace export). Polling for completion goes through `page.request` and is not a user
 * action. Training is REAL (tiny CPU run on cached SmolLM2-135M-Instruct), so
 * eval and export have weights.
 *
 * Needs the isolated stack with a Celery worker AND a brand-new empty DATA_DIR
 * (a reused one hands the fresh DB's run #1 an old run directory, whose old
 * adapter marks the run completed at once); see agent-notes/testing.md
 * ("Main-path counter"). Run: `npm run test:e2e:main-path`.
 *
 * Freshness checks (review fix, phase 2): the run started after this test began,
 * was seen running before it completed, trained for real (steps, loss, weights
 * files written after the test began when the output dir is on this machine);
 * the stored eval result is the one the job the form queued wrote; the export is
 * the one the form created, for this run, and was written after training ended.
 */

const ADMIN = 'admin';
const ADMIN_KEY = 'sk-mock-admin-key';
const BASE_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const FIXTURES = path.join(__dirname, 'fixtures');

const TERMINAL = ['completed', 'failed', 'cancelled'];
const JOB_TERMINAL = ['succeeded', 'failed', 'cancelled'];
/** A tiny CPU run takes minutes; a run "completed" in seconds reused an old run directory. */
const MIN_TRAINING_SECONDS = 30;

/** Server timestamps are UTC; SQLite drops the offset. */
function utcMs(value: unknown): number {
  const text = String(value ?? '');
  return Date.parse(/[zZ]|[+-]\d\d:\d\d$/.test(text) ? text : `${text}Z`);
}

/** Weight files (``*.safetensors`` / ``*.bin``) under ``dir`` with their mtimes; [] when not on this machine. */
function weightFiles(dir: string | null | undefined): Array<{ file: string; bytes: number; mtimeMs: number }> {
  if (!dir || !fs.existsSync(dir)) return [];
  const out: Array<{ file: string; bytes: number; mtimeMs: number }> = [];
  const walk = (d: string) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (/\.(safetensors|bin)$/.test(entry.name)) {
        const st = fs.statSync(full);
        out.push({ file: full, bytes: st.size, mtimeMs: st.mtimeMs });
      }
    }
  };
  walk(dir);
  return out;
}

async function getJson<T>(page: Page, url: string, headers: Record<string, string>): Promise<T> {
  const res = await page.request.get(url, { headers });
  expect(res.ok(), `GET ${url}`).toBeTruthy();
  return (await res.json()) as T;
}

test('main path: new project -> own files -> train -> evaluate -> export, counted', async ({ page }) => {
  const counter = createPathCounter(page);
  const started = Date.now();
  // Everything this test makes (run, eval result, export) must be newer than this (1 s slack for clock rounding).
  const testStartedMs = started - 1000;
  const stamp = (label: string) => console.log(`[${Math.round((Date.now() - started) / 1000)}s] ${label}`);

  // ── Setup: login (not part of the counted path) ───────────────────────
  await page.goto('/login');
  await counter.fill(page.getByPlaceholder('Enter your username'), ADMIN, { uncounted: 'setup' });
  await counter.fill(page.getByPlaceholder('API Key or Password'), ADMIN_KEY, { uncounted: 'setup' });
  await counter.click(page.getByRole('button', { name: /^Sign in$/ }), { uncounted: 'setup' });
  await page.waitForURL((url) => url.pathname === '/', { timeout: 20_000 });
  const token = await page.evaluate(() => localStorage.getItem('slm_token'));
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  counter.start();

  // ── 1. Create the project (Projects) ──────────────────────────────────
  await counter.click(page.getByRole('button', { name: /^\+ New Project$/ }));
  // Phase 6: the name is optional; left empty, it is derived from the brief (shown in the dialog) and
  // sent like a typed name.
  await counter.fill(
    page.getByTestId('create-project-brief'),
    'Answer customer support questions about accounts, billing and teams from the help center.',
  );
  await expect(page.getByTestId('create-project-derived-name')).toContainText('Answer customer support questions about accounts');
  // A tiny base model so training and export run on CPU. Phase 7b: the field is on the main step (it was
  // under Advanced options, uncounted then too). A user who keeps the suggested model leaves it empty, so
  // when the hint already names BASE_MODEL nothing is typed; otherwise the CPU model is test setup, not a
  // product action (rule in pathCounter.ts), and the log says so.
  const baseHint = page.getByTestId('create-project-base-model-hint');
  await expect(baseHint).toContainText('Leave it empty to use', { timeout: 10_000 });
  await expect(baseHint).toContainText(/suggested from the brief|domain preset/, { timeout: 10_000 }).catch(() => undefined);
  if (!((await baseHint.textContent()) ?? '').includes(BASE_MODEL)) {
    await counter.fill(page.getByTestId('create-project-base-model'), BASE_MODEL, { uncounted: 'setup' });
  }
  const createdProject = page.waitForResponse((r) => new URL(r.url()).pathname === '/api/projects' && r.request().method() === 'POST');
  await counter.click(page.getByTestId('create-project-submit'));
  const createdProjectRes = await createdProject;
  expect(createdProjectRes.ok(), 'project created').toBeTruthy();
  expect((createdProjectRes.request().postDataJSON() as { name?: string }).name, 'derived name sent').toBe('Answer customer support questions about accounts');
  // Phase 6: a new project (no data yet) lands on its first step, Data › Sources — no Overview screen on the path.
  await page.waitForURL(/\/project\/\d+\/data/, { timeout: 60_000 });
  const projectId = Number(page.url().match(/\/project\/(\d+)/)![1]);
  const api = `/api/projects/${projectId}`;
  stamp('project created');

  // ── 2. Data: import my own train / val / test files (step 1, Sources; phase 3) ───
  // One multi-file pick (roles from the file names) + Import: the real multipart upload.
  await counter.setFiles(page.getByTestId('pdsplit-files'), ['train.jsonl', 'val.jsonl', 'test.jsonl'].map((f) => path.join(FIXTURES, f)));
  await expect(page.getByTestId('pdsplit-role-0')).toHaveValue('train');
  await expect(page.getByTestId('pdsplit-role-1')).toHaveValue('val');
  await expect(page.getByTestId('pdsplit-role-2')).toHaveValue('test');
  const importRes = page.waitForResponse((r) => r.url().includes('/dataset/predefined-split') && r.request().method() === 'POST');
  await counter.click(page.getByTestId('pdsplit-submit'));
  expect((await importRes).ok(), 'predefined-split import').toBeTruthy();
  await expect(page.getByTestId('pdsplit-problem')).toHaveCount(0);
  await expect(page.getByTestId('pdsplit-result')).toBeVisible({ timeout: 60_000 });
  // The import result's leak line (the plan's "unseen G5 sentence"): seen on the result card.
  await expect(page.getByTestId('pdsplit-leak')).toContainText('No train, validation or test rows are exact repeats of each other');
  console.log(`import: ${(await page.getByTestId('pdsplit-result').textContent()) ?? ''}`);
  await expect(page.getByTestId('own-files-continue-train')).toBeEnabled();
  stamp('files imported');

  // ── 3. Train: Train, Launch ───────────────────────────────────────────
  await counter.click(page.getByTestId('nav-train'), { uncounted: 'navigation' });
  await counter.click(page.getByTestId('train-form-train'));
  await counter.click(page.getByRole('button', { name: 'Launch', exact: true }));

  interface RunRow {
    id: number; status: string; started_at: string | null; completed_at: string | null; created_at: string;
    total_steps: number | null; final_train_loss: number | null; output_dir: string | null;
  }
  let experimentId = 0;
  let status = 'unknown';
  let run: RunRow | null = null;
  const statusesSeen: string[] = [];
  const trainDeadline = Date.now() + 1_200_000;
  while (Date.now() < trainDeadline) {
    const runs = await getJson<RunRow[]>(page, `${api}/training/experiments`, headers);
    expect(runs.length, 'runs in this new project').toBeLessThanOrEqual(1);
    run = runs[runs.length - 1] ?? null;
    if (run) {
      experimentId = run.id;
      status = run.status;
      if (statusesSeen[statusesSeen.length - 1] !== status) statusesSeen.push(status);
      if (TERMINAL.includes(status)) break;
    }
    await page.waitForTimeout(3000);
  }
  expect(status, 'training run status').toBe('completed');
  // The run is this test's and trained for real: started after the test began, seen running before it completed,
  // took at least MIN_TRAINING_SECONDS, logged steps and a loss, and wrote its weights after the test began.
  expect(run).not.toBeNull();
  const runStartedMs = utcMs(run!.started_at);
  const runCompletedMs = utcMs(run!.completed_at);
  const trainingSeconds = (runCompletedMs - runStartedMs) / 1000;
  console.log(`run #${experimentId}: statuses ${statusesSeen.join(' > ')}; started ${run!.started_at}, completed ${run!.completed_at} (${trainingSeconds.toFixed(0)} s); steps ${run!.total_steps}, final train loss ${run!.final_train_loss}`);
  expect(runStartedMs, 'run started after the test began').toBeGreaterThan(testStartedMs);
  expect(statusesSeen, 'run seen running before it completed').toContain('running');
  expect(trainingSeconds, 'real training time').toBeGreaterThanOrEqual(MIN_TRAINING_SECONDS);
  expect(run!.total_steps ?? 0, 'training steps').toBeGreaterThan(0);
  expect(typeof run!.final_train_loss, 'final train loss').toBe('number');
  const weights = weightFiles(run!.output_dir);
  if (run!.output_dir && fs.existsSync(run!.output_dir)) {
    expect(weights.length, `weight files under ${run!.output_dir}`).toBeGreaterThan(0);
    for (const w of weights) {
      console.log(`  weights: ${w.file} ${w.bytes} bytes, written ${new Date(w.mtimeMs).toISOString()}`);
      expect(w.mtimeMs, `${w.file} written by this run`).toBeGreaterThan(testStartedMs);
    }
  } else {
    console.log(`  weights: output dir ${run!.output_dir} is not on this machine; file freshness not checked`);
  }
  stamp(`training completed (run #${experimentId})`);

  // ── 4. Results: Score this run (short form), read the Verdict card ───
  await counter.click(page.getByTestId('nav-results'), { uncounted: 'navigation' });
  // Phase 7e (gap G3): a trained run with no evals opens the form by default, so there is no open click.
  await expect(page.getByTestId('eval-run-form')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('eval-score-run-open')).toHaveAttribute('data-open', 'true');
  // UI-only prefills (decision 3): the 7-row test split is scored in full ("All 7"), max new tokens is
  // suggested from its longest answer, temperature stays 0. Nothing to fill.
  await expect(page.getByTestId('eval-run-rows-all')).toBeChecked({ timeout: 30_000 });
  await expect(page.getByTestId('eval-run-rows-provenance')).toContainText('Test examples: 7 rows');
  // Phase 7c: nothing picked before and no base-model score on the test split (auto-lift is off on this stack), so
  // the form starts on Both and says why; the user keeps it. One click: Run both.
  await expect(page.getByTestId('eval-run-model-both')).toBeChecked({ timeout: 30_000 });
  await expect(page.getByTestId('eval-run-model-default-both')).toContainText('Both is the default for a first eval');
  await expect(page.getByTestId('eval-run-submit')).toHaveText('Run both');
  console.log(`score this run: ${(await page.getByTestId('eval-run-rows-provenance').textContent()) ?? ''} | ${(await page.getByTestId('eval-run-model-default-both').textContent()) ?? ''} | ${(await page.getByTestId('eval-run-sends').textContent()) ?? ''}`);
  // The jobs the form queues (each 202 body is the Job): the fine-tuned leg first, then the base leg.
  const queuedResponses: Array<Response> = [];
  const onResponse = (r: Response) => {
    if (r.url().includes('/evaluation/run-heldout') && r.request().method() === 'POST') queuedResponses.push(r);
  };
  page.on('response', onResponse);
  await counter.click(page.getByTestId('eval-run-submit'));
  await expect.poll(() => queuedResponses.length, { timeout: 60_000 }).toBe(2);
  page.off('response', onResponse);
  const [queuedRes, queuedBaseRes] = queuedResponses;
  expect(queuedRes.status(), 'fine-tuned eval queued').toBe(202);
  expect(queuedBaseRes.status(), 'base eval queued').toBe(202);
  const evalJobId = ((await queuedRes.json()) as { id: number }).id;
  const baseJobId = ((await queuedBaseRes.json()) as { id: number }).id;
  const sentBody = queuedRes.request().postDataJSON() as Record<string, unknown>;
  const sentBaseBody = queuedBaseRes.request().postDataJSON() as Record<string, unknown>;
  console.log(`eval job #${evalJobId} queued with ${JSON.stringify(sentBody)}`);
  console.log(`base eval job #${baseJobId} queued with ${JSON.stringify(sentBaseBody)}`);
  expect(sentBody.experiment_id).toBe(experimentId);
  expect(sentBody).not.toHaveProperty('model_path');
  expect(sentBaseBody.experiment_id).toBe(experimentId);
  expect(sentBaseBody.model_path, 'the base leg scores the run\'s base model').toBe(BASE_MODEL);
  expect(sentBaseBody.max_samples, 'both legs, same rows').toBe(sentBody.max_samples);
  expect(sentBaseBody.max_new_tokens, 'both legs, same max new tokens').toBe(sentBody.max_new_tokens);

  interface JobRow { id: number; status: string; error: string | null; result: { eval_result_id?: number } | null }
  const waitJob = async (id: number): Promise<JobRow | null> => {
    let row: JobRow | null = null;
    const deadline = Date.now() + 900_000;
    while (Date.now() < deadline) {
      row = await getJson<JobRow>(page, `/api/jobs/${id}`, headers);
      if (JOB_TERMINAL.includes(row.status)) break;
      await page.waitForTimeout(3000);
    }
    return row;
  };
  const job = await waitJob(evalJobId);
  expect(job?.status, `eval job #${evalJobId} (${job?.error ?? ''})`).toBe('succeeded');
  stamp('fine-tuned evaluation done');
  const baseJob = await waitJob(baseJobId);
  expect(baseJob?.status, `base eval job #${baseJobId} (${baseJob?.error ?? ''})`).toBe('succeeded');
  stamp('base evaluation done');
  const evalResultId = job!.result?.eval_result_id;
  expect(typeof evalResultId, 'the job names its stored result').toBe('number');
  interface ResultRow { id: number; experiment_id: number; created_at: string; details?: { inference?: Record<string, unknown>; dataset?: Record<string, unknown> } }
  const results = await getJson<ResultRow[]>(page, `${api}/evaluation/results/${experimentId}`, headers);
  const stored = results.find((r) => r.id === evalResultId);
  expect(stored, `result #${evalResultId} under run #${experimentId}`).toBeTruthy();
  // The base leg's result is filed under the baseline run, not this one.
  expect(results.length, 'only the fine-tuned result this test queued').toBe(1);
  expect(utcMs(stored!.created_at), 'eval stored after training completed').toBeGreaterThanOrEqual(runCompletedMs - 1000);
  // Evidence for the prefills: rows answered and max new tokens as stored on the result the form's job wrote.
  const inference = stored!.details?.inference ?? {};
  console.log(`eval result #${evalResultId} (job #${evalJobId}): created ${stored!.created_at}, samples ${String(inference.samples)}, max_new_tokens ${String(inference.max_new_tokens)}, model_path ${String(inference.model_path)}`);
  expect(inference.samples).toBe(sentBody.max_samples);
  const baseResultId = baseJob!.result?.eval_result_id;
  expect(typeof baseResultId, 'the base job names its stored result').toBe('number');
  expect(baseResultId, 'the base leg stored its own result').not.toBe(evalResultId);
  console.log(`base eval result #${baseResultId} (job #${baseJobId})`);
  stamp('evaluation stored');
  // The verdict is read where the eval was run (no extra click): the Verdict card
  // shows the score of record with its rows and provenance once the eval Job lands.
  const score = page.getByTestId(`verdict-score-value-${evalResultId}`);
  await expect(score).toBeVisible({ timeout: 120_000 });
  const provenance = page.getByTestId(`verdict-score-provenance-${evalResultId}`);
  await expect(provenance).toContainText(`Run #${experimentId}`);
  await expect(page.getByTestId('verdict-rules')).toBeVisible();
  console.log(`verdict: ${(await score.textContent()) ?? ''} | ${(await provenance.textContent()) ?? ''} | ${(await page.getByTestId('verdict-rules').textContent()) ?? ''}`);

  // ── 5. Ship: export the run as HuggingFace, no quantization ───────────
  await counter.click(page.getByTestId('nav-ship'), { uncounted: 'navigation' });
  // Phase 5: the run is preselected (the Verdict's run, "#id · name") and HuggingFace prefills
  // quantization None, so Format is the only pick; both prefills are asserted, never clicked.
  const runSelect = page.getByTestId('export-run-select');
  await expect(runSelect).toHaveValue(String(experimentId), { timeout: 30_000 });
  await expect(runSelect.locator('option:checked')).toHaveText(new RegExp(`^#${experimentId} · `));
  await counter.select(page.getByTestId('export-format-select'), 'huggingface');
  await expect(page.getByTestId('export-quantization-select')).toHaveValue('none');
  // The export the form creates (its id) for this run is the one that must complete.
  const created = page.waitForResponse((r) => /\/export\/create$/.test(new URL(r.url()).pathname) && r.request().method() === 'POST');
  await counter.click(page.getByRole('button', { name: 'Run Export' }));
  const createdRes = await created;
  expect(createdRes.ok(), 'export created').toBeTruthy();
  const exportId = ((await createdRes.json()) as { id: number }).id;
  expect((createdRes.request().postDataJSON() as { experiment_id?: number }).experiment_id, 'export of this run').toBe(experimentId);

  interface ExportRow { id: number; status: string; output_path: string | null; created_at: string; completed_at: string | null }
  let exportRow: ExportRow | null = null;
  const exportDeadline = Date.now() + 600_000;
  while (Date.now() < exportDeadline) {
    const exports = await getJson<ExportRow[]>(page, `${api}/export/list`, headers);
    exportRow = exports.find((e) => e.id === exportId) ?? null;
    if (exportRow && TERMINAL.includes(exportRow.status)) break;
    await page.waitForTimeout(3000);
  }
  expect(exportRow?.status, `export #${exportId} status`).toBe('completed');
  expect(utcMs(exportRow!.created_at), 'export created after training completed').toBeGreaterThanOrEqual(runCompletedMs - 1000);
  const exported = weightFiles(exportRow!.output_path);
  for (const w of exported) {
    console.log(`  export: ${w.file} ${w.bytes} bytes, written ${new Date(w.mtimeMs).toISOString()}`);
    expect(w.mtimeMs, `${w.file} written after training completed`).toBeGreaterThanOrEqual(runCompletedMs - 1000);
  }
  console.log(`export #${exportId}: created ${exportRow!.created_at}, completed ${exportRow!.completed_at}, ${exported.length} weight file(s) at ${exportRow!.output_path}`);
  stamp('export completed');

  // ── Count + budget ────────────────────────────────────────────────────
  const result = counter.finish();
  console.log(`main path: ${result.actions} actions, ${result.screens.length} screens (reference ${REFERENCE_PATH.actions} / ${REFERENCE_PATH.screens})`);
  console.log(`  actions: ${result.log.join(' | ')}`);
  console.log(`  screens: ${result.screens.join(' > ')}`);
  expect(result.actions).toBeLessThanOrEqual(MAIN_PATH_BUDGET.maxActions);
  expect(result.screens.length).toBeLessThanOrEqual(MAIN_PATH_BUDGET.maxScreens);
});
