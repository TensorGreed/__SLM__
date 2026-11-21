import fs from 'node:fs';
import path from 'node:path';
import { test, expect, type Page } from '@playwright/test';

import { MAIN_PATH_BUDGET, REFERENCE_PATH } from './pathBudget';
import { createPathCounter } from './pathCounter';

/**
 * Main-path gate (UI simplification pass, phase 0b; phase 1 reads the verdict on Verdict).
 *
 * Drives the whole product path through the UI and counts it with
 * `pathCounter.ts` (rule in its header): New Project -> Data (own train / val /
 * test files) -> Train -> Results (Score this run: open, Run; then read the Verdict card:
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
  await counter.fill(page.getByTestId('create-project-name'), `main-path-${Date.now()}`);
  await counter.fill(
    page.getByTestId('create-project-brief'),
    'Answer customer support questions about accounts, billing and teams from the help center.',
  );
  // A tiny base model so training and export run on CPU; the Advanced options
  // are not counted (rule in pathCounter.ts).
  await counter.click(page.getByTestId('create-project-advanced-toggle'), { uncounted: 'advanced' });
  await counter.fill(page.getByPlaceholder('e.g. HuggingFaceTB/SmolLM2-135M-Instruct'), BASE_MODEL, { uncounted: 'advanced' });
  await counter.click(page.getByTestId('create-project-submit'));
  await page.waitForURL(/\/project\/\d+/, { timeout: 60_000 });
  const projectId = Number(page.url().match(/\/project\/(\d+)/)![1]);
  const api = `/api/projects/${projectId}`;
  stamp('project created');

  // ── 2. Data: import my own train / val / test files (step 1, Sources; phase 3) ───
  // One multi-file pick (roles from the file names) + Import: the real multipart upload.
  await counter.click(page.getByTestId('nav-data'), { uncounted: 'navigation' });
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
  await counter.click(page.getByTestId('eval-score-run-open'));
  // UI-only prefills (decision 3): the 7-row test split is scored in full ("All 7"), max new tokens is
  // suggested from its longest answer, temperature stays 0. Nothing to fill.
  await expect(page.getByTestId('eval-run-rows-all')).toBeChecked({ timeout: 30_000 });
  await expect(page.getByTestId('eval-run-rows-provenance')).toContainText('Test examples: 7 rows');
  await expect(page.getByTestId('eval-run-model-finetuned')).toBeChecked();
  console.log(`score this run: ${(await page.getByTestId('eval-run-rows-provenance').textContent()) ?? ''} | ${(await page.getByTestId('eval-run-sends').textContent()) ?? ''}`);
  // The job the form queues (the 202 body is the Job) is the one whose result the verdict must show.
  const queued = page.waitForResponse((r) => r.url().includes('/evaluation/run-heldout') && r.request().method() === 'POST');
  await counter.click(page.getByTestId('eval-run-submit'));
  const queuedRes = await queued;
  expect(queuedRes.status(), 'eval queued').toBe(202);
  const evalJobId = ((await queuedRes.json()) as { id: number }).id;
  const sentBody = queuedRes.request().postDataJSON() as Record<string, unknown>;
  console.log(`eval job #${evalJobId} queued with ${JSON.stringify(sentBody)}`);
  expect(sentBody.experiment_id).toBe(experimentId);
  expect(sentBody).not.toHaveProperty('model_path');

  interface JobRow { id: number; status: string; error: string | null; result: { eval_result_id?: number } | null }
  let job: JobRow | null = null;
  const evalDeadline = Date.now() + 900_000;
  while (Date.now() < evalDeadline) {
    job = await getJson<JobRow>(page, `/api/jobs/${evalJobId}`, headers);
    if (JOB_TERMINAL.includes(job.status)) break;
    await page.waitForTimeout(3000);
  }
  expect(job?.status, `eval job #${evalJobId} (${job?.error ?? ''})`).toBe('succeeded');
  const evalResultId = job!.result?.eval_result_id;
  expect(typeof evalResultId, 'the job names its stored result').toBe('number');
  interface ResultRow { id: number; experiment_id: number; created_at: string; details?: { inference?: Record<string, unknown>; dataset?: Record<string, unknown> } }
  const results = await getJson<ResultRow[]>(page, `${api}/evaluation/results/${experimentId}`, headers);
  const stored = results.find((r) => r.id === evalResultId);
  expect(stored, `result #${evalResultId} under run #${experimentId}`).toBeTruthy();
  expect(results.length, 'only the result this test queued').toBe(1);
  expect(utcMs(stored!.created_at), 'eval stored after training completed').toBeGreaterThanOrEqual(runCompletedMs - 1000);
  // Evidence for the prefills: rows answered and max new tokens as stored on the result the form's job wrote.
  const inference = stored!.details?.inference ?? {};
  console.log(`eval result #${evalResultId} (job #${evalJobId}): created ${stored!.created_at}, samples ${String(inference.samples)}, max_new_tokens ${String(inference.max_new_tokens)}, model_path ${String(inference.model_path)}`);
  expect(inference.samples).toBe(sentBody.max_samples);
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
