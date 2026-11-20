import path from 'node:path';
import { test, expect } from '@playwright/test';

import { MAIN_PATH_BUDGET, REFERENCE_PATH } from './pathBudget';
import { createPathCounter } from './pathCounter';

/**
 * Main-path gate (UI simplification pass, phase 0b).
 *
 * Drives the whole product path through the UI and counts it with
 * `pathCounter.ts` (rule in its header): New Project -> Data (own train / val /
 * test files) -> Train -> Results (Evaluate, then Failures) -> Ship (HuggingFace
 * export). Polling for completion goes through `page.request` and is not a user
 * action. Training is REAL (tiny CPU run on cached SmolLM2-135M-Instruct), so
 * eval and export have weights.
 *
 * Needs the isolated stack with a Celery worker; see agent-notes/testing.md
 * ("Main-path counter"). Run: `npm run test:e2e:main-path`.
 */

const ADMIN = 'admin';
const ADMIN_KEY = 'sk-mock-admin-key';
const BASE_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const FIXTURES = path.join(__dirname, 'fixtures');

const TERMINAL = ['completed', 'failed', 'cancelled'];

test('main path: new project -> own files -> train -> evaluate -> export, counted', async ({ page }) => {
  const counter = createPathCounter(page);
  const started = Date.now();
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

  // ── 2. Data: import my own train / val / test files (step 5, Split) ───
  await counter.click(page.getByTestId('nav-data'), { uncounted: 'navigation' });
  await counter.click(page.getByTestId('stage-step-dataprep'));
  await counter.click(page.getByTestId('split-own-files'));
  await counter.setFiles(page.getByTestId('pdsplit-train'), path.join(FIXTURES, 'train.jsonl'));
  await counter.setFiles(page.getByTestId('pdsplit-val'), path.join(FIXTURES, 'val.jsonl'));
  await counter.setFiles(page.getByTestId('pdsplit-test'), path.join(FIXTURES, 'test.jsonl'));
  await counter.click(page.getByTestId('pdsplit-submit'));
  await expect(page.getByTestId('pdsplit-problem')).toHaveCount(0);
  await expect(page.getByTestId('pdsplit-submit')).toHaveText('Import my files', { timeout: 60_000 });
  stamp('files imported');

  // ── 3. Train: Train, Launch ───────────────────────────────────────────
  await counter.click(page.getByTestId('nav-train'), { uncounted: 'navigation' });
  await counter.click(page.getByTestId('train-form-train'));
  await counter.click(page.getByRole('button', { name: 'Launch', exact: true }));

  let experimentId = 0;
  let status = 'unknown';
  const trainDeadline = Date.now() + 1_200_000;
  while (Date.now() < trainDeadline) {
    const res = await page.request.get(`${api}/training/experiments`, { headers });
    const runs = (await res.json()) as Array<{ id: number; status: string }>;
    const run = runs[runs.length - 1];
    if (run) {
      experimentId = run.id;
      status = run.status;
      if (TERMINAL.includes(status)) break;
    }
    await page.waitForTimeout(3000);
  }
  expect(status, 'training run status').toBe('completed');
  stamp(`training completed (run #${experimentId})`);

  // ── 4. Results: Evaluate on test examples, then Failures ──────────────
  await counter.click(page.getByTestId('nav-results'), { uncounted: 'navigation' });
  await counter.click(page.getByRole('button', { name: '+ Evaluate on test examples' }));
  const field = (label: string) => page.locator('.form-group', { has: page.locator('label', { hasText: new RegExp(`^${label}$`) }) }).locator('input');
  await counter.fill(field('Max Samples'), '7');
  await counter.fill(field('Max New Tokens'), '48');
  await counter.fill(field('Temperature'), '0');
  await counter.click(page.getByRole('button', { name: 'Run Evaluation' }));

  let evalRows = 0;
  const evalDeadline = Date.now() + 900_000;
  while (Date.now() < evalDeadline) {
    const res = await page.request.get(`${api}/evaluation/results/${experimentId}`, { headers });
    if (res.ok()) {
      const results = (await res.json()) as unknown[];
      evalRows = results.length;
      if (evalRows > 0) break;
    }
    await page.waitForTimeout(3000);
  }
  expect(evalRows, 'stored eval results').toBeGreaterThan(0);
  stamp('evaluation stored');
  await counter.click(page.getByTestId('results-tab-failures'));

  // ── 5. Ship: export the run as HuggingFace, no quantization ───────────
  await counter.click(page.getByTestId('nav-ship'), { uncounted: 'navigation' });
  const select = (label: string) => page.getByTestId('export-form-card').locator('.form-group', { has: page.locator('label', { hasText: new RegExp(`^${label}$`) }) }).locator('select');
  await counter.select(select('Experiment'), String(experimentId));
  await counter.select(select('Format'), 'huggingface');
  await counter.select(select('Quantization'), 'none');
  await counter.click(page.getByRole('button', { name: 'Run Export' }));

  let exportStatus = 'unknown';
  const exportDeadline = Date.now() + 600_000;
  while (Date.now() < exportDeadline) {
    const res = await page.request.get(`${api}/export/list`, { headers });
    const exports = (await res.json()) as Array<{ id: number; status: string }>;
    if (exports.length) {
      exportStatus = exports[exports.length - 1].status;
      if (TERMINAL.includes(exportStatus)) break;
    }
    await page.waitForTimeout(3000);
  }
  expect(exportStatus, 'export status').toBe('completed');
  stamp('export completed');

  // ── Count + budget ────────────────────────────────────────────────────
  const result = counter.finish();
  console.log(`main path: ${result.actions} actions, ${result.screens.length} screens (reference ${REFERENCE_PATH.actions} / ${REFERENCE_PATH.screens})`);
  console.log(`  actions: ${result.log.join(' | ')}`);
  console.log(`  screens: ${result.screens.join(' > ')}`);
  expect(result.actions).toBeLessThanOrEqual(MAIN_PATH_BUDGET.maxActions);
  expect(result.screens.length).toBeLessThanOrEqual(MAIN_PATH_BUDGET.maxScreens);
});
