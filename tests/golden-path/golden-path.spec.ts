import { test, expect } from '@playwright/test';

/**
 * Golden-path regression gate (Epic G, phase G1).
 *
 * Locks the newbie path against regression: a first-timer logs in, launches
 * a sample project, lands in the beginner-mode pipeline, and trains a model
 * that runs to completion. The high-regression-risk surface — auth →
 * project → config resolution → training runtime → async completion — is
 * exercised against the real stack.
 *
 * Training uses the **simulate runtime** (the decision for CI: no GPU), so
 * this is a *path-integrity* gate, not a model-quality gate. The eval→export
 * tail is deferred (G1-tail) — on a simulated checkpoint, eval needs real
 * inference, which is its own slice.
 *
 * The UI drives the user-facing entry (login, sample launch, pipeline
 * render); the training launch + completion poll go through the API
 * (`page.request`, robust against async timing) using the same endpoints
 * the Training UI calls.
 *
 * Uses the support-faq (QA / instruction_sft) sample — it trains cleanly on
 * the simulate runtime. (The sentiment/classification sample currently fails
 * the training data gate — tracked separately.)
 */

const ADMIN = 'admin';
const ADMIN_KEY = 'sk-mock-admin-key';

test('golden path: login → sample → beginner pipeline → train completes', async ({ page }) => {
  // ── 1. Login as the bootstrap admin (global project access) ──────────
  await page.goto('/login');
  await page.getByPlaceholder('Enter your username').fill(ADMIN);
  await page.getByPlaceholder('API Key or Password').fill(ADMIN_KEY);
  await page.getByRole('button', { name: /^Sign in$/ }).click();
  await page.waitForURL((url) => url.pathname === '/', { timeout: 20_000 });

  // The API calls below run via page.request (separate from the page's
  // localStorage), so carry the login token the UI stored. When auth is
  // disabled (the CI setup), there's no token and the API is open — so
  // the header is omitted. Works in both modes.
  const token = await page.evaluate(() => localStorage.getItem('slm_token'));
  const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};

  // ── 2. Launch the support-faq sample project from the dashboard ──────
  const sampleTile = page.locator(
    '[aria-label="Open the Demo · Support FAQ demo project"]',
  );
  await expect(sampleTile).toBeVisible();
  await sampleTile.click();

  // Land in the project workspace pipeline.
  await page.waitForURL(/\/project\/\d+/, { timeout: 20_000 });
  const projectId = Number(page.url().match(/\/project\/(\d+)/)![1]);
  expect(projectId).toBeGreaterThan(0);

  // ── 3. Beginner-mode pipeline renders with the golden-path stages ────
  // Sample projects launch in beginner mode; assert the core path stages
  // are present (the linear data → train → eval → export spine). The
  // pipeline stages render as buttons with emoji labels.
  await expect(page.getByRole('button', { name: '📂 Data' }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: '🔬 Training' }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: '📊 Evaluation' }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: '🚀 Export' }).first()).toBeVisible();

  // ── 4. Train on the simulate runtime, assert the run completes ───────
  const api = `/api/projects/${projectId}`;

  const cfgRes = await page.request.post(`${api}/training/experiments/effective-config`, {
    headers: authHeaders,
    data: {},
  });
  expect(cfgRes.ok()).toBeTruthy();
  const config = (await cfgRes.json()).resolved_training_config;
  expect(config?.base_model).toBeTruthy();

  const createRes = await page.request.post(`${api}/training/experiments`, {
    headers: authHeaders,
    data: { name: 'golden-path-e2e', config },
  });
  expect(createRes.ok()).toBeTruthy();
  const experimentId = (await createRes.json()).id;
  expect(experimentId).toBeGreaterThan(0);

  const startRes = await page.request.post(
    `${api}/training/experiments/${experimentId}/start`,
    { headers: authHeaders, data: { execution_mode: 'simulate' } },
  );
  expect(startRes.ok()).toBeTruthy();

  // Poll the experiments list until the run reaches a terminal status.
  let status = 'unknown';
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    const listRes = await page.request.get(`${api}/training/experiments`, {
      headers: authHeaders,
    });
    const experiments = (await listRes.json()) as Array<{ id: number; status: string }>;
    const run = experiments.find((e) => e.id === experimentId);
    status = run?.status ?? 'unknown';
    if (['completed', 'failed', 'cancelled'].includes(status)) break;
    await page.waitForTimeout(1500);
  }

  // The golden path produces a completed training run.
  expect(status).toBe('completed');
});
