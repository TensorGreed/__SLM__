import { defineConfig, devices } from '@playwright/test';

/**
 * Main-path E2E config (UI simplification pass, phase 0b).
 *
 * Runs ONLY `tests/golden-path/main-path.spec.ts`: the real path through the UI
 * (New Project -> own train / val / test files -> Train -> Evaluate -> Ship
 * export) with a tiny REAL CPU training run. Needs an isolated stack with a
 * Celery worker (TRAINING_BACKEND=external) and a cached SmolLM2-135M-Instruct;
 * setup in agent-notes/testing.md ("Main-path counter"). Not part of CI's
 * simulate-runtime golden-path job (that config ignores this file).
 */
export default defineConfig({
  testDir: './tests/golden-path',
  testMatch: /main-path\.spec\.ts$/,
  timeout: 1_800_000,
  expect: { timeout: 30_000 },
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:5173',
    headless: true,
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
