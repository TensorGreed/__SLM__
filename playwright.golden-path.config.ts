import { defineConfig, devices } from '@playwright/test';

/**
 * Golden-path E2E config (Epic G, phase G1).
 *
 * Distinct from the narrated demo-recording config (videos, slowMo): this
 * is a FAST regression gate that locks the newbie path — login → seed a
 * sample project → walk the beginner-mode pipeline → train (simulate) →
 * assert the run completes. No video, no slowMo; runs headless in CI.
 *
 * Backend must run with ALLOW_SIMULATED_TRAINING=true + TRAINING_BACKEND=simulate
 * + AUTH_ENABLED=false; frontend on :5173 (vite proxies /api → :8000).
 */
export default defineConfig({
  testDir: './tests/golden-path',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  retries: process.env.CI ? 1 : 0,
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
