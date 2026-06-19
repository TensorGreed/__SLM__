import { chromium, type FullConfig } from '@playwright/test';

/**
 * Global setup: warm the vite dev server before the suite runs.
 *
 * Vite compiles modules on-demand on the first browser request, which can
 * take 10–20s and makes the *first* test flaky (slow navigation / timeouts)
 * while later tests run against a warm server. Loading the app once here,
 * outside any test's timeout budget, removes that cold-start flake.
 */
export default async function globalSetup(config: FullConfig) {
  const baseURL =
    config.projects[0]?.use?.baseURL ||
    process.env.E2E_BASE_URL ||
    'http://localhost:5173';
  const browser = await chromium.launch();
  const page = await browser.newPage();
  try {
    await page.goto(baseURL, { waitUntil: 'load', timeout: 120_000 });
    // Touch the login route too — it's the suite's entry point.
    await page.goto(`${baseURL}/login`, { waitUntil: 'load', timeout: 120_000 });
  } finally {
    await browser.close();
  }
}
