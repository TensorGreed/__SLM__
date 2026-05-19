import { test } from '@playwright/test';

test('prototype browser recording smoke test', async ({ page }) => {
  // This test only proves that Playwright can open the app URL, record video,
  // and save a screenshot. Real selectors and login/setup behavior must be
  // added after the demo samples, UI routes, APIs, and pipeline steps are
  // fully mapped in docs-demo/evidence.
  await page.goto('/');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'docs-demo/screenshots/01-prototype-smoke.png', fullPage: true });
});

