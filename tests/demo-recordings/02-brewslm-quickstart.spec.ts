import { test, expect } from '@playwright/test';

// Video 02 — BrewSLM Quickstart
//
// Drives the 5-section arc documented at
//   docs-demo/videos/02-brewslm-quickstart/recording-plan.md
// against the running stack (vite on :5173, FastAPI on :8000).
//
// Selectors are all sourced from
//   docs-demo/evidence/11-selector-route-evidence.md
// and re-verified against the current frontend code (May 2026).
//
// Side effects: this spec calls POST /api/demo-projects/support-faq,
// which creates a new "Demo · Support FAQ" project row each time.
// Pre-existing rows can stack up; clean them via the project list
// or by running against a disposable SQLite DB.

const SCREENSHOT_DIR = 'docs-demo/screenshots';

test('Video 02 — BrewSLM Quickstart full arc', async ({ page }) => {
    test.setTimeout(120_000);

    // ── Section 1 — Login ─────────────────────────────────────────────
    await page.goto('/login');
    await expect(page.getByPlaceholder('Enter your username')).toBeVisible();
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v02-login.png`, fullPage: true });

    // Must be the bootstrap admin user, not just any username. Local
    // login auto-creates ENGINEER users on first sight; those users
    // have no project membership and hit 403 on /api/projects/<id>.
    // `admin` is the bootstrap user (AUTH_BOOTSTRAP_USERNAME in
    // backend/.env), which has global access.
    await page.getByPlaceholder('Enter your username').fill('admin');
    await page.getByPlaceholder('API Key or Password').fill('sk-mock-admin-key');
    await page.getByRole('button', { name: /^Sign in$/ }).click();

    // Land on project list ("/").
    await page.waitForURL((url) => url.pathname === '/', { timeout: 15_000 });

    // ── Section 2 — Project list + demo tiles ────────────────────────
    const tilesContainer = page.locator('.demo-project-tiles');
    await expect(tilesContainer).toBeVisible();

    const supportFaqTile = page.locator(
        '[aria-label="Open the Demo · Support FAQ demo project"]',
    );
    const piiTile = page.locator(
        '[aria-label="Open the Demo · PII / PCI Detector demo project"]',
    );
    const sentimentTile = page.locator(
        '[aria-label="Open the Demo · Sentiment classifier demo project"]',
    );
    await expect(supportFaqTile).toBeVisible();
    await expect(piiTile).toBeVisible();
    await expect(sentimentTile).toBeVisible();

    // Hover so the focus ring / cta swap is on-screen for the take.
    await supportFaqTile.hover();
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v02-tiles.png`, fullPage: true });

    // ── Section 3 — Seed support-faq ─────────────────────────────────
    await supportFaqTile.click();
    await page.waitForURL(/\/project\/\d+\/pipeline\/data/, { timeout: 30_000 });

    // The Data tab renders a `button.tab[title="Data"]` in the top tab bar.
    await expect(page.locator('button.tab[title="Data"]')).toBeVisible();
    // Let the document list settle (20 raw rows).
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v02-data-tab.png`, fullPage: true });

    // ── Section 4 — Pipeline tab tour ────────────────────────────────
    // Tabs are `button.tab[title="<label>"]` in the top tab bar
    // (frontend/src/pages/ProjectPipelinePage.tsx:285).
    const clickTab = async (
        label: string,
        screenshot?: string,
        fullPage = true,
    ) => {
        await page.locator(`button.tab[title="${label}"]`).click();
        await page.waitForTimeout(1000);
        if (screenshot) {
            await page.screenshot({
                path: `${SCREENSHOT_DIR}/${screenshot}`,
                fullPage,
            });
        }
    };

    await clickTab('Cleaning', 'v02-cleaning-tab.png');
    // Gold Set fullPage spans all 200 rows (~22000px). Capture viewport-
    // only so the screenshot is usable for narration / docs.
    await clickTab('Gold Set', 'v02-goldset-tab.png', false);
    await clickTab('Dataset Prep');
    await clickTab('Training', 'v02-training-tab-empty.png');
    await clickTab('Data');

    // ── Section 5 — Expand a raw row + wrap ──────────────────────────
    const firstExpander = page.locator('[data-testid^="expand-doc-"]').first();
    await expect(firstExpander).toBeVisible();
    await firstExpander.click();
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v02-expanded-row.png`, fullPage: true });
});
