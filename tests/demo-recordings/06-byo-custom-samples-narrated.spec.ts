import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 06 — BYO Custom Samples, narrated.
//
// Demonstrates the bring-your-own-data path: create a fresh
// non-demo project, upload a tiny custom CSV, see rows on the
// Data tab. Same Playwright/TTS pipeline as the seeded-sample
// videos (V03-V05), with the project creation + file upload
// driven through the UI instead of the demo seeder.

const DURATIONS_PATH = path.resolve(
    __dirname,
    '../../tts/audio/v06-durations.json',
);

const CSV_PATH = path.resolve(
    __dirname,
    '../../docs-demo/byo-sample/byo-coffee-shop-faq.csv',
);

type Durations = {
    cold_open: number;
    new_project: number;
    empty_data_tab: number;
    upload_csv: number;
    rows_imported: number;
    wrap: number;
};

const dur: Durations = JSON.parse(fs.readFileSync(DURATIONS_PATH, 'utf-8'));

const SCREENSHOT_DIR = 'docs-demo/screenshots';

async function padTo(
    page: import('@playwright/test').Page,
    sectionStartMs: number,
    sectionDurationSeconds: number,
) {
    const remaining = sectionDurationSeconds * 1000 - (Date.now() - sectionStartMs);
    if (remaining > 0) await page.waitForTimeout(remaining);
}

async function focusOn(
    page: import('@playwright/test').Page,
    selector: string,
) {
    const el = page.locator(selector).first();
    if ((await el.count()) > 0) {
        await el.evaluate((node) =>
            node.scrollIntoView({ block: 'center', behavior: 'instant' as ScrollBehavior }),
        );
        await page.waitForTimeout(300);
    }
}

test('Video 06 — BYO Custom Samples narrated', async ({ page, request }) => {
    test.setTimeout(10 * 60_000);

    expect(fs.existsSync(CSV_PATH), 'BYO sample CSV required').toBeTruthy();
    const projectName = `BYO Coffee Shop FAQ ${Date.now() % 100000}`;

    // Acquire JWT for the API-driven upload later (avoids fragile
    // UI file picker timing — same pattern as V09/V11).
    const loginResp = await request.post('http://localhost:8000/api/auth/local/login', {
        data: { username: 'admin', password: 'sk-mock-admin-key' },
    });
    expect(loginResp.ok()).toBeTruthy();
    const token = (await loginResp.json()).token as string;
    const authHeader = { Authorization: `Bearer ${token}` };

    // UI login
    await page.goto('/login');
    await page.getByPlaceholder('Enter your username').fill('admin');
    await page.getByPlaceholder('API Key or Password').fill('sk-mock-admin-key');
    await page.getByRole('button', { name: /^Sign in$/ }).click();
    await page.waitForURL((url) => url.pathname === '/', { timeout: 15_000 });
    await page.waitForTimeout(1000);

    // ── Section: cold open ───────────────────────────────────────────
    // Land on the project list. Focus on the existing tile strip /
    // header area so the viewer sees the platform's entry point.
    let sectionStart = Date.now();
    await focusOn(page, ':text("Try a demo project")');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v06-project-list.png`, fullPage: true });
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: new project (click + fill form) ────────────────────
    sectionStart = Date.now();
    await page.getByRole('button', { name: /^\+ New Project$/ }).click();
    await page.waitForTimeout(800);
    // The New Project modal defaults to Beginner Mode (a 3-step
    // brief-driven wizard). For the BYO walkthrough we want the
    // simple Name + Description + Create form, which lives behind
    // the "Advanced Mode" pill at the top of the modal.
    await page.getByRole('button', { name: /^Advanced Mode$/ }).click();
    await page.waitForTimeout(400);
    await page.getByPlaceholder('e.g. Legal Document Copilot').fill(projectName);
    await page.waitForTimeout(400);
    await page
        .getByPlaceholder('Brief description of the project goal')
        .fill('Six-row sample CSV for the BYO walkthrough.');
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v06-new-project-form.png`, fullPage: false });
    await page.getByRole('button', { name: /^Create Project$/ }).click();
    // Wait for navigation to the new project. Fresh projects land on
    // a guided-setup welcome card that doesn't render the inner tab
    // strip, so the URL match here is just the project root.
    await page.waitForURL(/\/project\/\d+/, { timeout: 30_000 });
    const projectId = (page.url().match(/\/project\/(\d+)/) || [])[1] ?? '';
    expect(projectId, 'Could not parse new project id from URL').not.toBe('');
    await padTo(page, sectionStart, dur.new_project);

    // ── Section: empty data tab ─────────────────────────────────────
    // Navigate directly to /pipeline/data (the SPA route the
    // IngestionPanel renders against). Skips the guided-setup
    // welcome card which would otherwise overlay the upload zone.
    sectionStart = Date.now();
    await page.goto(`/project/${projectId}/pipeline/data`);
    await page.waitForTimeout(1500);
    await focusOn(page, ':text("Drop files here or click to browse")');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v06-empty-data-tab.png`, fullPage: true });
    await padTo(page, sectionStart, dur.empty_data_tab);

    // ── Section: upload CSV ─────────────────────────────────────────
    sectionStart = Date.now();
    // Focus the upload zone so the viewer sees it
    await focusOn(page, ':text("Drop files here or click to browse")');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v06-upload-zone.png`, fullPage: false });
    // API-driven upload via multipart — same pattern as V09/V11.
    // The UI's setInputFiles path was unreliable here (the IngestionPanel's
    // change handler can stall on the FastAPI upload-batch endpoint when
    // the file picker is driven programmatically). The API call is the
    // canonical path the UI calls anyway.
    const csvBuffer = fs.readFileSync(CSV_PATH);
    const uploadResp = await request.post(
        `http://localhost:8000/api/projects/${projectId}/ingestion/upload-batch`,
        {
            headers: authHeader,
            multipart: {
                files: {
                    name: 'byo-coffee-shop-faq.csv',
                    mimeType: 'text/csv',
                    buffer: csvBuffer,
                },
            },
        },
    );
    expect(uploadResp.ok(), 'CSV upload should succeed').toBeTruthy();
    const uploadResult = await uploadResp.json();
    console.log(`[v06] uploaded ${uploadResult.uploaded ?? '?'} file(s)`);

    // Process the uploaded document so the CSV rows are actually
    // extracted. Without this step the document stays at status
    // "pending" and the project's progress_percent stays at 0 — which
    // keeps the GettingStartedWizard overlaying the IngestionPanel.
    const docsResp = await request.get(
        `http://localhost:8000/api/projects/${projectId}/ingestion/documents`,
        { headers: authHeader },
    );
    const docsData = await docsResp.json();
    const docs = Array.isArray(docsData) ? docsData : (docsData.documents ?? docsData.items ?? []);
    if (docs.length > 0) {
        const docId = docs[0].id;
        await request.post(
            `http://localhost:8000/api/projects/${projectId}/ingestion/documents/${docId}/process`,
            { headers: authHeader, data: {} },
        );
    }
    await padTo(page, sectionStart, dur.upload_csv);

    // ── Section: rows imported ──────────────────────────────────────
    sectionStart = Date.now();
    // Reload so the IngestionPanel re-fetches and lists the new rows.
    // Now that the doc is processed, progress_percent > 0 dismisses
    // the wizard and surfaces the actual document list.
    await page.reload();
    await page.waitForTimeout(2500);
    await focusOn(page, '[data-testid^="expand-doc-"]');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v06-rows-imported.png`, fullPage: true });
    await padTo(page, sectionStart, dur.rows_imported);

    // ── Section: wrap ───────────────────────────────────────────────
    sectionStart = Date.now();
    await padTo(page, sectionStart, dur.wrap);
});
