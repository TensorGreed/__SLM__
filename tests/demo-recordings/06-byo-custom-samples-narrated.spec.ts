import { test, expect, type Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 06 — BYO Custom Samples, narrated (Epic U workspace).
//
// The bring-your-own-data path: create a fresh non-demo project from the
// "+ New Project" dialog (name + what the model should do), land on its
// Overview, open Data · Sources (empty), upload a tiny custom CSV, see the
// rows listed under "Ingested documents". Project creation is driven
// through the UI; the file upload goes through the API (the same call the
// upload zone makes) because a programmatic file picker was unreliable.
// Audio durations come from tts/audio/v06-durations.json
// (tts/generate_v06_narration.py).
//
// Picture: the "spotlight + narrate" pattern (see video 03).

const CSV_PATH = path.resolve(__dirname, '../../docs-demo/byo-sample/byo-coffee-shop-faq.csv');


const DURATIONS_PATH = path.resolve(__dirname, '../../tts/audio/v06-durations.json');

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

// ── pacing ──────────────────────────────────────────────────────────────

async function padTo(page: Page, sectionStartMs: number, sectionDurationSeconds: number) {
    const remaining = sectionDurationSeconds * 1000 - (Date.now() - sectionStartMs);
    if (remaining > 0) await page.waitForTimeout(remaining);
}

function remainingMs(sectionStartMs: number, sectionDurationSeconds: number, marginMs = 400): number {
    return Math.max(0, sectionDurationSeconds * 1000 - (Date.now() - sectionStartMs) - marginMs);
}

// ── spotlight ───────────────────────────────────────────────────────────

// Passed to page.evaluate as a real function (a template-string body is
// evaluated as an expression and never runs).
function spotlightInPage(selector: string): Promise<boolean> {
    for (const d of Array.from(document.querySelectorAll('[data-spot-rect="1"]'))) d.remove();
    const el = document.querySelector(selector);
    if (!el) return Promise.resolve(false);
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    return new Promise<boolean>((resolve) => {
        setTimeout(() => {
            const rr = el.getBoundingClientRect();
            const pad = 10;
            const vw = window.innerWidth;
            const vh = window.innerHeight;
            const L = Math.max(0, rr.left - pad);
            const T = Math.max(0, rr.top - pad);
            const R = Math.min(vw, rr.right + pad);
            const B = Math.min(vh, rr.bottom + pad);
            const dim = 'rgba(2,6,23,0.78)';
            const Z = '2147483600';
            const rect = (l: number, t: number, w: number, h: number) => {
                const d = document.createElement('div');
                d.setAttribute('data-spot-rect', '1');
                d.style.cssText =
                    `position:fixed;left:${l}px;top:${t}px;width:${Math.max(0, w)}px;height:${Math.max(0, h)}px;` +
                    `background:${dim};z-index:${Z};pointer-events:none;`;
                document.body.appendChild(d);
            };
            rect(0, 0, vw, T);
            rect(0, B, vw, vh - B);
            rect(0, T, L, B - T);
            rect(R, T, vw - R, B - T);
            const ring = document.createElement('div');
            ring.setAttribute('data-spot-rect', '1');
            ring.style.cssText =
                `position:fixed;left:${L}px;top:${T}px;width:${R - L}px;height:${B - T}px;` +
                `z-index:${Z};pointer-events:none;border:3px solid #f5a623;border-radius:12px;` +
                'box-shadow:0 0 22px 4px rgba(245,166,35,0.55);';
            document.body.appendChild(ring);
            resolve(true);
        }, 650); // let the smooth scroll settle
    });
}

function clearSpotlightInPage(): void {
    for (const d of Array.from(document.querySelectorAll('[data-spot-rect="1"]'))) d.remove();
}

async function spotlight(page: Page, selector: string, holdMs: number) {
    let shown = false;
    try {
        shown = await page.evaluate(spotlightInPage, selector);
    } catch {
        shown = false;
    }
    if (!shown) console.warn(`[spotlight] no element for ${selector}`);
    await page.waitForTimeout(Math.max(300, holdMs - 1000));
    try {
        await page.evaluate(clearSpotlightInPage);
    } catch {
        /* page navigated away */
    }
    await page.waitForTimeout(300);
}

async function walkthrough(page: Page, totalMs: number, beats: Array<[string, number]>) {
    const wsum = beats.reduce((s, [, w]) => s + w, 0) || 1;
    for (const [selector, w] of beats) {
        const ms = Math.floor((totalMs * w) / wsum);
        if (!selector) await page.waitForTimeout(ms);
        else await spotlight(page, selector, ms);
    }
}

async function openDetails(page: Page, sectionId: string, testId = 'data-details') {
    const toggle = page.locator(`[data-testid="${testId}-toggle"]`);
    if ((await toggle.count()) > 0 && (await toggle.getAttribute('aria-expanded')) !== 'true') {
        await toggle.click();
        await page.waitForTimeout(400);
    }
    const section = page.locator(`[data-testid="${testId}-section-${sectionId}"]`);
    if ((await section.count()) > 0 && (await section.getAttribute('aria-expanded')) !== 'true') {
        await section.click();
        await page.waitForTimeout(600);
    }
}

// Click a toggle button only when it is not expanded yet.
async function expand(page: Page, selector: string) {
    const el = page.locator(selector).first();
    await expect(el).toBeVisible({ timeout: 15_000 });
    if ((await el.getAttribute('aria-expanded')) !== 'true') {
        await el.click();
        await page.waitForTimeout(600);
    }
}

async function screenshot(page: Page, name: string, fullPage = false) {
    await page.screenshot({ path: `${SCREENSHOT_DIR}/${name}.png`, fullPage });
}

// One beat on a tab/stage of a page: click the tab, let it render, take a
// screenshot, spotlight the first content card for the rest of the section.
async function tabBeat(
    page: Page,
    sectionStart: number,
    seconds: number,
    tabSelector: string,
    contentSelector: string,
    shot: string,
) {
    await page.locator(tabSelector).click();
    await page.waitForTimeout(1200);
    await screenshot(page, shot);
    await spotlight(page, tabSelector, Math.min(3000, remainingMs(sectionStart, seconds) / 3));
    await spotlight(page, contentSelector, remainingMs(sectionStart, seconds));
    await padTo(page, sectionStart, seconds);
}


test('Video 06 — BYO Custom Samples narrated', async ({ page, request }) => {
    test.setTimeout(10 * 60_000);

    expect(fs.existsSync(CSV_PATH), 'BYO sample CSV required').toBeTruthy();
    const projectName = `BYO Coffee Shop FAQ ${Date.now() % 100000}`;

    // JWT for the API-driven upload later.
    const loginResp = await request.post('http://localhost:8000/api/auth/local/login', {
        data: { username: 'admin', password: 'sk-mock-admin-key' },
    });
    expect(loginResp.ok()).toBeTruthy();
    const token = (await loginResp.json()).token as string;
    const authHeader = { Authorization: `Bearer ${token}` };

    // UI login (not narrated)
    await page.goto('/login');
    await page.getByPlaceholder('Enter your username').fill('admin');
    await page.getByPlaceholder('API Key or Password').fill('sk-mock-admin-key');
    await page.getByRole('button', { name: /^Sign in$/ }).click();
    await page.waitForURL((url) => url.pathname === '/', { timeout: 15_000 });
    await page.waitForTimeout(1000);

    // ── Section: cold open (the project list) ────────────────────────
    let sectionStart = Date.now();
    await screenshot(page, 'v06-project-list', true);
    await walkthrough(page, remainingMs(sectionStart, dur.cold_open), [
        ['.demo-project-tiles', 3],
        ['.project-list-toolbar', 1],
    ]);
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: new project (the dialog) ────────────────────────────
    sectionStart = Date.now();
    await page.getByRole('button', { name: /^\+ New Project$/ }).click();
    await expect(page.locator('[data-testid="create-project-name"]')).toBeVisible();
    await page.waitForTimeout(600);
    await page.locator('[data-testid="create-project-name"]').fill(projectName);
    await page
        .locator('[data-testid="create-project-brief"]')
        .fill('Answer customer questions about opening hours, the menu and the loyalty card, using our six-row coffee shop FAQ.');
    await page.waitForTimeout(600);
    await screenshot(page, 'v06-new-project-form');
    await spotlight(page, '.project-list-create-modal', Math.min(6000, remainingMs(sectionStart, dur.new_project) / 2));
    await page.locator('[data-testid="create-project-submit"]').click();
    // Every intake lands on the project's Overview.
    await page.waitForURL(/\/project\/\d+\/?$/, { timeout: 30_000 });
    await expect(page.locator('[data-testid="overview-page"]')).toBeVisible();
    const projectId = (page.url().match(/\/project\/(\d+)/) ?? [])[1] ?? '';
    expect(projectId, 'Could not parse new project id from URL').not.toBe('');
    await spotlight(page, '.overview-hero', remainingMs(sectionStart, dur.new_project));
    await padTo(page, sectionStart, dur.new_project);

    // ── Section: empty Data · Sources ────────────────────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-data"]').click();
    await page.waitForURL(/\/project\/\d+\/data/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="stage-step-data"]')).toBeVisible();
    await page.waitForTimeout(1200);
    await screenshot(page, 'v06-empty-data-tab', true);
    await walkthrough(page, remainingMs(sectionStart, dur.empty_data_tab), [
        ['.stage-page__steps', 1],
        ['.upload-zone', 3],
    ]);
    await padTo(page, sectionStart, dur.empty_data_tab);

    // ── Section: upload the CSV ──────────────────────────────────────
    sectionStart = Date.now();
    await screenshot(page, 'v06-upload-zone');
    // API-driven multipart upload — the call the upload zone makes.
    const csvBuffer = fs.readFileSync(CSV_PATH);
    const uploadResp = await request.post(
        `http://localhost:8000/api/projects/${projectId}/ingestion/upload-batch`,
        {
            headers: authHeader,
            multipart: {
                files: { name: 'byo-coffee-shop-faq.csv', mimeType: 'text/csv', buffer: csvBuffer },
            },
        },
    );
    expect(uploadResp.ok(), 'CSV upload should succeed').toBeTruthy();
    const uploadResult = await uploadResp.json();
    console.log(`[v06] uploaded ${uploadResult.uploaded ?? '?'} file(s)`);

    // Process the document so the CSV rows are extracted.
    const docsResp = await request.get(
        `http://localhost:8000/api/projects/${projectId}/ingestion/documents`,
        { headers: authHeader },
    );
    const docsData = await docsResp.json();
    const docs = Array.isArray(docsData) ? docsData : (docsData.documents ?? docsData.items ?? []);
    if (docs.length > 0) {
        await request.post(
            `http://localhost:8000/api/projects/${projectId}/ingestion/documents/${docs[0].id}/process`,
            { headers: authHeader, data: {} },
        );
    }
    await spotlight(page, '.upload-zone', remainingMs(sectionStart, dur.upload_csv));
    await padTo(page, sectionStart, dur.upload_csv);

    // ── Section: rows imported ───────────────────────────────────────
    sectionStart = Date.now();
    await page.reload();
    await expect(page.locator('[data-testid="stage-step-data"]')).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(1500);
    await openDetails(page, 'documents');
    const firstExpander = page.locator('[data-testid^="expand-doc-"]').first();
    await expect(firstExpander).toBeVisible({ timeout: 15_000 });
    await screenshot(page, 'v06-rows-imported', true);
    await spotlight(page, '#data-details-section-documents', remainingMs(sectionStart, dur.rows_imported));
    await padTo(page, sectionStart, dur.rows_imported);

    // ── Section: wrap (the Overview: stage and next step) ────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-overview"]').click();
    await page.waitForURL(/\/project\/\d+\/?$/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="overview-page"]')).toBeVisible();
    await page.waitForTimeout(800);
    await walkthrough(page, remainingMs(sectionStart, dur.wrap), [
        ['.overview-hero', 3],
        ['[data-testid="nav-data"]', 1],
        ['[data-testid="nav-train"]', 1],
        ['[data-testid="nav-results"]', 1],
        ['[data-testid="nav-ship"]', 1],
    ]);
    await padTo(page, sectionStart, dur.wrap);
});
