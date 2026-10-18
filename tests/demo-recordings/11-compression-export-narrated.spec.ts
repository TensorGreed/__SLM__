import { test, expect, type Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 11 — Compression + Export, narrated (Epic U workspace).
//
// Takes the LoRA adapter from Video 09's run, merges it into the
// SmolLM2-135M base, quantizes the merged weights to GGUF (llama.cpp), then
// registers the artifact as an export. All of it happens on Ship · Package:
// the export form on top, "Compress first" under the Options drawer. The
// merge, quantize and export calls are fired through the API (the UI forms
// only fill a result card when the panel itself fires the request).
// Pre-roll (not narrated): login + launch the sample, open Ship.
//
// Wall time: merge ~12 s, quantize ~5 s, export create + run ~3 s on the
// GB10. Durations come from tts/audio/v11-durations.json
// (tts/generate_v11_narration.py).

const DURATIONS_PATH = path.resolve(__dirname, '../../tts/audio/v11-durations.json');

type Durations = {
    cold_open: number;
    compression_setup: number;
    compress_run: number;
    compression_result: number;
    export_create: number;
    export_run: number;
    wrap: number;
};

const dur: Durations = JSON.parse(fs.readFileSync(DURATIONS_PATH, 'utf-8'));

const SCREENSHOT_DIR = 'docs-demo/screenshots';

// ── pacing ──────────────────────────────────────────────────────────────

// DEMO_STRICT_BEATS=1 (a check run, not a recording): a beat whose element is
// missing fails the run instead of logging a warning and holding on nothing.
const STRICT_BEATS = process.env.DEMO_STRICT_BEATS === '1';

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
    if (!shown) {
        if (STRICT_BEATS) throw new Error(`[spotlight] no element for ${selector}`);
        console.warn(`[spotlight] no element for ${selector}`);
    }
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
    if (STRICT_BEATS && (await section.count()) === 0) {
        throw new Error(`[openDetails] no ${testId} section "${sectionId}"`);
    }
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


async function apiAuth(page: Page): Promise<Record<string, string>> {
    const resp = await page.request.post('/api/auth/local/login', {
        data: { username: 'admin', password: 'sk-mock-admin-key' },
    });
    expect(resp.ok()).toBeTruthy();
    return { Authorization: `Bearer ${(await resp.json()).token as string}` };
}

// Poll a compression worker task until it finishes (or the budget runs out).
async function waitForTask(page: Page, projectId: string, taskId: string, headers: Record<string, string>, budgetMs: number) {
    const deadline = Date.now() + budgetMs;
    while (Date.now() < deadline) {
        const r = await page.request.get(`/api/projects/${projectId}/compression/jobs/tasks/${taskId}`, { headers });
        if (r.ok()) {
            const s = String((await r.json()).state ?? '').toLowerCase();
            if (s === 'success' || s === 'failure') return s;
        }
        await page.waitForTimeout(2000);
    }
    return 'timeout';
}

test('Video 11 — Compression + Export narrated', async ({ page }) => {
    test.setTimeout(15 * 60_000);
    const testStart = Date.now();

    // ── Pre-roll (not narrated) ──────────────────────────────────────
    const authHeader = await apiAuth(page);
    await page.goto('/login');
    await page.getByPlaceholder('Enter your username').fill('admin');
    await page.getByPlaceholder('API Key or Password').fill('sk-mock-admin-key');
    await page.getByRole('button', { name: /^Sign in$/ }).click();
    await page.waitForURL((url) => url.pathname === '/', { timeout: 15_000 });
    await page.locator('[aria-label="Open the Demo · Support FAQ starter project"]').click();
    await page.waitForURL(/\/project\/\d+\/?$/, { timeout: 30_000 });
    await expect(page.locator('[data-testid="overview-page"]')).toBeVisible();
    const projectId = (page.url().match(/\/project\/(\d+)/) ?? [])[1] ?? '';
    expect(projectId).not.toBe('');

    // Video 09's run supplies the adapter.
    const expsResp = await page.request.get(`/api/projects/${projectId}/training/experiments`, { headers: authHeader });
    const exps = (await expsResp.json()) as Array<{ id: number; name: string; status: string; output_dir: string }>;
    const v09 = exps.find((e) => e.name === 'v09-narrated-run' && e.status === 'completed');
    expect(v09, 'Video 09 run required; run 09-training-run-narrated.spec.ts first').toBeTruthy();
    const experimentId = (v09 as { id: number }).id;
    const adapterPath = (v09 as { output_dir: string }).output_dir + '/model';

    await page.locator('[data-testid="nav-ship"]').click();
    await page.waitForURL(/\/project\/\d+\/ship/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="ship-stage-package"]')).toBeVisible();
    await expect(page.locator('[data-testid="export-form-card"]')).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(1000);

    // The narration starts here; the recorder trims the pre-roll by this offset.
    fs.writeFileSync(
        path.resolve(__dirname, '../../tts/audio/v11-preroll.json'),
        JSON.stringify({ preroll_seconds: (Date.now() - testStart) / 1000 }),
    );

    // ── Section: cold open (Ship · Package) ──────────────────────────
    let sectionStart = Date.now();
    await screenshot(page, 'v11-ship-package', true);
    await walkthrough(page, remainingMs(sectionStart, dur.cold_open), [
        ['.stage-page__steps', 2],
        ['[data-testid="export-form-card"]', 2],
        ['[data-testid="ship-details"]', 2],
    ]);
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: compression setup (Options · Compress first) ────────
    sectionStart = Date.now();
    await openDetails(page, 'compress', 'ship-details');
    await expect(page.locator('#compression-run-picker')).toBeVisible({ timeout: 15_000 });
    // Picking the run fills the model and adapter paths from it.
    await page.locator('#compression-run-picker').selectOption(String(experimentId));
    await page.waitForTimeout(800);
    const loraInput = page.locator('label.form-label:has-text("LoRA Adapter Path") + input.input').first();
    if ((await loraInput.inputValue()) === '') await loraInput.fill(adapterPath);
    await screenshot(page, 'v11-compression-form');
    await walkthrough(page, remainingMs(sectionStart, dur.compression_setup), [
        ['#compression-run-picker', 2],
        ['#ship-details-section-compress', 3],
    ]);
    await padTo(page, sectionStart, dur.compression_setup);

    // ── Section: compress run — merge, then quantize, via the API ────
    sectionStart = Date.now();
    const mergeResp = await page.request.post(`/api/projects/${projectId}/compression/merge-lora`, {
        headers: authHeader,
        data: { base_model_path: 'HuggingFaceTB/SmolLM2-135M-Instruct', lora_adapter_path: adapterPath },
    });
    expect(mergeResp.ok()).toBeTruthy();
    const mergeJob = await mergeResp.json();
    await waitForTask(page, projectId, mergeJob.task_id as string, authHeader, 60_000);

    const quantResp = await page.request.post(`/api/projects/${projectId}/compression/quantize`, {
        headers: authHeader,
        data: { model_path: mergeJob.output_model_path as string, bits: 4, output_format: 'gguf' },
    });
    expect(quantResp.ok()).toBeTruthy();
    const quantJob = await quantResp.json();
    await waitForTask(page, projectId, quantJob.task_id as string, authHeader, 60_000);
    await spotlight(page, '#ship-details-section-compress', Math.max(1500, remainingMs(sectionStart, dur.compress_run)));
    await padTo(page, sectionStart, dur.compress_run);

    // ── Section: compression result ──────────────────────────────────
    // The Result card fills only for a run started from the panel itself;
    // the files are on disk regardless, so the picture stays on the form
    // while the narration carries the sizes.
    sectionStart = Date.now();
    await spotlight(page, '#ship-details-section-compress', remainingMs(sectionStart, dur.compression_result));
    await padTo(page, sectionStart, dur.compression_result);

    // ── Section: export create (the New export form) ─────────────────
    sectionStart = Date.now();
    const exportForm = page.locator('[data-testid="export-form-card"]');
    await exportForm.locator('select').nth(0).selectOption(String(experimentId));
    await exportForm.locator('select').nth(1).selectOption('gguf');
    await page.waitForTimeout(600);
    await screenshot(page, 'v11-export-form');
    await spotlight(page, '[data-testid="export-form-card"]', remainingMs(sectionStart, dur.export_create));
    await padTo(page, sectionStart, dur.export_create);

    // ── Section: export run — create + run via the API ───────────────
    sectionStart = Date.now();
    const exportCreateResp = await page.request.post(`/api/projects/${projectId}/export/create`, {
        headers: authHeader,
        data: { experiment_id: experimentId, export_format: 'gguf', quantization: 'Q4_K_M' },
    });
    expect(exportCreateResp.ok()).toBeTruthy();
    const exportId = (await exportCreateResp.json()).id as number;
    const exportRunResp = await page.request.post(`/api/projects/${projectId}/export/${exportId}/run`, {
        headers: authHeader,
        data: { run_smoke_tests: false },
    });
    // The run may fail validation when the GGUF is not placed in the export
    // directory; the recording's value is the Releases surface, not success.
    console.log(`[v11] export create id=${exportId} ok=${exportRunResp.ok()}`);
    await page.goto(`/project/${projectId}/ship?stage=package`);
    await expect(page.locator('[data-testid="export-releases"]')).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(1000);
    await screenshot(page, 'v11-export-releases', true);
    await walkthrough(page, remainingMs(sectionStart, dur.export_run), [
        ['[data-testid="export-releases"]', 2],
        ['.export-history-table', 3],
    ]);
    await padTo(page, sectionStart, dur.export_run);

    // ── Section: wrap ────────────────────────────────────────────────
    sectionStart = Date.now();
    await spotlight(page, '[data-testid="export-releases"]', remainingMs(sectionStart, dur.wrap));
    await padTo(page, sectionStart, dur.wrap);
});
