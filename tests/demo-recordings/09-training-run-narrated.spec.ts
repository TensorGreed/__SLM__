import { test, expect, type Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 09 — Training Run, narrated (Epic U workspace).
//
// First runtime-dependent video. Drives a real training run on the support-faq
// sample (SmolLM2-135M, 2 epochs, ~12 s on the GB10) through the real Celery
// worker. Pre-roll (not narrated): login + launch the sample, open Train.
// Then: the Train form (cold open) → More options (model + hyperparameters,
// LoRA) → the run is created + started through the API (a fixed config keeps
// the recording deterministic; the form sizes epochs itself) → the Runs list
// while it trains → the finished run's dashboard → wrap. Durations come from
// tts/audio/v09-durations.json (tts/generate_v09_narration.py).

const DURATIONS_PATH = path.resolve(__dirname, '../../tts/audio/v09-durations.json');

type Durations = {
    cold_open: number;
    config_recap: number;
    kickoff: number;
    watching: number;
    results: number;
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


// Log in through the API with the same local login the UI uses; page.request
// resolves relative URLs against the configured baseURL (the vite proxy), so
// the spec works against any backend the frontend points at.
async function apiAuth(page: Page): Promise<Record<string, string>> {
    const resp = await page.request.post('/api/auth/local/login', {
        data: { username: 'admin', password: 'sk-mock-admin-key' },
    });
    expect(resp.ok()).toBeTruthy();
    return { Authorization: `Bearer ${(await resp.json()).token as string}` };
}

test('Video 09 — Training Run narrated', async ({ page }) => {
    test.setTimeout(10 * 60_000);
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

    await page.locator('[data-testid="nav-train"]').click();
    await page.waitForURL(/\/project\/\d+\/train/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="train-form"]')).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(1500); // readiness preflight runs on mount

    // The narration starts here; the recorder trims the pre-roll by this offset.
    fs.writeFileSync(
        path.resolve(__dirname, '../../tts/audio/v09-preroll.json'),
        JSON.stringify({ preroll_seconds: (Date.now() - testStart) / 1000 }),
    );

    // ── Section: cold open (the Train form) ──────────────────────────
    let sectionStart = Date.now();
    await screenshot(page, 'v09-train-form', true);
    await walkthrough(page, remainingMs(sectionStart, dur.cold_open), [
        ['[data-testid="train-form-base-model"]', 2],
        ['[data-testid="train-form-readiness"]', 2],
        ['[data-testid="train-form-train"]', 1],
        ['.training-panel-title', 2],
    ]);
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: config recap (More options) ─────────────────────────
    sectionStart = Date.now();
    await expand(page, '[data-testid="train-form-more"]');
    await expect(page.locator('[data-testid="train-section-model"]')).toBeVisible({ timeout: 15_000 });
    await spotlight(page, '.train-form__sections', Math.min(3500, remainingMs(sectionStart, dur.config_recap) / 4));
    await expand(page, '[data-testid="train-section-model"]');
    await screenshot(page, 'v09-config-hyperparameters');
    await spotlight(page, '#train-section-model', remainingMs(sectionStart, dur.config_recap) * 0.55);
    await expand(page, '[data-testid="train-section-lora"]');
    await screenshot(page, 'v09-config-lora');
    await spotlight(page, '#train-section-lora', remainingMs(sectionStart, dur.config_recap));
    await padTo(page, sectionStart, dur.config_recap);

    // ── Section: kickoff — create + start through the API ────────────
    // The Runs list loads once on mount, so the run is created while the
    // viewer is still on the form and the page is reloaded afterwards: the
    // new row is already in the list the moment the narration says so.
    sectionStart = Date.now();
    const createResp = await page.request.post(`/api/projects/${projectId}/training/experiments`, {
        headers: authHeader,
        data: {
            name: 'v09-narrated-run',
            description: 'Video 09 narrated training run on support-faq.',
            config: {
                base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
                num_epochs: 2,
                batch_size: 2,
                gradient_accumulation_steps: 1,
                optimizer: 'adamw_torch',
                max_seq_length: 512,
                sequence_packing: false,
                warmup_ratio: 0.0,
            },
        },
    });
    expect(createResp.ok()).toBeTruthy();
    const expId = (await createResp.json()).id as number;
    const startResp = await page.request.post(`/api/projects/${projectId}/training/experiments/${expId}/start`, {
        headers: authHeader,
        data: {},
    });
    expect(startResp.ok()).toBeTruthy();

    await page.reload();
    await expect(page.locator('[data-testid="train-form"]')).toBeVisible({ timeout: 20_000 });
    await page.locator('.training-experiment-item').first().waitFor({ state: 'visible', timeout: 20_000 });
    await screenshot(page, 'v09-training-running', true);
    await spotlight(page, '.training-experiment-item', remainingMs(sectionStart, dur.kickoff));
    await padTo(page, sectionStart, dur.kickoff);

    // ── Section: watching — status until completed ───────────────────
    sectionStart = Date.now();
    await spotlight(page, '.training-experiment-item', Math.min(4000, remainingMs(sectionStart, dur.watching) / 2));
    const deadlineMs = sectionStart + dur.watching * 1000 + 60_000; // slack for a slower box
    let finalStatus = 'unknown';
    while (Date.now() < deadlineMs) {
        const statusResp = await page.request.get(
            `/api/projects/${projectId}/training/experiments/${expId}/status`,
            { headers: authHeader },
        );
        if (statusResp.ok()) {
            finalStatus = (await statusResp.json()).status as string;
            if (['completed', 'failed', 'cancelled'].includes(finalStatus)) break;
        }
        await page.waitForTimeout(2000);
    }
    expect(finalStatus).toBe('completed');
    // The Runs list is loaded once; reload so the row shows the final badge.
    await page.reload();
    await page.locator('.training-experiment-item').first().waitFor({ state: 'visible', timeout: 20_000 });
    await spotlight(page, '.training-experiment-item', remainingMs(sectionStart, dur.watching));
    await padTo(page, sectionStart, dur.watching);

    // ── Section: results (the run's dashboard) ───────────────────────
    sectionStart = Date.now();
    await page.locator('.training-experiment-item').first().getByRole('button', { name: /^Dashboard$/ }).click();
    await expect(page.locator('.metrics-grid')).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(1200);
    await screenshot(page, 'v09-training-completed', true);
    await walkthrough(page, remainingMs(sectionStart, dur.results), [
        ['.training-active-head', 2],
        ['.metrics-grid', 3],
    ]);
    await padTo(page, sectionStart, dur.results);

    // ── Section: wrap (back to the Runs list) ────────────────────────
    sectionStart = Date.now();
    await page.getByRole('button', { name: /Back to Experiments/ }).click();
    await page.locator('.training-experiment-item').first().waitFor({ state: 'visible', timeout: 15_000 });
    await spotlight(page, '.training-experiment-item', remainingMs(sectionStart, dur.wrap));
    await padTo(page, sectionStart, dur.wrap);
});
