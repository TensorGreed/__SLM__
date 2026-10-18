import { test, expect, type Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 05 — Sentiment Classifier workspace walkthrough, narrated (Epic U
// workspace).
//
// Overview (cold open) → Data · Sources → Data · Labels (the answer key's
// label mix) → Data · Split → Train (More options: Tokenization, the mobile
// target) → Train (the form; LoRA and advanced training) → Results (what the
// classification checks report) → Ship · Package (compress first, then the
// export form — ONNX with eight-bit quantization is the natural target for
// the mobile CPU profile) → wrap. Cleaning and Generate are not their own
// sections (videos 03 and 04 cover them). No training, no generation.
// Audio durations come from tts/audio/v05-durations.json
// (tts/generate_v05_narration.py).
//
// Picture: the "spotlight + narrate" pattern (see video 03).


const DURATIONS_PATH = path.resolve(__dirname, '../../tts/audio/v05-durations.json');

type Durations = {
    cold_open: number;
    data: number;
    goldset: number;
    dataprep: number;
    tokenization: number;
    training_config: number;
    eval: number;
    compression_export: number;
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


test('Video 05 — Sentiment Classifier workspace narrated', async ({ page }) => {
    test.setTimeout(12 * 60_000);

    // Pre-roll: log in, launch the sample (not narrated).
    await page.goto('/login');
    await page.getByPlaceholder('Enter your username').fill('admin');
    await page.getByPlaceholder('API Key or Password').fill('sk-mock-admin-key');
    await page.getByRole('button', { name: /^Sign in$/ }).click();
    await page.waitForURL((url) => url.pathname === '/', { timeout: 15_000 });
    await page.locator('[aria-label="Open the Demo · Sentiment classifier starter project"]').click();
    await page.waitForURL(/\/project\/\d+\/?$/, { timeout: 30_000 });
    await expect(page.locator('[data-testid="overview-page"]')).toBeVisible();
    const projectId = (page.url().match(/\/project\/(\d+)/) ?? [])[1] ?? '';
    expect(projectId).not.toBe('');

    // ── Section: cold open (the Overview) ────────────────────────────
    let sectionStart = Date.now();
    await page.waitForTimeout(1000);
    await screenshot(page, 'v05-overview');
    await walkthrough(page, remainingMs(sectionStart, dur.cold_open), [
        ['.overview-hero', 3],
        ['[data-testid="nav-data"]', 1],
        ['[data-testid="nav-train"]', 1],
        ['[data-testid="nav-results"]', 1],
        ['[data-testid="nav-ship"]', 1],
    ]);
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: Data · Sources (expand a document) ──────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-data"]').click();
    await page.waitForURL(/\/project\/\d+\/data/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="stage-step-data"]')).toBeVisible();
    await page.waitForTimeout(800);
    await spotlight(page, '.stage-page__steps', Math.min(3500, remainingMs(sectionStart, dur.data) / 3));
    await openDetails(page, 'documents');
    const firstExpander = page.locator('[data-testid^="expand-doc-"]').first();
    await expect(firstExpander).toBeVisible({ timeout: 15_000 });
    await firstExpander.scrollIntoViewIfNeeded();
    await firstExpander.click();
    await page.waitForTimeout(1200);
    await screenshot(page, 'v05-expanded-row');
    await spotlight(page, '#data-details-section-documents', remainingMs(sectionStart, dur.data));
    await padTo(page, sectionStart, dur.data);

    // ── Section: Data · Labels (the answer key's label mix) ──────────
    sectionStart = Date.now();
    await page.locator('[data-testid="stage-step-goldset"]').click();
    await page.waitForTimeout(1000);
    await openDetails(page, 'entries');
    await page.waitForTimeout(600);
    await screenshot(page, 'v05-goldset');
    await walkthrough(page, remainingMs(sectionStart, dur.goldset), [
        ['[data-testid="gold-entries-header"]', 3],
        ['[data-testid="gold-entries-mix-summary"]', 2],
    ]);
    await padTo(page, sectionStart, dur.goldset);

    // ── Section: Data · Split (labels and the three splits) ──────────
    sectionStart = Date.now();
    await page.locator('[data-testid="stage-step-dataprep"]').click();
    await page.waitForTimeout(1200);
    await screenshot(page, 'v05-schema-profile');
    await walkthrough(page, remainingMs(sectionStart, dur.dataprep), [
        ['.stage-page__content .card', 3],
        ['[data-testid="split-keep-test-label"]', 1],
    ]);
    await padTo(page, sectionStart, dur.dataprep);

    // ── Section: Train · More options · Tokenization (mobile angle) ──
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-train"]').click();
    await page.waitForURL(/\/project\/\d+\/train/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="train-form"]')).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(1200);
    await expand(page, '[data-testid="train-form-more"]');
    await expect(page.locator('[data-testid="train-section-tokenization"]')).toBeVisible({ timeout: 15_000 });
    await spotlight(page, '.train-form__sections', Math.min(3500, remainingMs(sectionStart, dur.tokenization) / 3));
    await expand(page, '[data-testid="train-section-tokenization"]');
    await screenshot(page, 'v05-tokenization');
    await spotlight(page, '#train-section-tokenization', remainingMs(sectionStart, dur.tokenization));
    await padTo(page, sectionStart, dur.tokenization);

    // ── Section: Train (the form; LoRA and advanced training) ────────
    sectionStart = Date.now();
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior }));
    await page.waitForTimeout(500);
    await screenshot(page, 'v05-train-form');
    await spotlight(page, '[data-testid="train-form-task"]', Math.min(4000, remainingMs(sectionStart, dur.training_config) / 3));
    await expand(page, '[data-testid="train-section-lora"]');
    await screenshot(page, 'v05-training-config-mobile');
    await spotlight(page, '#train-section-lora', remainingMs(sectionStart, dur.training_config));
    await padTo(page, sectionStart, dur.training_config);

    // ── Section: Results (classification checks) ─────────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-results"]').click();
    await page.waitForURL(/\/project\/\d+\/results/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="results-tab-verdict"]')).toBeVisible();
    await tabBeat(page, sectionStart, dur.eval, '[data-testid="results-tab-verdict"]', '.results-page .card', 'v05-results-verdict');

    // ── Section: Ship · Package (compress first, then export) ────────
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-ship"]').click();
    await page.waitForURL(/\/project\/\d+\/ship/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="ship-stage-package"]')).toBeVisible();
    await page.waitForTimeout(1000);
    await openDetails(page, 'compress', 'ship-details');
    await screenshot(page, 'v05-compression');
    await spotlight(page, '#ship-details-section-compress', Math.min(5000, remainingMs(sectionStart, dur.compression_export) / 2));
    await screenshot(page, 'v05-export-onnx', true);
    await spotlight(page, '[data-testid="export-form-card"]', remainingMs(sectionStart, dur.compression_export));
    await padTo(page, sectionStart, dur.compression_export);

    // ── Section: wrap (hold on the Ship page, then the Overview) ─────
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
