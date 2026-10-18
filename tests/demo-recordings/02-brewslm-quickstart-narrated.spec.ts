import { test, expect, type Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 02 — BrewSLM Quickstart, narrated take (Epic U workspace).
//
// login → project list tiles → launch the Support FAQ sample → the Overview
// (stage, next step, the plan) → Data · Sources (expand a document) → Data ·
// Prepare → Data · Labels (the answer key) → Data · Split → Train (the form)
// → wrap. Each section is paced to the duration of its narration WAV
// (tts/generate_narration.py → tts/audio/v02-durations.json) and the final
// video is muxed against the concatenated narration, so the on-screen
// action lines up with the spoken text.
//
// The picture follows the "spotlight + narrate" pattern: the element the
// narration is about is scrolled to the centre, the rest of the page is
// dimmed and the element gets a ring (ported from
// docs-demo/videos/13-pii-success-story/build.py `_spotlight`). No
// continuous scrolling, no static screens.

const DURATIONS_PATH = path.resolve(__dirname, '../../tts/audio/v02-durations.json');

type Durations = {
    cold_open: number;
    login: number;
    tiles: number;
    overview: number;
    sources: number;
    prepare: number;
    labels: number;
    split: number;
    train: number;
    wrap: number;
};

const dur: Durations = JSON.parse(fs.readFileSync(DURATIONS_PATH, 'utf-8'));

const SCREENSHOT_DIR = 'docs-demo/screenshots';

// ── pacing ──────────────────────────────────────────────────────────────

// Pad a section so its wall-clock matches the audio length. If the action
// took longer than the audio we simply do not pad (the audio drifts a tiny
// bit but never cuts off mid-action).
async function padTo(page: Page, sectionStartMs: number, sectionDurationSeconds: number) {
    const remaining = sectionDurationSeconds * 1000 - (Date.now() - sectionStartMs);
    if (remaining > 0) await page.waitForTimeout(remaining);
}

// Milliseconds of the section still to run, less a margin so the spotlight
// is cleared before the next section's action starts.
function remainingMs(sectionStartMs: number, sectionDurationSeconds: number, marginMs = 400): number {
    return Math.max(0, sectionDurationSeconds * 1000 - (Date.now() - sectionStartMs) - marginMs);
}

// ── spotlight ───────────────────────────────────────────────────────────

// Scroll the first element matching `selector` to the viewport centre, dim
// everything else with four fixed rects appended to <body> (a nesting
// wrapper failed to composite reliably under software rendering) and draw
// a ring around it. Returns false when the selector matches nothing.
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

// Spotlight a sequence of anchors across what is left of the section:
// `beats` = [selector, weight]; an empty selector is a plain hold.
async function walkthrough(page: Page, totalMs: number, beats: Array<[string, number]>) {
    const wsum = beats.reduce((s, [, w]) => s + w, 0) || 1;
    for (const [selector, w] of beats) {
        const ms = Math.floor((totalMs * w) / wsum);
        if (!selector) await page.waitForTimeout(ms);
        else await spotlight(page, selector, ms);
    }
}

// Open a Details-drawer section (`data-details-section-<id>`) if it is not
// open yet. The drawer's own toggle must be open first.
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

async function screenshot(page: Page, name: string, fullPage = false) {
    await page.screenshot({ path: `${SCREENSHOT_DIR}/${name}.png`, fullPage });
}

test('Video 02 — narrated take', async ({ page }) => {
    test.setTimeout(8 * 60_000); // total audio ≈ 2m40s + slack

    // ── Section: cold open (hold the login page) ─────────────────────
    let sectionStart = Date.now();
    await page.goto('/login');
    await expect(page.getByPlaceholder('Enter your username')).toBeVisible();
    await screenshot(page, 'v02n-login');
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: login ───────────────────────────────────────────────
    sectionStart = Date.now();
    await page.getByPlaceholder('Enter your username').fill('admin');
    await page.getByPlaceholder('API Key or Password').fill('sk-mock-admin-key');
    await page.getByRole('button', { name: /^Sign in$/ }).click();
    await page.waitForURL((url) => url.pathname === '/', { timeout: 15_000 });
    await padTo(page, sectionStart, dur.login);

    // ── Section: starter-project tiles ───────────────────────────────
    sectionStart = Date.now();
    await expect(page.locator('.demo-project-tiles')).toBeVisible();
    const supportFaqTile = page.locator('[aria-label="Open the Demo · Support FAQ starter project"]');
    await expect(supportFaqTile).toBeVisible();
    await supportFaqTile.hover();
    await screenshot(page, 'v02n-tiles', true);
    await spotlight(page, '.demo-project-tiles', remainingMs(sectionStart, dur.tiles));
    await padTo(page, sectionStart, dur.tiles);

    // ── Section: seed Support FAQ → the Overview ─────────────────────
    sectionStart = Date.now();
    await supportFaqTile.click();
    await page.waitForURL(/\/project\/\d+\/?$/, { timeout: 30_000 });
    await expect(page.locator('[data-testid="overview-page"]')).toBeVisible();
    await page.waitForTimeout(1200);
    const projectId = (page.url().match(/\/project\/(\d+)/) ?? [])[1] ?? '';
    expect(projectId).not.toBe('');
    await screenshot(page, 'v02n-overview');
    // stage → next step → the plan (opened) — the narration names them in
    // that order.
    const overviewPlanToggle = page.locator('[data-testid="overview-plan-toggle"]');
    await walkthrough(page, remainingMs(sectionStart, dur.overview) - 1200, [
        ['[data-testid="overview-stage"]', 1],
        ['[data-testid="overview-next"]', 1],
    ]);
    if ((await overviewPlanToggle.getAttribute('aria-expanded')) !== 'true') {
        await overviewPlanToggle.click();
        await page.waitForTimeout(400);
    }
    await screenshot(page, 'v02n-overview-plan');
    await spotlight(page, '.overview-plan', remainingMs(sectionStart, dur.overview));
    await padTo(page, sectionStart, dur.overview);

    // ── Section: Data · Sources, expand a document ───────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-data"]').click();
    await page.waitForURL(/\/project\/\d+\/data/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="stage-step-data"]')).toBeVisible();
    await page.waitForTimeout(800);
    await spotlight(page, '.stage-page__steps', Math.min(4500, remainingMs(sectionStart, dur.sources) / 3));
    await openDetails(page, 'documents');
    const firstExpander = page.locator('[data-testid^="expand-doc-"]').first();
    await expect(firstExpander).toBeVisible({ timeout: 15_000 });
    await firstExpander.scrollIntoViewIfNeeded();
    await firstExpander.click();
    await page.waitForTimeout(1200);
    await screenshot(page, 'v02n-data-sources-expanded');
    await spotlight(page, '#data-details-section-documents', remainingMs(sectionStart, dur.sources));
    await padTo(page, sectionStart, dur.sources);

    // ── Section: Data · Prepare ──────────────────────────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="stage-step-cleaning"]').click();
    await page.waitForTimeout(1000);
    await screenshot(page, 'v02n-data-prepare');
    await spotlight(page, '.stage-page__content', remainingMs(sectionStart, dur.prepare));
    await padTo(page, sectionStart, dur.prepare);

    // ── Section: Data · Labels (the answer key) ──────────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="stage-step-goldset"]').click();
    await page.waitForTimeout(1000);
    await openDetails(page, 'entries');
    await page.waitForTimeout(600);
    await screenshot(page, 'v02n-data-labels');
    await spotlight(page, '[data-testid="gold-entries-header"]', remainingMs(sectionStart, dur.labels));
    await padTo(page, sectionStart, dur.labels);

    // ── Section: Data · Split ────────────────────────────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="stage-step-dataprep"]').click();
    await page.waitForTimeout(1000);
    await screenshot(page, 'v02n-data-split');
    await spotlight(page, '.stage-page__content .card', remainingMs(sectionStart, dur.split));
    await padTo(page, sectionStart, dur.split);

    // ── Section: Train (the form) ────────────────────────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-train"]').click();
    await page.waitForURL(/\/project\/\d+\/train/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="train-form"]')).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(1500); // readiness preflight runs on mount
    await screenshot(page, 'v02n-train-form');
    // base model → task → seeds → readiness + Train → More options
    await walkthrough(page, remainingMs(sectionStart, dur.train), [
        ['[data-testid="train-form-base-model"]', 2],
        ['[data-testid="train-form-task"]', 2],
        ['[role="radiogroup"][aria-label="Seeds"]', 2],
        ['[data-testid="train-form-readiness"]', 3],
        ['[data-testid="train-form-train"]', 2],
        ['[data-testid="train-form-more"]', 2],
    ]);
    await padTo(page, sectionStart, dur.train);

    // ── Section: wrap (back on the Overview) ─────────────────────────
    sectionStart = Date.now();
    await page.locator('[data-testid="nav-overview"]').click();
    await page.waitForURL(/\/project\/\d+\/?$/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="overview-page"]')).toBeVisible();
    await page.waitForTimeout(800);
    await spotlight(page, '.overview-hero', remainingMs(sectionStart, dur.wrap));
    await padTo(page, sectionStart, dur.wrap);
});
