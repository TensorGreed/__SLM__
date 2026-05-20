import { test } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 01 — SLM 101, narrated.
//
// Slide-based conceptual intro. No product UI. The deck is a static
// HTML file at docs-demo/slides/v01-slm-101.html. The spec loads it
// via the file:// protocol and advances slides by calling
// window.showSlide(n) from page.evaluate at each section boundary.
//
// Durations from tts/audio/v01-durations.json.

const DURATIONS_PATH = path.resolve(
    __dirname,
    '../../tts/audio/v01-durations.json',
);

const SLIDE_DECK_PATH = path.resolve(
    __dirname,
    '../../docs-demo/slides/v01-slm-101.html',
);

type Durations = {
    title: number;
    what_is_slm: number;
    why_matter: number;
    lifecycle: number;
    brewslm_fits: number;
    wrap: number;
};

const dur: Durations = JSON.parse(fs.readFileSync(DURATIONS_PATH, 'utf-8'));

async function padTo(
    page: import('@playwright/test').Page,
    sectionStartMs: number,
    sectionDurationSeconds: number,
) {
    const remaining = sectionDurationSeconds * 1000 - (Date.now() - sectionStartMs);
    if (remaining > 0) await page.waitForTimeout(remaining);
}

async function showSlide(
    page: import('@playwright/test').Page,
    n: number,
) {
    await page.evaluate((slideNum) => {
        // @ts-expect-error window.showSlide is defined in the deck.
        window.showSlide(slideNum);
    }, n);
    await page.waitForTimeout(250); // settle the opacity transition
}

test('Video 01 — SLM 101 narrated', async ({ page }) => {
    test.setTimeout(10 * 60_000);

    // Load the slide deck. Use file:// — no backend needed.
    await page.goto(`file://${SLIDE_DECK_PATH}`);
    await page.waitForLoadState('domcontentloaded');
    // Show the title slide explicitly to be safe (deck initial state).
    await showSlide(page, 1);

    // ── Section: title (slide 1) ─────────────────────────────────────
    let sectionStart = Date.now();
    await padTo(page, sectionStart, dur.title);

    // ── Section: what is an SLM? (slide 2) ───────────────────────────
    sectionStart = Date.now();
    await showSlide(page, 2);
    await padTo(page, sectionStart, dur.what_is_slm);

    // ── Section: why SLMs matter (slide 3) ───────────────────────────
    sectionStart = Date.now();
    await showSlide(page, 3);
    await padTo(page, sectionStart, dur.why_matter);

    // ── Section: lifecycle (slide 4) ─────────────────────────────────
    sectionStart = Date.now();
    await showSlide(page, 4);
    await padTo(page, sectionStart, dur.lifecycle);

    // ── Section: where BrewSLM fits (slide 5) ────────────────────────
    sectionStart = Date.now();
    await showSlide(page, 5);
    await padTo(page, sectionStart, dur.brewslm_fits);

    // ── Section: wrap (slide 6) ──────────────────────────────────────
    sectionStart = Date.now();
    await showSlide(page, 6);
    await padTo(page, sectionStart, dur.wrap);
});
