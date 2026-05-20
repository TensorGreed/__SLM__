import { test } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 14 — Architecture, narrated.
//
// Companion slide video to V01. Same shape: static HTML deck loaded
// via file://, slides driven by window.showSlide(n).

const DURATIONS_PATH = path.resolve(
    __dirname,
    '../../tts/audio/v14-durations.json',
);

const SLIDE_DECK_PATH = path.resolve(
    __dirname,
    '../../docs-demo/slides/v14-architecture.html',
);

type Durations = {
    title: number;
    stack: number;
    data_flow: number;
    where_things_run: number;
    trust_boundaries: number;
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
    await page.waitForTimeout(250);
}

test('Video 14 — Architecture narrated', async ({ page }) => {
    test.setTimeout(10 * 60_000);
    await page.goto(`file://${SLIDE_DECK_PATH}`);
    await page.waitForLoadState('domcontentloaded');
    await showSlide(page, 1);

    let sectionStart = Date.now();
    await padTo(page, sectionStart, dur.title);

    sectionStart = Date.now();
    await showSlide(page, 2);
    await padTo(page, sectionStart, dur.stack);

    sectionStart = Date.now();
    await showSlide(page, 3);
    await padTo(page, sectionStart, dur.data_flow);

    sectionStart = Date.now();
    await showSlide(page, 4);
    await padTo(page, sectionStart, dur.where_things_run);

    sectionStart = Date.now();
    await showSlide(page, 5);
    await padTo(page, sectionStart, dur.trust_boundaries);

    sectionStart = Date.now();
    await showSlide(page, 6);
    await padTo(page, sectionStart, dur.wrap);
});
