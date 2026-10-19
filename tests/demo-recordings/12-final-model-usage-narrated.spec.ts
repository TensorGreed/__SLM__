import { test, expect, type Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

// Video 12 — Final Model Usage, narrated (Epic U workspace).
//
// Closes the runtime arc. Takes Video 11's GGUF, registers it with Ollama
// (`ollama create`, in the pre-roll), then sends a prompt through the chat
// playground on Results · Try it and waits for the reply. Pre-roll (not
// narrated): the Ollama registration, login, launch the sample, open
// Results · Try it. Durations come from tts/audio/v12-durations.json
// (tts/generate_v12_narration.py).
//
// Override the GGUF location with V12_GGUF_PATH; the default is where
// Video 11 leaves it under the stack's data directory (DEMO_DATA_DIR — the
// scratch stack's /tmp/brewslm-demo-stack/data; record-narrated.sh sets it).

const DURATIONS_PATH = path.resolve(__dirname, '../../tts/audio/v12-durations.json');

type Durations = {
    cold_open: number;
    ollama_register: number;
    playground_setup: number;
    send_prompt: number;
    response: number;
    wrap: number;
};

const dur: Durations = JSON.parse(fs.readFileSync(DURATIONS_PATH, 'utf-8'));

const OLLAMA_MODEL_ALIAS = 'slm-supportfaq';
const OLLAMA_STAGE_DIR = '/tmp/v12-ollama';

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


test('Video 12 — Final Model Usage narrated', async ({ page }) => {
    test.setTimeout(15 * 60_000);
    const testStart = Date.now();

    // ── Pre-roll (not narrated): log in, open the sample ─────────────
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

    // Stage Video 11's GGUF + a one-line Modelfile and register it with
    // Ollama before the narration starts; the alias is what the video serves.
    // Compression writes a run's GGUF under compressed/exp-<run>/ — take the
    // newest one there (video 11 compresses exactly one run).
    const compressedDir = path.join(
        process.env.DEMO_DATA_DIR ?? path.resolve(__dirname, '../../data'),
        `projects/${projectId}/compressed`,
    );
    const runGgufs = fs.existsSync(compressedDir)
        ? fs.readdirSync(compressedDir)
              .filter((name) => /^exp-\d+$/.test(name))
              .map((name) => path.join(compressedDir, name, 'quantized_4bit.gguf'))
              .filter((file) => fs.existsSync(file))
              .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)
        : [];
    const ggufPath = process.env.V12_GGUF_PATH ?? runGgufs[0] ?? path.join(compressedDir, 'quantized_4bit.gguf');
    expect(fs.existsSync(ggufPath), `Video 11 GGUF artifact required at ${ggufPath}`).toBeTruthy();
    fs.mkdirSync(OLLAMA_STAGE_DIR, { recursive: true });
    fs.copyFileSync(ggufPath, `${OLLAMA_STAGE_DIR}/model.gguf`);
    fs.writeFileSync(`${OLLAMA_STAGE_DIR}/Modelfile`, 'FROM ./model.gguf\n');
    execSync(`ollama create ${OLLAMA_MODEL_ALIAS} -f Modelfile`, { cwd: OLLAMA_STAGE_DIR, encoding: 'utf-8' });
    console.log(`[v12] Ollama model "${OLLAMA_MODEL_ALIAS}" created from GGUF`);

    await page.locator('[data-testid="nav-results"]').click();
    await page.waitForURL(/\/project\/\d+\/results/, { timeout: 15_000 });
    await page.locator('[data-testid="results-tab-try"]').click();
    await expect(page.locator('[data-testid="results-playground"]')).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(1500);

    // The narration starts here; the recorder trims the pre-roll by this offset.
    fs.writeFileSync(
        path.resolve(__dirname, '../../tts/audio/v12-preroll.json'),
        JSON.stringify({ preroll_seconds: (Date.now() - testStart) / 1000 }),
    );

    // ── Section: cold open (Results · Try it) ────────────────────────
    let sectionStart = Date.now();
    await screenshot(page, 'v12-playground-empty', true);
    await walkthrough(page, remainingMs(sectionStart, dur.cold_open), [
        ['[data-testid="results-tab-try"]', 2],
        ['.playground-panel__head', 2],
    ]);
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: Ollama register (narration carries; done in pre-roll)
    sectionStart = Date.now();
    await spotlight(page, '.playground-settings', remainingMs(sectionStart, dur.ollama_register));
    await padTo(page, sectionStart, dur.ollama_register);

    // ── Section: playground setup (provider + model) ─────────────────
    sectionStart = Date.now();
    const providerSelect = page.locator('[data-testid="playground-provider"]');
    await providerSelect.selectOption({ value: 'openai_compatible' });
    await page.waitForTimeout(500);
    const modelInput = page.locator('[data-testid="playground-model"]');
    await expect(modelInput).toBeVisible({ timeout: 10_000 });
    await modelInput.fill(OLLAMA_MODEL_ALIAS);
    await page.waitForTimeout(500);
    await screenshot(page, 'v12-playground-config');
    await walkthrough(page, remainingMs(sectionStart, dur.playground_setup), [
        ['[data-testid="playground-provider"]', 2],
        ['[data-testid="playground-model"]', 2],
    ]);
    await padTo(page, sectionStart, dur.playground_setup);

    // ── Section: send prompt ─────────────────────────────────────────
    sectionStart = Date.now();
    const promptArea = page.locator('textarea[placeholder="Write your prompt..."]').first();
    await expect(promptArea).toBeVisible();
    await promptArea.scrollIntoViewIfNeeded();
    await promptArea.fill('How do I reset my password?');
    await page.waitForTimeout(800);
    await page.getByRole('button', { name: /^Send$/ }).click();
    await spotlight(page, '.playground-composer', remainingMs(sectionStart, dur.send_prompt));
    await padTo(page, sectionStart, dur.send_prompt);

    // ── Section: response (wait for the model to finish) ─────────────
    sectionStart = Date.now();
    await expect(page.locator('.playground-message.assistant').first()).toBeVisible({ timeout: 90_000 });
    await expect(page.getByRole('button', { name: /^Send$/ })).toBeVisible({ timeout: 90_000 }); // reply finished
    await page.waitForTimeout(800);
    await screenshot(page, 'v12-response', true);
    await spotlight(page, '.playground-message.assistant', remainingMs(sectionStart, dur.response));
    await padTo(page, sectionStart, dur.response);

    // ── Section: wrap ────────────────────────────────────────────────
    sectionStart = Date.now();
    await spotlight(page, '.playground-message.assistant', remainingMs(sectionStart, dur.wrap));
    await padTo(page, sectionStart, dur.wrap);
});
