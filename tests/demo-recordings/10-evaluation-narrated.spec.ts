import { test, expect, type Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Video 10 — Evaluation, narrated (Epic U workspace).
//
// Scores the run Video 09 trained (v09-narrated-run) on the practice answer
// key (gold_dev) with the held-out evaluation endpoint, and shows it on
// Results: Verdict (the evaluate form, the summary card, pass/fail rules,
// scorecard) then Failures (sample predictions). Pre-roll (not narrated):
// login + launch the sample, open Results. The eval itself is fired through
// the API so the recording stays deterministic. Durations come from
// tts/audio/v10-durations.json (tts/generate_v10_narration.py).

const DURATIONS_PATH = path.resolve(__dirname, '../../tts/audio/v10-durations.json');

type Durations = {
    cold_open: number;
    setup: number;
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


async function apiAuth(page: Page): Promise<Record<string, string>> {
    const resp = await page.request.post('/api/auth/local/login', {
        data: { username: 'admin', password: 'sk-mock-admin-key' },
    });
    expect(resp.ok()).toBeTruthy();
    return { Authorization: `Bearer ${(await resp.json()).token as string}` };
}

test('Video 10 — Evaluation narrated', async ({ page }) => {
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

    // Video 09's run must still exist; fail early with a clear message.
    const expsResp = await page.request.get(`/api/projects/${projectId}/training/experiments`, { headers: authHeader });
    const exps = (await expsResp.json()) as Array<{ id: number; name: string; status: string }>;
    const v09 = exps.find((e) => e.name === 'v09-narrated-run' && e.status === 'completed');
    expect(v09, 'Video 09 run must exist; run 09-training-run-narrated.spec.ts first').toBeTruthy();
    const experimentId = (v09 as { id: number }).id;

    await page.locator('[data-testid="nav-results"]').click();
    await page.waitForURL(/\/project\/\d+\/results/, { timeout: 15_000 });
    await expect(page.locator('[data-testid="results-tab-verdict"]')).toBeVisible();
    await page.waitForTimeout(2000); // the summary card resolves the run

    // The narration starts here; the recorder trims the pre-roll by this offset.
    fs.writeFileSync(
        path.resolve(__dirname, '../../tts/audio/v10-preroll.json'),
        JSON.stringify({ preroll_seconds: (Date.now() - testStart) / 1000 }),
    );

    // ── Section: cold open (Results · Verdict) ───────────────────────
    let sectionStart = Date.now();
    await screenshot(page, 'v10-results-before', true);
    await walkthrough(page, remainingMs(sectionStart, dur.cold_open), [
        ['.results-page__tabs', 2],
        ['[data-testid="eval-summary"]', 3],
    ]);
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: setup (the Evaluate form) ───────────────────────────
    sectionStart = Date.now();
    const evaluateButton = page.getByRole('button', { name: /\+ Evaluate on test examples/ });
    await expect(evaluateButton).toBeVisible({ timeout: 20_000 });
    await evaluateButton.click();
    const form = page.locator('.eval-panel__card--accent');
    await expect(form).toBeVisible();
    // Show the settings the API call below uses: the practice answer key,
    // scored by exact match.
    await form.locator('select').nth(0).selectOption('gold_dev');
    await form.locator('select').nth(1).selectOption('exact_match');
    await page.waitForTimeout(500);
    await screenshot(page, 'v10-evaluate-form');
    await spotlight(page, '.eval-panel__card--accent', remainingMs(sectionStart, dur.setup));
    await padTo(page, sectionStart, dur.setup);

    // ── Section: kickoff — the held-out evaluation, through the API ──
    sectionStart = Date.now();
    const evalPromise = page.request.post(`/api/projects/${projectId}/evaluation/run-heldout`, {
        headers: authHeader,
        data: {
            experiment_id: experimentId,
            dataset_name: 'gold_dev',
            eval_type: 'exact_match',
            max_samples: 20,
            max_new_tokens: 64,
            temperature: 0.0,
        },
        timeout: 5 * 60_000,
    });
    await spotlight(page, '.eval-panel__card--accent', remainingMs(sectionStart, dur.kickoff));
    await padTo(page, sectionStart, dur.kickoff);

    // ── Section: watching — block on the evaluation ──────────────────
    sectionStart = Date.now();
    const evalResp = await evalPromise;
    expect(evalResp.ok()).toBeTruthy();
    const evalResult = await evalResp.json();
    console.log(
        `[v10] eval result id=${evalResult.id} exact_match=${evalResult.metrics?.exact_match} ` +
            `f1=${evalResult.metrics?.f1} pass_rate=${evalResult.pass_rate}`,
    );
    // Reload so the summary card, the pass/fail rules and the scorecard
    // read the new result.
    await page.goto(`/project/${projectId}/results?tab=verdict`);
    await expect(page.locator('[data-testid="eval-summary"]')).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(1500);
    await spotlight(page, '[data-testid="eval-summary"]', remainingMs(sectionStart, dur.watching));
    await padTo(page, sectionStart, dur.watching);

    // ── Section: results (rules, scorecard, then the predictions) ────
    sectionStart = Date.now();
    await screenshot(page, 'v10-results-after', true);
    await walkthrough(page, remainingMs(sectionStart, dur.results) * 0.55, [
        ['[data-testid="results-auto-gate"]', 2],
        ['#experiment-scorecard', 2],
    ]);
    await page.locator('[data-testid="results-tab-failures"]').click();
    await page.waitForTimeout(1500);
    await screenshot(page, 'v10-results-failures');
    await spotlight(page, '.eval-sample-predictions', remainingMs(sectionStart, dur.results));
    await padTo(page, sectionStart, dur.results);

    // ── Section: wrap (hold on the sample predictions) ───────────────
    sectionStart = Date.now();
    await spotlight(page, '.eval-sample-predictions', remainingMs(sectionStart, dur.wrap));
    await padTo(page, sectionStart, dur.wrap);
});
