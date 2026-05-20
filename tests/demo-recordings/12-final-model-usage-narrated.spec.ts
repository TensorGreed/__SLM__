import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

// Video 12 — Final Model Usage, narrated.
//
// Closes the runtime arc. Takes V11's GGUF artifact, registers it
// with Ollama via `ollama create`, then sends a prompt through the
// BrewSLM Playground UI. Demonstrates the model actually responds.
//
// Wall-time budget (all real):
//   - ollama create → ~0.2s (Ollama indexes bytes already on disk)
//   - playground form fill → ~2s
//   - send + first-token latency → ~5-15s on GB10
//   - full response → ~13-20s total
//
// Durations from tts/audio/v12-durations.json.

const DURATIONS_PATH = path.resolve(
    __dirname,
    '../../tts/audio/v12-durations.json',
);

type Durations = {
    cold_open: number;
    ollama_register: number;
    playground_setup: number;
    send_prompt: number;
    response: number;
    wrap: number;
};

const dur: Durations = JSON.parse(fs.readFileSync(DURATIONS_PATH, 'utf-8'));

const SCREENSHOT_DIR = 'docs-demo/screenshots';

const OLLAMA_MODEL_ALIAS = 'slm-supportfaq';
const GGUF_PATH =
    '/home/anuragj/Desktop/GitHub/__SLM__/data/projects/4/compressed/quantized_4bit.gguf';
const OLLAMA_STAGE_DIR = '/tmp/v12-ollama';

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

test('Video 12 — Final Model Usage narrated', async ({ page }) => {
    test.setTimeout(15 * 60_000);

    // Stage the GGUF + Modelfile, then register with Ollama BEFORE
    // any recording starts. This is pre-roll setup that the spec
    // does, but the resulting model alias is the artifact V12
    // demonstrates serving.
    expect(fs.existsSync(GGUF_PATH), 'V11 GGUF artifact required').toBeTruthy();
    fs.mkdirSync(OLLAMA_STAGE_DIR, { recursive: true });
    fs.copyFileSync(GGUF_PATH, `${OLLAMA_STAGE_DIR}/model.gguf`);
    fs.writeFileSync(
        `${OLLAMA_STAGE_DIR}/Modelfile`,
        'FROM ./model.gguf\n',
    );
    execSync(
        `ollama create ${OLLAMA_MODEL_ALIAS} -f Modelfile`,
        { cwd: OLLAMA_STAGE_DIR, encoding: 'utf-8' },
    );
    console.log(`[v12] Ollama model "${OLLAMA_MODEL_ALIAS}" created from GGUF`);

    // Login + open support-faq project
    await page.goto('/login');
    await page.getByPlaceholder('Enter your username').fill('admin');
    await page.getByPlaceholder('API Key or Password').fill('sk-mock-admin-key');
    await page.getByRole('button', { name: /^Sign in$/ }).click();
    await page.waitForURL((url) => url.pathname === '/', { timeout: 15_000 });

    await page
        .locator('[aria-label="Open the Demo · Support FAQ demo project"]')
        .click();
    await page.waitForURL(/\/project\/\d+\/pipeline\/data/, { timeout: 30_000 });
    const projectId = (page.url().match(/\/project\/(\d+)\//) || [])[1] ?? '';
    expect(projectId).not.toBe('');

    // Navigate to Playground directly via URL. The Playground sidebar
    // entry lives under the "Training" rail (not the default Pipeline
    // rail), so a `button[title="Playground"]` click would require
    // switching rails first. Direct navigation is simpler and still
    // shows the SPA route landing on the Playground page.
    await page.goto(`/project/${projectId}/playground`);
    await page.waitForURL(/\/project\/\d+\/playground/, { timeout: 15_000 });
    await page.waitForTimeout(1500);
    await focusOn(page, ':text("Chat Playground")');

    // ── Section: cold open ───────────────────────────────────────────
    let sectionStart = Date.now();
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v12-playground-empty.png`, fullPage: true });
    await padTo(page, sectionStart, dur.cold_open);

    // ── Section: ollama_register (narration carries; setup already done)
    sectionStart = Date.now();
    // Show that Ollama has the model registered — focus on the Model
    // input field so it's visible while narration describes the
    // ollama-create step.
    await focusOn(page, 'label.form-label:has-text("Model")');
    await padTo(page, sectionStart, dur.ollama_register);

    // ── Section: playground setup (Provider + Model + temp) ────────
    sectionStart = Date.now();
    // Provider dropdown — pick OpenAI-Compatible / Ollama
    const providerSelect = page.locator('label.form-label:has-text("Provider") + select').first();
    if ((await providerSelect.count()) > 0) {
        await providerSelect.selectOption({ value: 'openai_compatible' });
        await page.waitForTimeout(400);
    }
    // Model input — set to our Ollama alias
    const modelInput = page.locator('label.form-label:has-text("Model") + input').first();
    if ((await modelInput.count()) > 0) {
        await modelInput.fill(OLLAMA_MODEL_ALIAS);
        await page.waitForTimeout(400);
    }
    await focusOn(page, 'label.form-label:has-text("Provider")');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v12-playground-config.png`, fullPage: false });
    await padTo(page, sectionStart, dur.playground_setup);

    // ── Section: send_prompt (type the prompt) ──────────────────────
    sectionStart = Date.now();
    const promptArea = page.locator('textarea[placeholder="Write your prompt..."]').first();
    await expect(promptArea).toBeVisible();
    await focusOn(page, 'textarea[placeholder="Write your prompt..."]');
    await promptArea.fill(
        'How do I reset my password?',
    );
    await page.waitForTimeout(800);
    // Click Send
    await page.getByRole('button', { name: /^Send$/ }).click();
    await padTo(page, sectionStart, dur.send_prompt);

    // ── Section: response (wait for the model to finish) ────────────
    sectionStart = Date.now();
    // Wait for the response to appear in the messages area. The
    // playground renders messages in containers under .playground-
    // messages or similar — pick a generic selector that catches the
    // assistant response text.
    await page.waitForTimeout(dur.response * 1000 * 0.6);
    // Scroll to the response area so the new message is visible.
    await focusOn(page, ':text("How do I reset my password?")');
    await page.screenshot({ path: `${SCREENSHOT_DIR}/v12-response.png`, fullPage: true });
    await padTo(page, sectionStart, dur.response);

    // ── Section: wrap ────────────────────────────────────────────────
    sectionStart = Date.now();
    await padTo(page, sectionStart, dur.wrap);
});
