import type { Locator, Page } from '@playwright/test';

/**
 * Main-path counter for the golden-path E2E (UI simplification pass, phase 0).
 *
 * Counting rule (the same one the UI simplification plan and the Model 6 v2
 * run use):
 *  - An ACTION is one user input on the first successful path: a click, a
 *    fill, a select, a check or a file pick, made through `counter.click(...)`
 *    / `fill` / `select` / `check` / `setFiles`.
 *  - NOT counted: navigation (`page.goto`, waits), retries, opening an
 *    "Advanced" / "More options" disclosure (pass `{ uncounted: 'advanced' }`),
 *    and setup that is not part of the product path such as logging in (pass
 *    `{ uncounted: 'setup' }`). Calls made through `page.request` (API) are
 *    not user actions and are not counted either.
 *  - A SCREEN is a distinct route the user sees after `counter.start()`: the
 *    pathname (ids folded to `:id`) plus the `step` / `tab` / `stage` query
 *    value. `/login` and anything before `start()` are setup, not screens.
 *
 * The budget lives in `pathBudget.ts`; later phases lower it.
 */

export type Uncounted = 'advanced' | 'setup' | 'navigation';

export interface CountOptions {
    /** Why this input is not part of the counted path. */
    uncounted?: Uncounted;
}

export function screenKey(rawUrl: string): string {
    const url = new URL(rawUrl, 'http://local');
    const path = url.pathname.replace(/\/\d+(?=\/|$)/g, '/:id');
    const parts = ['step', 'tab', 'stage']
        .map((name) => url.searchParams.get(name))
        .filter((value): value is string => Boolean(value));
    return parts.length ? `${path}?${parts.join('&')}` : path;
}

export interface PathCounter {
    /** Begin counting screens (call after setup such as login). */
    start(): void;
    click(locator: Locator, options?: CountOptions): Promise<void>;
    fill(locator: Locator, value: string, options?: CountOptions): Promise<void>;
    select(locator: Locator, value: string, options?: CountOptions): Promise<void>;
    check(locator: Locator, options?: CountOptions): Promise<void>;
    setFiles(locator: Locator, files: Parameters<Locator['setInputFiles']>[0], options?: CountOptions): Promise<void>;
    /** Snapshot, including the screen currently shown. */
    finish(): { actions: number; screens: string[]; log: string[] };
}

export function createPathCounter(page: Page): PathCounter {
    let started = false;
    let actions = 0;
    const screens: string[] = [];
    const log: string[] = [];

    const addScreen = (url: string) => {
        if (!started) return;
        const key = screenKey(url);
        if (key === '/login' || screens.includes(key)) return;
        screens.push(key);
    };
    page.on('framenavigated', (frame) => {
        if (frame === page.mainFrame()) addScreen(frame.url());
    });

    const note = (kind: string, locator: Locator, options?: CountOptions) => {
        if (options?.uncounted) {
            log.push(`(${options.uncounted}) ${kind}`);
            return;
        }
        actions += 1;
        log.push(`${actions}. ${kind} ${locator.toString()}`);
    };

    return {
        start() {
            started = true;
            addScreen(page.url());
        },
        async click(locator, options) {
            note('click', locator, options);
            await locator.click();
        },
        async fill(locator, value, options) {
            note('fill', locator, options);
            await locator.fill(value);
        },
        async select(locator, value, options) {
            note('select', locator, options);
            await locator.selectOption(value);
        },
        async check(locator, options) {
            note('check', locator, options);
            await locator.check();
        },
        async setFiles(locator, files, options) {
            note('files', locator, options);
            await locator.setInputFiles(files);
        },
        finish() {
            addScreen(page.url());
            return { actions, screens: [...screens], log: [...log] };
        },
    };
}
