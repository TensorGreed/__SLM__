import type { Locator, Page } from '@playwright/test';

/**
 * Main-path counter for the golden-path E2E (UI simplification pass, phase 0).
 *
 * Counting rule (the same one the UI simplification plan and the 2026-11-19
 * UI-only reference run use):
 *  - An ACTION is one user input on the first successful path: a click, a
 *    fill, a select, a check or a file pick, made through `counter.click(...)`
 *    / `fill` / `select` / `check` / `setFiles`.
 *  - NOT counted: waits and polling, retries, opening an "Advanced" /
 *    "More options" disclosure (pass `{ uncounted: 'advanced' }`), setup that
 *    is not part of the product path such as logging in (`'setup'`), and
 *    NAVIGATION (`'navigation'`). Navigation means moving between the
 *    workspace destinations: `page.goto`, and the sidebar / top-bar links
 *    (Overview, Data, Train, Results, Ship, Settings, the project list).
 *    Switching a step, tab or stage INSIDE a page (Data steps, Results tabs,
 *    Ship stages) is a counted click. Calls made through `page.request` (API)
 *    are not user actions and are not counted either.
 *  - A SCREEN is a distinct view the user sees after `counter.start()`: the
 *    pathname (numeric ids and `annotate/<job>` folded to `:id`) plus the
 *    `step` / `tab` / `stage` value, where the page's default value folds to
 *    the bare path (so `/results` and `/results?tab=verdict` are one view).
 *    `/login`, anything before `start()`, and the transient URLs of legacy
 *    redirects are not screens.
 *
 * The budget lives in `pathBudget.ts`; later phases lower it.
 */

export type Uncounted = 'advanced' | 'setup' | 'navigation';

export interface CountOptions {
    /** Why this input is not part of the counted path. */
    uncounted?: Uncounted;
}

/** Default value of the view parameter per destination (folds to the bare path). */
const DEFAULT_VIEW: Record<string, { param: string; value: string }> = {
    data: { param: 'step', value: 'data' },
    results: { param: 'tab', value: 'verdict' },
    ship: { param: 'stage', value: 'package' },
};

/** Routes that only redirect (LegacyRouteRedirect); their URLs are never a screen. */
const LEGACY_SEGMENTS = new Set([
    'workflow', 'domain', 'autopilot', 'extensions', 'guide', 'wizard', 'pipeline', 'data-studio', 'annotate-list',
    'recipes', 'recipe-picker', 'training-config', 'models', 'adapter-studio', 'playground', 'deployments',
    'observability', 'manifest',
]);

/** Null when the URL is not a screen (login, a legacy redirect bounce). */
export function screenKey(rawUrl: string): string | null {
    const url = new URL(rawUrl, 'http://local');
    if (url.pathname === '/login') return null;
    const segments = url.pathname.split('/').filter(Boolean);
    if (segments[0] === 'project' && segments[2]) {
        const second = segments[2];
        const isAnnotateList = second === 'annotate' && !segments[3];
        const isEvalCompare = second === 'eval' && segments[3] === 'compare';
        if (LEGACY_SEGMENTS.has(second) || isAnnotateList || isEvalCompare) return null;
    }
    const folded = segments.map((segment, index) => (/^\d+$/.test(segment) || segments[index - 1] === 'annotate' ? ':id' : segment));
    const path = `/${folded.join('/')}`;
    const destination = segments[0] === 'project' ? segments[2] : undefined;
    const view = destination ? DEFAULT_VIEW[destination] : undefined;
    const parts = ['step', 'tab', 'stage']
        .filter((name) => !(view && view.param === name && url.searchParams.get(name) === view.value))
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
        if (key === null || screens.includes(key)) return;
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
