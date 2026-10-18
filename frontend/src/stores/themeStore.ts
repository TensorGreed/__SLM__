/**
 * The colour theme (Epic U phase U7): ``light`` (the default), ``dark``,
 * or ``system`` (follow the OS). The resolved value is written to
 * ``<html data-theme="light|dark">`` — ``index.css`` keys the dark token
 * overrides on it. Persisted to localStorage; the switch lives on the
 * project Settings page (Appearance) and in the ⌘K palette.
 */

import { create } from 'zustand';

export type ThemePreference = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

export const THEME_STORAGE_KEY = 'brewslm:theme';
export const THEME_OPTIONS: ReadonlyArray<{ value: ThemePreference; label: string; hint: string }> = [
    { value: 'light', label: 'Light', hint: 'The default.' },
    { value: 'dark', label: 'Dark', hint: 'Dark surfaces, the same layout.' },
    { value: 'system', label: 'Match system', hint: 'Follows the operating system setting.' },
];

const DARK_QUERY = '(prefers-color-scheme: dark)';

export function isThemePreference(value: unknown): value is ThemePreference {
    return value === 'light' || value === 'dark' || value === 'system';
}

function loadPreference(): ThemePreference {
    try {
        const raw = localStorage.getItem(THEME_STORAGE_KEY);
        if (isThemePreference(raw)) return raw;
    } catch {
        // Unavailable storage — light.
    }
    return 'light';
}

function savePreference(value: ThemePreference): void {
    try {
        localStorage.setItem(THEME_STORAGE_KEY, value);
    } catch {
        // Quota / private mode — the in-memory value still applies.
    }
}

function systemPrefersDark(): boolean {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    try {
        return window.matchMedia(DARK_QUERY).matches;
    } catch {
        return false;
    }
}

/** Pure: what a preference resolves to given the OS setting. */
export function resolveTheme(preference: ThemePreference, prefersDark: boolean): ResolvedTheme {
    if (preference === 'system') return prefersDark ? 'dark' : 'light';
    return preference;
}

function applyToDocument(resolved: ResolvedTheme): void {
    if (typeof document === 'undefined') return;
    document.documentElement.dataset.theme = resolved;
}

interface ThemeState {
    preference: ThemePreference;
    resolved: ResolvedTheme;
    setPreference: (value: ThemePreference) => void;
    /** Flip between light and dark (a ``system`` preference becomes the opposite of what it resolves to). */
    toggle: () => void;
}

export const useThemeStore = create<ThemeState>((set, get) => ({
    preference: 'light',
    resolved: 'light',

    setPreference: (value) => {
        const resolved = resolveTheme(value, systemPrefersDark());
        savePreference(value);
        applyToDocument(resolved);
        set({ preference: value, resolved });
    },

    toggle: () => {
        get().setPreference(get().resolved === 'dark' ? 'light' : 'dark');
    },
}));

let mediaListenerBound = false;

/**
 * Read the saved preference, apply it to ``<html>`` and keep a ``system``
 * preference in step with the OS. Called once from ``main.tsx`` before the
 * first render so the page never flashes the wrong theme.
 */
export function initTheme(): ResolvedTheme {
    const preference = loadPreference();
    const resolved = resolveTheme(preference, systemPrefersDark());
    applyToDocument(resolved);
    useThemeStore.setState({ preference, resolved });

    if (!mediaListenerBound && typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
        try {
            const query = window.matchMedia(DARK_QUERY);
            const onChange = () => {
                if (useThemeStore.getState().preference !== 'system') return;
                const next = resolveTheme('system', query.matches);
                applyToDocument(next);
                useThemeStore.setState({ resolved: next });
            };
            if (typeof query.addEventListener === 'function') query.addEventListener('change', onChange);
            mediaListenerBound = true;
        } catch {
            // No matchMedia (jsdom) — the preference still applies.
        }
    }
    return resolved;
}

/** Test-only: clear storage and the document attribute. */
export function _resetThemeStoreForTests(): void {
    try {
        localStorage.removeItem(THEME_STORAGE_KEY);
    } catch {
        // ignore
    }
    if (typeof document !== 'undefined') delete document.documentElement.dataset.theme;
    useThemeStore.setState({ preference: 'light', resolved: 'light' });
}
