import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { _resetThemeStoreForTests, initTheme, resolveTheme, THEME_STORAGE_KEY, useThemeStore } from './themeStore';

describe('themeStore', () => {
    beforeEach(() => _resetThemeStoreForTests());
    afterEach(() => _resetThemeStoreForTests());

    it('resolveTheme follows the OS only for "system"', () => {
        expect(resolveTheme('light', true)).toBe('light');
        expect(resolveTheme('dark', false)).toBe('dark');
        expect(resolveTheme('system', true)).toBe('dark');
        expect(resolveTheme('system', false)).toBe('light');
    });

    it('defaults to light and writes the resolved theme onto <html>', () => {
        expect(initTheme()).toBe('light');
        expect(document.documentElement.dataset.theme).toBe('light');
        expect(useThemeStore.getState().preference).toBe('light');
    });

    it('setPreference persists and applies; toggle flips the resolved theme', () => {
        initTheme();
        useThemeStore.getState().setPreference('dark');
        expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
        expect(document.documentElement.dataset.theme).toBe('dark');
        useThemeStore.getState().toggle();
        expect(useThemeStore.getState().resolved).toBe('light');
        expect(document.documentElement.dataset.theme).toBe('light');
    });

    it('initTheme reads a saved preference and ignores junk', () => {
        localStorage.setItem(THEME_STORAGE_KEY, 'dark');
        expect(initTheme()).toBe('dark');
        localStorage.setItem(THEME_STORAGE_KEY, 'neon');
        expect(initTheme()).toBe('light');
    });
});
