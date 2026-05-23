import { beforeEach, describe, expect, it } from 'vitest';

import {
    _resetCoachModeStoreForTests,
    computeCoachDefaultOn,
    useCoachModeStore,
} from './coachModeStore';

describe('coachModeStore', () => {
    beforeEach(() => {
        _resetCoachModeStoreForTests();
    });

    it('computeCoachDefaultOn returns true for null level (no progression yet)', () => {
        expect(computeCoachDefaultOn(null)).toBe(true);
        expect(computeCoachDefaultOn(undefined)).toBe(true);
    });

    it('computeCoachDefaultOn returns true for level < 3 (onboarding) and false for >= 3', () => {
        expect(computeCoachDefaultOn(1)).toBe(true);
        expect(computeCoachDefaultOn(2)).toBe(true);
        expect(computeCoachDefaultOn(3)).toBe(false);
        expect(computeCoachDefaultOn(5)).toBe(false);
    });

    it('isOn returns the default when no override is set', () => {
        const { isOn } = useCoachModeStore.getState();
        // No override yet — falls back to whatever default we pass in.
        expect(isOn(1, true)).toBe(true);
        expect(isOn(1, false)).toBe(false);
    });

    it('setOverride pins the effective state regardless of default', () => {
        const { setOverride, isOn } = useCoachModeStore.getState();
        setOverride(1, 'on');
        // Default-off but the override wins.
        expect(useCoachModeStore.getState().isOn(1, false)).toBe(true);
        setOverride(1, 'off');
        // Default-on but the override wins.
        expect(useCoachModeStore.getState().isOn(1, true)).toBe(false);
        // Sanity: explicitly called isOn from the destructured handle too.
        expect(isOn(1, true)).toBe(false);
    });

    it('toggle flips from default-on → explicit off, then off → on', () => {
        const { toggle } = useCoachModeStore.getState();
        // Project starts default-on, no override.
        expect(useCoachModeStore.getState().isOn(7, true)).toBe(true);
        toggle(7, true);
        expect(useCoachModeStore.getState().isOn(7, true)).toBe(false);
        toggle(7, true);
        expect(useCoachModeStore.getState().isOn(7, true)).toBe(true);
    });

    it('clearOverride restores default behavior', () => {
        const { setOverride, clearOverride } = useCoachModeStore.getState();
        setOverride(5, 'off');
        expect(useCoachModeStore.getState().isOn(5, true)).toBe(false);
        clearOverride(5);
        expect(useCoachModeStore.getState().isOn(5, true)).toBe(true);
    });

    it('overrides are persisted to localStorage', () => {
        const { setOverride } = useCoachModeStore.getState();
        setOverride(99, 'off');
        const raw = localStorage.getItem('brewslm:coach-mode-overrides');
        expect(raw).not.toBeNull();
        const parsed = JSON.parse(raw!);
        expect(parsed['99']).toBe('off');
    });
});
