/**
 * Zustand store for the per-project "inline tips" switch (the Coach),
 * persisted to localStorage.
 *
 * Tips are on by default for every project. The store holds the explicit
 * per-project overrides; a project not present in ``manualOverrides``
 * falls back to on. (Until Epic U phase U1 the default was gated by the
 * removed gamification level.)
 */

import { create } from 'zustand';

const STORAGE_KEY = 'brewslm:coach-mode-overrides';

type Override = 'on' | 'off';

/** Tips default to on; an override flips them per project. */
export const COACH_DEFAULT_ON = true;

function loadOverrides(): Record<string, Override> {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            return parsed as Record<string, Override>;
        }
    } catch {
        // Corrupt or unavailable — fall back to no overrides.
    }
    return {};
}

function saveOverrides(overrides: Record<string, Override>): void {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    } catch {
        // Quota / SSR / private-mode — silent fail keeps the toggle
        // functional in-memory for the session.
    }
}

interface CoachModeState {
    manualOverrides: Record<string, Override>;
    setOverride: (projectId: number, value: Override) => void;
    clearOverride: (projectId: number) => void;
    isOn: (projectId: number, defaultOn?: boolean) => boolean;
    toggle: (projectId: number, defaultOn?: boolean) => void;
}

export const useCoachModeStore = create<CoachModeState>((set, get) => ({
    manualOverrides: loadOverrides(),

    setOverride: (projectId, value) => {
        set((state) => {
            const next = { ...state.manualOverrides, [String(projectId)]: value };
            saveOverrides(next);
            return { manualOverrides: next };
        });
    },

    clearOverride: (projectId) => {
        set((state) => {
            const next = { ...state.manualOverrides };
            delete next[String(projectId)];
            saveOverrides(next);
            return { manualOverrides: next };
        });
    },

    isOn: (projectId, defaultOn = COACH_DEFAULT_ON) => {
        const override = get().manualOverrides[String(projectId)];
        if (override === 'on') return true;
        if (override === 'off') return false;
        return defaultOn;
    },

    toggle: (projectId, defaultOn = COACH_DEFAULT_ON) => {
        const currentlyOn = get().isOn(projectId, defaultOn);
        get().setOverride(projectId, currentlyOn ? 'off' : 'on');
    },
}));

/**
 * Reset all in-memory + persisted overrides. Test-only helper.
 */
export function _resetCoachModeStoreForTests(): void {
    try {
        localStorage.removeItem(STORAGE_KEY);
    } catch {
        // ignore
    }
    useCoachModeStore.setState({ manualOverrides: {} });
}
