/**
 * Zustand store for per-project Coach Mode toggle with XP-gated defaults and localStorage persistence.
 */

import { create } from 'zustand';

// Coach Mode toggle is per-project. The store holds explicit user
// overrides — a project not present in ``manualOverrides`` falls back
// to a level-derived default (computed by callers via
// ``computeCoachDefaultOn(level)``).
//
// We persist overrides to localStorage manually rather than via
// ``zustand/middleware`` to stay consistent with the codebase's
// existing stores (toastStore, projectStore — none use persist()).
//
// Threshold for the XP-gated default (Phase 4 — flipped from 3 to 5):
// - Level < 5 → default ON (the user is still ramping up; Intern
//   through ML Engineer per the gamification level titles).
// - Level >= 5 → default OFF (Senior ML Engineer and above —
//   "power-user" zone).
// Users can always flip the default via the per-project toggle.

const STORAGE_KEY = 'brewslm:coach-mode-overrides';

type Override = 'on' | 'off';

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
    isOn: (projectId: number, defaultOn: boolean) => boolean;
    toggle: (projectId: number, defaultOn: boolean) => void;
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

    isOn: (projectId, defaultOn) => {
        const override = get().manualOverrides[String(projectId)];
        if (override === 'on') return true;
        if (override === 'off') return false;
        return defaultOn;
    },

    toggle: (projectId, defaultOn) => {
        const currentlyOn = get().isOn(projectId, defaultOn);
        get().setOverride(projectId, currentlyOn ? 'off' : 'on');
    },
}));

/**
 * Coach Mode is default-on while the user is still ramping up
 * (gamification level < 5) and default-off once they've crossed into
 * the "Senior" tier. ``null`` (no progression data yet) is treated as
 * "early-stage user, default on" — better to over-coach a brand-new
 * project than to silently leave a beginner without guidance.
 */
export function computeCoachDefaultOn(level: number | null | undefined): boolean {
    if (level == null) return true;
    return level < 5;
}

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
